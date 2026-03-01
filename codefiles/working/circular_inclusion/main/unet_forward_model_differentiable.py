"""
U-Net Forward Model Testing
Iterate through dataset, predict with U-Net, solve FEM, compute L2 loss
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from fem_utils import *
from unet import *

torch.set_default_dtype(torch.float64)

# ============================================================================
# PARAMETERS
# ============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

dataset_filename = 'angular_scanning_dataset.pt'
output_dir = Path('unet_test_frames')
grid_size = 64

R_outer = 1.0
R_inner = 0.3
n_radial = 20
n_angular = 40

E_b = 1.0
E_i = 10.0
nu = 0.3

print("="*70)
print("U-NET FORWARD MODEL TESTING")
print("="*70)

# ============================================================================
# LOAD DATASET
# ============================================================================
print(f"\nLoading dataset: {dataset_filename}")
dataset = torch.load(dataset_filename)

force_vectors = dataset['force_vectors'].to(device)
boundary_displacements_measured = dataset['boundary_displacements'].to(device)
n_pairs = dataset['n_pairs']
boundary_nodes = dataset['boundary_nodes'].to(device)
points = dataset['points'].to(device)

n_samples = force_vectors.shape[0]
print(f"  Samples: {n_samples}")
print(f"  Boundary nodes: {len(boundary_nodes)}")

# ============================================================================
# CREATE POLAR MESH
# ============================================================================
print("\nCreating polar mesh...")

# Create radial and angular coordinates
r = torch.linspace(0, R_outer, n_radial, dtype=torch.float64, device=device)
theta = torch.linspace(0, 2*torch.pi, n_angular+1, dtype=torch.float64, device=device)[:-1]

R, Theta = torch.meshgrid(r, theta, indexing='ij')
X = R * torch.cos(Theta)
Y = R * torch.sin(Theta)

points = torch.stack([X.ravel(), Y.ravel()], dim=1)

# Handle center point
center_indices = torch.where(torch.sqrt(points[:, 0]**2 + points[:, 1]**2) < 1e-10)[0]
if len(center_indices) > 0:
    points_list = []
    center_added = False
    for i in range(len(points)):
        if torch.sqrt(points[i, 0]**2 + points[i, 1]**2) < 1e-10:
            if not center_added:
                points_list.append(points[i])
                center_added = True
        else:
            points_list.append(points[i])
    points = torch.stack(points_list)

n_nodes = len(points)
n_dof = 2 * n_nodes

print(f"  Nodes: {n_nodes}")

# ============================================================================
# CREATE ELEMENTS
# ============================================================================
print("Creating elements...")

elements = []

def node_idx(i, j):
    if i == 0:
        return 0
    else:
        return 1 + (i - 1) * n_angular + (j % n_angular)

for i in range(n_radial - 1):
    for j in range(n_angular):
        if i == 0:
            n0 = 0
            n1 = node_idx(1, j)
            n2 = node_idx(1, j + 1)
            elements.append([n0, n1, n2])
        else:
            n0 = node_idx(i, j)
            n1 = node_idx(i, j + 1)
            n2 = node_idx(i + 1, j + 1)
            n3 = node_idx(i + 1, j)
            
            elements.append([n0, n1, n2])
            elements.append([n0, n2, n3])

elements = torch.tensor(elements, dtype=torch.long, device=device)
n_elements = len(elements)

print(f"  Elements: {n_elements}")

# Extract element centroids
centroids = torch.mean(points[elements], dim=1)

# ============================================================================
# DIFFERENTIABLE UTILS
# ============================================================================

def batched_element_stiffness(coords, E, nu):
    # coords: (n_elem, 3, 2)
    x1, y1 = coords[:, 0, 0], coords[:, 0, 1]
    x2, y2 = coords[:, 1, 0], coords[:, 1, 1]
    x3, y3 = coords[:, 2, 0], coords[:, 2, 1]

    area = 0.5 * torch.abs((x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1))
    area = torch.clamp(area, min=1e-10)

    # Compute b1..c3 for all elements (shape: n_elem,)
    b1, b2, b3 = y2 - y3, y3 - y1, y1 - y2
    c1, c2, c3 = x3 - x2, x1 - x3, x2 - x1

    # Build B-matrices for all elements: (n_elem, 3, 6)
    B = torch.stack([
        torch.stack([b1, torch.zeros_like(b1), b2, torch.zeros_like(b2), b3, torch.zeros_like(b3)], dim=-1),
        torch.stack([torch.zeros_like(c1), c1, torch.zeros_like(c2), c2, torch.zeros_like(c3), c3], dim=-1),
        torch.stack([c1, b1, c2, b2, c3, b3], dim=-1)
    ], dim=1) / (2 * area).view(-1, 1, 1)

    # Plane stress D-matrices for each element: (n_elem, 3, 3)
    D = (E / (1 - nu**2)).view(-1, 1, 1) * torch.tensor(
        [[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]],
        dtype=torch.float64, device=coords.device
    )

    # Batched K_e = area * Bᵀ D B   → shape (n_elem, 6, 6)
    K_e = area.view(-1, 1, 1) * torch.matmul(B.transpose(1, 2), torch.matmul(D, B))

    return K_e


def assemble_stiffness_differentiable(points, elements, element_materials, nu):
    n_elems = elements.shape[0]
    n_nodes = points.shape[0]
    n_dof = 2 * n_nodes

    # --- 1. Compute all element stiffness matrices (batched) ---
    coords = points[elements]                 # (n_elems, 3, 2)
    E = element_materials.view(-1, 1, 1)      # (n_elems, 1, 1)
    K_e = batched_element_stiffness(coords, E, nu)  # (n_elems, 6, 6)

    # --- 2. Compute global DOF indices for each element ---
    dof_ids = torch.stack([
        2*elements, 2*elements+1
    ], dim=-1).view(n_elems, 6)  # (n_elems, 6)

    # --- 3. Scatter-add all contributions into global K ---
    K = torch.zeros((n_dof, n_dof), dtype=torch.float64, device=points.device)
    for local_i in range(6):
        for local_j in range(6):
            idx_i = dof_ids[:, local_i]
            idx_j = dof_ids[:, local_j]
            K.index_put_(
                (idx_i, idx_j),
                K_e[:, local_i, local_j],
                accumulate=True
            )

    return K

def solve_fem_soft(K, F, fixed_weights, free_weights, n_dof, device, penalty=1e6):
    """
    Differentiable FEM solve with soft boundary constraints.

    Args:
        K: (n_dof, n_dof) global stiffness matrix
        F: (n_dof,) global force vector
        fixed_weights: (n_dof,) weights in [0,1] indicating fixed DOFs (1=fixed)
        free_weights: (n_dof,) weights in [0,1] indicating free DOFs (1=free)
        penalty: scalar multiplier for enforcing displacement ~0 at fixed DOFs

    Returns:
        U: (n_dof,) displacement vector
        U_nodes: (n_nodes, 2) reshaped displacement field
    """

    # Build soft penalty matrix: K_eff = K + diag(penalty * fixed_weights)
    K_eff = K + torch.diag(penalty * fixed_weights)

    # Modify force vector to zero-out fixed DOFs smoothly
    F_eff = F * free_weights

    # Solve full system (no slicing, fully differentiable)
    U = torch.linalg.solve(K_eff, F_eff)

    # Reshape
    n_nodes = n_dof // 2
    U_nodes = U.reshape(n_nodes, 2)

    return U, U_nodes

def soft_min(a, b, temperature=0.1):
    """
    Differentiable soft minimum using LogSumExp trick.
    As temperature → 0, this approaches min(a, b).
    
    Args:
        a, b: scalar tensors
        temperature: smoothness parameter (smaller = closer to true min)
    
    Returns:
        Smooth approximation of min(a, b)
    """
    return -temperature * torch.logsumexp(
        torch.stack([-a/temperature, -b/temperature]), dim=0
    )

# ============================================================================
# INITIALIZE U-NET
# ============================================================================
print("\nInitializing U-Net...")

model = UNet(in_channels=2, out_channels=1, base_features=32)
model = model.to(device).to(torch.float64)
model.eval()

n_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {n_params:,}")

# Create coordinate grid input
x_grid = torch.linspace(-R_outer, R_outer, grid_size, dtype=torch.float64, device=device)
y_grid = torch.linspace(-R_outer, R_outer, grid_size, dtype=torch.float64, device=device)
X_grid, Y_grid = torch.meshgrid(x_grid, y_grid, indexing='ij')
coords_input = torch.stack([X_grid, Y_grid], dim=0).unsqueeze(0)

print(f"  Input shape: {coords_input.shape}")

# ============================================================================
# U-NET FORWARD PASS (ONCE)
# ============================================================================
print("\nRunning U-Net forward pass...")

model.train()                     # enables gradient flow
output_mask = model(coords_input) # NO torch.no_grad()

print(f"  Output shape: {output_mask.shape}")
print(f"  Output range: [{output_mask.min():.4f}, {output_mask.max():.4f}]")

# ============================================================================
# SAMPLE MASK AT ELEMENT CENTROIDS
# ============================================================================
print("\nSampling U-Net mask at element centroids...")

# element centroids (n_elements, 2)
centroids = torch.mean(points[elements], dim=1)  # (n_elem, 2)

# normalize centroids to [-1, 1] for grid_sample
grid = centroids / R_outer                       # (n_elem, 2)
grid = grid.unsqueeze(0).unsqueeze(1)            # (1, n_elem, 1, 2)

# sample differentiably from output_mask (1,1,H,W)
sampled = F.grid_sample(
    output_mask, grid, mode='bilinear', align_corners=True
)  # (1,1,n_elem,1)

element_mask_raw = sampled.view(-1)  # Raw U-Net outputs at element centroids

print(f"  Element mask raw range: [{element_mask_raw.min():.4f}, {element_mask_raw.max():.4f}]")

# ============================================================================
# THRESHOLD AROUND MEAN TO CREATE BINARY MASK
# ============================================================================
print("\nThresholding U-Net output around mean...")

# Compute mean (differentiable)
mask_mean = element_mask_raw.mean()
print(f"  Mask mean: {mask_mean:.4f}")

# Center around mean and apply sigmoid with temperature
temperature = 1000.0  # Higher = sharper binary separation
element_mask_values = torch.sigmoid(temperature * (element_mask_raw - mask_mean))

print(f"  Element mask (after threshold) range: [{element_mask_values.min():.4f}, {element_mask_values.max():.4f}]")

# ============================================================================
# ASSIGN MATERIALS BASED ON U-NET MASK
# ============================================================================
print("\nAssigning materials based on U-Net mask...")

unet_materials = E_b + (E_i - E_b) * element_mask_values

print(f"  Material range: [{unet_materials.min():.4f}, {unet_materials.max():.4f}]")
print(f"  Background elements (E≈{E_b}): {torch.sum(element_mask_values < 0.1)}")
print(f"  Inclusion elements (E≈{E_i}): {torch.sum(element_mask_values > 0.9)}")

# ============================================================================
# ASSEMBLE STIFFNESS MATRIX WITH U-NET MATERIALS
# ============================================================================
print("\nAssembling stiffness matrix with U-Net materials...")

K = assemble_stiffness_differentiable(points, elements, unet_materials, nu)

print(f"  K shape: {K.shape}")
print(f"  K symmetric: {torch.allclose(K, K.T, atol=1e-10)}")

# ============================================================================
# ESTIMATE INCLUSION CENTER AND SIZE FROM U-NET MASK
# ============================================================================
print("\nEstimating inclusion geometry from U-Net mask...")

# Step 1: Find center of inclusion (weighted by mask values)
total_mask = element_mask_values.sum()
center_x = (centroids[:, 0] * element_mask_values).sum() / total_mask
center_y = (centroids[:, 1] * element_mask_values).sum() / total_mask

print(f"  Estimated center: ({center_x:.4f}, {center_y:.4f})")

# Step 2: Compute distance from center to origin
center_dist_to_origin = torch.sqrt(center_x**2 + center_y**2)
print(f"  Distance to origin: {center_dist_to_origin:.4f}")

# Step 3: Compute distance from center to outer boundary
center_dist_to_boundary = R_outer - center_dist_to_origin
print(f"  Distance to boundary: {center_dist_to_boundary:.4f}")

# Step 4: Soft minimum (fully differentiable!)
min_dist = soft_min(center_dist_to_origin, center_dist_to_boundary, temperature=0.1)
fixed_region_radius = min_dist / 2.0

print(f"  Fixed region radius: {fixed_region_radius:.4f}")

# ============================================================================
# APPLY BOUNDARY CONDITIONS BASED ON ESTIMATED INCLUSION
# ============================================================================
print("\nApplying boundary conditions...")

# Compute distance of each node from estimated center
node_distances = torch.sqrt(
    (points[:, 0] - center_x)**2 + 
    (points[:, 1] - center_y)**2
)

# Nodes within fixed_region_radius are fixed (soft transition)
sharpness = 20.0
fixed_node_mask = torch.sigmoid(sharpness * (fixed_region_radius - node_distances))

print(f"  Node mask range: [{fixed_node_mask.min():.4f}, {fixed_node_mask.max():.4f}]")
print(f"  Strongly fixed nodes (>0.9): {(fixed_node_mask > 0.9).sum()}")
print(f"  Strongly free nodes (<0.1): {(fixed_node_mask < 0.1).sum()}")

# Create DOF weights
fixed_weights = fixed_node_mask.unsqueeze(1).repeat(1, 2).view(-1)  # (n_dof,)
free_weights = 1.0 - fixed_weights                                   # (n_dof,)

print(f"  Mean fixed weight: {fixed_weights.mean():.4f}")
print(f"  Mean free weight: {free_weights.mean():.4f}")

# ============================================================================
# GROUND TRUTH VISUALIZATION WITH SSIM
# ============================================================================
print("\n" + "="*70)
print("GROUND TRUTH VISUALIZATION")
print("="*70)

print("\nCreating ground truth materials...")

# Assign ground truth materials to elements
gt_materials = torch.zeros(n_elements, dtype=torch.float64, device=device)
for i in range(n_elements):
    elem = elements[i]
    centroid = torch.mean(points[elem], dim=0)
    dist_to_center = torch.sqrt(centroid[0]**2 + centroid[1]**2)
    
    if dist_to_center <= R_inner:
        gt_materials[i] = E_i
    else:
        gt_materials[i] = E_b

print(f"  Background elements (E={E_b}): {torch.sum(gt_materials == E_b)}")
print(f"  Inclusion elements (E={E_i}): {torch.sum(gt_materials == E_i)}")

# ============================================================================
# COMPUTE SSIM BETWEEN GROUND TRUTH AND U-NET PREDICTION
# ============================================================================
print("\nComputing SSIM...")

# Normalize materials to [0, 1] range
gt_norm = (gt_materials - E_b) / (E_i - E_b)
pred_norm = (unet_materials - E_b) / (E_i - E_b)

# SSIM components
C1 = 0.01 ** 2
C2 = 0.03 ** 2

mu_gt = gt_norm.mean()
mu_pred = pred_norm.mean()

sigma_gt = ((gt_norm - mu_gt) ** 2).mean()
sigma_pred = ((pred_norm - mu_pred) ** 2).mean()
sigma_gt_pred = ((gt_norm - mu_gt) * (pred_norm - mu_pred)).mean()

# SSIM formula
luminance = (2 * mu_gt * mu_pred + C1) / (mu_gt ** 2 + mu_pred ** 2 + C1)
contrast = (2 * torch.sqrt(sigma_gt) * torch.sqrt(sigma_pred) + C2) / (sigma_gt + sigma_pred + C2)
structure = (sigma_gt_pred + C2 / 2) / (torch.sqrt(sigma_gt) * torch.sqrt(sigma_pred) + C2 / 2)

ssim_value = (luminance * contrast * structure).item()

print(f"  SSIM: {ssim_value:.4f}")

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\nCreating visualization...")

# Convert to numpy for plotting
points_np = points.cpu().numpy()
elements_np = elements.cpu().numpy()
gt_materials_np = gt_materials.cpu().numpy()
unet_materials_np = unet_materials.detach().cpu().numpy()
element_mask_values_np = element_mask_values.detach().cpu().numpy()
fixed_node_mask_np = fixed_node_mask.detach().cpu().numpy()

fig, axes = plt.subplots(1, 3, figsize=(24, 7))

# Left panel: Ground truth material properties
ax = axes[0]

# Plot ground truth mesh
for e, elem in enumerate(elements_np):
    elem_coords = points_np[elem]
    color = 'red' if gt_materials_np[e] == E_i else 'blue'
    tri = plt.Polygon(elem_coords, facecolor=color, edgecolor='black', 
                      alpha=0.5, linewidth=0.3)
    ax.add_patch(tri)

# Draw boundaries
theta_plot = np.linspace(0, 2*np.pi, 100)
ax.plot(R_outer*np.cos(theta_plot), R_outer*np.sin(theta_plot), 'k-', 
        linewidth=2, label='Outer boundary')
ax.plot(R_inner*np.cos(theta_plot), R_inner*np.sin(theta_plot), 'r--', 
        linewidth=2, label='Inclusion boundary', alpha=0.7)

ax.set_xlim(-R_outer*1.1, R_outer*1.1)
ax.set_ylim(-R_outer*1.1, R_outer*1.1)
ax.set_aspect('equal')
ax.set_title('Ground Truth Material Properties\n(Red=Inclusion E=10, Blue=Background E=1)', 
             fontsize=12, fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
ax.grid(True, alpha=0.3)

# Middle panel: U-Net predicted binary mask
ax = axes[1]
for e, elem in enumerate(elements_np):
    elem_coords = points_np[elem]
    # Color by thresholded mask value
    mask_val = element_mask_values_np[e]
    color = 'red' if mask_val > 0.5 else 'blue'
    tri = plt.Polygon(elem_coords, facecolor=color, edgecolor='black', 
                      alpha=0.5, linewidth=0.3)
    ax.add_patch(tri)

# Draw estimated inclusion region
center_x_np = center_x.detach().cpu().numpy()
center_y_np = center_y.detach().cpu().numpy()
fixed_region_radius_np = fixed_region_radius.detach().cpu().numpy()

theta_plot = np.linspace(0, 2*np.pi, 100)
estimated_circle_x = center_x_np + fixed_region_radius_np * np.cos(theta_plot)
estimated_circle_y = center_y_np + fixed_region_radius_np * np.sin(theta_plot)

ax.plot(R_outer*np.cos(theta_plot), R_outer*np.sin(theta_plot), 'k-', 
        linewidth=2, label='Outer boundary')
ax.plot(estimated_circle_x, estimated_circle_y, 'g--', 
        linewidth=2, label=f'Fixed region (r={fixed_region_radius_np:.3f})', alpha=0.7)
ax.scatter([center_x_np], [center_y_np], c='green', s=200, marker='x', 
           linewidths=3, label='Estimated center', zorder=10)

ax.set_xlim(-R_outer*1.1, R_outer*1.1)
ax.set_ylim(-R_outer*1.1, R_outer*1.1)
ax.set_aspect('equal')
ax.set_title('U-Net Binary Mask + Estimated Fixed Region\n(Red=Inclusion, Blue=Background)', 
             fontsize=12, fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
ax.grid(True, alpha=0.3)

# Right panel: U-Net predicted materials
ax = axes[2]
for e, elem in enumerate(elements_np):
    elem_coords = points_np[elem]
    # Color by continuous material value
    mat_val = unet_materials_np[e]
    intensity = (mat_val - E_b) / (E_i - E_b)
    color = plt.cm.RdYlBu_r(intensity)
    tri = plt.Polygon(elem_coords, facecolor=color, edgecolor='black', 
                      alpha=0.7, linewidth=0.3)
    ax.add_patch(tri)

# Draw boundaries
theta_plot = np.linspace(0, 2*np.pi, 100)
ax.plot(R_outer*np.cos(theta_plot), R_outer*np.sin(theta_plot), 'k-', 
        linewidth=2, label='Outer boundary')

ax.set_xlim(-R_outer*1.1, R_outer*1.1)
ax.set_ylim(-R_outer*1.1, R_outer*1.1)
ax.set_aspect('equal')
ax.set_title('U-Net Predicted Materials (Continuous)\n(Blue=E≈1, Red=E≈10)', 
             fontsize=12, fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
ax.grid(True, alpha=0.3)

# Add colorbar for right panel
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
sm = ScalarMappable(cmap=plt.cm.RdYlBu_r, norm=Normalize(vmin=E_b, vmax=E_i))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Young's Modulus (E)", rotation=270, labelpad=20)

# Add main title with SSIM
fig.suptitle(f'Material Property Segmentation | SSIM: {ssim_value:.4f}', 
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout()
plt.savefig('unet_forward_pass_viz.png', dpi=150, bbox_inches='tight')
print("Saved: unet_forward_pass_viz.png")

plt.close()

# ============================================================================
# ITERATE THROUGH DATASET AND COMPUTE LOSSES
# ============================================================================
print("\n" + "="*70)
print("TESTING ON DATASET")
print("="*70)

output_dir.mkdir(exist_ok=True)
print(f"\nOutput directory: {output_dir}")

l2_losses = []
frame_files = []

for i in range(n_samples):
    print(f"\n[Sample {i+1}/{n_samples}] n_pairs = {n_pairs[i].item()}")
    
    # Get force vector for this sample
    F = force_vectors[i]
    
    # Get measured boundary displacements
    boundary_disp_measured = boundary_displacements_measured[i]  # (n_boundary, 2)
    
    # Solve FEM with U-Net materials
    U, U_nodes = solve_fem_soft(K, F, fixed_weights, free_weights, n_dof, device)
    
    # Get predicted boundary displacements
    boundary_disp_predicted = U_nodes[boundary_nodes]  # (n_boundary, 2)
    
    # Compute L2 loss (scaled by R_outer and number of boundary nodes)
    diff = (boundary_disp_predicted - boundary_disp_measured) / R_outer
    l2_loss = torch.sqrt(torch.sum(diff ** 2) / len(boundary_nodes))
    l2_losses.append(l2_loss)
    
    print(f"  L2 loss: {l2_loss:.3f}")
    
    # Create visualization
    frame_filename = output_dir / f'sample_{i:03d}_pairs_{n_pairs[i].item():02d}.png'
    
    # Reconstruct force node pairs from force vector
    F_np = F.cpu().numpy()
    F_2d = F_np.reshape(-1, 2)
    force_magnitudes = np.sqrt(F_2d[:, 0]**2 + F_2d[:, 1]**2)
    force_node_indices = np.where(force_magnitudes > 1e-10)[0]
    
    force_nodes_list = []
    for j in range(0, len(force_node_indices), 2):
        if j + 1 < len(force_node_indices):
            node1 = force_node_indices[j]
            node2 = force_node_indices[j + 1]
            force_nodes_list.append((node1, node2))
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    # Reference circles
    theta_plot = np.linspace(0, 2*np.pi, 200)
    ax.plot(R_outer*np.cos(theta_plot), R_outer*np.sin(theta_plot), 'k-', 
            linewidth=1.5, alpha=0.3, label='Outer boundary (reference)', zorder=1)
    
    # Draw estimated fixed region
    estimated_circle_x = center_x_np + fixed_region_radius_np * np.cos(theta_plot)
    estimated_circle_y = center_y_np + fixed_region_radius_np * np.sin(theta_plot)
    ax.plot(estimated_circle_x, estimated_circle_y, 'g-', 
            linewidth=2, alpha=0.5, label='Estimated fixed region', zorder=2)
    
    # Convert to numpy
    points_np = points.cpu().numpy()
    boundary_nodes_np = boundary_nodes.cpu().numpy()
    boundary_positions = points_np[boundary_nodes_np]
    
    measured_np = boundary_disp_measured.cpu().numpy()
    predicted_np = boundary_disp_predicted.detach().cpu().numpy()
    
    # Original boundary nodes
    ax.scatter(boundary_positions[:, 0], boundary_positions[:, 1], 
               c='gray', s=60, alpha=0.4, label='Original boundary nodes', 
               zorder=3, marker='o')
    
    # Deformed boundary - measured (green)
    deformed_measured = boundary_positions + measured_np
    ax.scatter(deformed_measured[:, 0], deformed_measured[:, 1], 
               c='green', s=80, alpha=0.8, label='Measured boundary', 
               zorder=5, marker='o', edgecolors='darkgreen', linewidths=2)
    
    # Deformed boundary line - measured
    angles = np.arctan2(boundary_positions[:, 1], boundary_positions[:, 0])
    sort_idx = np.argsort(angles)
    deformed_measured_sorted = deformed_measured[sort_idx]
    deformed_measured_closed = np.vstack([deformed_measured_sorted, deformed_measured_sorted[0]])
    
    ax.plot(deformed_measured_closed[:, 0], deformed_measured_closed[:, 1], 
            'g-', linewidth=2, alpha=0.6, label='Measured boundary (line)', zorder=4.5)
    
    # Deformed boundary - predicted (blue)
    deformed_predicted = boundary_positions + predicted_np
    ax.scatter(deformed_predicted[:, 0], deformed_predicted[:, 1], 
               c='blue', s=80, alpha=0.8, label='Predicted boundary', 
               zorder=4, marker='s', edgecolors='darkblue', linewidths=2)
    
    # Deformed boundary line - predicted
    deformed_predicted_sorted = deformed_predicted[sort_idx]
    deformed_predicted_closed = np.vstack([deformed_predicted_sorted, deformed_predicted_sorted[0]])
    
    ax.plot(deformed_predicted_closed[:, 0], deformed_predicted_closed[:, 1], 
            'b--', linewidth=2, alpha=0.6, label='Predicted boundary (line)', zorder=3.5)
    
    # Force nodes and vectors (different colors)
    force_colors = ['red', 'orange', 'purple', 'brown', 'pink', 'olive', 'cyan', 'magenta']
    force_scale = 3.0
    
    for pair_idx, (node1, node2) in enumerate(force_nodes_list):
        pos1_np = points_np[node1]
        pos2_np = points_np[node2]
        
        color = force_colors[pair_idx % len(force_colors)]
        
        # Force nodes
        ax.scatter([pos1_np[0], pos2_np[0]], [pos1_np[1], pos2_np[1]], 
                   c=color, s=300, zorder=6, 
                   edgecolors='darkred', linewidths=3, marker='o',
                   label=f'Force pair {pair_idx+1}' if pair_idx == 0 else '')
        
        # Force vectors
        force1_np = np.array([F_np[2*node1], F_np[2*node1+1]])
        force2_np = np.array([F_np[2*node2], F_np[2*node2+1]])
        
        ax.arrow(pos1_np[0], pos1_np[1], force_scale*force1_np[0], force_scale*force1_np[1],
                 head_width=0.15, head_length=0.12, fc=color, ec='darkred', 
                 linewidth=3, zorder=7)
        ax.arrow(pos2_np[0], pos2_np[1], force_scale*force2_np[0], force_scale*force2_np[1],
                 head_width=0.15, head_length=0.12, fc=color, ec='darkred', 
                 linewidth=3, zorder=7)
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.set_title(f'Sample {i+1}/{n_samples} | n_pairs={n_pairs[i].item()} | L2 Loss: {l2_loss:.3f}', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(frame_filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    frame_files.append(str(frame_filename))
    print(f"  Saved: {frame_filename.name}")

# ============================================================================
# COMPUTE AVERAGE LOSS
# ============================================================================
print("\n" + "="*70)
print("RESULTS")
print("="*70)

# --- Average loss across all force instances ---
loss = torch.stack(l2_losses).mean()

print(f"Total averaged loss: {loss.item():.6f}")

print(f"\nIndividual L2 losses:")
for i, l in enumerate(l2_losses):
    print(f"  Sample {i+1}: {l.item():.3f}")

print(f"\nAll {len(frame_files)} frames saved to {output_dir}/")

print("\n" + "="*70)
print("DONE")
print("="*70)
