"""
U-Net Forward Model Testing
Iterate through dataset, predict with U-Net, solve FEM, compute L2 loss
"""

import torch
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
centroids = torch.zeros((n_elements, 2), dtype=torch.float64, device=device)
for i in range(n_elements):
    elem = elements[i]
    centroids[i] = torch.mean(points[elem], dim=0)

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

with torch.no_grad():
    output_mask = model(coords_input)

print(f"  Output shape: {output_mask.shape}")
print(f"  Output range: [{output_mask.min():.4f}, {output_mask.max():.4f}]")

# ============================================================================
# SAMPLE MASK AT ELEMENT CENTROIDS
# ============================================================================
print("\nSampling U-Net mask at element centroids...")

mask_2d = output_mask.squeeze()
element_mask_values = torch.zeros(n_elements, dtype=torch.float64, device=device)

x_min, x_max = -R_outer, R_outer
y_min, y_max = -R_outer, R_outer

for i in range(n_elements):
    centroid = centroids[i]
    
    x_idx = int(((centroid[0] - x_min) / (x_max - x_min) * (grid_size - 1)).item())
    y_idx = int(((centroid[1] - y_min) / (y_max - y_min) * (grid_size - 1)).item())
    
    x_idx = max(0, min(grid_size - 1, x_idx))
    y_idx = max(0, min(grid_size - 1, y_idx))
    
    element_mask_values[i] = mask_2d[y_idx, x_idx]

print(f"  Element mask value range: [{element_mask_values.min():.4f}, {element_mask_values.max():.4f}]")

# ============================================================================
# ASSIGN MATERIALS BASED ON U-NET MASK
# ============================================================================
print("\nAssigning materials based on U-Net mask...")

threshold = element_mask_values.median().item()
print(f"  Using median threshold: {threshold:.4f}")

above_threshold = element_mask_values > threshold
below_threshold = ~above_threshold

n_above = above_threshold.sum().item()
n_below = below_threshold.sum().item()

print(f"  Elements above threshold: {n_above}")
print(f"  Elements below threshold: {n_below}")

if n_above < n_below:
    print(f"  Assigning inclusion (E={E_i}) to above-threshold elements")
    unet_materials = torch.where(above_threshold,
                              torch.tensor(E_i, device=device), 
                              torch.tensor(E_b, device=device))
else:
    print(f"  Assigning inclusion (E={E_i}) to below-threshold elements")
    unet_materials = torch.where(below_threshold,
                              torch.tensor(E_i, device=device), 
                              torch.tensor(E_b, device=device))

print(f"  Background elements (E={E_b}): {torch.sum(unet_materials == E_b)}")
print(f"  Inclusion elements (E={E_i}): {torch.sum(unet_materials == E_i)}")

# ============================================================================
# ASSEMBLE STIFFNESS MATRIX WITH U-NET MATERIALS
# ============================================================================
print("\nAssembling stiffness matrix with U-Net materials...")

K = assemble_stiffness(points, elements, unet_materials, nu)

print(f"  K shape: {K.shape}")
print(f"  K symmetric: {torch.allclose(K, K.T, atol=1e-10)}")

# ============================================================================
# GET INCLUSION NODES FROM MATERIAL PROPERTIES
# ============================================================================
print("\nIdentifying inclusion nodes from material properties...")

# Find elements with inclusion material
inclusion_elements = torch.where(unet_materials == E_i)[0]

# Get all nodes from inclusion elements
inclusion_nodes_set = set()
for elem_idx in inclusion_elements:
    elem = elements[elem_idx]
    for node in elem:
        inclusion_nodes_set.add(node.item())

inclusion_nodes = torch.tensor(list(inclusion_nodes_set), dtype=torch.long, device=device)
print(f"  Inclusion nodes: {len(inclusion_nodes)}")

# Setup boundary conditions
fixed_dofs_list = []
for node in inclusion_nodes:
    fixed_dofs_list.append(2 * node)
    fixed_dofs_list.append(2 * node + 1)

fixed_dofs = torch.tensor(fixed_dofs_list, dtype=torch.long, device=device)
all_dofs = torch.arange(n_dof, device=device)
free_dofs = torch.tensor([dof for dof in all_dofs if dof not in fixed_dofs], 
                        dtype=torch.long, device=device)

print(f"  Fixed DOFs: {len(fixed_dofs)}")
print(f"  Free DOFs: {len(free_dofs)}")

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
unet_materials_np = unet_materials.cpu().numpy()

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

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

# Right panel: U-Net predicted materials
ax = axes[1]
for e, elem in enumerate(elements_np):
    elem_coords = points_np[elem]
    color = 'red' if unet_materials_np[e] == E_i else 'blue'
    tri = plt.Polygon(elem_coords, facecolor=color, edgecolor='black', 
                      alpha=0.5, linewidth=0.3)
    ax.add_patch(tri)

# Draw boundaries
theta_plot = np.linspace(0, 2*np.pi, 100)
ax.plot(R_outer*np.cos(theta_plot), R_outer*np.sin(theta_plot), 'k-', 
        linewidth=2, label='Outer boundary')

ax.set_xlim(-R_outer*1.1, R_outer*1.1)
ax.set_ylim(-R_outer*1.1, R_outer*1.1)
ax.set_aspect('equal')
ax.set_title('U-Net Predicted Materials (Random Init)\n(Red=Inclusion E=10, Blue=Background E=1)', 
             fontsize=12, fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
ax.grid(True, alpha=0.3)

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
    U, U_nodes = solve_fem(K, F, fixed_dofs, free_dofs, n_dof, device)
    
    # Get predicted boundary displacements
    boundary_disp_predicted = U_nodes[boundary_nodes]  # (n_boundary, 2)
    
    # Compute L2 loss (scaled by R_outer and number of boundary nodes)
    diff = (boundary_disp_predicted - boundary_disp_measured) / R_outer
    l2_loss = torch.sqrt(torch.sum(diff ** 2) / len(boundary_nodes)).item()
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
    ax.plot(R_inner*np.cos(theta_plot), R_inner*np.sin(theta_plot), 'r-', 
            linewidth=2, alpha=0.5, label='Inclusion (fixed)', zorder=2)
    
    # Convert to numpy
    points_np = points.cpu().numpy()
    boundary_nodes_np = boundary_nodes.cpu().numpy()
    boundary_positions = points_np[boundary_nodes_np]
    
    measured_np = boundary_disp_measured.cpu().numpy()
    predicted_np = boundary_disp_predicted.cpu().numpy()
    
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

average_l2_loss = np.mean(l2_losses)

print(f"\nIndividual L2 losses:")
for i, loss in enumerate(l2_losses):
    print(f"  Sample {i+1}: {loss:.3f}")

print(f"\nAverage L2 Loss: {average_l2_loss:.3f}")

print(f"\nAll {len(frame_files)} frames saved to {output_dir}/")

print("\n" + "="*70)
print("DONE")
print("="*70)