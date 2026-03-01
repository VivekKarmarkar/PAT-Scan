"""
U-Net Training Script for Inverse Problem
Train U-Net to predict material properties from boundary measurements
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from fem_utils import *
from unet import *

torch.set_default_dtype(torch.float64)

# ============================================================================
# PARAMETERS
# ============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Dataset
dataset_filename = 'angular_scanning_dataset.pt'

# Geometry
grid_size = 64
R_outer = 1.0
R_inner = 0.3
n_radial = 20
n_angular = 40

# Material properties
E_b = 1.0
E_i = 10.0
nu = 0.3

# Training
num_iterations = 1000
learning_rate = 1e-3  # Increased from 1e-4 for more aggressive exploration
log_every = 10
checkpoint_every = 100

# Loss function
# Using square root of L2 loss (more balanced gradients than squared L2)

# Regularization
lambda_tv = 0.01  # Total variation regularization weight (encourages sharp boundaries)

# Thresholding
temperature = 1000.0
bc_sharpness = 20.0
soft_min_temp = 0.1

print("="*70)
print("U-NET INVERSE PROBLEM TRAINING")
print("="*70)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def soft_min(a, b, temperature=0.1):
    """Differentiable soft minimum using LogSumExp"""
    return -temperature * torch.logsumexp(
        torch.stack([-a/temperature, -b/temperature]), dim=0
    )

def batched_element_stiffness(coords, E, nu):
    """Compute element stiffness matrices for all elements (batched)"""
    x1, y1 = coords[:, 0, 0], coords[:, 0, 1]
    x2, y2 = coords[:, 1, 0], coords[:, 1, 1]
    x3, y3 = coords[:, 2, 0], coords[:, 2, 1]

    area = 0.5 * torch.abs((x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1))
    area = torch.clamp(area, min=1e-10)

    b1, b2, b3 = y2 - y3, y3 - y1, y1 - y2
    c1, c2, c3 = x3 - x2, x1 - x3, x2 - x1

    B = torch.stack([
        torch.stack([b1, torch.zeros_like(b1), b2, torch.zeros_like(b2), b3, torch.zeros_like(b3)], dim=-1),
        torch.stack([torch.zeros_like(c1), c1, torch.zeros_like(c2), c2, torch.zeros_like(c3), c3], dim=-1),
        torch.stack([c1, b1, c2, b2, c3, b3], dim=-1)
    ], dim=1) / (2 * area).view(-1, 1, 1)

    D = (E / (1 - nu**2)).view(-1, 1, 1) * torch.tensor(
        [[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]],
        dtype=torch.float64, device=coords.device
    )

    K_e = area.view(-1, 1, 1) * torch.matmul(B.transpose(1, 2), torch.matmul(D, B))
    return K_e

def assemble_stiffness_differentiable(points, elements, element_materials, nu):
    """Assemble global stiffness matrix (differentiable)"""
    n_elems = elements.shape[0]
    n_nodes = points.shape[0]
    n_dof = 2 * n_nodes

    coords = points[elements]
    E = element_materials.view(-1, 1, 1)
    K_e = batched_element_stiffness(coords, E, nu)

    dof_ids = torch.stack([
        2*elements, 2*elements+1
    ], dim=-1).view(n_elems, 6)

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
    """Differentiable FEM solve with soft boundary constraints"""
    K_eff = K + torch.diag(penalty * fixed_weights)
    F_eff = F * free_weights
    U = torch.linalg.solve(K_eff, F_eff)
    n_nodes = n_dof // 2
    U_nodes = U.reshape(n_nodes, 2)
    return U, U_nodes

def sample_mask_at_centroids(output_mask, centroids, R_outer):
    """Sample U-Net output at element centroids using grid_sample"""
    grid = centroids / R_outer
    grid = grid.unsqueeze(0).unsqueeze(1)
    sampled = F.grid_sample(
        output_mask, grid, mode='bilinear', align_corners=True
    )
    return sampled.view(-1)

def threshold_mask(element_mask_raw, temperature):
    """Threshold mask around mean with sigmoid"""
    mask_mean = element_mask_raw.mean()
    element_mask_values = torch.sigmoid(temperature * (element_mask_raw - mask_mean))
    return element_mask_values

def estimate_inclusion_geometry(element_mask_values, centroids, R_outer):
    """Estimate inclusion center and safe fixed region radius"""
    total_mask = element_mask_values.sum()
    center_x = (centroids[:, 0] * element_mask_values).sum() / total_mask
    center_y = (centroids[:, 1] * element_mask_values).sum() / total_mask
    
    center_dist_to_origin = torch.sqrt(center_x**2 + center_y**2)
    center_dist_to_boundary = R_outer - center_dist_to_origin
    
    min_dist = soft_min(center_dist_to_origin, center_dist_to_boundary, temperature=soft_min_temp)
    fixed_region_radius = min_dist / 2.0
    
    return center_x, center_y, fixed_region_radius

def compute_boundary_conditions(points, center_x, center_y, fixed_region_radius, sharpness=20.0):
    """Compute soft boundary condition weights based on distance from center"""
    node_distances = torch.sqrt(
        (points[:, 0] - center_x)**2 + 
        (points[:, 1] - center_y)**2
    )
    
    fixed_node_mask = torch.sigmoid(sharpness * (fixed_region_radius - node_distances))
    fixed_weights = fixed_node_mask.unsqueeze(1).repeat(1, 2).view(-1)
    free_weights = 1.0 - fixed_weights
    
    return fixed_weights, free_weights

def compute_ssim(gt_materials, pred_materials, E_b, E_i):
    """Compute SSIM between ground truth and predicted materials"""
    gt_norm = (gt_materials - E_b) / (E_i - E_b)
    pred_norm = (pred_materials - E_b) / (E_i - E_b)
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    mu_gt = gt_norm.mean()
    mu_pred = pred_norm.mean()
    
    sigma_gt = ((gt_norm - mu_gt) ** 2).mean()
    sigma_pred = ((pred_norm - mu_pred) ** 2).mean()
    sigma_gt_pred = ((gt_norm - mu_gt) * (pred_norm - mu_pred)).mean()
    
    luminance = (2 * mu_gt * mu_pred + C1) / (mu_gt ** 2 + mu_pred ** 2 + C1)
    contrast = (2 * torch.sqrt(sigma_gt) * torch.sqrt(sigma_pred) + C2) / (sigma_gt + sigma_pred + C2)
    structure = (sigma_gt_pred + C2 / 2) / (torch.sqrt(sigma_gt) * torch.sqrt(sigma_pred) + C2 / 2)
    
    ssim = luminance * contrast * structure
    return ssim

def compute_tv_loss(output_mask):
    """
    Compute Total Variation loss on the U-Net output mask.
    TV encourages sharp boundaries by penalizing spatial gradients.
    
    TV = sum(|grad_x|) + sum(|grad_y|)
    
    Args:
        output_mask: [1, 1, H, W] U-Net output
    
    Returns:
        tv_loss: scalar tensor
    """
    # Compute spatial gradients
    diff_x = output_mask[:, :, :, 1:] - output_mask[:, :, :, :-1]  # Horizontal differences
    diff_y = output_mask[:, :, 1:, :] - output_mask[:, :, :-1, :]  # Vertical differences
    
    # Total variation = sum of absolute gradients
    tv_loss = torch.mean(torch.abs(diff_x)) + torch.mean(torch.abs(diff_y))
    
    return tv_loss

# ============================================================================
# FORWARD PASS AND LOSS COMPUTATION
# ============================================================================

def forward_pass_and_loss(model, coords_input, points, elements, centroids,
                         force_vectors, boundary_displacements_measured,
                         boundary_nodes, gt_materials, n_samples, n_dof, device, lambda_tv):
    """
    Complete forward pass: U-Net → materials → FEM → loss
    
    Returns:
        total_loss: scalar tensor for backprop
        metrics: dict with logging metrics
    """
    
    # 1. U-Net forward pass
    output_mask = model(coords_input)
    
    # 2. Sample at element centroids
    element_mask_raw = sample_mask_at_centroids(output_mask, centroids, R_outer)
    
    # 3. Threshold around mean
    element_mask_values = threshold_mask(element_mask_raw, temperature)
    
    # 4. Assign materials
    unet_materials = E_b + (E_i - E_b) * element_mask_values
    
    # 5. Assemble stiffness matrix
    K = assemble_stiffness_differentiable(points, elements, unet_materials, nu)
    
    # 6. Estimate inclusion geometry and boundary conditions
    center_x, center_y, fixed_region_radius = estimate_inclusion_geometry(
        element_mask_values, centroids, R_outer
    )
    fixed_weights, free_weights = compute_boundary_conditions(
        points, center_x, center_y, fixed_region_radius, bc_sharpness
    )
    
    # 7. Solve FEM for all force cases and compute loss
    total_loss = 0.0
    for i in range(n_samples):
        U, U_nodes = solve_fem_soft(
            K, force_vectors[i], fixed_weights, free_weights, n_dof, device
        )
        boundary_disp_predicted = U_nodes[boundary_nodes]
        diff = (boundary_disp_predicted - boundary_displacements_measured[i]) / R_outer
        
        # Square root of L2 loss per sample
        # This provides more balanced gradients than squared L2
        l2_loss = torch.sqrt(torch.sum(torch.square(diff))) / len(boundary_nodes)
        total_loss += l2_loss
    
    # Average loss over all samples
    total_loss = total_loss / n_samples
    
    # Add TV regularization to encourage sharp boundaries
    tv_loss = compute_tv_loss(output_mask)
    total_loss = total_loss + lambda_tv * tv_loss
    
    # 8. Compute metrics for logging
    ssim = compute_ssim(gt_materials, unet_materials, E_b, E_i)
    
    metrics = {
        'loss': total_loss.item(),
        'data_loss': (total_loss - lambda_tv * tv_loss).item(),
        'tv_loss': tv_loss.item(),
        'ssim': ssim.item(),
        'center_x': center_x.item(),
        'center_y': center_y.item(),
        'radius': fixed_region_radius.item(),
        'mean_fixed_weight': fixed_weights.mean().item(),
        'material_range': (unet_materials.min().item(), unet_materials.max().item()),
        'n_background': (element_mask_values < 0.1).sum().item(),
        'n_inclusion': (element_mask_values > 0.9).sum().item(),
    }
    
    return total_loss, metrics

# ============================================================================
# LOAD DATASET
# ============================================================================
print(f"\nLoading dataset: {dataset_filename}")
dataset = torch.load(dataset_filename)

force_vectors = dataset['force_vectors'].to(device)
boundary_displacements_measured = dataset['boundary_displacements'].to(device)
n_pairs = dataset['n_pairs']
boundary_nodes = dataset['boundary_nodes'].to(device)
points_dataset = dataset['points'].to(device)

n_samples = force_vectors.shape[0]
print(f"  Samples: {n_samples}")
print(f"  Boundary nodes: {len(boundary_nodes)}")

# ============================================================================
# CREATE MESH
# ============================================================================
print("\nCreating polar mesh...")

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

# Compute element centroids
centroids = torch.mean(points[elements], dim=1)

# ============================================================================
# CREATE GROUND TRUTH MATERIALS
# ============================================================================
print("\nCreating ground truth materials...")

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
# INITIALIZE U-NET
# ============================================================================
print("\nInitializing U-Net...")

model = UNet(in_channels=2, out_channels=1, base_features=32)
model = model.to(device).to(torch.float64)
model.train()

n_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {n_params:,}")

# Create coordinate grid input
x_grid = torch.linspace(-R_outer, R_outer, grid_size, dtype=torch.float64, device=device)
y_grid = torch.linspace(-R_outer, R_outer, grid_size, dtype=torch.float64, device=device)
X_grid, Y_grid = torch.meshgrid(x_grid, y_grid, indexing='ij')
coords_input = torch.stack([X_grid, Y_grid], dim=0).unsqueeze(0)

print(f"  Input shape: {coords_input.shape}")

# ============================================================================
# SETUP OPTIMIZER AND SCHEDULER
# ============================================================================
print("\nSetting up optimizer...")
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
print(f"  Optimizer: Adam")
print(f"  Initial learning rate: {learning_rate}")
print(f"  Loss function: Square root of L2 (balanced gradients)")
print(f"  Regularization: TV (lambda={lambda_tv}) - encourages sharp boundaries")

# Two-phase learning: constant LR for first 500 iters, then exponential decay
# Phase 1 (0-500): Explore aggressively to find solution
# Phase 2 (500-1000): Fine-tune with decaying LR for precision
final_lr = 1e-5
decay_start = 500
decay_iterations = num_iterations - decay_start
gamma = np.exp(np.log(final_lr / learning_rate) / decay_iterations)

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda iteration: 1.0 if iteration < decay_start else gamma ** (iteration - decay_start)
)

print(f"  Phase 1 (iter 0-{decay_start}): Constant LR = {learning_rate}")
print(f"  Phase 2 (iter {decay_start}-{num_iterations}): Decay to {final_lr}")
print(f"  Decay gamma: {gamma:.6f}")

# ============================================================================
# TRAINING LOOP
# ============================================================================
print("\n" + "="*70)
print("TRAINING")
print("="*70)
print(f"  Iterations: {num_iterations}")
print(f"  Log every: {log_every}")
print(f"  Checkpoint every: {checkpoint_every}")
print()

# Create checkpoint directory
checkpoint_dir = Path('checkpoints')
checkpoint_dir.mkdir(exist_ok=True)

# Training loop
for iteration in range(num_iterations):
    # Forward pass and compute loss
    loss, metrics = forward_pass_and_loss(
        model, coords_input, points, elements, centroids,
        force_vectors, boundary_displacements_measured,
        boundary_nodes, gt_materials, n_samples, n_dof, device, lambda_tv
    )
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    # Logging
    if iteration % log_every == 0:
        current_lr = scheduler.get_last_lr()[0]
        print(f"Iter {iteration:4d} | "
              f"Loss: {metrics['loss']:.6f} | "
              f"Data: {metrics['data_loss']:.6f} | "
              f"TV: {metrics['tv_loss']:.6f} | "
              f"SSIM: {metrics['ssim']:.4f} | "
              f"Center: ({metrics['center_x']:+.3f}, {metrics['center_y']:+.3f}) | "
              f"Radius: {metrics['radius']:.3f} | "
              f"LR: {current_lr:.2e}")
    
    # Checkpointing
    if iteration % checkpoint_every == 0:
        checkpoint_path = checkpoint_dir / f'checkpoint_{iteration:04d}.pt'
        torch.save({
            'iteration': iteration,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
            'metrics': metrics,
        }, checkpoint_path)
        print(f"  → Saved checkpoint: {checkpoint_path.name}")

print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)

# Save final model
final_model_path = checkpoint_dir / 'final_model.pt'
torch.save({
    'model_state_dict': model.state_dict(),
    'metrics': metrics,
}, final_model_path)
print(f"\nFinal model saved: {final_model_path}")

# ============================================================================
# GENERATE FINAL VISUALIZATION
# ============================================================================
print("\nGenerating final visualization...")

# Run one final forward pass to get all outputs
model.eval()
with torch.no_grad():
    output_mask = model(coords_input)
    element_mask_raw = sample_mask_at_centroids(output_mask, centroids, R_outer)
    element_mask_values = threshold_mask(element_mask_raw, temperature)
    unet_materials = E_b + (E_i - E_b) * element_mask_values
    center_x, center_y, fixed_region_radius = estimate_inclusion_geometry(
        element_mask_values, centroids, R_outer
    )

# Compute final SSIM
final_ssim = compute_ssim(gt_materials, unet_materials, E_b, E_i)

# ============================================================================
# POST-PROCESSING: Binary thresholding at material midpoint
# ============================================================================
print("\nPost-processing: Creating binary prediction...")

# Threshold at midpoint between E_b and E_i in material space
E_midpoint = (E_b + E_i) / 2.0

# Assign binary materials based on which is closer
binary_materials = torch.where(
    unet_materials > E_midpoint,
    torch.tensor(E_i, dtype=torch.float64, device=device),
    torch.tensor(E_b, dtype=torch.float64, device=device)
)

# Compute SSIM for binary prediction
binary_ssim = compute_ssim(gt_materials, binary_materials, E_b, E_i)

print(f"  Material midpoint: {E_midpoint:.1f}")
print(f"  Binary elements (E={E_i}): {(binary_materials == E_i).sum().item()}")
print(f"  Binary elements (E={E_b}): {(binary_materials == E_b).sum().item()}")
print(f"  Continuous SSIM: {final_ssim.item():.4f}")
print(f"  Binary SSIM: {binary_ssim.item():.4f}")

# Compute accuracy
correct_elements = (binary_materials == gt_materials).sum().item()
accuracy = correct_elements / n_elements
print(f"  Element accuracy: {accuracy:.2%} ({correct_elements}/{n_elements})")

# Convert to numpy
points_np = points.cpu().numpy()
elements_np = elements.cpu().numpy()
gt_materials_np = gt_materials.cpu().numpy()
unet_materials_np = unet_materials.cpu().numpy()
binary_materials_np = binary_materials.cpu().numpy()
element_mask_values_np = element_mask_values.cpu().numpy()
center_x_np = center_x.cpu().numpy()
center_y_np = center_y.cpu().numpy()
fixed_region_radius_np = fixed_region_radius.cpu().numpy()

fig, axes = plt.subplots(1, 3, figsize=(24, 7))

# Left panel: Ground truth
ax = axes[0]
for e, elem in enumerate(elements_np):
    elem_coords = points_np[elem]
    color = 'red' if gt_materials_np[e] == E_i else 'blue'
    tri = plt.Polygon(elem_coords, facecolor=color, edgecolor='black', 
                      alpha=0.5, linewidth=0.3)
    ax.add_patch(tri)

theta_plot = np.linspace(0, 2*np.pi, 100)
ax.plot(R_outer*np.cos(theta_plot), R_outer*np.sin(theta_plot), 'k-', 
        linewidth=2, label='Outer boundary')
ax.plot(R_inner*np.cos(theta_plot), R_inner*np.sin(theta_plot), 'r--', 
        linewidth=2, label='True inclusion (r=0.3)', alpha=0.7)

ax.set_xlim(-R_outer*1.1, R_outer*1.1)
ax.set_ylim(-R_outer*1.1, R_outer*1.1)
ax.set_aspect('equal')
ax.set_title('Ground Truth Material Properties\n(Red=Inclusion E=10, Blue=Background E=1)', 
             fontsize=12, fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
ax.grid(True, alpha=0.3)

# Middle panel: U-Net continuous prediction
ax = axes[1]
for e, elem in enumerate(elements_np):
    elem_coords = points_np[elem]
    mat_val = unet_materials_np[e]
    intensity = (mat_val - E_b) / (E_i - E_b)
    color = plt.cm.RdYlBu_r(intensity)
    tri = plt.Polygon(elem_coords, facecolor=color, edgecolor='black', 
                      alpha=0.7, linewidth=0.3)
    ax.add_patch(tri)

ax.plot(R_outer*np.cos(theta_plot), R_outer*np.sin(theta_plot), 'k-', 
        linewidth=2, label='Outer boundary')
ax.scatter([center_x_np], [center_y_np], c='green', s=200, marker='x', 
           linewidths=3, label='Predicted center', zorder=10)

ax.set_xlim(-R_outer*1.1, R_outer*1.1)
ax.set_ylim(-R_outer*1.1, R_outer*1.1)
ax.set_aspect('equal')
ax.set_title(f'U-Net Continuous Prediction\n(SSIM: {final_ssim.item():.4f})', 
             fontsize=12, fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
ax.grid(True, alpha=0.3)

# Add colorbar for middle panel
sm = ScalarMappable(cmap=plt.cm.RdYlBu_r, norm=Normalize(vmin=E_b, vmax=E_i))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Young's Modulus (E)", rotation=270, labelpad=20)

# Right panel: Binary thresholded prediction
ax = axes[2]
for e, elem in enumerate(elements_np):
    elem_coords = points_np[elem]
    color = 'red' if binary_materials_np[e] == E_i else 'blue'
    tri = plt.Polygon(elem_coords, facecolor=color, edgecolor='black', 
                      alpha=0.5, linewidth=0.3)
    ax.add_patch(tri)

ax.plot(R_outer*np.cos(theta_plot), R_outer*np.sin(theta_plot), 'k-', 
        linewidth=2, label='Outer boundary')
ax.scatter([center_x_np], [center_y_np], c='green', s=200, marker='x', 
           linewidths=3, label='Predicted center', zorder=10)

ax.set_xlim(-R_outer*1.1, R_outer*1.1)
ax.set_ylim(-R_outer*1.1, R_outer*1.1)
ax.set_aspect('equal')
ax.set_title(f'Binary Prediction (Material Threshold E={E_midpoint:.1f})\n(SSIM: {binary_ssim.item():.4f}, Acc: {accuracy:.1%})', 
             fontsize=12, fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
ax.grid(True, alpha=0.3)

# Main title
fig.suptitle(f'Final Results After {num_iterations} Iterations (√L2 Loss + TV Regularization)', 
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout()
final_viz_path = 'final_training_result_v1.png'
plt.savefig(final_viz_path, dpi=150, bbox_inches='tight')
print(f"\nSaved final visualization: {final_viz_path}")
plt.close()

print("\nDone!")