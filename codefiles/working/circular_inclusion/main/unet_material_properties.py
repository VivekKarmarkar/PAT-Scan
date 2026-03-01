"""
U-Net Forward Pass Visualization
Create polar mesh, run U-Net forward pass, visualize output
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from unet import *

torch.set_default_dtype(torch.float64)


# ============================================================================
# PARAMETERS
# ============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

R_outer = 1.0
n_radial = 20
n_angular = 40

print("="*70)
print("POLAR MESH CREATION")
print("="*70)

# ============================================================================
# CREATE STRUCTURED POLAR GRID
# ============================================================================
print("\nCreating polar grid...")

# Create radial and angular coordinates
r = torch.linspace(0, R_outer, n_radial, dtype=torch.float64, device=device)
theta = torch.linspace(0, 2*torch.pi, n_angular+1, dtype=torch.float64, device=device)[:-1]

# Create meshgrid
R, Theta = torch.meshgrid(r, theta, indexing='ij')

# Convert to Cartesian coordinates
X = R * torch.cos(Theta)
Y = R * torch.sin(Theta)

# Flatten to get all points
points = torch.stack([X.ravel(), Y.ravel()], dim=1)

# Add center point explicitly (remove r=0 duplicates)
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

print(f"  Radial divisions: {n_radial}")
print(f"  Angular divisions: {n_angular}")
print(f"  Total nodes: {n_nodes}")

# ============================================================================
# CREATE TRIANGLES FROM POLAR GRID
# ============================================================================
print("\nCreating triangles from polar grid...")

elements = []

def node_idx(i, j):
    """Get node index for radial index i and angular index j"""
    if i == 0:
        return 0  # Center point
    else:
        return 1 + (i - 1) * n_angular + (j % n_angular)

# Create triangles
for i in range(n_radial - 1):
    for j in range(n_angular):
        if i == 0:
            # Triangles connected to center
            n0 = 0
            n1 = node_idx(1, j)
            n2 = node_idx(1, j + 1)
            elements.append([n0, n1, n2])
        else:
            # Quadrilaterals split into 2 triangles
            n0 = node_idx(i, j)
            n1 = node_idx(i, j + 1)
            n2 = node_idx(i + 1, j + 1)
            n3 = node_idx(i + 1, j)
            
            elements.append([n0, n1, n2])
            elements.append([n0, n2, n3])

elements = torch.tensor(elements, dtype=torch.long, device=device)
n_elements = len(elements)

print(f"  Triangles: {n_elements}")

# ============================================================================
# EXTRACT ELEMENT CENTROIDS
# ============================================================================
print("\nExtracting element centroids...")

centroids = torch.zeros((n_elements, 2), dtype=torch.float64, device=device)

for i in range(n_elements):
    elem = elements[i]
    centroid = torch.mean(points[elem], dim=0)
    centroids[i] = centroid

print(f"  Centroid shape: {centroids.shape}")
print(f"  Centroid range X: [{centroids[:, 0].min():.4f}, {centroids[:, 0].max():.4f}]")
print(f"  Centroid range Y: [{centroids[:, 1].min():.4f}, {centroids[:, 1].max():.4f}]")

# ============================================================================
# PREPARE U-NET INPUT
# ============================================================================
print("\n" + "="*70)
print("U-NET FORWARD PASS")
print("="*70)

print("\nPreparing U-Net input...")

# Reshape centroids to 2D grid (approximate for visualization)
# We'll create a square grid and interpolate centroids onto it
grid_size = 64

# Create grid
x_grid = torch.linspace(-R_outer, R_outer, grid_size, dtype=torch.float64, device=device)
y_grid = torch.linspace(-R_outer, R_outer, grid_size, dtype=torch.float64, device=device)
X_grid, Y_grid = torch.meshgrid(x_grid, y_grid, indexing='ij')

# Stack to create input (2 channels: x and y coordinates)
coords_input = torch.stack([X_grid, Y_grid], dim=0)  # (2, grid_size, grid_size)
coords_input = coords_input.unsqueeze(0)  # (1, 2, grid_size, grid_size) - batch dimension

print(f"  Input shape: {coords_input.shape}")

# ============================================================================
# INITIALIZE U-NET
# ============================================================================
print("\nInitializing U-Net...")

model = UNet(in_channels=2, out_channels=1, base_features=32)
model = model.to(device).to(torch.float64)

n_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {n_params:,}")

# ============================================================================
# FORWARD PASS
# ============================================================================
print("\nRunning forward pass...")

model.eval()
with torch.no_grad():
    output_mask = model(coords_input)

print(f"  Output shape: {output_mask.shape}")
print(f"  Output range: [{output_mask.min():.4f}, {output_mask.max():.4f}]")

# ============================================================================
# SAMPLE U-NET MASK AT ELEMENT CENTROIDS
# ============================================================================
print("\nSampling U-Net mask at element centroids...")

mask_2d = output_mask.squeeze()  # (grid_size, grid_size)

# Map centroids to mask grid coordinates
x_min, x_max = -R_outer, R_outer
y_min, y_max = -R_outer, R_outer

element_mask_values = torch.zeros(n_elements, dtype=torch.float64, device=device)

for i in range(n_elements):
    centroid = centroids[i]
    
    # Normalize to [0, grid_size-1]
    x_idx = int(((centroid[0] - x_min) / (x_max - x_min) * (grid_size - 1)).item())
    y_idx = int(((centroid[1] - y_min) / (y_max - y_min) * (grid_size - 1)).item())
    
    # Clamp to valid range
    x_idx = max(0, min(grid_size - 1, x_idx))
    y_idx = max(0, min(grid_size - 1, y_idx))
    
    element_mask_values[i] = mask_2d[y_idx, x_idx]

print(f"  Element mask value range: [{element_mask_values.min():.4f}, {element_mask_values.max():.4f}]")

# Use median as threshold to get approximately 50/50 split
threshold = element_mask_values.median().item()
print(f"  Using median threshold: {threshold:.4f}")

# ============================================================================
# ASSIGN MATERIALS BASED ON U-NET MASK
# ============================================================================
print("\nAssigning materials based on U-Net mask...")

E_b = 1.0   # Background Young's modulus
E_i = 10.0  # Inclusion Young's modulus

# Split elements based on threshold
above_threshold = element_mask_values > threshold
below_threshold = ~above_threshold

n_above = above_threshold.sum().item()
n_below = below_threshold.sum().item()

print(f"  Elements above threshold: {n_above}")
print(f"  Elements below threshold: {n_below}")

# Assign E_i (inclusion) to SMALLER group, E_b (background) to LARGER group
if n_above < n_below:
    # Above threshold is smaller -> inclusion
    print(f"  Assigning inclusion (E={E_i}) to above-threshold elements")
    unet_materials = torch.where(above_threshold, 
                              torch.tensor(E_i, device=device), 
                              torch.tensor(E_b, device=device))
else:
    # Below threshold is smaller -> inclusion
    print(f"  Assigning inclusion (E={E_i}) to below-threshold elements")
    unet_materials = torch.where(below_threshold,
                              torch.tensor(E_i, device=device), 
                              torch.tensor(E_b, device=device))

print(f"  Background elements (E={E_b}): {torch.sum(unet_materials == E_b)}")
print(f"  Inclusion elements (E={E_i}): {torch.sum(unet_materials == E_i)}")

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\n" + "="*70)
print("VISUALIZATION")
print("="*70)

print("\nCreating visualization...")

# Convert to numpy for plotting
points_np = points.cpu().numpy()
elements_np = elements.cpu().numpy()
unet_materials_np = unet_materials.cpu().numpy()

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Left panel: Ground truth material properties
ax = axes[0]

# Ground truth: circular inclusion at R_inner
R_inner = 0.3

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

gt_materials_np = gt_materials.cpu().numpy()

# ============================================================================
# COMPUTE SSIM BETWEEN GROUND TRUTH AND U-NET PREDICTION
# ============================================================================
print("\nComputing SSIM...")

# Simple SSIM computation on element materials
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

gt_materials_np = gt_materials.cpu().numpy()

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

print("\n" + "="*70)
print("DONE")
print("="*70)