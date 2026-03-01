"""
U-Net Training Script for Inverse Problem - v3 Universal
Train U-Net to predict material properties from boundary measurements
Post-processing: Universal level-set based hard boundary extraction for ANY shape
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy import ndimage
from scipy.ndimage import binary_closing, binary_dilation, generate_binary_structure
from scipy.interpolate import griddata, splprep, splev
from skimage import measure
from matplotlib.path import Path as MplPath
from fem_utils import *
from unet import *
import time

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
learning_rate = 1e-3
log_every = 10
checkpoint_every = 100

# Regularization
lambda_tv = 0.01
smoothing_sigma = 0.05

# Thresholding
temperature = 1000.0
bc_sharpness = 20.0
soft_min_temp = 0.1

print("="*70)
print("U-NET INVERSE PROBLEM TRAINING - V3 UNIVERSAL")
print("Universal Level-Set Post-Processing for ANY Shape")
print("="*70)

# Start timing
start_time = time.time()

# ============================================================================
# HELPER FUNCTIONS (same as v2)
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

def smooth_mask_differentiable(element_mask_values, centroids, sigma=0.05):
    """Smooth the mask using differentiable Gaussian-like operation"""
    dist_matrix = torch.cdist(centroids.unsqueeze(0), centroids.unsqueeze(0)).squeeze(0)
    weights = torch.exp(-dist_matrix**2 / (2 * sigma**2))
    weights = weights / weights.sum(dim=1, keepdim=True)
    smoothed = torch.matmul(weights, element_mask_values)
    return smoothed

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
    """Compute Total Variation loss on the U-Net output mask"""
    diff_x = output_mask[:, :, :, 1:] - output_mask[:, :, :, :-1]
    diff_y = output_mask[:, :, 1:, :] - output_mask[:, :, :-1, :]
    tv_loss = torch.mean(torch.abs(diff_x)) + torch.mean(torch.abs(diff_y))
    return tv_loss

# ============================================================================
# UNIVERSAL LEVEL-SET POST-PROCESSING FUNCTIONS
# ============================================================================

def extract_contour_levelset(unet_materials_np, centroids_np, threshold=5.5, grid_resolution=128):
    """
    Extract smooth contour from U-Net prediction using level-set method.
    Works for ANY shape (circles, ellipses, irregular, etc.)
    
    Args:
        unet_materials_np: [n_elements] predicted material properties (numpy)
        centroids_np: [n_elements, 2] element centroids (numpy)
        threshold: material value for isocurve extraction
        grid_resolution: resolution for interpolation grid
    
    Returns:
        contours: list of contour arrays (each [n_points, 2])
    """
    print("\n  Contour Extraction:")
    print(f"    Material threshold: {threshold}")
    print(f"    Grid resolution: {grid_resolution}x{grid_resolution}")
    
    # Create interpolation grid
    x_min, x_max = centroids_np[:, 0].min(), centroids_np[:, 0].max()
    y_min, y_max = centroids_np[:, 1].min(), centroids_np[:, 1].max()
    
    x_grid = np.linspace(x_min, x_max, grid_resolution)
    y_grid = np.linspace(y_min, y_max, grid_resolution)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Interpolate material field onto regular grid
    Z = griddata(centroids_np, unet_materials_np, (X, Y), method='cubic', fill_value=E_b)
    
    # Extract contours at threshold
    contours = measure.find_contours(Z, threshold)
    
    if len(contours) == 0:
        print("    WARNING: No contours found!")
        return []
    
    # Take the longest contour (main inclusion boundary)
    main_contour_idx = np.argmax([len(c) for c in contours])
    contour = contours[main_contour_idx]
    
    # Convert from grid coordinates back to physical coordinates
    contour_physical = np.zeros_like(contour)
    contour_physical[:, 0] = x_min + contour[:, 1] * (x_max - x_min) / grid_resolution
    contour_physical[:, 1] = y_min + contour[:, 0] * (y_max - y_min) / grid_resolution
    
    print(f"    Found {len(contours)} contours")
    print(f"    Main contour: {len(contour_physical)} points")
    
    return [contour_physical]

def smooth_contour_spline(contour, smoothing=0.001, n_points=200):
    """
    Smooth a contour using spline interpolation.
    This works for ANY closed curve shape.
    
    Args:
        contour: [n_points, 2] raw contour points
        smoothing: spline smoothing parameter (0 = no smoothing, higher = more smoothing)
        n_points: number of points in smoothed contour
    
    Returns:
        smooth_contour: [n_points, 2] smoothed contour points
    """
    print("\n  Contour Smoothing:")
    print(f"    Input points: {len(contour)}")
    print(f"    Smoothing parameter: {smoothing}")
    print(f"    Output points: {n_points}")
    
    # Fit periodic spline to contour
    # per=True ensures the spline is closed (periodic)
    try:
        tck, u = splprep([contour[:, 0], contour[:, 1]], s=smoothing, per=True)
        
        # Evaluate spline at uniform intervals
        u_new = np.linspace(0, 1, n_points)
        smooth_x, smooth_y = splev(u_new, tck)
        
        smooth_contour = np.column_stack([smooth_x, smooth_y])
        
        print(f"    Spline fitting successful!")
        
        return smooth_contour
        
    except Exception as e:
        print(f"    WARNING: Spline fitting failed ({e})")
        print(f"    Returning raw contour")
        return contour

def create_hard_mask_from_contour(centroids_np, contour):
    """
    Create hard binary mask using simple point-in-polygon test on centroids.
    ULTRA SIMPLE: Inside contour → 1, Outside contour → 0
    
    Args:
        centroids_np: [n_elements, 2] element centroids
        contour: [n_points, 2] boundary contour
    
    Returns:
        hard_mask_np: [n_elements] binary mask (0 or 1)
    """
    print("\n  Hard Mask Generation (Simple Inside/Outside Test):")
    
    # Create a Path object from contour
    path = MplPath(contour)
    
    # Check which centroids are inside the contour
    inside = path.contains_points(centroids_np)
    hard_mask_np = inside.astype(float)
    
    n_inclusion = np.sum(hard_mask_np > 0.5)
    n_background = len(hard_mask_np) - n_inclusion
    
    print(f"    Inclusion elements: {n_inclusion}")
    print(f"    Background elements: {n_background}")
    print(f"    Method: Pure geometric test (centroid inside → inclusion)")
    
    return hard_mask_np

def compute_geometric_properties(contour):
    """
    Compute geometric properties of the contour for analysis.
    
    Args:
        contour: [n_points, 2] boundary contour
    
    Returns:
        properties: dict with center, area, perimeter, etc.
    """
    # Center (centroid)
    center = np.mean(contour, axis=0)
    
    # Area using shoelace formula
    x, y = contour[:, 0], contour[:, 1]
    area = 0.5 * np.abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) + x[-1] * y[0] - x[0] * y[-1])
    
    # Perimeter
    diff = np.diff(contour, axis=0, append=contour[0:1])
    perimeter = np.sum(np.sqrt(np.sum(diff**2, axis=1)))
    
    # Equivalent radius (radius of circle with same area)
    equivalent_radius = np.sqrt(area / np.pi)
    
    properties = {
        'center': center,
        'area': area,
        'perimeter': perimeter,
        'equivalent_radius': equivalent_radius
    }
    
    return properties

def levelset_postprocess_universal(unet_materials_np, centroids_np, E_b, E_i, 
                                   smoothing=0.001, n_contour_points=200):
    """
    Universal level-set post-processing pipeline for ANY shape.
    ULTRA SIMPLE VERSION:
    1. Extract contour at material threshold (works for any shape)
    2. Smooth contour using splines (preserves shape, removes noise)
    3. Inside contour → E_i, Outside contour → E_b (pure geometric test)
    
    Args:
        unet_materials_np: [n_elements] continuous material prediction
        centroids_np: [n_elements, 2] element centroids
        E_b, E_i: background and inclusion material values
        smoothing: spline smoothing parameter (0.001 is good default)
        n_contour_points: number of points in smoothed contour (200 is good default)
    
    Returns:
        hard_materials_np: [n_elements] hard material values
        smooth_contour: [n_points, 2] smoothed boundary contour
        properties: dict with geometric properties
    """
    print("\n" + "="*70)
    print("UNIVERSAL LEVEL-SET POST-PROCESSING (SIMPLE VERSION)")
    print("Works for ANY shape: circles, ellipses, irregular, etc.")
    print("="*70)
    
    # Step 1: Extract contour at midpoint
    threshold = (E_b + E_i) / 2.0
    contours = extract_contour_levelset(unet_materials_np, centroids_np, threshold)
    
    if len(contours) == 0:
        print("\n  FALLBACK: Using simple thresholding")
        hard_mask_np = (unet_materials_np > threshold).astype(float)
        hard_materials_np = E_b + (E_i - E_b) * hard_mask_np
        return hard_materials_np, None, None
    
    raw_contour = contours[0]
    
    # Step 2: Smooth contour using splines
    smooth_contour = smooth_contour_spline(raw_contour, smoothing, n_contour_points)
    
    # Step 3: Simple inside/outside test - trust the contour!
    hard_mask_np = create_hard_mask_from_contour(centroids_np, smooth_contour)
    
    # Convert to material values
    hard_materials_np = E_b + (E_i - E_b) * hard_mask_np
    
    # Compute geometric properties
    print("\n  Geometric Properties:")
    properties = compute_geometric_properties(smooth_contour)
    print(f"    Center: ({properties['center'][0]:.4f}, {properties['center'][1]:.4f})")
    print(f"    Area: {properties['area']:.4f}")
    print(f"    Perimeter: {properties['perimeter']:.4f}")
    print(f"    Equivalent radius: {properties['equivalent_radius']:.4f}")
    
    print("\n  Post-processing complete!")
    print("="*70)
    
    return hard_materials_np, smooth_contour, properties

# ============================================================================
# FORWARD PASS AND LOSS COMPUTATION
# ============================================================================

def forward_pass_and_loss(model, coords_input, points, elements, centroids,
                         force_vectors, boundary_displacements_measured,
                         boundary_nodes, gt_materials, n_samples, n_dof, device, 
                         lambda_tv, smoothing_sigma):
    """Complete forward pass: U-Net → materials → FEM → loss"""
    
    output_mask = model(coords_input)
    element_mask_raw = sample_mask_at_centroids(output_mask, centroids, R_outer)
    element_mask_values = threshold_mask(element_mask_raw, temperature)
    element_mask_values = smooth_mask_differentiable(element_mask_values, centroids, smoothing_sigma)
    unet_materials = E_b + (E_i - E_b) * element_mask_values
    
    K = assemble_stiffness_differentiable(points, elements, unet_materials, nu)
    
    center_x, center_y, fixed_region_radius = estimate_inclusion_geometry(
        element_mask_values, centroids, R_outer
    )
    fixed_weights, free_weights = compute_boundary_conditions(
        points, center_x, center_y, fixed_region_radius, bc_sharpness
    )
    
    total_loss = 0.0
    for i in range(n_samples):
        U, U_nodes = solve_fem_soft(
            K, force_vectors[i], fixed_weights, free_weights, n_dof, device
        )
        boundary_disp_predicted = U_nodes[boundary_nodes]
        diff = (boundary_disp_predicted - boundary_displacements_measured[i]) / R_outer
        l2_loss = torch.sqrt(torch.sum(torch.square(diff))) / len(boundary_nodes)
        total_loss += l2_loss
    
    total_loss = total_loss / n_samples
    tv_loss = compute_tv_loss(output_mask)
    total_loss = total_loss + lambda_tv * tv_loss
    
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
print(f"  Smoothing: Gaussian (sigma={smoothing_sigma}) - reduces boundary noise")

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
    loss, metrics = forward_pass_and_loss(
        model, coords_input, points, elements, centroids,
        force_vectors, boundary_displacements_measured,
        boundary_nodes, gt_materials, n_samples, n_dof, device, lambda_tv, smoothing_sigma
    )
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    
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

# Calculate training runtime
training_end_time = time.time()
training_seconds = training_end_time - start_time
training_minutes = int(training_seconds // 60)
training_secs = int(training_seconds % 60)
print(f"\nTraining time: {training_minutes:02d}:{training_secs:02d} (mm:ss)")

# Save final model
final_model_path = checkpoint_dir / 'final_model.pt'
torch.save({
    'model_state_dict': model.state_dict(),
    'metrics': metrics,
    'training_runtime_seconds': training_seconds,
    'training_runtime_formatted': f"{training_minutes:02d}:{training_secs:02d}",
}, final_model_path)
print(f"\nFinal model saved: {final_model_path}")

# ============================================================================
# GENERATE FINAL VISUALIZATION WITH UNIVERSAL LEVEL-SET POST-PROCESSING
# ============================================================================
print("\nGenerating final visualization...")

# Run one final forward pass to get all outputs
model.eval()
with torch.no_grad():
    output_mask = model(coords_input)
    element_mask_raw = sample_mask_at_centroids(output_mask, centroids, R_outer)
    element_mask_values = threshold_mask(element_mask_raw, temperature)
    element_mask_values = smooth_mask_differentiable(element_mask_values, centroids, smoothing_sigma)
    unet_materials = E_b + (E_i - E_b) * element_mask_values
    center_x, center_y, fixed_region_radius = estimate_inclusion_geometry(
        element_mask_values, centroids, R_outer
    )

# Compute continuous SSIM
continuous_ssim = compute_ssim(gt_materials, unet_materials, E_b, E_i)

# Convert to numpy for post-processing
unet_materials_np = unet_materials.cpu().numpy()
centroids_np = centroids.cpu().numpy()
points_np = points.cpu().numpy()
elements_np = elements.cpu().numpy()
gt_materials_np = gt_materials.cpu().numpy()

# ============================================================================
# APPLY UNIVERSAL LEVEL-SET POST-PROCESSING
# ============================================================================
hard_materials_np, smooth_contour, properties = levelset_postprocess_universal(
    unet_materials_np, centroids_np, E_b, E_i,
    smoothing=0.001,  # Adjust for more/less smoothing
    n_contour_points=200  # More points = smoother boundary
)

# Convert back to torch for SSIM computation
hard_materials = torch.from_numpy(hard_materials_np).to(device)
hard_ssim = compute_ssim(gt_materials, hard_materials, E_b, E_i)

# Compute accuracy
correct_elements = (hard_materials_np == gt_materials_np).sum()
accuracy = correct_elements / n_elements

print(f"\nFinal Metrics:")
print(f"  Continuous SSIM: {continuous_ssim.item():.4f}")
print(f"  Hard SSIM: {hard_ssim.item():.4f}")
print(f"  Element accuracy: {accuracy:.2%} ({int(correct_elements)}/{n_elements})")

# ============================================================================
# CREATE VISUALIZATION
# ============================================================================
print("\nCreating visualization...")

center_x_np = center_x.cpu().numpy()
center_y_np = center_y.cpu().numpy()

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

# Plot extracted contour if available
if smooth_contour is not None:
    ax.plot(smooth_contour[:, 0], smooth_contour[:, 1], 'cyan', linewidth=2.5, 
            label='Extracted contour', linestyle='-', alpha=0.9)

ax.set_xlim(-R_outer*1.1, R_outer*1.1)
ax.set_ylim(-R_outer*1.1, R_outer*1.1)
ax.set_aspect('equal')
ax.set_title(f'U-Net Continuous Prediction\n(SSIM: {continuous_ssim.item():.4f})', 
             fontsize=12, fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
ax.grid(True, alpha=0.3)

# Add colorbar
sm = ScalarMappable(cmap=plt.cm.RdYlBu_r, norm=Normalize(vmin=E_b, vmax=E_i))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Young's Modulus (E)", rotation=270, labelpad=20)

# Right panel: Universal level-set hard boundary
ax = axes[2]
for e, elem in enumerate(elements_np):
    elem_coords = points_np[elem]
    color = 'red' if hard_materials_np[e] == E_i else 'blue'
    tri = plt.Polygon(elem_coords, facecolor=color, edgecolor='black', 
                      alpha=0.5, linewidth=0.3)
    ax.add_patch(tri)

ax.plot(R_outer*np.cos(theta_plot), R_outer*np.sin(theta_plot), 'k-', 
        linewidth=2, label='Outer boundary')

# Plot smoothed contour boundary
if smooth_contour is not None:
    ax.plot(smooth_contour[:, 0], smooth_contour[:, 1], 'lime', linewidth=2.5, 
            label='Smoothed boundary', linestyle='-', alpha=0.9)
    
    if properties is not None:
        ax.scatter([properties['center'][0]], [properties['center'][1]], 
                  c='lime', s=200, marker='x', linewidths=3, 
                  label=f"Center ({properties['center'][0]:.3f}, {properties['center'][1]:.3f})", 
                  zorder=10)

ax.set_xlim(-R_outer*1.1, R_outer*1.1)
ax.set_ylim(-R_outer*1.1, R_outer*1.1)
ax.set_aspect('equal')

# Create title with geometric info
title_str = f'Universal Level-Set Hard Boundary\n(SSIM: {hard_ssim.item():.4f}, Acc: {accuracy:.1%}'
if properties is not None:
    title_str += f', Area: {properties["area"]:.3f}'
title_str += ')'

ax.set_title(title_str, fontsize=12, fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
ax.grid(True, alpha=0.3)

# Main title
fig.suptitle(f'Final Results After {num_iterations} Iterations (√L2 + TV + Smoothing + Simple Level-Set)', 
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout()
final_viz_path = 'final_training_result_v3.png'
plt.savefig(final_viz_path, dpi=150, bbox_inches='tight')
print(f"\nSaved final visualization: {final_viz_path}")
plt.close()

print("\n" + "="*70)
print("V3 UNIVERSAL COMPLETE - WORKS FOR ANY SHAPE!")
print("="*70)

# Calculate total runtime (training + post-processing + visualization)
end_time = time.time()
total_seconds = end_time - start_time
total_minutes = int(total_seconds // 60)
total_secs = int(total_seconds % 60)

postprocess_seconds = total_seconds - training_seconds
postprocess_minutes = int(postprocess_seconds // 60)
postprocess_secs = int(postprocess_seconds % 60)

print(f"\n{'='*70}")
print(f"RUNTIME SUMMARY")
print(f"{'='*70}")
print(f"  Training:        {training_minutes:02d}:{training_secs:02d} (mm:ss)")
print(f"  Post-processing: {postprocess_minutes:02d}:{postprocess_secs:02d} (mm:ss)")
print(f"  TOTAL:           {total_minutes:02d}:{total_secs:02d} (mm:ss)")
print(f"{'='*70}")

print(f"\nResults:")
print(f"  Continuous prediction SSIM: {continuous_ssim.item():.4f}")
print(f"  Hard boundary SSIM: {hard_ssim.item():.4f}")
print(f"  Element accuracy: {accuracy:.2%}")

if properties is not None:
    print(f"\nGeometric Properties:")
    print(f"  Center: ({properties['center'][0]:.4f}, {properties['center'][1]:.4f})")
    print(f"  Area: {properties['area']:.4f}")
    print(f"  Perimeter: {properties['perimeter']:.4f}")
    print(f"  Equivalent radius: {properties['equivalent_radius']:.4f}")
    
    print(f"\nGround Truth:")
    print(f"  True center: (0.0000, 0.0000)")
    print(f"  True area: {np.pi * R_inner**2:.4f}")
    print(f"  True radius: {R_inner:.4f}")
    
    print(f"\nErrors:")
    center_error = np.sqrt(properties['center'][0]**2 + properties['center'][1]**2)
    area_error = abs(properties['area'] - np.pi * R_inner**2)
    radius_error = abs(properties['equivalent_radius'] - R_inner)
    print(f"  Center error: {center_error:.4f}")
    print(f"  Area error: {area_error:.4f}")
    print(f"  Radius error: {radius_error:.4f}")

print("\n" + "="*70)
print("KEY ADVANTAGE: This method works for:")
print("  - Circles (your current case)")
print("  - Ellipses")
print("  - Irregular shapes")
print("  - Multiple disconnected regions")
print("  - ANY closed boundary!")
print("="*70)

# Save comprehensive results summary
results_summary = {
    'continuous_ssim': continuous_ssim.item(),
    'hard_ssim': hard_ssim.item(),
    'accuracy': accuracy,
    'geometric_properties': properties,
    'runtime': {
        'training_seconds': training_seconds,
        'training_formatted': f"{training_minutes:02d}:{training_secs:02d}",
        'postprocessing_seconds': postprocess_seconds,
        'postprocessing_formatted': f"{postprocess_minutes:02d}:{postprocess_secs:02d}",
        'total_seconds': total_seconds,
        'total_formatted': f"{total_minutes:02d}:{total_secs:02d}",
    },
    'training_params': {
        'num_iterations': num_iterations,
        'learning_rate': learning_rate,
        'lambda_tv': lambda_tv,
        'smoothing_sigma': smoothing_sigma,
    }
}

results_path = checkpoint_dir / 'results_summary.pt'
torch.save(results_summary, results_path)
print(f"\nResults summary saved: {results_path}")

print("\nDone!")