"""
U-Net Training Script with Grid Search - v5
Performs grid search over hyperparameters INCLUDING LEARNING RATE, then trains with best config
Keeps all visualization and animation functionality intact
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
from PIL import Image
import imageio
import os
import itertools
from copy import deepcopy

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

# Grid Search Parameters
GRID_SEARCH_ENABLED = True
GRID_SEARCH_ITERATIONS = 10

grid_search_space = {
    'learning_rate': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
    'lambda_tv': [0.001, 0.005, 0.01, 0.02, 0.05],
    'smoothing_sigma': [0.01, 0.03, 0.05, 0.07, 0.1],
    'temperature': [100, 500, 1000, 2000, 5000],
    'bc_sharpness': [10, 20, 40]
}

# Training (AFTER grid search)
num_iterations = 200  # Changed from 1000 to 200
learning_rate = 1e-3  # Default value (will be overridden by grid search)
log_every = 10

# Animation checkpoints: first 20 iterations individually, then every 50
animation_checkpoints = list(range(21)) + list(range(50, num_iterations + 1, 50))
animation_checkpoints = sorted(set(animation_checkpoints))

# Animation parameters
frame_duration = 0.5  # seconds per frame
animation_fps = int(1.0 / frame_duration)

print("="*70)
print("U-NET INVERSE PROBLEM TRAINING - V5 WITH LR IN GRID SEARCH")
print("="*70)
print(f"Grid Search: {GRID_SEARCH_ENABLED}")
if GRID_SEARCH_ENABLED:
    print(f"  - Grid search iterations: {GRID_SEARCH_ITERATIONS}")
    print(f"  - learning_rate values: {grid_search_space['learning_rate']}")
    print(f"  - lambda_tv values: {grid_search_space['lambda_tv']}")
    print(f"  - smoothing_sigma values: {grid_search_space['smoothing_sigma']}")
    print(f"  - temperature values: {grid_search_space['temperature']}")
    print(f"  - bc_sharpness values: {grid_search_space['bc_sharpness']}")
    total_configs = np.prod([len(v) for v in grid_search_space.values()])
    print(f"  - Total configurations: {total_configs}")
print(f"Final training iterations: {num_iterations}")
print("="*70)

# Start timing
start_time = time.time()

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
    K_eff = K.clone()
    K_eff.diagonal().add_(penalty * fixed_weights)
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

def precompute_smoothing_weights(centroids, sigma=0.05):
    """Precompute Gaussian smoothing weights (constant for a fixed mesh)"""
    dist_matrix = torch.cdist(centroids.unsqueeze(0), centroids.unsqueeze(0)).squeeze(0)
    weights = torch.exp(-dist_matrix**2 / (2 * sigma**2))
    weights = weights / weights.sum(dim=1, keepdim=True)
    return weights

def smooth_mask_differentiable(element_mask_values, smoothing_weights):
    """Smooth the mask using precomputed Gaussian weights"""
    return torch.matmul(smoothing_weights, element_mask_values)

def estimate_inclusion_geometry(element_mask_values, centroids, R_outer, soft_min_temp=0.1):
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
# LEVEL-SET POST-PROCESSING FUNCTIONS
# ============================================================================

def extract_contour_levelset(unet_materials_np, centroids_np, threshold=5.5, grid_resolution=128):
    """Extract smooth contour from U-Net prediction using level-set method"""
    x_min, x_max = centroids_np[:, 0].min(), centroids_np[:, 0].max()
    y_min, y_max = centroids_np[:, 1].min(), centroids_np[:, 1].max()
    
    x_grid = np.linspace(x_min, x_max, grid_resolution)
    y_grid = np.linspace(y_min, y_max, grid_resolution)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    Z = griddata(centroids_np, unet_materials_np, (X, Y), method='cubic', fill_value=E_b)
    
    contours = measure.find_contours(Z, threshold)
    
    if len(contours) == 0:
        return []
    
    main_contour_idx = np.argmax([len(c) for c in contours])
    contour = contours[main_contour_idx]
    
    contour_physical = np.zeros_like(contour)
    contour_physical[:, 0] = x_min + contour[:, 1] * (x_max - x_min) / grid_resolution
    contour_physical[:, 1] = y_min + contour[:, 0] * (y_max - y_min) / grid_resolution
    
    return [contour_physical]

def smooth_contour_spline(contour, smoothing=0.001, n_points=200):
    """Smooth a contour using spline interpolation"""
    try:
        tck, u = splprep([contour[:, 0], contour[:, 1]], s=smoothing, per=True)
        u_new = np.linspace(0, 1, n_points)
        smooth_x, smooth_y = splev(u_new, tck)
        smooth_contour = np.column_stack([smooth_x, smooth_y])
        return smooth_contour
    except:
        return contour

def create_hard_mask_from_contour(centroids_np, contour):
    """Create hard binary mask using simple point-in-polygon test on centroids"""
    path = MplPath(contour)
    inside = path.contains_points(centroids_np)
    hard_mask_np = inside.astype(float)
    return hard_mask_np

def compute_geometric_properties(contour):
    """Compute geometric properties of the contour"""
    center = np.mean(contour, axis=0)
    
    x, y = contour[:, 0], contour[:, 1]
    area = 0.5 * np.abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) + x[-1] * y[0] - x[0] * y[-1])
    
    diff = np.diff(contour, axis=0, append=contour[0:1])
    perimeter = np.sum(np.sqrt(np.sum(diff**2, axis=1)))
    
    equivalent_radius = np.sqrt(area / np.pi)
    
    properties = {
        'center': center,
        'area': area,
        'perimeter': perimeter,
        'equivalent_radius': equivalent_radius
    }
    
    return properties

def levelset_postprocess_universal(unet_materials_np, centroids_np, E_b, E_i, 
                                   smoothing=0.001, n_contour_points=200, verbose=False):
    """Universal level-set post-processing pipeline (silent version for animation)"""
    threshold = (E_b + E_i) / 2.0
    contours = extract_contour_levelset(unet_materials_np, centroids_np, threshold)
    
    if len(contours) == 0:
        hard_mask_np = (unet_materials_np > threshold).astype(float)
        hard_materials_np = E_b + (E_i - E_b) * hard_mask_np
        return hard_materials_np, None, None
    
    raw_contour = contours[0]
    smooth_contour = smooth_contour_spline(raw_contour, smoothing, n_contour_points)
    hard_mask_np = create_hard_mask_from_contour(centroids_np, smooth_contour)
    hard_materials_np = E_b + (E_i - E_b) * hard_mask_np
    
    try:
        properties = compute_geometric_properties(smooth_contour)
    except:
        properties = None
    
    return hard_materials_np, smooth_contour, properties

# ============================================================================
# FORWARD PASS AND LOSS COMPUTATION
# ============================================================================

def forward_pass_and_loss(model, coords_input, points, elements, centroids,
                         force_vectors_angular, boundary_displacements_angular,
                         force_vectors_symmetric, boundary_displacements_symmetric,
                         boundary_nodes, gt_materials, n_samples_angular, n_samples_symmetric,
                         n_dof, device, lambda_tv, smoothing_weights, temperature, bc_sharpness,
                         use_symmetric, soft_min_temp=0.1):
    """
    Complete forward pass: U-Net → materials → FEM → loss
    Now handles BOTH angular scanning and symmetric scanning datasets
    """

    # U-Net forward pass (same material prediction for all samples)
    output_mask = model(coords_input)
    element_mask_raw = sample_mask_at_centroids(output_mask, centroids, R_outer)
    element_mask_values = threshold_mask(element_mask_raw, temperature)
    element_mask_values = smooth_mask_differentiable(element_mask_values, smoothing_weights)
    unet_materials = E_b + (E_i - E_b) * element_mask_values
    
    # Assemble stiffness matrix (same for all samples)
    K = assemble_stiffness_differentiable(points, elements, unet_materials, nu)
    
    # Estimate boundary conditions (same for all samples)
    center_x, center_y, fixed_region_radius = estimate_inclusion_geometry(
        element_mask_values, centroids, R_outer, soft_min_temp
    )
    fixed_weights, free_weights = compute_boundary_conditions(
        points, center_x, center_y, fixed_region_radius, bc_sharpness
    )
    
    # Build effective stiffness once (shared across all samples)
    K_eff = K.clone()
    K_eff.diagonal().add_(1e6 * fixed_weights)

    # ========================================================================
    # LOSS 1: Angular scanning dataset (diverse force configurations)
    # ========================================================================
    # Batch solve: stack all force vectors and solve in one call (reuses LU factorization)
    F_angular = torch.stack([force_vectors_angular[i] * free_weights for i in range(n_samples_angular)], dim=1)
    U_angular = torch.linalg.solve(K_eff, F_angular)  # (n_dof, n_samples)
    n_nodes = n_dof // 2
    U_nodes_angular = U_angular.reshape(n_nodes, 2, n_samples_angular)  # (n_nodes, 2, n_samples)

    angular_loss = 0.0
    for i in range(n_samples_angular):
        boundary_disp_predicted = U_nodes_angular[boundary_nodes, :, i]
        diff = (boundary_disp_predicted - boundary_displacements_angular[i]) / R_outer
        l2_loss = torch.sqrt(torch.sum(torch.square(diff))) / len(boundary_nodes)
        angular_loss += l2_loss

    angular_loss = angular_loss / n_samples_angular

    # ========================================================================
    # LOSS 2: Symmetric scanning dataset (rotational invariance)
    # ========================================================================
    symmetric_loss = 0.0
    if use_symmetric and n_samples_symmetric > 0:
        F_symmetric = torch.stack([force_vectors_symmetric[i] * free_weights for i in range(n_samples_symmetric)], dim=1)
        U_symmetric = torch.linalg.solve(K_eff, F_symmetric)
        U_nodes_symmetric = U_symmetric.reshape(n_nodes, 2, n_samples_symmetric)

        for i in range(n_samples_symmetric):
            boundary_disp_predicted = U_nodes_symmetric[boundary_nodes, :, i]
            diff = (boundary_disp_predicted - boundary_displacements_symmetric[i]) / R_outer
            l2_loss = torch.sqrt(torch.sum(torch.square(diff))) / len(boundary_nodes)
            symmetric_loss += l2_loss

        symmetric_loss = symmetric_loss / n_samples_symmetric
    
    # ========================================================================
    # TOTAL LOSS: Data (angular + symmetric) + TV regularization
    # ========================================================================
    data_loss = angular_loss + symmetric_loss
    tv_loss = compute_tv_loss(output_mask)
    total_loss = data_loss + lambda_tv * tv_loss
    
    # Compute SSIM for monitoring
    ssim = compute_ssim(gt_materials, unet_materials, E_b, E_i)
    
    metrics = {
        'loss': total_loss.item(),
        'data_loss': data_loss.item(),
        'angular_loss': angular_loss.item(),
        'symmetric_loss': symmetric_loss.item() if use_symmetric else 0.0,
        'tv_loss': tv_loss.item(),
        'ssim': ssim.item(),
        'center_x': center_x.item(),
        'center_y': center_y.item(),
        'radius': fixed_region_radius.item(),
    }
    
    return total_loss, metrics

# ============================================================================
# VISUALIZATION FUNCTION FOR ANIMATION FRAMES
# ============================================================================

def create_animation_frame(iteration, current_time, loss_history, model, coords_input,
                          points_np, elements_np, centroids_np, gt_materials_np,
                          temperature, smoothing_weights, save_path):
    """Create a single frame for the animation with 2-row layout"""

    # Run inference
    model.eval()
    with torch.no_grad():
        output_mask = model(coords_input)
        element_mask_raw = sample_mask_at_centroids(output_mask, centroids, R_outer)
        element_mask_values = threshold_mask(element_mask_raw, temperature)
        element_mask_values = smooth_mask_differentiable(element_mask_values, smoothing_weights)
        unet_materials = E_b + (E_i - E_b) * element_mask_values
        center_x, center_y, fixed_region_radius = estimate_inclusion_geometry(
            element_mask_values, centroids, R_outer
        )
    
    # Compute SSIM
    gt_materials_torch = torch.from_numpy(gt_materials_np).to(device)
    continuous_ssim = compute_ssim(gt_materials_torch, unet_materials, E_b, E_i)
    
    # Convert to numpy
    unet_materials_np = unet_materials.cpu().numpy()
    center_x_np = center_x.cpu().numpy()
    center_y_np = center_y.cpu().numpy()
    
    # Post-process
    hard_materials_np, smooth_contour, properties = levelset_postprocess_universal(
        unet_materials_np, centroids_np, E_b, E_i, smoothing=0.001, n_contour_points=200
    )
    
    if hard_materials_np is not None:
        hard_materials_torch = torch.from_numpy(hard_materials_np).to(device)
        hard_ssim = compute_ssim(gt_materials_torch, hard_materials_torch, E_b, E_i)
        correct_elements = (hard_materials_np == gt_materials_np).sum()
        accuracy = correct_elements / len(gt_materials_np)
    else:
        hard_ssim = torch.tensor(0.0)
        accuracy = 0.0
        properties = None
    
    # Create figure with 2 rows
    fig = plt.figure(figsize=(24, 14))
    
    # Top row: 3 panels (ground truth, continuous, hard)
    gs_top = fig.add_gridspec(2, 3, height_ratios=[2, 1], hspace=0.3, wspace=0.3,
                              left=0.05, right=0.95, top=0.92, bottom=0.08)
    
    axes_top = [fig.add_subplot(gs_top[0, i]) for i in range(3)]
    ax_loss = fig.add_subplot(gs_top[1, :])
    
    theta_plot = np.linspace(0, 2*np.pi, 100)
    
    # Panel 1: Ground truth
    ax = axes_top[0]
    for e, elem in enumerate(elements_np):
        elem_coords = points_np[elem]
        color = 'red' if gt_materials_np[e] == E_i else 'blue'
        tri = plt.Polygon(elem_coords, facecolor=color, edgecolor='black', 
                          alpha=0.5, linewidth=0.3)
        ax.add_patch(tri)
    
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
    
    # Panel 2: U-Net continuous
    ax = axes_top[1]
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
    
    sm = ScalarMappable(cmap=plt.cm.RdYlBu_r, norm=Normalize(vmin=E_b, vmax=E_i))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Young's Modulus (E)", rotation=270, labelpad=20)
    
    # Panel 3: Level-set hard
    ax = axes_top[2]
    if hard_materials_np is not None:
        for e, elem in enumerate(elements_np):
            elem_coords = points_np[elem]
            color = 'red' if hard_materials_np[e] == E_i else 'blue'
            tri = plt.Polygon(elem_coords, facecolor=color, edgecolor='black', 
                              alpha=0.5, linewidth=0.3)
            ax.add_patch(tri)
    
    ax.plot(R_outer*np.cos(theta_plot), R_outer*np.sin(theta_plot), 'k-', 
            linewidth=2, label='Outer boundary')
    
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
    
    title_str = f'Level-Set Hard Boundary\n(SSIM: {hard_ssim.item():.4f}, Acc: {accuracy:.1%}'
    if properties is not None:
        title_str += f', Area: {properties["area"]:.3f}'
    title_str += ')'
    
    ax.set_title(title_str, fontsize=12, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Loss curve panel
    if len(loss_history) > 0:
        iterations = [item[0] for item in loss_history]
        losses = [item[1] for item in loss_history]
        
        ax_loss.plot(iterations, losses, 'b-', linewidth=2, label='Total Loss')
        ax_loss.scatter([iteration], [loss_history[-1][1]], c='red', s=200, 
                       marker='o', zorder=10, label=f'Current (iter {iteration})')
        ax_loss.axvline(x=iteration, color='red', linestyle='--', alpha=0.5)
        
        ax_loss.set_xlabel('Iteration', fontsize=12)
        ax_loss.set_ylabel('Loss', fontsize=12)
        ax_loss.set_title('Training Loss Curve', fontsize=12, fontweight='bold')
        ax_loss.legend()
        ax_loss.grid(True, alpha=0.3)
        ax_loss.set_xlim(0, num_iterations)
        
        if len(losses) > 0:
            y_min, y_max = min(losses), max(losses)
            y_range = y_max - y_min
            ax_loss.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
    
    # Super title
    minutes = int(current_time // 60)
    seconds = int(current_time % 60)
    current_loss = loss_history[-1][1] if len(loss_history) > 0 else 0.0
    
    fig.suptitle(f'Iteration: {iteration}/{num_iterations}  |  Loss: {current_loss:.6f}  |  Time: {minutes:02d}:{seconds:02d}', 
                 fontsize=16, fontweight='bold')
    
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    model.train()

# ============================================================================
# LOAD DATASETS (ANGULAR + SYMMETRIC)
# ============================================================================
print(f"\nLoading angular scanning dataset: {dataset_filename}")
dataset_angular = torch.load(dataset_filename)

force_vectors_angular = dataset_angular['force_vectors'].to(device)
boundary_displacements_angular = dataset_angular['boundary_displacements'].to(device)
n_pairs = dataset_angular['n_pairs']
boundary_nodes = dataset_angular['boundary_nodes'].to(device)
points_dataset = dataset_angular['points'].to(device)

n_samples_angular = force_vectors_angular.shape[0]
print(f"  Angular samples: {n_samples_angular}")
print(f"  Boundary nodes: {len(boundary_nodes)}")

# Load symmetric scanning dataset
symmetric_dataset_filename = 'symmetric_scanning_dataset.pt'
print(f"\nLoading symmetric scanning dataset: {symmetric_dataset_filename}")
try:
    dataset_symmetric = torch.load(symmetric_dataset_filename)
    
    force_vectors_symmetric = dataset_symmetric['force_vectors'].to(device)
    boundary_displacements_symmetric = dataset_symmetric['boundary_displacements'].to(device)
    rotation_angles = dataset_symmetric['rotation_angles']
    
    n_samples_symmetric = force_vectors_symmetric.shape[0]
    print(f"  Symmetric samples: {n_samples_symmetric}")
    print(f"  Rotation angles: {rotation_angles[0]:.1f}° to {rotation_angles[-1]:.1f}°")
    
    use_symmetric = True
except FileNotFoundError:
    print(f"  ⚠ Symmetric dataset not found - training without symmetry data")
    print(f"  Run symmetric_scanning.py to generate it!")
    use_symmetric = False
    force_vectors_symmetric = None
    boundary_displacements_symmetric = None
    n_samples_symmetric = 0

print(f"\nTotal training samples: {n_samples_angular + n_samples_symmetric}")
print(f"  - Angular scanning: {n_samples_angular} samples (diverse force configs)")
print(f"  - Symmetric scanning: {n_samples_symmetric} samples (rotational invariance)")

# Keep backward compatibility variable names for mesh creation
force_vectors = force_vectors_angular  # Used for n_dof calculation
boundary_displacements_measured = boundary_displacements_angular
n_samples = n_samples_angular

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

centroids = torch.mean(points[elements], dim=1)

# ============================================================================
# CREATE GROUND TRUTH MATERIALS
# ============================================================================
print("\nCreating ground truth materials...")

dists = torch.sqrt(centroids[:, 0]**2 + centroids[:, 1]**2)
gt_materials = torch.where(dists <= R_inner,
                           torch.tensor(E_i, dtype=torch.float64, device=device),
                           torch.tensor(E_b, dtype=torch.float64, device=device))

print(f"  Background elements (E={E_b}): {torch.sum(gt_materials == E_b)}")
print(f"  Inclusion elements (E={E_i}): {torch.sum(gt_materials == E_i)}")

# Convert to numpy for visualization
points_np = points.cpu().numpy()
elements_np = elements.cpu().numpy()
centroids_np = centroids.cpu().numpy()
gt_materials_np = gt_materials.cpu().numpy()

# ============================================================================
# SETUP INPUT GRID FOR U-NET
# ============================================================================
x_grid = torch.linspace(-R_outer, R_outer, grid_size, dtype=torch.float64, device=device)
y_grid = torch.linspace(-R_outer, R_outer, grid_size, dtype=torch.float64, device=device)
X_grid, Y_grid = torch.meshgrid(x_grid, y_grid, indexing='ij')
coords_input = torch.stack([X_grid, Y_grid], dim=0).unsqueeze(0)

# ============================================================================
# GRID SEARCH FOR HYPERPARAMETERS (INCLUDING LEARNING RATE)
# ============================================================================

if GRID_SEARCH_ENABLED:
    print("\n" + "="*70)
    print("GRID SEARCH FOR HYPERPARAMETERS (INCLUDING LEARNING RATE)")
    print("="*70)
    
    # Generate all combinations
    param_names = list(grid_search_space.keys())
    param_values = list(grid_search_space.values())
    all_combinations = list(itertools.product(*param_values))
    
    print(f"\nTesting {len(all_combinations)} configurations for {GRID_SEARCH_ITERATIONS} iterations each")
    print(f"Total grid search iterations: {len(all_combinations) * GRID_SEARCH_ITERATIONS}")
    
    grid_search_results = []
    grid_search_start_time = time.time()
    
    for config_idx, config in enumerate(all_combinations):
        config_dict = dict(zip(param_names, config))
        
        print(f"\n[{config_idx+1}/{len(all_combinations)}] Testing config:")
        print(f"  learning_rate={config_dict['learning_rate']:.2e}, lambda_tv={config_dict['lambda_tv']}")
        print(f"  smoothing_sigma={config_dict['smoothing_sigma']}, temperature={config_dict['temperature']}")
        print(f"  bc_sharpness={config_dict['bc_sharpness']}")
        
        # Initialize fresh model for this configuration
        model_test = UNet(in_channels=2, out_channels=1, base_features=32)
        model_test = model_test.to(device).to(torch.float64)
        model_test.train()
        
        # Use the learning rate from this configuration
        optimizer_test = torch.optim.Adam(model_test.parameters(), lr=config_dict['learning_rate'])
        
        # Run training for GRID_SEARCH_ITERATIONS
        total_loss_sum = 0.0
        final_ssim = 0.0
        
        # Precompute smoothing weights for this config's sigma
        gs_smoothing_weights = precompute_smoothing_weights(centroids, config_dict['smoothing_sigma'])

        for iteration in range(GRID_SEARCH_ITERATIONS):
            loss, metrics = forward_pass_and_loss(
                model_test, coords_input, points, elements, centroids,
                force_vectors_angular, boundary_displacements_angular,
                force_vectors_symmetric, boundary_displacements_symmetric,
                boundary_nodes, gt_materials, n_samples_angular, n_samples_symmetric,
                n_dof, device,
                config_dict['lambda_tv'],
                gs_smoothing_weights,
                config_dict['temperature'],
                config_dict['bc_sharpness'],
                use_symmetric
            )
            
            optimizer_test.zero_grad()
            loss.backward()
            optimizer_test.step()
            
            total_loss_sum += metrics['loss']
            if iteration == GRID_SEARCH_ITERATIONS - 1:
                final_ssim = metrics['ssim']
        
        avg_loss = total_loss_sum / GRID_SEARCH_ITERATIONS
        
        result = {
            'config': config_dict,
            'avg_loss': avg_loss,
            'final_ssim': final_ssim,
            'final_loss': metrics['loss']
        }
        grid_search_results.append(result)
        
        print(f"  → Avg Loss: {avg_loss:.6f}, Final SSIM: {final_ssim:.4f}")
        
        # Clean up
        del model_test
        del optimizer_test
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Find best configuration
    best_result = min(grid_search_results, key=lambda x: x['avg_loss'])
    best_config = best_result['config']
    
    grid_search_end_time = time.time()
    grid_search_duration = grid_search_end_time - grid_search_start_time
    
    print("\n" + "="*70)
    print("GRID SEARCH COMPLETE")
    print("="*70)
    print(f"Time: {grid_search_duration:.1f}s ({grid_search_duration/60:.1f} min)")
    print(f"\nBest configuration (by average loss):")
    print(f"  learning_rate: {best_config['learning_rate']:.2e}")
    print(f"  lambda_tv: {best_config['lambda_tv']}")
    print(f"  smoothing_sigma: {best_config['smoothing_sigma']}")
    print(f"  temperature: {best_config['temperature']}")
    print(f"  bc_sharpness: {best_config['bc_sharpness']}")
    print(f"  Average Loss: {best_result['avg_loss']:.6f}")
    print(f"  Final SSIM: {best_result['final_ssim']:.4f}")
    
    # Save grid search results
    grid_search_save_path = 'grid_search_results_with_lr.txt'
    with open(grid_search_save_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("GRID SEARCH RESULTS (INCLUDING LEARNING RATE)\n")
        f.write("="*70 + "\n\n")
        
        # Sort by average loss
        sorted_results = sorted(grid_search_results, key=lambda x: x['avg_loss'])
        
        for idx, result in enumerate(sorted_results):
            f.write(f"Rank {idx+1}:\n")
            f.write(f"  learning_rate: {result['config']['learning_rate']:.2e}\n")
            f.write(f"  lambda_tv: {result['config']['lambda_tv']}\n")
            f.write(f"  smoothing_sigma: {result['config']['smoothing_sigma']}\n")
            f.write(f"  temperature: {result['config']['temperature']}\n")
            f.write(f"  bc_sharpness: {result['config']['bc_sharpness']}\n")
            f.write(f"  Average Loss: {result['avg_loss']:.6f}\n")
            f.write(f"  Final SSIM: {result['final_ssim']:.4f}\n")
            f.write(f"  Final Loss: {result['final_loss']:.6f}\n")
            f.write("\n")
    
    print(f"\n✓ Grid search results saved to: {grid_search_save_path}")
    
    # Use best hyperparameters for main training
    learning_rate = best_config['learning_rate']
    lambda_tv = best_config['lambda_tv']
    smoothing_sigma = best_config['smoothing_sigma']
    temperature = best_config['temperature']
    bc_sharpness = best_config['bc_sharpness']
    soft_min_temp = 0.1  # Keep this fixed

else:
    # Use default hyperparameters if grid search is disabled
    learning_rate = 1e-3
    lambda_tv = 0.01
    smoothing_sigma = 0.05
    temperature = 1000.0
    bc_sharpness = 20.0
    soft_min_temp = 0.1

# Precompute smoothing weights for main training (constant for fixed mesh + sigma)
smoothing_weights = precompute_smoothing_weights(centroids, smoothing_sigma)

print("\n" + "="*70)
print("STARTING MAIN TRAINING WITH SELECTED HYPERPARAMETERS")
print("="*70)
print(f"  learning_rate: {learning_rate:.2e}")
print(f"  lambda_tv: {lambda_tv}")
print(f"  smoothing_sigma: {smoothing_sigma}")
print(f"  temperature: {temperature}")
print(f"  bc_sharpness: {bc_sharpness}")
print(f"  Iterations: {num_iterations}")
print("="*70)

# ============================================================================
# INITIALIZE U-NET FOR MAIN TRAINING
# ============================================================================
print("\nInitializing U-Net for main training...")

model = UNet(in_channels=2, out_channels=1, base_features=32)
model = model.to(device).to(torch.float64)
model.train()

n_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {n_params:,}")
print(f"  Input shape: {coords_input.shape}")

# ============================================================================
# SETUP OPTIMIZER AND SCHEDULER
# ============================================================================
print("\nSetting up optimizer...")
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

final_lr = learning_rate / 100  # Decay to 1% of initial learning rate
decay_start = int(num_iterations * 0.5)  # Start decay at 50% of training
decay_iterations = num_iterations - decay_start
gamma = np.exp(np.log(final_lr / learning_rate) / decay_iterations) if decay_iterations > 0 else 1.0

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda iteration: 1.0 if iteration < decay_start else gamma ** (iteration - decay_start)
)

# ============================================================================
# SETUP ANIMATION
# ============================================================================
frames_dir = Path('animation_frames')
frames_dir.mkdir(exist_ok=True)

checkpoint_dir = Path('checkpoints')
checkpoint_dir.mkdir(exist_ok=True)

loss_history = []

print("\n" + "="*70)
print("TRAINING WITH ANIMATION")
print("="*70)
print(f"  Iterations: {num_iterations}")
print(f"  Animation checkpoints: First 20 iterations + every 50 after")
print(f"  Total frames: {len(animation_checkpoints)}")
print(f"  Animation FPS: {animation_fps}")
print()

# ============================================================================
# PRE-TRAINING STANDALONE VISUALIZATION
# ============================================================================
print("\n" + "="*70)
print("SAVING PRE-TRAINING VISUALIZATION")
print("="*70)

print("Creating pre-training visualization (iteration 0)...")
main_training_start_time = time.time()
current_time = 0.0
loss_history.append((0, 0.0))  # Placeholder loss

# Create standalone pre-training image (no loss curve, just 3 panels)
model.eval()
with torch.no_grad():
    output_mask = model(coords_input)
    element_mask_raw = sample_mask_at_centroids(output_mask, centroids, R_outer)
    element_mask_values = threshold_mask(element_mask_raw, temperature)
    element_mask_values = smooth_mask_differentiable(element_mask_values, smoothing_weights)
    unet_materials = E_b + (E_i - E_b) * element_mask_values
    center_x, center_y, fixed_region_radius = estimate_inclusion_geometry(
        element_mask_values, centroids, R_outer
    )

gt_materials_torch = torch.from_numpy(gt_materials_np).to(device)
continuous_ssim = compute_ssim(gt_materials_torch, unet_materials, E_b, E_i)

unet_materials_np_pre = unet_materials.cpu().numpy()
center_x_np = center_x.cpu().numpy()
center_y_np = center_y.cpu().numpy()

hard_materials_np_pre, smooth_contour_pre, properties_pre = levelset_postprocess_universal(
    unet_materials_np_pre, centroids_np, E_b, E_i, smoothing=0.001, n_contour_points=200
)

if hard_materials_np_pre is not None:
    hard_materials_torch = torch.from_numpy(hard_materials_np_pre).to(device)
    hard_ssim = compute_ssim(gt_materials_torch, hard_materials_torch, E_b, E_i)
    correct_elements = (hard_materials_np_pre == gt_materials_np).sum()
    accuracy = correct_elements / len(gt_materials_np)
else:
    hard_ssim = torch.tensor(0.0)
    accuracy = 0.0
    properties_pre = None

# Create single-row figure (just the 3 panels)
fig, axes = plt.subplots(1, 3, figsize=(24, 7))
theta_plot = np.linspace(0, 2*np.pi, 100)

# Panel 1: Ground truth
ax = axes[0]
for e, elem in enumerate(elements_np):
    elem_coords = points_np[elem]
    color = 'red' if gt_materials_np[e] == E_i else 'blue'
    tri = plt.Polygon(elem_coords, facecolor=color, edgecolor='black', 
                      alpha=0.5, linewidth=0.3)
    ax.add_patch(tri)

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

# Panel 2: U-Net continuous (pre-training)
ax = axes[1]
for e, elem in enumerate(elements_np):
    elem_coords = points_np[elem]
    mat_val = unet_materials_np_pre[e]
    intensity = (mat_val - E_b) / (E_i - E_b)
    color = plt.cm.RdYlBu_r(intensity)
    tri = plt.Polygon(elem_coords, facecolor=color, edgecolor='black', 
                      alpha=0.7, linewidth=0.3)
    ax.add_patch(tri)

ax.plot(R_outer*np.cos(theta_plot), R_outer*np.sin(theta_plot), 'k-', 
        linewidth=2, label='Outer boundary')
ax.scatter([center_x_np], [center_y_np], c='green', s=200, marker='x', 
           linewidths=3, label='Predicted center', zorder=10)

if smooth_contour_pre is not None:
    ax.plot(smooth_contour_pre[:, 0], smooth_contour_pre[:, 1], 'cyan', linewidth=2.5, 
            label='Extracted contour', linestyle='-', alpha=0.9)

ax.set_xlim(-R_outer*1.1, R_outer*1.1)
ax.set_ylim(-R_outer*1.1, R_outer*1.1)
ax.set_aspect('equal')
ax.set_title(f'U-Net Continuous Prediction (Pre-Training)\n(SSIM: {continuous_ssim.item():.4f})', 
             fontsize=12, fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
ax.grid(True, alpha=0.3)

sm = ScalarMappable(cmap=plt.cm.RdYlBu_r, norm=Normalize(vmin=E_b, vmax=E_i))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Young's Modulus (E)", rotation=270, labelpad=20)

# Panel 3: Level-set hard (pre-training)
ax = axes[2]
if hard_materials_np_pre is not None:
    for e, elem in enumerate(elements_np):
        elem_coords = points_np[elem]
        color = 'red' if hard_materials_np_pre[e] == E_i else 'blue'
        tri = plt.Polygon(elem_coords, facecolor=color, edgecolor='black', 
                          alpha=0.5, linewidth=0.3)
        ax.add_patch(tri)

ax.plot(R_outer*np.cos(theta_plot), R_outer*np.sin(theta_plot), 'k-', 
        linewidth=2, label='Outer boundary')

if smooth_contour_pre is not None:
    ax.plot(smooth_contour_pre[:, 0], smooth_contour_pre[:, 1], 'lime', linewidth=2.5, 
            label='Smoothed boundary', linestyle='-', alpha=0.9)
    
    if properties_pre is not None:
        ax.scatter([properties_pre['center'][0]], [properties_pre['center'][1]], 
                  c='lime', s=200, marker='x', linewidths=3, 
                  label=f"Center ({properties_pre['center'][0]:.3f}, {properties_pre['center'][1]:.3f})", 
                  zorder=10)

ax.set_xlim(-R_outer*1.1, R_outer*1.1)
ax.set_ylim(-R_outer*1.1, R_outer*1.1)
ax.set_aspect('equal')

title_str = f'Level-Set Hard Boundary (Pre-Training)\n(SSIM: {hard_ssim.item():.4f}, Acc: {accuracy:.1%}'
if properties_pre is not None:
    title_str += f', Area: {properties_pre["area"]:.3f}'
title_str += ')'

ax.set_title(title_str, fontsize=12, fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
ax.grid(True, alpha=0.3)

fig.suptitle('Pre-Training Results (Iteration 0) - Random Initialization', 
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout()
pre_training_path = 'pre_training_visualization.png'
plt.savefig(pre_training_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {pre_training_path}")
plt.close()

model.train()

# Create initial animation frame
create_animation_frame(
    0, current_time, loss_history, model, coords_input,
    points_np, elements_np, centroids_np, gt_materials_np,
    temperature, smoothing_weights,
    frames_dir / 'frame_0000.png'
)

# ============================================================================
# TRAINING LOOP
# ============================================================================
for iteration in range(num_iterations):
    loss, metrics = forward_pass_and_loss(
        model, coords_input, points, elements, centroids,
        force_vectors_angular, boundary_displacements_angular,
        force_vectors_symmetric, boundary_displacements_symmetric,
        boundary_nodes, gt_materials, n_samples_angular, n_samples_symmetric,
        n_dof, device, lambda_tv, smoothing_weights, temperature, bc_sharpness,
        use_symmetric
    )
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    # Track loss history
    loss_history.append((iteration + 1, metrics['loss']))
    
    if iteration % log_every == 0:
        current_lr = scheduler.get_last_lr()[0]
        log_str = (f"Iter {iteration:4d} | "
                  f"Loss: {metrics['loss']:.6f} "
                  f"(Ang: {metrics['angular_loss']:.6f}")
        
        if use_symmetric:
            log_str += f", Sym: {metrics['symmetric_loss']:.6f}"
        
        log_str += (f", TV: {metrics['tv_loss']:.6f}) | "
                   f"SSIM: {metrics['ssim']:.4f} | "
                   f"LR: {current_lr:.2e}")
        
        print(log_str)
    
    # Check if this iteration is a checkpoint for animation
    if (iteration + 1) in animation_checkpoints:
        checkpoint_path = checkpoint_dir / f'checkpoint_{iteration+1:04d}.pt'
        torch.save({
            'iteration': iteration + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
            'metrics': metrics,
            'hyperparameters': {
                'learning_rate': learning_rate,
                'lambda_tv': lambda_tv,
                'smoothing_sigma': smoothing_sigma,
                'temperature': temperature,
                'bc_sharpness': bc_sharpness
            }
        }, checkpoint_path)
        
        # Create animation frame
        current_time = time.time() - main_training_start_time
        frame_path = frames_dir / f'frame_{iteration+1:04d}.png'
        print(f"  → Creating frame: {frame_path.name}")
        create_animation_frame(
            iteration + 1, current_time, loss_history, model, coords_input,
            points_np, elements_np, centroids_np, gt_materials_np,
            temperature, smoothing_weights, frame_path
        )

print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)

training_end_time = time.time()
training_seconds = training_end_time - main_training_start_time
training_minutes = int(training_seconds // 60)
training_secs = int(training_seconds % 60)
print(f"\nMain training time: {training_minutes:02d}:{training_secs:02d} (mm:ss)")

# ============================================================================
# POST-TRAINING STANDALONE VISUALIZATION
# ============================================================================
print("\n" + "="*70)
print("SAVING POST-TRAINING VISUALIZATION")
print("="*70)

print("Creating post-training visualization (final results)...")

# Run final inference
model.eval()
with torch.no_grad():
    output_mask = model(coords_input)
    element_mask_raw = sample_mask_at_centroids(output_mask, centroids, R_outer)
    element_mask_values = threshold_mask(element_mask_raw, temperature)
    element_mask_values = smooth_mask_differentiable(element_mask_values, smoothing_weights)
    unet_materials = E_b + (E_i - E_b) * element_mask_values
    center_x, center_y, fixed_region_radius = estimate_inclusion_geometry(
        element_mask_values, centroids, R_outer
    )

# Compute continuous SSIM
continuous_ssim = compute_ssim(gt_materials_torch, unet_materials, E_b, E_i)

# Convert to numpy for post-processing
unet_materials_np = unet_materials.cpu().numpy()
center_x_np = center_x.cpu().numpy()
center_y_np = center_y.cpu().numpy()

# Apply level-set post-processing
hard_materials_np, smooth_contour, properties = levelset_postprocess_universal(
    unet_materials_np, centroids_np, E_b, E_i,
    smoothing=0.001,
    n_contour_points=200
)

# Convert back to torch for SSIM computation
hard_materials = torch.from_numpy(hard_materials_np).to(device)
hard_ssim = compute_ssim(gt_materials_torch, hard_materials, E_b, E_i)

# Compute accuracy
correct_elements = (hard_materials_np == gt_materials_np).sum()
accuracy = correct_elements / n_elements

print(f"\nFinal Metrics:")
print(f"  Continuous SSIM: {continuous_ssim.item():.4f}")
print(f"  Hard SSIM: {hard_ssim.item():.4f}")
print(f"  Element accuracy: {accuracy:.2%} ({int(correct_elements)}/{n_elements})")

# Create single-row figure (just the 3 panels)
print("\nCreating visualization...")
fig, axes = plt.subplots(1, 3, figsize=(24, 7))
theta_plot = np.linspace(0, 2*np.pi, 100)

# Panel 1: Ground truth
ax = axes[0]
for e, elem in enumerate(elements_np):
    elem_coords = points_np[elem]
    color = 'red' if gt_materials_np[e] == E_i else 'blue'
    tri = plt.Polygon(elem_coords, facecolor=color, edgecolor='black', 
                      alpha=0.5, linewidth=0.3)
    ax.add_patch(tri)

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

# Panel 2: U-Net continuous prediction
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

sm = ScalarMappable(cmap=plt.cm.RdYlBu_r, norm=Normalize(vmin=E_b, vmax=E_i))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Young's Modulus (E)", rotation=270, labelpad=20)

# Panel 3: Level-set hard boundary
ax = axes[2]
for e, elem in enumerate(elements_np):
    elem_coords = points_np[elem]
    color = 'red' if hard_materials_np[e] == E_i else 'blue'
    tri = plt.Polygon(elem_coords, facecolor=color, edgecolor='black', 
                      alpha=0.5, linewidth=0.3)
    ax.add_patch(tri)

ax.plot(R_outer*np.cos(theta_plot), R_outer*np.sin(theta_plot), 'k-', 
        linewidth=2, label='Outer boundary')

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

title_str = f'Level-Set Hard Boundary\n(SSIM: {hard_ssim.item():.4f}, Acc: {accuracy:.1%}'
if properties is not None:
    title_str += f', Area: {properties["area"]:.3f}'
title_str += ')'

ax.set_title(title_str, fontsize=12, fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
ax.grid(True, alpha=0.3)

fig.suptitle(f'Final Results After {num_iterations} Iterations (Grid Search Optimized with LR)', 
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout()
post_training_path = 'post_training_visualization.png'
plt.savefig(post_training_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {post_training_path}")
plt.close()

# ============================================================================
# CREATE ANIMATIONS (GIF AND MP4)
# ============================================================================
print("\n" + "="*70)
print("CREATING ANIMATIONS")
print("="*70)

# Collect all frames
frame_files = sorted(frames_dir.glob('frame_*.png'))
print(f"\nFound {len(frame_files)} frames")

if len(frame_files) > 0:
    # Load all frames as numpy arrays
    print(f"\nLoading {len(frame_files)} frames...")
    images = []
    for frame_file in frame_files:
        img = imageio.imread(frame_file)
        images.append(img)
    
    # Create GIF
    print("\nCreating GIF...")
    gif_path = 'training_animation.gif'
    duration_ms = int(frame_duration * 1000)
    
    pil_images = [Image.fromarray(img) for img in images]
    pil_images[0].save(
        gif_path,
        save_all=True,
        append_images=pil_images[1:],
        duration=duration_ms,
        loop=0
    )
    print(f"✓ GIF created: {gif_path}")
    
    gif_size_mb = os.path.getsize(gif_path) / (1024 * 1024)
    print(f"  File size: {gif_size_mb:.2f} MB")
    print(f"  Duration: {len(images) * frame_duration:.1f} seconds")
    print(f"  Frame duration: {frame_duration}s ({animation_fps} fps)")
    
    # Create MP4
    print("\n" + "="*70)
    print("CREATING MP4 ANIMATION")
    print("="*70)
    
    try:
        mp4_path = 'training_animation.mp4'
        print(f"Writing MP4: {mp4_path}")
        
        writer = imageio.get_writer(mp4_path, fps=animation_fps, codec='libx264', quality=8)
        
        for img in images:
            writer.append_data(img)
        
        writer.close()
        print(f"✓ MP4 created: {mp4_path}")
        
        mp4_size_mb = os.path.getsize(mp4_path) / (1024 * 1024)
        print(f"  File size: {mp4_size_mb:.2f} MB")
        
    except (ValueError, ImportError) as e:
        print(f"⚠ Could not create MP4: {str(e)}")
        print("\nTo enable MP4 creation, install ffmpeg plugin:")
        print("  pip install imageio[ffmpeg]")
        print("  or: pip install imageio-ffmpeg")

end_time = time.time()
total_seconds = end_time - start_time
total_minutes = int(total_seconds // 60)
total_secs = int(total_seconds % 60)

print("\n" + "="*70)
print("ALL DONE!")
print("="*70)
print(f"\nTotal time: {total_minutes:02d}:{total_secs:02d} (mm:ss)")
if GRID_SEARCH_ENABLED:
    gs_minutes = int(grid_search_duration // 60)
    gs_secs = int(grid_search_duration % 60)
    print(f"  Grid search: {gs_minutes:02d}:{gs_secs:02d}")
print(f"  Main training: {training_minutes:02d}:{training_secs:02d}")
animation_time = total_seconds - training_seconds - (grid_search_duration if GRID_SEARCH_ENABLED else 0)
anim_minutes = int(animation_time // 60)
anim_secs = int(animation_time % 60)
print(f"  Post-processing + Animation: {anim_minutes:02d}:{anim_secs:02d}")

print("\nOutput files:")
if GRID_SEARCH_ENABLED:
    print(f"  - grid_search_results_with_lr.txt (all {len(all_combinations)} configurations)")
print(f"  - pre_training_visualization.png (before training)")
print(f"  - post_training_visualization.png (after training)")
print(f"  - training_animation.gif")
print(f"  - training_animation.mp4")
print(f"  - animation_frames/ ({len(animation_checkpoints)} frames)")
print(f"  - checkpoints/ ({len(animation_checkpoints)} model checkpoints)")

if GRID_SEARCH_ENABLED:
    print(f"\nBest hyperparameters used:")
    print(f"  learning_rate: {learning_rate:.2e}")
    print(f"  lambda_tv: {lambda_tv}")
    print(f"  smoothing_sigma: {smoothing_sigma}")
    print(f"  temperature: {temperature}")
    print(f"  bc_sharpness: {bc_sharpness}")

print("\nDone!")