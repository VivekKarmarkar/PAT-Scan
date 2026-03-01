"""
U-Net Training Script for Inverse Problem - v5 UPGRADED FOR IRREGULAR INCLUSIONS
Train U-Net to predict material properties from boundary measurements
Post-processing: Universal level-set based hard boundary extraction for ANY shape
Visualization: Creates animated GIF and MP4 showing training evolution
NOW SUPPORTS: Both circular and irregular inclusions automatically!
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

# Default circular parameters (will be overwritten if using irregular geometry)
R_inner = 0.3
n_radial = 20
n_angular = 40

# Irregular hardcoded mesh parameters
n_radial_irregular = n_radial
n_angular_irregular = n_angular

# Material properties
E_b = 1.0
E_i = 10.0
nu = 0.3

# Training
num_iterations = 1000
learning_rate = 1e-4
log_every = 10

# Animation checkpoints: first 20 iterations individually, then every 50
animation_checkpoints = list(range(21)) + list(range(50, num_iterations + 1, 50))
animation_checkpoints = sorted(set(animation_checkpoints))

# Regularization
lambda_tv=0.001
#smoothing_sigma=0.01
smoothing_sigma=0.07

# Thresholding
#temperature=2000.0
temperature=5000.0
bc_sharpness = 40.0
soft_min_temp = 0.1

# Animation parameters
frame_duration = 0.5  # seconds per frame
animation_fps = int(1.0 / frame_duration)

print("="*70)
print("U-NET INVERSE PROBLEM TRAINING - V5 UPGRADED")
print("Supports both circular and irregular inclusions!")
print("Creating training evolution animation (GIF + MP4)")
print("="*70)

# Start timing
start_time = time.time()

# ============================================================================
# UTILITY FUNCTIONS FOR IRREGULAR GEOMETRY
# ============================================================================

def is_inside_irregular_inclusion_vectorized(x, y, center_x, center_y, R_base, a_coeffs, b_coeffs, n_modes):
    """Check if points are inside irregular inclusion"""
    x_local = x - center_x
    y_local = y - center_y
    r = torch.sqrt(x_local**2 + y_local**2)
    theta = torch.atan2(y_local, x_local)
    
    r_boundary = R_base * torch.ones_like(theta)
    for n in range(1, n_modes + 1):
        r_boundary = r_boundary + R_base * (a_coeffs[n-1] * torch.cos(n * theta) + 
                                            b_coeffs[n-1] * torch.sin(n * theta))
    r_boundary = torch.clamp(r_boundary, R_base * 0.5, R_base * 1.5)
    
    return r < r_boundary

def plot_irregular_boundary(ax, center_x, center_y, R_base, a_coeffs, b_coeffs, n_modes, 
                            device, **plot_kwargs):
    """Plot irregular inclusion boundary on given axis"""
    theta_sample = torch.linspace(0, 2*torch.pi, 200, dtype=torch.float64, device=device)
    r_sample = R_base * torch.ones_like(theta_sample)
    for n in range(1, n_modes + 1):
        r_sample = r_sample + R_base * (a_coeffs[n-1] * torch.cos(n * theta_sample) + 
                                        b_coeffs[n-1] * torch.sin(n * theta_sample))
    r_sample = torch.clamp(r_sample, R_base * 0.5, R_base * 1.5)
    
    theta_np = theta_sample.cpu().numpy()
    r_np = r_sample.cpu().numpy()
    x_boundary = center_x + r_np * np.cos(theta_np)
    y_boundary = center_y + r_np * np.sin(theta_np)
    
    return ax.plot(x_boundary, y_boundary, **plot_kwargs)

# ============================================================================
# HELPER FUNCTIONS (same as before)
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
                         n_dof, device, lambda_tv, smoothing_sigma, use_symmetric):
    """
    Complete forward pass: U-Net → materials → FEM → loss
    Now handles BOTH angular scanning and symmetric scanning datasets
    """
    
    # U-Net forward pass (same material prediction for all samples)
    output_mask = model(coords_input)
    element_mask_raw = sample_mask_at_centroids(output_mask, centroids, R_outer)
    element_mask_values = threshold_mask(element_mask_raw, temperature)
    element_mask_values = smooth_mask_differentiable(element_mask_values, centroids, smoothing_sigma)
    unet_materials = E_b + (E_i - E_b) * element_mask_values
    
    # Assemble stiffness matrix (same for all samples)
    K = assemble_stiffness_differentiable(points, elements, unet_materials, nu)
    
    # Estimate boundary conditions (same for all samples)
    center_x, center_y, fixed_region_radius = estimate_inclusion_geometry(
        element_mask_values, centroids, R_outer
    )
    fixed_weights, free_weights = compute_boundary_conditions(
        points, center_x, center_y, fixed_region_radius, bc_sharpness
    )
    
    # ========================================================================
    # LOSS 1: Angular scanning dataset (diverse force configurations)
    # ========================================================================
    angular_loss = 0.0
    for i in range(n_samples_angular):
        U, U_nodes = solve_fem_soft(
            K, force_vectors_angular[i], fixed_weights, free_weights, n_dof, device
        )
        boundary_disp_predicted = U_nodes[boundary_nodes]
        diff = (boundary_disp_predicted - boundary_displacements_angular[i]) / R_outer
        l2_loss = torch.sqrt(torch.sum(torch.square(diff))) / len(boundary_nodes)
        angular_loss += l2_loss
    
    angular_loss = angular_loss / n_samples_angular
    
    # ========================================================================
    # LOSS 2: Symmetric scanning dataset (rotational invariance)
    # ========================================================================
    symmetric_loss = 0.0
    if use_symmetric and n_samples_symmetric > 0:
        for i in range(n_samples_symmetric):
            U, U_nodes = solve_fem_soft(
                K, force_vectors_symmetric[i], fixed_weights, free_weights, n_dof, device
            )
            boundary_disp_predicted = U_nodes[boundary_nodes]
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
                          save_path, is_irregular, geometry_params):
    """Create a single frame for the animation with 2-row layout"""
    
    # Run inference
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
    
    if is_irregular:
        plot_irregular_boundary(ax, geometry_params['center_x'], geometry_params['center_y'],
                               geometry_params['R_base'], geometry_params['a_coeffs'],
                               geometry_params['b_coeffs'], geometry_params['n_modes'],
                               device, linewidth=2, color='r', linestyle='--',
                               label='True irregular inclusion', alpha=0.7)
    else:
        ax.plot(R_inner*np.cos(theta_plot), R_inner*np.sin(theta_plot), 'r--', 
                linewidth=2, label=f'True inclusion (r={R_inner})', alpha=0.7)
    
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
    
    geom_str = "Irregular" if is_irregular else "Circular"
    fig.suptitle(f'Iteration: {iteration}/{num_iterations}  |  Loss: {current_loss:.6f}  |  Time: {minutes:02d}:{seconds:02d}  |  Geometry: {geom_str}', 
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

# Check geometry type
metadata = dataset_angular.get('metadata', {})
geometry_type = metadata.get('geometry_type', 'circular')
is_irregular = (geometry_type == 'irregular')

print(f"\n🔧 Detected geometry type: {geometry_type.upper()}")

# Load geometry-specific parameters
geometry_params = {}
if is_irregular:
    print("  Loading irregular inclusion parameters...")
    geometry_params['center_x'] = dataset_angular['center_x']
    geometry_params['center_y'] = dataset_angular['center_y']
    geometry_params['R_base'] = dataset_angular['R_base']
    geometry_params['a_coeffs'] = dataset_angular['a_coeffs'].to(device)
    geometry_params['b_coeffs'] = dataset_angular['b_coeffs'].to(device)
    geometry_params['n_modes'] = dataset_angular['n_modes']
    print(f"  Inclusion center: ({geometry_params['center_x']}, {geometry_params['center_y']})")
    print(f"  Base radius: {geometry_params['R_base']}")
    print(f"  Fourier modes: {geometry_params['n_modes']}")
else:
    R_inner = dataset_angular['R_inner']
    print(f"  Inclusion radius: {R_inner}")

n_samples_angular = force_vectors_angular.shape[0]
print(f"  Angular samples: {n_samples_angular}")
print(f"  Boundary nodes: {len(boundary_nodes)}")

# Load symmetric scanning dataset
symmetric_dataset_filename = 'symmetric_scanning_dataset.pt'
print(f"\nLoading symmetric scanning dataset: {symmetric_dataset_filename}")
try:
    dataset_symmetric = torch.load(symmetric_dataset_filename)
    
    # Verify geometry type matches
    sym_metadata = dataset_symmetric.get('metadata', {})
    sym_geometry_type = sym_metadata.get('geometry_type', 'circular')
    
    if sym_geometry_type != geometry_type:
        print(f"  ⚠ WARNING: Symmetric dataset geometry ({sym_geometry_type}) doesn't match angular ({geometry_type})")
        print(f"  Skipping symmetric dataset")
        use_symmetric = False
        force_vectors_symmetric = None
        boundary_displacements_symmetric = None
        n_samples_symmetric = 0
    else:
        force_vectors_symmetric = dataset_symmetric['force_vectors'].to(device)
        boundary_displacements_symmetric = dataset_symmetric['boundary_displacements'].to(device)
        rotation_angles = dataset_symmetric['rotation_angles']
        
        n_samples_symmetric = force_vectors_symmetric.shape[0]
        print(f"  Symmetric samples: {n_samples_symmetric}")
        print(f"  Rotation angles: {rotation_angles[0]:.1f}° to {rotation_angles[-1]:.1f}°")
        
        use_symmetric = True
except FileNotFoundError:
    print(f"  ⚠ Symmetric dataset not found - training without symmetry data")
    print(f"  Run symmetric_scanning_upgraded.py to generate it!")
    use_symmetric = False
    force_vectors_symmetric = None
    boundary_displacements_symmetric = None
    n_samples_symmetric = 0

print(f"\nTotal training samples: {n_samples_angular + n_samples_symmetric}")
print(f"  - Angular scanning: {n_samples_angular} samples (diverse force configs)")
print(f"  - Symmetric scanning: {n_samples_symmetric} samples (rotational invariance)")

# Keep backward compatibility variable names for mesh creation
force_vectors = force_vectors_angular
boundary_displacements_measured = boundary_displacements_angular
n_samples = n_samples_angular

# ============================================================================
# CREATE OR LOAD MESH
# ============================================================================
if is_irregular:
    # For irregular geometry, regenerate mesh using same parameters
    print("\nRegenerating mesh for irregular geometry...")
    
    # Get mesh parameters from dataset
    #n_radial_dataset = dataset_angular['metadata'].get('n_radial', 20)
    #n_angular_dataset = dataset_angular['metadata'].get('n_angular', 40)
    n_radial_dataset = dataset_angular['metadata'].get('n_radial', n_radial_irregular)
    n_angular_dataset = dataset_angular['metadata'].get('n_angular', n_angular_irregular)
    
    print(f"  Using dataset mesh parameters: n_radial={n_radial_dataset}, n_angular={n_angular_dataset}")
    
    # Regenerate the same polar mesh
    r = torch.linspace(0, R_outer, n_radial_dataset, dtype=torch.float64, device=device)
    theta = torch.linspace(0, 2*torch.pi, n_angular_dataset+1, dtype=torch.float64, device=device)[:-1]
    
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
    
    # Create elements using same structure
    print("Creating elements...")
    
    elements = []
    
    def node_idx(i, j):
        if i == 0:
            return 0
        else:
            return 1 + (i - 1) * n_angular_dataset + (j % n_angular_dataset)
    
    for i in range(n_radial_dataset - 1):
        for j in range(n_angular_dataset):
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
    
else:
    # For circular geometry, create fresh polar mesh (original behavior)
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

gt_materials = torch.zeros(n_elements, dtype=torch.float64, device=device)

if is_irregular:
    # Use irregular inclusion boundary
    for i in range(n_elements):
        elem = elements[i]
        centroid = torch.mean(points[elem], dim=0)
        
        inside = is_inside_irregular_inclusion_vectorized(
            centroid[0], centroid[1],
            geometry_params['center_x'], geometry_params['center_y'],
            geometry_params['R_base'], geometry_params['a_coeffs'],
            geometry_params['b_coeffs'], geometry_params['n_modes']
        )
        
        if inside:
            gt_materials[i] = E_i
        else:
            gt_materials[i] = E_b
else:
    # Use circular inclusion
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

# Convert to numpy for visualization
points_np = points.cpu().numpy()
elements_np = elements.cpu().numpy()
centroids_np = centroids.cpu().numpy()
gt_materials_np = gt_materials.cpu().numpy()

# ============================================================================
# INITIALIZE U-NET
# ============================================================================
print("\nInitializing U-Net...")

model = UNet(in_channels=2, out_channels=1, base_features=32)
model = model.to(device).to(torch.float64)
model.train()

n_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {n_params:,}")

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

final_lr = 1e-5
decay_start = 500
decay_iterations = num_iterations - decay_start
gamma = np.exp(np.log(final_lr / learning_rate) / decay_iterations)

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
current_time = time.time() - start_time
loss_history.append((0, 0.0))  # Placeholder loss

# Create standalone pre-training image (no loss curve, just 3 panels)
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

if is_irregular:
    plot_irregular_boundary(ax, geometry_params['center_x'], geometry_params['center_y'],
                           geometry_params['R_base'], geometry_params['a_coeffs'],
                           geometry_params['b_coeffs'], geometry_params['n_modes'],
                           device, linewidth=2, color='r', linestyle='--',
                           label='True irregular inclusion', alpha=0.7)
else:
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

geom_str = "Irregular" if is_irregular else "Circular"
fig.suptitle(f'Pre-Training Results (Iteration 0) - Random Initialization ({geom_str} Inclusion)', 
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout()
pre_training_path = 'pre_training_visualization.png'
plt.savefig(pre_training_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {pre_training_path}")
plt.close()

model.train()

# Create initial animation frame
create_animation_frame(
    iteration=0,
    current_time=current_time,
    loss_history=loss_history,
    model=model,
    coords_input=coords_input,
    points_np=points_np,
    elements_np=elements_np,
    centroids_np=centroids_np,
    gt_materials_np=gt_materials_np,
    save_path=str(frames_dir / 'frame_00000.png'),
    is_irregular=is_irregular,
    geometry_params=geometry_params
)

# ============================================================================
# TRAINING LOOP WITH ANIMATION CHECKPOINTS
# ============================================================================
print("\n" + "="*70)
print("STARTING TRAINING")
print("="*70)

training_start = time.time()

for iteration in range(num_iterations):
    # Forward pass and compute loss
    loss, metrics = forward_pass_and_loss(
        model, coords_input, points, elements, centroids,
        force_vectors_angular, boundary_displacements_angular,
        force_vectors_symmetric, boundary_displacements_symmetric,
        boundary_nodes, gt_materials, n_samples_angular, n_samples_symmetric,
        n_dof, device, lambda_tv, smoothing_sigma, use_symmetric
    )
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    # Track loss history
    loss_history.append((iteration + 1, metrics['loss']))
    
    # Logging
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
    
    # Save animation checkpoints
    if (iteration + 1) in animation_checkpoints:
        current_time = time.time() - training_start
        frame_path = frames_dir / f'frame_{iteration+1:05d}.png'
        checkpoint_path = checkpoint_dir / f'checkpoint_{iteration+1:05d}.pt'
        
        create_animation_frame(
            iteration=iteration+1,
            current_time=current_time,
            loss_history=loss_history,
            model=model,
            coords_input=coords_input,
            points_np=points_np,
            elements_np=elements_np,
            centroids_np=centroids_np,
            gt_materials_np=gt_materials_np,
            save_path=str(frame_path),
            is_irregular=is_irregular,
            geometry_params=geometry_params
        )
        
        torch.save({
            'iteration': iteration + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': metrics['loss'],
            'ssim': metrics['ssim']
        }, checkpoint_path)

training_end = time.time()
training_time = training_end - training_start
training_minutes = int(training_time // 60)
training_secs = int(training_time % 60)

print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)
print(f"Training time: {training_minutes:02d}:{training_secs:02d} (mm:ss)")

# ============================================================================
# FINAL POST-TRAINING VISUALIZATION
# ============================================================================
print("\n" + "="*70)
print("SAVING POST-TRAINING VISUALIZATION")
print("="*70)

# Final inference
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
gt_materials_torch = torch.from_numpy(gt_materials_np).to(device)
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

if is_irregular:
    plot_irregular_boundary(ax, geometry_params['center_x'], geometry_params['center_y'],
                           geometry_params['R_base'], geometry_params['a_coeffs'],
                           geometry_params['b_coeffs'], geometry_params['n_modes'],
                           device, linewidth=2, color='r', linestyle='--',
                           label='True irregular inclusion', alpha=0.7)
else:
    ax.plot(R_inner*np.cos(theta_plot), R_inner*np.sin(theta_plot), 'r--', 
            linewidth=2, label=f'True inclusion (r={R_inner})', alpha=0.7)

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

geom_str = "Irregular" if is_irregular else "Circular"
fig.suptitle(f'Final Results After {num_iterations} Iterations ({geom_str} Inclusion)', 
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
print(f"\nGeometry Type: {geom_str}")
print(f"Total time: {total_minutes:02d}:{total_secs:02d} (mm:ss)")
print(f"Training: {training_minutes:02d}:{training_secs:02d}")
print(f"Post-processing + Animation: {total_minutes - training_minutes:02d}:{total_secs - training_secs:02d}")
print("\nOutput files:")
print(f"  - pre_training_visualization.png (before training)")
print(f"  - post_training_visualization.png (after training)")
print(f"  - training_animation.gif")
print(f"  - training_animation.mp4")
print(f"  - animation_frames/ ({len(animation_checkpoints)} frames)")
print(f"  - checkpoints/ ({len(animation_checkpoints)} model checkpoints)")

print("\nDone!")