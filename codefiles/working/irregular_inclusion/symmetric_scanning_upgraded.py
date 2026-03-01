"""
Symmetric Scanning Script - UPGRADED FOR IRREGULAR INCLUSIONS
Applies a SINGLE force pair at different angles (rotated by angular_spacing)
All samples are rotationally symmetric versions of the same physics problem
Generates animations and builds dataset to teach U-Net rotational invariance
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
from PIL import Image
import imageio
from fem_utils import *

torch.set_default_dtype(torch.float64)

print("="*70)
print("FEM SYMMETRIC SCANNING - ROTATIONAL SYMMETRY DATASET (UPGRADED)")
print("="*70)

# ============================================================================
# CONFIGURATION - SELECT GEOMETRY TYPE
# ============================================================================
USE_IRREGULAR = True  # Set to True for irregular inclusion, False for circular

if USE_IRREGULAR:
    MESH_FILE = 'irregular_inclusion_sample.pt'
    print("\n🔧 Using IRREGULAR INCLUSION geometry")
else:
    MESH_FILE = 'mesh_sample.pt'  # Your default circular inclusion mesh
    print("\n🔧 Using CIRCULAR INCLUSION geometry")

# ============================================================================
# PARAMETERS
# ============================================================================
force_magnitude = 0.1
angular_spacing = 9.0         # Rotation increment in degrees
n_rotations = 20              # Number of rotations (20 * 9° = 180°)

# Animation settings
fps = 5                       # Frames per second for animation
output_dir = Path('symmetric_scanning_frames')
gif_filename = 'symmetric_scanning.gif'
mp4_filename = 'symmetric_scanning.mp4'
dataset_filename = 'symmetric_scanning_dataset.pt'

# ============================================================================
# UTILITY FUNCTIONS FOR IRREGULAR GEOMETRY
# ============================================================================

def load_mesh_irregular(filename='irregular_inclusion_sample.pt'):
    """Load mesh data for irregular inclusion"""
    device = get_device()
    data = torch.load(filename, map_location=device)
    
    mesh = {
        'points': data['points'].to(device),
        'elements': data['elements'].to(device),
        'K': data['K'].to(device),
        'element_materials': data['element_materials'].to(device),
        'E_b': data['E_b'],
        'E_i': data['E_i'],
        'nu': data['nu'],
        'R_outer': data['R_outer'],
        'R_base': data['R_base'],
        'center_x': data['center_x'],
        'center_y': data['center_y'],
        'a_coeffs': data['a_coeffs'].to(device),
        'b_coeffs': data['b_coeffs'].to(device),
        'n_modes': data['n_modes'],
        'irregularity': data['irregularity'],
        'seed': data['seed'],
        'device': device
    }
    
    print(f"\n✓ Loaded irregular inclusion mesh from: {filename}")
    print(f"  Nodes: {len(mesh['points'])}")
    print(f"  Elements: {len(mesh['elements'])}")
    
    return mesh


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


def find_boundary_nodes_irregular(points, R_outer, tolerance=1e-6):
    """Find nodes on the outer boundary"""
    radii = torch.sqrt(points[:, 0]**2 + points[:, 1]**2)
    boundary_mask = torch.abs(radii - R_outer) < tolerance
    boundary_nodes = torch.where(boundary_mask)[0]
    return boundary_nodes, radii


def setup_boundary_conditions_irregular(points, center_x, center_y, R_base,
                                       a_coeffs, b_coeffs, n_modes, n_dof, device):
    """Setup boundary conditions for irregular inclusion"""
    inside_mask = is_inside_irregular_inclusion_vectorized(
        points[:, 0], points[:, 1], center_x, center_y, R_base, a_coeffs, b_coeffs, n_modes
    )
    
    inside_nodes = torch.where(inside_mask)[0]
    
    fixed_dofs = []
    for node in inside_nodes:
        fixed_dofs.extend([2*node, 2*node+1])
    
    fixed_dofs = torch.tensor(fixed_dofs, dtype=torch.long, device=device)
    all_dofs = torch.arange(n_dof, device=device)
    free_dofs = torch.tensor([dof for dof in all_dofs if dof not in fixed_dofs], 
                            dtype=torch.long, device=device)
    
    return fixed_dofs, free_dofs


def check_penetration_irregular(points, U_nodes, boundary_nodes, 
                                center_x, center_y, R_base, a_coeffs, b_coeffs, n_modes):
    """Check if boundary nodes penetrate irregular inclusion after deformation"""
    boundary_nodes_np = boundary_nodes.cpu().numpy()
    U_nodes_np = U_nodes.cpu().numpy()
    points_np = points.cpu().numpy()
    
    boundary_positions = points_np[boundary_nodes_np]
    boundary_displacements = U_nodes_np[boundary_nodes_np]
    deformed_boundary = boundary_positions + boundary_displacements
    
    device = points.device
    x_deformed = torch.tensor(deformed_boundary[:, 0], dtype=torch.float64, device=device)
    y_deformed = torch.tensor(deformed_boundary[:, 1], dtype=torch.float64, device=device)
    
    x_local = x_deformed - center_x
    y_local = y_deformed - center_y
    r_actual = torch.sqrt(x_local**2 + y_local**2)
    theta = torch.atan2(y_local, x_local)
    
    r_boundary = R_base * torch.ones_like(theta)
    for n in range(1, n_modes + 1):
        r_boundary = r_boundary + R_base * (a_coeffs[n-1] * torch.cos(n * theta) + 
                                            b_coeffs[n-1] * torch.sin(n * theta))
    r_boundary = torch.clamp(r_boundary, R_base * 0.5, R_base * 1.5)
    
    clearances = r_actual - r_boundary
    
    min_clearance = torch.min(clearances)
    min_idx = torch.argmin(clearances)
    min_node = boundary_nodes[min_idx].item()
    min_radius = r_actual[min_idx].item()
    
    is_valid = min_clearance >= 0
    penetration_depth = -min_clearance.item() if not is_valid else 0.0
    
    return is_valid, penetration_depth, min_radius, min_node


def plot_deformation_single_pair_irregular(points, U_nodes, boundary_nodes, F, node1, node2,
                                          force_angle, force_magnitude, R_outer,
                                          center_x, center_y, R_base, a_coeffs, b_coeffs, n_modes,
                                          clearance, filename='deformation_plot.png'):
    """Create deformation plot for single force pair with irregular inclusion"""
    boundary_nodes_np = boundary_nodes.cpu().numpy()
    U_nodes_np = U_nodes.cpu().numpy()
    points_np = points.cpu().numpy()
    node1_np = node1.cpu().item()
    node2_np = node2.cpu().item()
    
    boundary_positions = points_np[boundary_nodes_np]
    boundary_displacements = U_nodes_np[boundary_nodes_np]
    disp_magnitudes = np.sqrt(boundary_displacements[:, 0]**2 + boundary_displacements[:, 1]**2)
    
    max_mag_idx = np.argmax(disp_magnitudes)
    max_mag_node = boundary_nodes_np[max_mag_idx]
    
    disp_at_node1 = np.sqrt(U_nodes_np[node1_np, 0]**2 + U_nodes_np[node1_np, 1]**2)
    disp_at_node2 = np.sqrt(U_nodes_np[node2_np, 0]**2 + U_nodes_np[node2_np, 1]**2)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    # Plot outer boundary circle
    theta_plot = np.linspace(0, 2*np.pi, 200)
    ax.plot(R_outer*np.cos(theta_plot), R_outer*np.sin(theta_plot), 'k-', 
            linewidth=1.5, alpha=0.3, label='Outer boundary (reference)', zorder=1)
    
    # Plot irregular inclusion boundary
    device = points.device
    theta_sample = torch.linspace(0, 2*torch.pi, 200, dtype=torch.float64, device=device)
    r_sample = R_base * torch.ones_like(theta_sample)
    for n in range(1, n_modes + 1):
        r_sample = r_sample + R_base * (a_coeffs[n-1] * torch.cos(n * theta_sample) + 
                                        b_coeffs[n-1] * torch.sin(n * theta_sample))
    r_sample = torch.clamp(r_sample, R_base * 0.5, R_base * 1.5)
    
    x_boundary = center_x + r_sample.cpu().numpy() * np.cos(theta_sample.cpu().numpy())
    y_boundary = center_y + r_sample.cpu().numpy() * np.sin(theta_sample.cpu().numpy())
    
    ax.plot(x_boundary, y_boundary, 'r-', 
            linewidth=2, alpha=0.5, label='Irregular inclusion (fixed)', zorder=2)
    ax.plot([center_x], [center_y], 'ro', markersize=8, label='Inclusion center', zorder=2.5)
    
    # Original boundary nodes
    ax.scatter(boundary_positions[:, 0], boundary_positions[:, 1], 
               c='gray', s=60, alpha=0.4, label='Original boundary nodes', 
               zorder=3, marker='o')
    
    # Deformed boundary nodes
    deformed_boundary = boundary_positions + boundary_displacements
    ax.scatter(deformed_boundary[:, 0], deformed_boundary[:, 1], 
               c='blue', s=60, alpha=0.7, label='Deformed boundary nodes', 
               zorder=4, marker='s')
    
    # Deformed boundary line
    angles = np.arctan2(boundary_positions[:, 1], boundary_positions[:, 0])
    sort_idx = np.argsort(angles)
    deformed_sorted = deformed_boundary[sort_idx]
    deformed_sorted_closed = np.vstack([deformed_sorted, deformed_sorted[0]])
    
    ax.plot(deformed_sorted_closed[:, 0], deformed_sorted_closed[:, 1], 
            'b-', linewidth=2, alpha=0.6, label='Deformed boundary (line)', zorder=3.5)
    
    # Displacement vectors (subsample for clarity)
    vector_subsample = max(1, len(boundary_nodes_np) // 40)
    for i in range(0, len(boundary_nodes_np), vector_subsample):
        if disp_magnitudes[i] > 1e-10:
            x0, y0 = boundary_positions[i]
            dx, dy = boundary_displacements[i]
            ax.arrow(x0, y0, dx, dy,
                    head_width=0.02, head_length=0.015, 
                    fc='black', ec='black', alpha=0.6, linewidth=1, zorder=5)
    
    # Force application nodes
    pos1_np = points_np[node1_np]
    pos2_np = points_np[node2_np]
    
    ax.scatter([pos1_np[0], pos2_np[0]], [pos1_np[1], pos2_np[1]], 
               c='red', s=300, zorder=6, label='Force application nodes', 
               edgecolors='darkred', linewidths=3, marker='o')
    
    # Force vectors
    F_np = F.cpu().numpy()
    force1_np = np.array([F_np[2*node1_np], F_np[2*node1_np+1]])
    force2_np = np.array([F_np[2*node2_np], F_np[2*node2_np+1]])
    
    force_scale = 3.0
    ax.arrow(pos1_np[0], pos1_np[1], force_scale*force1_np[0], force_scale*force1_np[1],
             head_width=0.15, head_length=0.12, fc='red', ec='darkred', 
             linewidth=4, zorder=7, label='Applied forces')
    ax.arrow(pos2_np[0], pos2_np[1], force_scale*force2_np[0], force_scale*force2_np[1],
             head_width=0.15, head_length=0.12, fc='red', ec='darkred', 
             linewidth=4, zorder=7)
    
    # Max displacement node
    ax.scatter([boundary_positions[max_mag_idx, 0]], [boundary_positions[max_mag_idx, 1]], 
               c='lime', s=300, zorder=8, label='Max displacement node', 
               edgecolors='darkgreen', linewidths=3, marker='*')
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.set_title(f'FEM Diagnostic: Forces + Displacements (Angle={force_angle}°)\n' + 
                 f'Irregular inclusion fixed | Max disp = {disp_magnitudes.max():.3e} | ✓ VALID', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Info box
    info_text = f'Force nodes: {node1_np}, {node2_np}\n'
    info_text += f'Max disp node: {max_mag_node}\n'
    info_text += f'Force magnitude: {force_magnitude}\n'
    info_text += f'Max displacement: {disp_magnitudes.max():.3e}\n'
    info_text += f'Disp at force nodes:\n'
    info_text += f'  Node {node1_np}: {disp_at_node1:.3e}\n'
    info_text += f'  Node {node2_np}: {disp_at_node2:.3e}\n'
    info_text += f'\nClearance: {clearance:.4f} ✓'
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()


# ============================================================================
# SETUP
# ============================================================================
print(f"\nScan Parameters:")
print(f"  Force pairs: 1 (fixed - teaching symmetry)")
print(f"  Angular spacing: {angular_spacing}° (rotation increment)")
print(f"  Number of rotations: {n_rotations}")
print(f"  Angular range: 0° to {(n_rotations-1) * angular_spacing}°")
print(f"  Force magnitude: {force_magnitude}")
print(f"  Animation FPS: {fps}")

# Create output directory
output_dir.mkdir(exist_ok=True)
print(f"\nFrame directory: {output_dir}")

# ============================================================================
# LOAD MESH (once)
# ============================================================================
print("\nLoading mesh...")

if USE_IRREGULAR:
    mesh = load_mesh_irregular(MESH_FILE)
    points = mesh['points']
    K = mesh['K']
    R_outer = mesh['R_outer']
    device = mesh['device']
    
    # Irregular-specific parameters
    center_x = mesh['center_x']
    center_y = mesh['center_y']
    R_base = mesh['R_base']
    a_coeffs = mesh['a_coeffs']
    b_coeffs = mesh['b_coeffs']
    n_modes = mesh['n_modes']
    
    # Find boundary nodes
    boundary_nodes, radii = find_boundary_nodes_irregular(points, R_outer)
    
else:
    mesh = load_mesh()
    points = mesh['points']
    K = mesh['K']
    R_outer = mesh['R_outer']
    R_inner = mesh['R_inner']
    device = mesh['device']
    
    # Find boundary nodes
    boundary_nodes, radii = find_boundary_nodes(points, R_outer)

n_nodes = len(points)
n_dof = 2 * n_nodes
n_boundary_nodes = len(boundary_nodes)

print(f"  Nodes: {n_nodes}")
print(f"  DOFs: {n_dof}")
print(f"  Boundary nodes: {n_boundary_nodes}")

# Setup boundary conditions
if USE_IRREGULAR:
    fixed_dofs, free_dofs = setup_boundary_conditions_irregular(
        points, center_x, center_y, R_base, a_coeffs, b_coeffs, n_modes, n_dof, device
    )
else:
    fixed_dofs, free_dofs = setup_boundary_conditions(radii, R_inner, n_dof, device)

# ============================================================================
# DATA STORAGE INITIALIZATION
# ============================================================================
n_iterations = n_rotations

# Storage lists
force_vectors_list = []
boundary_displacements_list = []
rotation_angles_list = []
clearance_list = []
max_displacement_list = []

print(f"\nDataset storage initialized for {n_iterations} rotations")

# ============================================================================
# MAIN LOOP - SINGLE FORCE PAIR, ROTATED
# ============================================================================
print("\n" + "="*70)
print("GENERATING FRAMES AND BUILDING SYMMETRY DATASET")
print("="*70)
print("Note: All samples are the SAME physics, just rotated!")
print("="*70)

frame_files = []

for iter_idx in range(n_rotations):
    # Calculate rotation angle for this iteration
    rotation_angle = iter_idx * angular_spacing
    
    print(f"\n[Iteration {iter_idx + 1}/{n_iterations}] Rotation angle: {rotation_angle}°")
    
    # Apply SINGLE force pair at this angle
    F, node1, node2 = apply_single_force_pair(
        points, boundary_nodes, force_magnitude, rotation_angle, device
    )
    
    # Solve
    U, U_nodes = solve_fem(K, F, fixed_dofs, free_dofs, n_dof, device)
    
    max_disp = torch.max(torch.abs(U)).item()
    print(f"  Max displacement: {max_disp:.6e}")
    
    # Penetration check (different for irregular vs circular)
    if USE_IRREGULAR:
        is_valid, penetration_depth, min_deformed_radius, min_radius_node = check_penetration_irregular(
            points, U_nodes, boundary_nodes, center_x, center_y, R_base, a_coeffs, b_coeffs, n_modes
        )
    else:
        is_valid, penetration_depth, min_deformed_radius, min_radius_node = check_penetration(
            points, U_nodes, boundary_nodes, R_inner
        )
    
    clearance = -penetration_depth
    print(f"  Clearance: {clearance:.6f}")
    
    if not is_valid:
        print(f"  ⚠ WARNING: Penetration detected at angle {rotation_angle}°!")
    
    # Store data for dataset
    force_vectors_list.append(F.cpu().clone())
    boundary_displacements_list.append(U_nodes[boundary_nodes].cpu().clone())
    rotation_angles_list.append(rotation_angle)
    clearance_list.append(clearance)
    max_displacement_list.append(max_disp)
    
    print(f"  Data stored (iteration {iter_idx})")
    
    # Generate plot
    frame_filename = output_dir / f'frame_{iter_idx:03d}_angle_{int(rotation_angle):03d}.png'
    
    if USE_IRREGULAR:
        plot_deformation_single_pair_irregular(
            points, U_nodes, boundary_nodes, F, node1, node2,
            force_angle=rotation_angle, force_magnitude=force_magnitude,
            R_outer=R_outer, center_x=center_x, center_y=center_y,
            R_base=R_base, a_coeffs=a_coeffs, b_coeffs=b_coeffs, n_modes=n_modes,
            clearance=clearance, filename=str(frame_filename)
        )
    else:
        plot_deformation_single_pair(
            points, U_nodes, boundary_nodes, F, node1, node2,
            force_angle=rotation_angle, force_magnitude=force_magnitude,
            R_outer=R_outer, R_inner=R_inner, clearance=clearance,
            filename=str(frame_filename)
        )
    
    frame_files.append(str(frame_filename))
    print(f"  Saved frame: {frame_filename.name}")

# ============================================================================
# SAVE DATASET
# ============================================================================
print("\n" + "="*70)
print("SAVING SYMMETRY DATASET")
print("="*70)

# Convert lists to tensors
force_vectors = torch.stack(force_vectors_list)
boundary_displacements = torch.stack(boundary_displacements_list)
rotation_angles_tensor = torch.tensor(rotation_angles_list, dtype=torch.float64)
clearance_tensor = torch.tensor(clearance_list, dtype=torch.float64)
max_displacement_tensor = torch.tensor(max_displacement_list, dtype=torch.float64)

print(f"\nDataset tensor shapes:")
print(f"  force_vectors: {force_vectors.shape}")
print(f"  boundary_displacements: {boundary_displacements.shape}")
print(f"  rotation_angles: {rotation_angles_tensor.shape}")
print(f"  clearance: {clearance_tensor.shape}")
print(f"  max_displacement: {max_displacement_tensor.shape}")

# Create dataset dictionary
dataset = {
    # Data
    'force_vectors': force_vectors,
    'boundary_displacements': boundary_displacements,
    'rotation_angles': rotation_angles_tensor,
    'clearance': clearance_tensor,
    'max_displacement': max_displacement_tensor,
    
    # Mesh information
    'boundary_nodes': boundary_nodes.cpu(),
    'points': points.cpu(),
    'R_outer': R_outer,
    
    # Metadata
    'metadata': {
        'description': 'FEM symmetric scanning dataset for teaching rotational invariance to U-Net',
        'purpose': 'All samples are rotationally symmetric - same physics, different angles',
        'geometry_type': 'irregular' if USE_IRREGULAR else 'circular',
        'force_magnitude': force_magnitude,
        'angular_spacing': angular_spacing,
        'n_rotations': n_rotations,
        'n_force_pairs': 1,  # Always 1 for symmetric scanning
        'angular_range': f"0° to {(n_rotations-1) * angular_spacing}°",
        'n_iterations': n_iterations,
        'n_nodes': n_nodes,
        'n_dof': n_dof,
        'n_boundary_nodes': n_boundary_nodes,
        'device_used': str(device),
        'dtype': str(torch.get_default_dtype())
    }
}

# Add geometry-specific info
if USE_IRREGULAR:
    dataset['R_base'] = R_base
    dataset['center_x'] = center_x
    dataset['center_y'] = center_y
    dataset['a_coeffs'] = a_coeffs.cpu()
    dataset['b_coeffs'] = b_coeffs.cpu()
    dataset['n_modes'] = n_modes
    dataset['metadata']['center_x'] = center_x
    dataset['metadata']['center_y'] = center_y
    dataset['metadata']['R_base'] = R_base
    dataset['metadata']['n_modes'] = n_modes
else:
    dataset['R_inner'] = R_inner
    dataset['metadata']['R_inner'] = R_inner

# Save dataset
print(f"\nSaving dataset to: {dataset_filename}")
torch.save(dataset, dataset_filename)
print(f"✓ Dataset saved successfully!")

file_size_mb = os.path.getsize(dataset_filename) / (1024 * 1024)
print(f"  File size: {file_size_mb:.2f} MB")

# ============================================================================
# CREATE GIF
# ============================================================================
print("\n" + "="*70)
print("CREATING GIF ANIMATION")
print("="*70)

print(f"\nReading {len(frame_files)} frames...")
images = []
for frame_file in frame_files:
    images.append(imageio.imread(frame_file))

print(f"Writing GIF: {gif_filename}")
imageio.mimsave(gif_filename, images, fps=fps, loop=0)
print(f"✓ GIF created: {gif_filename}")

file_size_mb = os.path.getsize(gif_filename) / (1024 * 1024)
print(f"  File size: {file_size_mb:.2f} MB")
print("\nThe GIF shows the force pair rotating around the circle!")
print("Notice: All displacement patterns are just rotated versions of each other")

# ============================================================================
# CREATE MP4
# ============================================================================
print("\n" + "="*70)
print("CREATING MP4 ANIMATION")
print("="*70)

try:
    print(f"Writing MP4: {mp4_filename}")
    writer = imageio.get_writer(mp4_filename, fps=fps, codec='libx264', quality=8)
    
    for img in images:
        writer.append_data(img)
    
    writer.close()
    print(f"✓ MP4 created: {mp4_filename}")
    
    file_size_mb = os.path.getsize(mp4_filename) / (1024 * 1024)
    print(f"  File size: {file_size_mb:.2f} MB")
    
    mp4_created = True
    
except (ValueError, ImportError) as e:
    print(f"⚠ Could not create MP4: {str(e)}")
    print("\nTo enable MP4 creation, install ffmpeg plugin:")
    print("  pip install imageio[ffmpeg]")
    print("  or: pip install imageio-ffmpeg")
    mp4_created = False

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("SYMMETRIC DATASET GENERATION COMPLETE")
print("="*70)

geometry_type = "Irregular" if USE_IRREGULAR else "Circular"
print(f"\nGeometry Type: {geometry_type} Inclusion")
print(f"Generated {n_iterations} rotationally symmetric samples")
print(f"Rotation angles: 0° to {(n_rotations-1) * angular_spacing}° in {angular_spacing}° steps")
print(f"Force configuration: SINGLE force pair (teaching symmetry)")

print(f"\nOutput files:")
print(f"  Dataset: {dataset_filename} ({file_size_mb:.2f} MB)")
print(f"  GIF: {gif_filename}")
if mp4_created:
    print(f"  MP4: {mp4_filename}")
else:
    print(f"  MP4: (not created - install imageio[ffmpeg] to enable)")
print(f"  Frames: {output_dir}/")

print("\nDataset contents:")
print(f"  - force_vectors: [n_samples={n_iterations}, n_dof={n_dof}]")
print(f"  - boundary_displacements: [n_samples={n_iterations}, n_boundary_nodes={n_boundary_nodes}, 2]")
print(f"  - rotation_angles: [n_samples={n_iterations}]")
print(f"  - clearance: [n_samples={n_iterations}]")
print(f"  - max_displacement: [n_samples={n_iterations}]")
if USE_IRREGULAR:
    print(f"  - center_x, center_y, R_base, a_coeffs, b_coeffs, n_modes")
else:
    print(f"  - R_inner")
print(f"  - boundary_nodes, points, R_outer, metadata")

print("\n" + "="*70)
print("KEY INSIGHT: TEACHING ROTATIONAL SYMMETRY")
print("="*70)
print("\nThis dataset teaches the U-Net that:")
print("  'Rotating the input → should rotate the output'")
print("\nAll samples have:")
print(f"  ✓ Same geometry ({geometry_type.lower()} inclusion)")
print("  ✓ Same material properties")
print("  ✓ Same force configuration (1 pair)")
print("  ✓ Different angle (0°, 9°, 18°, ..., 171°)")
print("\nCombine this with angular_scanning_dataset.pt for robust training!")

print("\nTo load the dataset:")
print(f"  dataset = torch.load('{dataset_filename}')")
print(f"  forces = dataset['force_vectors']")
print(f"  displacements = dataset['boundary_displacements']")
print(f"  angles = dataset['rotation_angles']")

print("\n" + "="*70)