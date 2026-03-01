"""
Symmetric Scanning Script
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
print("FEM SYMMETRIC SCANNING - ROTATIONAL SYMMETRY DATASET")
print("="*70)

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
mesh = load_mesh()

points = mesh['points']
elements = mesh['elements']
K = mesh['K']
R_outer = mesh['R_outer']
R_inner = mesh['R_inner']
device = mesh['device']

n_nodes = len(points)
n_dof = 2 * n_nodes

print(f"  Nodes: {n_nodes}")
print(f"  DOFs: {n_dof}")

# Get boundary nodes (same for all iterations)
boundary_nodes, radii = find_boundary_nodes(points, R_outer)
n_boundary_nodes = len(boundary_nodes)

print(f"  Boundary nodes: {n_boundary_nodes}")

# Setup boundary conditions (same for all iterations)
fixed_dofs, free_dofs = setup_boundary_conditions(radii, R_inner, n_dof, device)

# ============================================================================
# DATA STORAGE INITIALIZATION
# ============================================================================
n_iterations = n_rotations

# Storage lists (will convert to tensors later)
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
    
    # Penetration check
    is_valid, penetration_depth, min_deformed_radius, min_radius_node = check_penetration(
        points, U_nodes, boundary_nodes, R_inner
    )
    
    clearance = -penetration_depth
    print(f"  Clearance: {clearance:.6f}")
    
    if not is_valid:
        print(f"  ⚠ WARNING: Penetration detected at angle {rotation_angle}°!")
    
    # Store data for dataset (keep as tensors)
    force_vectors_list.append(F.cpu().clone())
    boundary_displacements_list.append(U_nodes[boundary_nodes].cpu().clone())
    rotation_angles_list.append(rotation_angle)
    clearance_list.append(clearance)
    max_displacement_list.append(max_disp)
    
    print(f"  Data stored (iteration {iter_idx})")
    
    # Generate plot
    frame_filename = output_dir / f'frame_{iter_idx:03d}_angle_{int(rotation_angle):03d}.png'
    
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
force_vectors = torch.stack(force_vectors_list)  # Shape: [n_rotations, n_dof]
boundary_displacements = torch.stack(boundary_displacements_list)  # Shape: [n_rotations, n_boundary_nodes, 2]
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
    'R_inner': R_inner,
    
    # Metadata
    'metadata': {
        'description': 'FEM symmetric scanning dataset for teaching rotational invariance to U-Net',
        'purpose': 'All samples are rotationally symmetric - same physics, different angles',
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

print(f"\nGenerated {n_iterations} rotationally symmetric samples")
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
print(f"  - boundary_nodes, points, R_outer, R_inner, metadata")

print("\n" + "="*70)
print("KEY INSIGHT: TEACHING ROTATIONAL SYMMETRY")
print("="*70)
print("\nThis dataset teaches the U-Net that:")
print("  'Rotating the input → should rotate the output'")
print("\nAll samples have:")
print("  ✓ Same geometry (circular inclusion)")
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