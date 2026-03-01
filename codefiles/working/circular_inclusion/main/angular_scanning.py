"""
Angular Scanning Script
Loops over number of force pairs (1-20) with fixed angular spacing
Generates animations and builds dataset for inverse problem ML workflow
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
print("FEM ANGULAR SCANNING - DATASET GENERATION")
print("="*70)

# ============================================================================
# PARAMETERS
# ============================================================================
force_magnitude = 0.1
initial_angle = 0.0           # Fixed at 0 degrees
angular_spacing = 9.0         # Fixed at 9 degrees
n_pairs_start = 1             # Start with 1 force pair
n_pairs_end = 20              # End with 20 force pairs
n_pairs_increment = 1         # Increment by 1

# Animation settings
fps = 5                       # Frames per second for animation
output_dir = Path('angular_scanning_frames')
gif_filename = 'angular_scanning.gif'
mp4_filename = 'angular_scanning.mp4'
dataset_filename = 'angular_scanning_dataset.pt'

# ============================================================================
# SETUP
# ============================================================================
print(f"\nScan Parameters:")
print(f"  Angular spacing: {angular_spacing}° (fixed)")
print(f"  Initial angle: {initial_angle}° (fixed)")
print(f"  Force magnitude: {force_magnitude}")
print(f"  Number of pairs range: {n_pairs_start} to {n_pairs_end}")
print(f"  Increment: {n_pairs_increment}")
print(f"  Total iterations: {(n_pairs_end - n_pairs_start) // n_pairs_increment + 1}")
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
n_iterations = (n_pairs_end - n_pairs_start) // n_pairs_increment + 1

# Storage lists (will convert to tensors later)
force_vectors_list = []
boundary_displacements_list = []
n_pairs_list = []
clearance_list = []
max_displacement_list = []

print(f"\nDataset storage initialized for {n_iterations} iterations")

# ============================================================================
# MAIN LOOP
# ============================================================================
print("\n" + "="*70)
print("GENERATING FRAMES AND BUILDING DATASET")
print("="*70)

frame_files = []
n_pairs_values = range(n_pairs_start, n_pairs_end + 1, n_pairs_increment)

for iter_idx, n_force_pairs in enumerate(n_pairs_values):
    print(f"\n[Iteration {iter_idx + 1}/{n_iterations}] Number of force pairs: {n_force_pairs}")
    
    # Apply forces
    F, force_nodes_list = apply_multiple_force_pairs(
        points, boundary_nodes, force_magnitude, 
        initial_angle, angular_spacing, n_force_pairs, device
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
        print(f"  ⚠ WARNING: Penetration detected with {n_force_pairs} pairs!")
    
    # Store data for dataset (keep as tensors)
    force_vectors_list.append(F.cpu().clone())
    boundary_displacements_list.append(U_nodes[boundary_nodes].cpu().clone())
    n_pairs_list.append(n_force_pairs)
    clearance_list.append(clearance)
    max_displacement_list.append(max_disp)
    
    print(f"  Data stored (iteration {iter_idx})")
    
    # Generate plot
    frame_filename = output_dir / f'frame_{iter_idx:03d}_pairs_{n_force_pairs:02d}.png'
    
    plot_deformation_multiple_pairs(
        points, U_nodes, boundary_nodes, F, force_nodes_list,
        n_force_pairs, angular_spacing, force_magnitude,
        R_outer, R_inner, clearance,
        filename=str(frame_filename)
    )
    
    frame_files.append(str(frame_filename))
    print(f"  Saved frame: {frame_filename.name}")

# ============================================================================
# SAVE DATASET
# ============================================================================
print("\n" + "="*70)
print("SAVING DATASET")
print("="*70)

# Convert lists to tensors
force_vectors = torch.stack(force_vectors_list)  # Shape: [n_iterations, n_dof]
boundary_displacements = torch.stack(boundary_displacements_list)  # Shape: [n_iterations, n_boundary_nodes, 2]
n_pairs_tensor = torch.tensor(n_pairs_list, dtype=torch.long)
clearance_tensor = torch.tensor(clearance_list, dtype=torch.float64)
max_displacement_tensor = torch.tensor(max_displacement_list, dtype=torch.float64)

print(f"\nDataset tensor shapes:")
print(f"  force_vectors: {force_vectors.shape}")
print(f"  boundary_displacements: {boundary_displacements.shape}")
print(f"  n_pairs: {n_pairs_tensor.shape}")
print(f"  clearance: {clearance_tensor.shape}")
print(f"  max_displacement: {max_displacement_tensor.shape}")

# Create dataset dictionary
dataset = {
    # Data
    'force_vectors': force_vectors,
    'boundary_displacements': boundary_displacements,
    'n_pairs': n_pairs_tensor,
    'clearance': clearance_tensor,
    'max_displacement': max_displacement_tensor,
    
    # Mesh information
    'boundary_nodes': boundary_nodes.cpu(),
    'points': points.cpu(),
    'R_outer': R_outer,
    'R_inner': R_inner,
    
    # Metadata
    'metadata': {
        'description': 'FEM forward problem dataset for inverse problem ML workflow',
        'force_magnitude': force_magnitude,
        'initial_angle': initial_angle,
        'angular_spacing': angular_spacing,
        'n_pairs_start': n_pairs_start,
        'n_pairs_end': n_pairs_end,
        'n_pairs_increment': n_pairs_increment,
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
print("DATASET GENERATION COMPLETE")
print("="*70)

print(f"\nGenerated {n_iterations} samples")
print(f"Number of force pairs: {n_pairs_start} to {n_pairs_end}")
print(f"Angular spacing: {angular_spacing}° (fixed)")

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
print(f"  - n_pairs: [n_samples={n_iterations}]")
print(f"  - clearance: [n_samples={n_iterations}]")
print(f"  - max_displacement: [n_samples={n_iterations}]")
print(f"  - boundary_nodes, points, R_outer, R_inner, metadata")

print("\nTo load the dataset:")
print(f"  dataset = torch.load('{dataset_filename}')")
print(f"  forces = dataset['force_vectors']")
print(f"  displacements = dataset['boundary_displacements']")

print("\n" + "="*70)