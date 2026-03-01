"""
Radial Scanning Script
Applies FIXED force configuration (20 pairs) at different MAGNITUDES
All samples are magnitude-scaled versions of the same force pattern
Generates animations and builds dataset to teach U-Net magnitude invariance
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
print("FEM RADIAL SCANNING - MAGNITUDE INVARIANCE DATASET")
print("="*70)

# ============================================================================
# PARAMETERS
# ============================================================================
n_force_pairs = 20            # FIXED - maximum force coverage
initial_angle = 0.0           # FIXED - start at 0 degrees
angular_spacing = 9.0         # FIXED - standard spacing
n_magnitude_samples = 10      # Number of different magnitudes to test

# Force magnitude range
force_mag_min = 0.01
force_mag_max = 0.1
force_magnitudes = np.linspace(force_mag_min, force_mag_max, n_magnitude_samples)

# Animation settings
fps = 3                       # Slower to see magnitude changes
output_dir = Path('radial_scanning_frames')
gif_filename = 'radial_scanning.gif'
mp4_filename = 'radial_scanning.mp4'
dataset_filename = 'radial_scanning_dataset.pt'

# ============================================================================
# SETUP
# ============================================================================
print(f"\nScan Parameters:")
print(f"  Force pairs: {n_force_pairs} (fixed - teaching magnitude invariance)")
print(f"  Angular spacing: {angular_spacing}° (fixed)")
print(f"  Initial angle: {initial_angle}° (fixed)")
print(f"  Magnitude samples: {n_magnitude_samples}")
print(f"  Magnitude range: {force_mag_min} to {force_mag_max}")
print(f"  Magnitudes: {force_magnitudes}")
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
n_iterations = n_magnitude_samples

# Storage lists (will convert to tensors later)
force_vectors_list = []
boundary_displacements_list = []
force_magnitudes_list = []
clearance_list = []
max_displacement_list = []

print(f"\nDataset storage initialized for {n_iterations} magnitude samples")

# ============================================================================
# MAIN LOOP - FIXED FORCE PATTERN, VARYING MAGNITUDE
# ============================================================================
print("\n" + "="*70)
print("GENERATING FRAMES AND BUILDING MAGNITUDE DATASET")
print("="*70)
print(f"Note: All samples use {n_force_pairs} force pairs at {angular_spacing}° spacing")
print("Only the MAGNITUDE varies - teaching scale invariance!")
print("="*70)

frame_files = []

for iter_idx, force_magnitude in enumerate(force_magnitudes):
    print(f"\n[Iteration {iter_idx + 1}/{n_iterations}] Force magnitude: {force_magnitude:.4f}")
    
    # Apply FIXED force pattern (20 pairs) at THIS magnitude
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
        print(f"  ⚠ WARNING: Penetration detected at magnitude {force_magnitude:.4f}!")
    
    # Store data for dataset (keep as tensors)
    force_vectors_list.append(F.cpu().clone())
    boundary_displacements_list.append(U_nodes[boundary_nodes].cpu().clone())
    force_magnitudes_list.append(force_magnitude)
    clearance_list.append(clearance)
    max_displacement_list.append(max_disp)
    
    print(f"  Data stored (iteration {iter_idx})")
    
    # Generate plot
    frame_filename = output_dir / f'frame_{iter_idx:03d}_mag_{force_magnitude:.4f}.png'
    
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
print("SAVING MAGNITUDE DATASET")
print("="*70)

# Convert lists to tensors
force_vectors = torch.stack(force_vectors_list)  # Shape: [n_magnitudes, n_dof]
boundary_displacements = torch.stack(boundary_displacements_list)  # Shape: [n_magnitudes, n_boundary_nodes, 2]
force_magnitudes_tensor = torch.tensor(force_magnitudes_list, dtype=torch.float64)
clearance_tensor = torch.tensor(clearance_list, dtype=torch.float64)
max_displacement_tensor = torch.tensor(max_displacement_list, dtype=torch.float64)

print(f"\nDataset tensor shapes:")
print(f"  force_vectors: {force_vectors.shape}")
print(f"  boundary_displacements: {boundary_displacements.shape}")
print(f"  force_magnitudes: {force_magnitudes_tensor.shape}")
print(f"  clearance: {clearance_tensor.shape}")
print(f"  max_displacement: {max_displacement_tensor.shape}")

# Create dataset dictionary
dataset = {
    # Data
    'force_vectors': force_vectors,
    'boundary_displacements': boundary_displacements,
    'force_magnitudes': force_magnitudes_tensor,
    'clearance': clearance_tensor,
    'max_displacement': max_displacement_tensor,
    
    # Mesh information
    'boundary_nodes': boundary_nodes.cpu(),
    'points': points.cpu(),
    'R_outer': R_outer,
    'R_inner': R_inner,
    
    # Metadata
    'metadata': {
        'description': 'FEM radial scanning dataset for teaching magnitude invariance to U-Net',
        'purpose': 'All samples are magnitude-scaled - same force pattern, different amplitudes',
        'force_mag_min': force_mag_min,
        'force_mag_max': force_mag_max,
        'n_magnitude_samples': n_magnitude_samples,
        'n_force_pairs': n_force_pairs,
        'initial_angle': initial_angle,
        'angular_spacing': angular_spacing,
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
print("\nThe GIF shows force magnitude increasing from 0.01 to 0.1!")
print("Notice: Displacement patterns scale linearly with force magnitude")

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
print("RADIAL DATASET GENERATION COMPLETE")
print("="*70)

print(f"\nGenerated {n_iterations} magnitude-scaled samples")
print(f"Force magnitudes: {force_mag_min:.4f} to {force_mag_max:.4f}")
print(f"Force configuration: {n_force_pairs} pairs at {angular_spacing}° spacing (fixed)")

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
print(f"  - force_magnitudes: [n_samples={n_iterations}]")
print(f"  - clearance: [n_samples={n_iterations}]")
print(f"  - max_displacement: [n_samples={n_iterations}]")
print(f"  - boundary_nodes, points, R_outer, R_inner, metadata")

print("\n" + "="*70)
print("KEY INSIGHT: TEACHING MAGNITUDE INVARIANCE")
print("="*70)
print("\nThis dataset teaches the U-Net that:")
print("  'Scaling the force → scales displacement → material stays SAME'")
print("\nAll samples have:")
print("  ✓ Same geometry (circular inclusion)")
print("  ✓ Same material properties")
print("  ✓ Same force configuration (20 pairs at 9° spacing)")
print("  ✓ Different force magnitude (0.01 to 0.1)")
print("\nLinear elasticity: U = K^(-1) * F")
print("  → Double force → Double displacement")
print("  → But K (material) stays the same!")
print("\nCombine with angular + symmetric datasets for robust training!")

print("\nTo load the dataset:")
print(f"  dataset = torch.load('{dataset_filename}')")
print(f"  forces = dataset['force_vectors']")
print(f"  displacements = dataset['boundary_displacements']")
print(f"  magnitudes = dataset['force_magnitudes']")

print("\n" + "="*70)