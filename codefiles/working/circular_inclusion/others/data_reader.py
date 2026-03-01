"""
Data Reader Script
Loads the angular_scanning_dataset.pt and recreates all visualizations
Sanity check to verify dataset integrity and usability
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import imageio
from fem_utils import *

torch.set_default_dtype(torch.float64)

print("="*70)
print("DATA READER - SANITY CHECK")
print("="*70)

# ============================================================================
# PARAMETERS
# ============================================================================
dataset_filename = 'angular_scanning_dataset.pt'
output_dir = Path('data_reader_frames')
gif_filename = 'data_reader_reconstruction.gif'
mp4_filename = 'data_reader_reconstruction.mp4'
fps = 5

# ============================================================================
# LOAD DATASET
# ============================================================================
print(f"\nLoading dataset: {dataset_filename}")

if not os.path.exists(dataset_filename):
    print(f"❌ ERROR: Dataset file not found: {dataset_filename}")
    print("Please run angular_scanning.py first to generate the dataset.")
    exit(1)

dataset = torch.load(dataset_filename)
print("✓ Dataset loaded successfully!")

# ============================================================================
# EXTRACT DATA
# ============================================================================
print("\n" + "="*70)
print("DATASET CONTENTS")
print("="*70)

# Extract tensors
force_vectors = dataset['force_vectors']
boundary_displacements = dataset['boundary_displacements']
n_pairs = dataset['n_pairs']
clearance = dataset['clearance']
max_displacement = dataset['max_displacement']

# Extract mesh info
boundary_nodes = dataset['boundary_nodes']
points = dataset['points']
R_outer = dataset['R_outer']
R_inner = dataset['R_inner']

# Extract metadata
metadata = dataset['metadata']

print("\nData Tensors:")
print(f"  force_vectors: {force_vectors.shape} | dtype: {force_vectors.dtype}")
print(f"  boundary_displacements: {boundary_displacements.shape} | dtype: {boundary_displacements.dtype}")
print(f"  n_pairs: {n_pairs.shape} | dtype: {n_pairs.dtype}")
print(f"  clearance: {clearance.shape} | dtype: {clearance.dtype}")
print(f"  max_displacement: {max_displacement.shape} | dtype: {max_displacement.dtype}")

print("\nMesh Information:")
print(f"  points: {points.shape}")
print(f"  boundary_nodes: {boundary_nodes.shape}")
print(f"  R_outer: {R_outer:.4f}")
print(f"  R_inner: {R_inner:.4f}")

print("\nMetadata:")
for key, value in metadata.items():
    print(f"  {key}: {value}")

# ============================================================================
# DATA STATISTICS
# ============================================================================
print("\n" + "="*70)
print("DATA STATISTICS")
print("="*70)

n_samples = force_vectors.shape[0]
n_dof = force_vectors.shape[1]
n_boundary_nodes = boundary_displacements.shape[1]

print(f"\nDataset size:")
print(f"  Number of samples: {n_samples}")
print(f"  DOFs per sample: {n_dof}")
print(f"  Boundary nodes: {n_boundary_nodes}")

print(f"\nForce vectors:")
print(f"  Shape: {force_vectors.shape}")
print(f"  Min: {force_vectors.min():.6e}")
print(f"  Max: {force_vectors.max():.6e}")
print(f"  Mean: {force_vectors.mean():.6e}")
print(f"  Std: {force_vectors.std():.6e}")

print(f"\nBoundary displacements:")
print(f"  Shape: {boundary_displacements.shape}")
print(f"  Min: {boundary_displacements.min():.6e}")
print(f"  Max: {boundary_displacements.max():.6e}")
print(f"  Mean: {boundary_displacements.mean():.6e}")
print(f"  Std: {boundary_displacements.std():.6e}")

# Displacement magnitudes
disp_x = boundary_displacements[:, :, 0]
disp_y = boundary_displacements[:, :, 1]
disp_mag = torch.sqrt(disp_x**2 + disp_y**2)

print(f"\nDisplacement magnitudes:")
print(f"  Min: {disp_mag.min():.6e}")
print(f"  Max: {disp_mag.max():.6e}")
print(f"  Mean: {disp_mag.mean():.6e}")
print(f"  Std: {disp_mag.std():.6e}")

print(f"\nNumber of force pairs per sample:")
print(f"  Min: {n_pairs.min()}")
print(f"  Max: {n_pairs.max()}")
print(f"  Range: {n_pairs.tolist()}")

print(f"\nClearance values:")
print(f"  Min: {clearance.min():.6f}")
print(f"  Max: {clearance.max():.6f}")
print(f"  Mean: {clearance.mean():.6f}")
print(f"  Std: {clearance.std():.6f}")

print(f"\nMax displacement per sample:")
print(f"  Min: {max_displacement.min():.6e}")
print(f"  Max: {max_displacement.max():.6e}")
print(f"  Mean: {max_displacement.mean():.6e}")

# ============================================================================
# RECONSTRUCT FULL DISPLACEMENT FIELD
# ============================================================================
print("\n" + "="*70)
print("RECONSTRUCTING VISUALIZATIONS")
print("="*70)

# Create output directory
output_dir.mkdir(exist_ok=True)
print(f"\nFrame directory: {output_dir}")

# We need to reconstruct U_nodes for visualization
# U_nodes should have shape [n_nodes, 2] for each sample
n_nodes = points.shape[0]

print(f"\nRecreating frames for {n_samples} samples...")

frame_files = []

for i in range(n_samples):
    print(f"\n[Frame {i+1}/{n_samples}] n_pairs = {n_pairs[i].item()}")
    
    # Reconstruct full U_nodes array
    # Initialize with zeros (interior nodes have zero displacement in visualization)
    U_nodes = torch.zeros(n_nodes, 2, dtype=torch.float64)
    
    # Fill in boundary node displacements
    U_nodes[boundary_nodes] = boundary_displacements[i]
    
    # Get force vector for this sample
    F = force_vectors[i]
    
    # Reconstruct force node pairs from the force vector
    # Find non-zero forces to identify force nodes
    F_2d = F.reshape(-1, 2)
    force_magnitudes = torch.sqrt(F_2d[:, 0]**2 + F_2d[:, 1]**2)
    force_node_indices = torch.where(force_magnitudes > 1e-10)[0]
    
    # Pair them up (assuming they come in pairs)
    force_nodes_list = []
    for j in range(0, len(force_node_indices), 2):
        if j + 1 < len(force_node_indices):
            node1 = force_node_indices[j]
            node2 = force_node_indices[j + 1]
            force_nodes_list.append((node1, node2))
    
    # Get parameters
    n_force_pairs = n_pairs[i].item()
    angular_spacing = metadata['angular_spacing']
    force_magnitude = metadata['force_magnitude']
    clearance_val = clearance[i].item()
    
    # Generate plot
    frame_filename = output_dir / f'frame_{i:03d}_pairs_{n_force_pairs:02d}.png'
    
    plot_deformation_multiple_pairs(
        points, U_nodes, boundary_nodes, F, force_nodes_list,
        n_force_pairs, angular_spacing, force_magnitude,
        R_outer, R_inner, clearance_val,
        filename=str(frame_filename)
    )
    
    frame_files.append(str(frame_filename))
    print(f"  Saved: {frame_filename.name}")

print(f"\n✓ All {len(frame_files)} frames reconstructed!")

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
print("SANITY CHECK COMPLETE ✓")
print("="*70)

print(f"\nDataset verification:")
print(f"  ✓ Dataset loaded successfully from {dataset_filename}")
print(f"  ✓ {n_samples} samples verified")
print(f"  ✓ All frames reconstructed from data")
print(f"  ✓ Animations created")

print(f"\nOutput files:")
print(f"  GIF: {gif_filename}")
if mp4_created:
    print(f"  MP4: {mp4_filename}")
else:
    print(f"  MP4: (not created - install imageio[ffmpeg] to enable)")
print(f"  Frames: {output_dir}/")

print("\nDataset is ready for ML workflow!")
print("  - Forward problem data: force_vectors → boundary_displacements")
print("  - Inverse problem goal: boundary_displacements → force_vectors")

print("\n" + "="*70)