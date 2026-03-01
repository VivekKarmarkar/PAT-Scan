"""
Angular Resolution Animation Script
Generates animation showing deformation as angular spacing varies from 0° to 90°
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
print("FEM ANGULAR RESOLUTION ANIMATION")
print("="*70)

# ============================================================================
# PARAMETERS
# ============================================================================
force_magnitude = 0.1
initial_angle = 0.0           # Fixed at 0 degrees
n_force_pairs = 2             # Fixed at 2 pairs
angle_start = 0               # Start angle
angle_end = 90                # End angle
angle_increment = 1           # Increment in degrees

# Animation settings
fps = 10                      # Frames per second for animation
output_dir = Path('angular_resolution_frames')
gif_filename = 'angular_resolution.gif'
mp4_filename = 'angular_resolution.mp4'

# ============================================================================
# SETUP
# ============================================================================
print(f"\nAnimation Parameters:")
print(f"  Force pairs: {n_force_pairs}")
print(f"  Initial angle: {initial_angle}°")
print(f"  Angular spacing range: {angle_start}° to {angle_end}°")
print(f"  Increment: {angle_increment}°")
print(f"  Total frames: {(angle_end - angle_start) // angle_increment + 1}")
print(f"  FPS: {fps}")

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

# ============================================================================
# MAIN LOOP
# ============================================================================
print("\n" + "="*70)
print("GENERATING FRAMES")
print("="*70)

frame_files = []
angle_values = range(angle_start, angle_end + 1, angle_increment)

for frame_idx, angular_spacing in enumerate(angle_values):
    print(f"\n[Frame {frame_idx + 1}/{len(angle_values)}] Angular spacing: {angular_spacing}°")
    
    # Apply forces
    boundary_nodes, radii = find_boundary_nodes(points, R_outer)
    F, force_nodes_list = apply_multiple_force_pairs(
        points, boundary_nodes, force_magnitude, 
        initial_angle, angular_spacing, n_force_pairs, device
    )
    
    # Boundary conditions
    fixed_dofs, free_dofs = setup_boundary_conditions(radii, R_inner, n_dof, device)
    
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
        print(f"  ⚠ WARNING: Penetration detected at {angular_spacing}°!")
    
    # Generate plot
    frame_filename = output_dir / f'frame_{frame_idx:03d}_angle_{angular_spacing:03d}.png'
    
    plot_deformation_multiple_pairs(
        points, U_nodes, boundary_nodes, F, force_nodes_list,
        n_force_pairs, angular_spacing, force_magnitude,
        R_outer, R_inner, clearance,
        filename=str(frame_filename)
    )
    
    frame_files.append(str(frame_filename))
    print(f"  Saved: {frame_filename.name}")

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
print("ANIMATION COMPLETE")
print("="*70)

print(f"\nGenerated {len(frame_files)} frames")
print(f"Angular spacing: {angle_start}° to {angle_end}° (increment: {angle_increment}°)")
print(f"\nOutput files:")
print(f"  GIF: {gif_filename}")
if mp4_created:
    print(f"  MP4: {mp4_filename}")
else:
    print(f"  MP4: (not created - install imageio[ffmpeg] to enable)")
print(f"  Frames: {output_dir}/")

print("\n" + "="*70)