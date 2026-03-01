"""
Single Force Pair FEM Solver
Interactive script for exploring single force pair configurations
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
from fem_utils import *

torch.set_default_dtype(torch.float64)

print("="*70)
print("FEM Solver - Single Force Pair")
print("="*70)

# ============================================================================
# PARAMETERS (EDIT THESE)
# ============================================================================
force_magnitude = 0.1
force_angle = 45.0  # degrees

# ============================================================================
# LOAD MESH
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
# APPLY FORCES
# ============================================================================
print("\nApplying forces...")
print(f"  Force magnitude: {force_magnitude}")
print(f"  Force angle: {force_angle}°")

boundary_nodes, radii = find_boundary_nodes(points, R_outer)
F, node1, node2 = apply_single_force_pair(points, boundary_nodes, force_magnitude, force_angle, device)

print(f"  Boundary nodes: {len(boundary_nodes)}")
print(f"  Force node 1: {node1.item()}")
print(f"    Position: ({points[node1, 0]:.4f}, {points[node1, 1]:.4f})")
print(f"    Angle: {torch.atan2(points[node1, 1], points[node1, 0]).item() * 180 / torch.pi:.1f}°")
print(f"  Force node 2: {node2.item()}")
print(f"    Position: ({points[node2, 0]:.4f}, {points[node2, 1]:.4f})")
print(f"    Angle: {torch.atan2(points[node2, 1], points[node2, 0]).item() * 180 / torch.pi:.1f}°")
print(f"  Net force: ({F[::2].sum():.10f}, {F[1::2].sum():.10f})")

# ============================================================================
# BOUNDARY CONDITIONS
# ============================================================================
print("\nApplying boundary conditions...")

fixed_dofs, free_dofs = setup_boundary_conditions(radii, R_inner, n_dof, device)

print(f"  Fixed DOFs: {len(fixed_dofs)}")
print(f"  Free DOFs: {len(free_dofs)}")

# ============================================================================
# SOLVE
# ============================================================================
print("\nSolving KU = F...")

K_cond = torch.linalg.cond(K[free_dofs][:, free_dofs])
print(f"  Condition number: {K_cond:.2e}")

U, U_nodes = solve_fem(K, F, fixed_dofs, free_dofs, n_dof, device)

print(f"  Solution computed!")
print(f"  Max displacement: {torch.max(torch.abs(U)):.6e}")

# ============================================================================
# PENETRATION CHECK
# ============================================================================
print("\n" + "="*70)
print("PENETRATION CHECK")
print("="*70)

is_valid, penetration_depth, min_deformed_radius, min_radius_node = check_penetration(
    points, U_nodes, boundary_nodes, R_inner
)

print(f"\nInclusion radius: {R_inner:.4f}")
print(f"Min deformed radius: {min_deformed_radius:.4f}")
print(f"  at node: {min_radius_node}")

if not is_valid:
    print(f"\n" + "="*70)
    print(f"❌ INVALID CONFIGURATION: PENETRATION DETECTED")
    print(f"="*70)
    print(f"Penetration depth: {penetration_depth:.6f}")
    print(f"Boundary penetrated {penetration_depth/R_inner*100:.2f}% into inclusion")
    print(f"\nForce configuration REJECTED. Terminating without visualization.")
    print("="*70)
    sys.exit(1)

clearance = -penetration_depth
print(f"\n✓ VALID CONFIGURATION: Clearance of {clearance:.6f}")
print(f"   Min radius is {clearance/R_inner*100:.2f}% above inclusion boundary")

# ============================================================================
# DIAGNOSTIC ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("DIAGNOSTIC ANALYSIS")
print("="*70)

boundary_nodes_np = boundary_nodes.cpu().numpy()
U_nodes_np = U_nodes.cpu().numpy()
points_np = points.cpu().numpy()

boundary_positions = points_np[boundary_nodes_np]
boundary_displacements = U_nodes_np[boundary_nodes_np]
disp_magnitudes = np.sqrt(boundary_displacements[:, 0]**2 + boundary_displacements[:, 1]**2)

max_mag_idx = np.argmax(disp_magnitudes)
max_mag_node = boundary_nodes_np[max_mag_idx]

print(f"\nMax displacement magnitude: {disp_magnitudes[max_mag_idx]:.6e}")
print(f"  at node: {max_mag_node}")
print(f"  position: ({points_np[max_mag_node, 0]:.4f}, {points_np[max_mag_node, 1]:.4f})")

node1_np = node1.cpu().item()
node2_np = node2.cpu().item()

disp_at_node1 = np.sqrt(U_nodes_np[node1_np, 0]**2 + U_nodes_np[node1_np, 1]**2)
disp_at_node2 = np.sqrt(U_nodes_np[node2_np, 0]**2 + U_nodes_np[node2_np, 1]**2)

print(f"\nDisplacement at force nodes:")
print(f"  Node {node1_np}: {disp_at_node1:.6e}")
print(f"  Node {node2_np}: {disp_at_node2:.6e}")

print(f"\nIs max at force node? {max_mag_node in [node1_np, node2_np]}")

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\nCreating diagnostic plot...")

plot_deformation_single_pair(points, U_nodes, boundary_nodes, F, node1, node2,
                             force_angle, force_magnitude, R_outer, R_inner,
                             clearance, filename='deformation_plot.png')

print("Saved: deformation_plot.png")

# ============================================================================
# DISPLACEMENT MAGNITUDE PLOT
# ============================================================================
print("\nCreating displacement magnitude plot...")

plot_displacement_magnitude(points, U_nodes, R_outer, R_inner, 
                           title_suffix=f' (Angle={force_angle}°)',
                           filename='displacement_magnitude.png')

print("Saved: displacement_magnitude.png")

print("\n" + "="*70)
print("DONE - VALID CONFIGURATION")
print("="*70)