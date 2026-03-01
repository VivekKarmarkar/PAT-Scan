"""
Automated Tests for FEM Solver
Test 1: Force magnitude sweep until penetration
Test 2: Angular sweep with validation and plot saving
"""

import torch
import numpy as np
import os
import sys
from fem_utils import *

torch.set_default_dtype(torch.float64)

# Create output directories
os.makedirs('test_results/deformation_plots', exist_ok=True)
os.makedirs('test_results/displacement_magnitude', exist_ok=True)

print("="*70)
print("AUTOMATED FEM TESTS")
print("="*70)

# ============================================================================
# TEST 1: FORCE MAGNITUDE SWEEP (Until Penetration)
# ============================================================================
print("\n" + "="*70)
print("TEST 1: Force Magnitude Sweep Until Penetration")
print("="*70)

# Load mesh
mesh = load_mesh()
points = mesh['points']
K = mesh['K']
R_outer = mesh['R_outer']
R_inner = mesh['R_inner']
device = mesh['device']

# Test parameters
force_angle = 0.0  # Fixed angle
magnitude_start = 0.1
magnitude_increment = 0.1

# Find boundary nodes
boundary_nodes, radii = find_boundary_nodes(points, R_outer)
n_dof = 2 * len(points)

print(f"\nTest parameters:")
print(f"  Fixed angle: {force_angle}°")
print(f"  Starting magnitude: {magnitude_start}")
print(f"  Increment: {magnitude_increment}")

# Setup boundary conditions (same for all iterations)
fixed_dofs, free_dofs = setup_boundary_conditions(radii, R_inner, n_dof, device)

# Run sweep
test1_results = []
magnitude = magnitude_start
iteration = 0

while True:
    iteration += 1
    print(f"\n--- Iteration {iteration}: magnitude = {magnitude:.2f} ---")
    
    # Apply forces
    F, node1, node2 = apply_single_force_pair(points, boundary_nodes, magnitude, force_angle, device)
    
    # Solve
    U, U_nodes = solve_fem(K, F, fixed_dofs, free_dofs, n_dof, device)
    
    # Check penetration
    is_valid, penetration_depth, min_radius, min_node = check_penetration(points, U_nodes, boundary_nodes, R_inner)
    
    max_disp = torch.max(torch.abs(U)).item()
    
    if is_valid:
        clearance = -penetration_depth
        print(f"  ✓ VALID: clearance = {clearance:.6f}, max_disp = {max_disp:.6e}")
        test1_results.append({
            'magnitude': magnitude,
            'valid': True,
            'clearance': clearance,
            'max_disp': max_disp
        })
        magnitude += magnitude_increment
    else:
        print(f"  ❌ PENETRATION: depth = {penetration_depth:.6f}")
        test1_results.append({
            'magnitude': magnitude,
            'valid': False,
            'penetration': penetration_depth,
            'max_disp': max_disp
        })
        print(f"\n✓ Test 1 Complete: Penetration occurred at magnitude = {magnitude:.2f}")
        print(f"  Maximum valid magnitude: {magnitude - magnitude_increment:.2f}")
        break

# Save Test 1 results
with open('test_results/test1_magnitude_sweep.txt', 'w') as f:
    f.write("TEST 1: Force Magnitude Sweep Results\n")
    f.write("="*60 + "\n\n")
    f.write(f"Fixed angle: {force_angle}°\n")
    f.write(f"Starting magnitude: {magnitude_start}\n")
    f.write(f"Increment: {magnitude_increment}\n\n")
    f.write("Results:\n")
    f.write("-" * 60 + "\n")
    for result in test1_results:
        if result['valid']:
            f.write(f"Magnitude {result['magnitude']:.2f}: VALID (clearance={result['clearance']:.6f}, max_disp={result['max_disp']:.6e})\n")
        else:
            f.write(f"Magnitude {result['magnitude']:.2f}: INVALID (penetration={result['penetration']:.6f})\n")

print(f"\nTest 1 results saved to: test_results/test1_magnitude_sweep.txt")

# ============================================================================
# TEST 2: ANGULAR SWEEP WITH VALIDATION
# ============================================================================
print("\n" + "="*70)
print("TEST 2: Angular Sweep with Validation")
print("="*70)

# Test parameters
force_magnitude = 0.1  # Fixed magnitude
angle_start = 0.0
angle_increment = 9.0  # Match mesh spacing
angle_end = 180.0

print(f"\nTest parameters:")
print(f"  Fixed magnitude: {force_magnitude}")
print(f"  Angle range: {angle_start}° to {angle_end}°")
print(f"  Increment: {angle_increment}°")

# Run sweep
test2_results = []
current_angle = angle_start
previous_max_disp = None
tolerance = 0.05  # 5% tolerance for max displacement consistency

while current_angle <= angle_end:
    print(f"\n--- Angle = {current_angle:.1f}° ---")
    
    # Apply forces
    F, node1, node2 = apply_single_force_pair(points, boundary_nodes, force_magnitude, current_angle, device)
    
    # Solve
    U, U_nodes = solve_fem(K, F, fixed_dofs, free_dofs, n_dof, device)
    
    # Check penetration
    is_valid, penetration_depth, min_radius, min_node = check_penetration(points, U_nodes, boundary_nodes, R_inner)
    
    if not is_valid:
        print(f"  ❌ PENETRATION: depth = {penetration_depth:.6f}")
        print(f"  Stopping test - invalid configuration")
        break
    
    # Get boundary displacements and find max
    boundary_nodes_np = boundary_nodes.cpu().numpy()
    U_nodes_np = U_nodes.cpu().numpy()
    boundary_displacements = U_nodes_np[boundary_nodes_np]
    disp_magnitudes = np.sqrt(boundary_displacements[:, 0]**2 + boundary_displacements[:, 1]**2)
    
    max_mag_idx = np.argmax(disp_magnitudes)
    max_mag_node = boundary_nodes_np[max_mag_idx]
    max_disp = disp_magnitudes[max_mag_idx]
    
    # Check if max is at force node
    node1_np = node1.cpu().item()
    node2_np = node2.cpu().item()
    max_at_force_node = max_mag_node in [node1_np, node2_np]
    
    print(f"  Max displacement: {max_disp:.6e}")
    print(f"  Max at force node: {max_at_force_node}")
    
    # Check consistency with previous angle
    if previous_max_disp is not None:
        relative_diff = abs(max_disp - previous_max_disp) / previous_max_disp
        is_consistent = relative_diff < tolerance
        print(f"  Relative change from previous: {relative_diff*100:.2f}%")
        print(f"  Consistent: {is_consistent}")
    else:
        is_consistent = True
        relative_diff = None
    
    # Save plots
    deform_filename = f'test_results/deformation_plots/angle_{int(current_angle):03d}.png'
    disp_mag_filename = f'test_results/displacement_magnitude/angle_{int(current_angle):03d}.png'
    
    # Create deformation plot
    plot_deformation_single_pair(points, U_nodes, boundary_nodes, F, node1, node2,
                                 current_angle, force_magnitude, R_outer, R_inner,
                                 -penetration_depth, filename=deform_filename)
    
    # Create displacement magnitude plot
    plot_displacement_magnitude(points, U_nodes, R_outer, R_inner, 
                               title_suffix=f' (Angle={current_angle:.1f}°)',
                               filename=disp_mag_filename)
    
    print(f"  Saved plots")
    
    # Record results
    test2_results.append({
        'angle': current_angle,
        'max_disp': max_disp,
        'max_at_force_node': max_at_force_node,
        'is_consistent': is_consistent,
        'relative_diff': relative_diff,
        'clearance': -penetration_depth
    })
    
    previous_max_disp = max_disp
    current_angle += angle_increment

# Save Test 2 results
with open('test_results/test2_angular_sweep.txt', 'w') as f:
    f.write("TEST 2: Angular Sweep Results\n")
    f.write("="*60 + "\n\n")
    f.write(f"Fixed magnitude: {force_magnitude}\n")
    f.write(f"Angle range: {angle_start}° to {angle_end}°\n")
    f.write(f"Increment: {angle_increment}°\n")
    f.write(f"Tolerance: {tolerance*100:.1f}%\n\n")
    f.write("Results:\n")
    f.write("-" * 60 + "\n")
    f.write(f"{'Angle':>8} | {'Max Disp':>12} | {'At Force':>10} | {'Consistent':>10} | {'Rel Diff':>10} | {'Clearance':>10}\n")
    f.write("-" * 60 + "\n")
    
    for result in test2_results:
        angle_str = f"{result['angle']:6.1f}°"
        max_disp_str = f"{result['max_disp']:.6e}"
        at_force_str = "Yes" if result['max_at_force_node'] else "No"
        consistent_str = "Yes" if result['is_consistent'] else "No"
        rel_diff_str = f"{result['relative_diff']*100:.2f}%" if result['relative_diff'] is not None else "N/A"
        clearance_str = f"{result['clearance']:.6f}"
        
        f.write(f"{angle_str:>8} | {max_disp_str:>12} | {at_force_str:>10} | {consistent_str:>10} | {rel_diff_str:>10} | {clearance_str:>10}\n")

print(f"\nTest 2 results saved to: test_results/test2_angular_sweep.txt")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("TEST SUMMARY")
print("="*70)

print(f"\nTest 1: Force Magnitude Sweep")
print(f"  Valid magnitudes tested: {len([r for r in test1_results if r['valid']])}")
print(f"  Maximum valid magnitude: {max(r['magnitude'] for r in test1_results if r['valid']):.2f}")
print(f"  Penetration magnitude: {next(r['magnitude'] for r in test1_results if not r['valid']):.2f}")

print(f"\nTest 2: Angular Sweep")
print(f"  Angles tested: {len(test2_results)}")
print(f"  All max at force nodes: {all(r['max_at_force_node'] for r in test2_results)}")
print(f"  All consistent: {all(r['is_consistent'] for r in test2_results)}")
print(f"  Plots saved in:")
print(f"    - test_results/deformation_plots/")
print(f"    - test_results/displacement_magnitude/")

print("\n" + "="*70)
print("ALL TESTS COMPLETE")
print("="*70)