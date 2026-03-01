"""
Automated Tests for FEM Solver - UPGRADED VERSION
Now supports both circular and irregular inclusions
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
print("AUTOMATED FEM TESTS - UPGRADED FOR IRREGULAR INCLUSIONS")
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
# UTILITY FUNCTIONS FOR IRREGULAR GEOMETRY
# ============================================================================

def load_mesh_irregular(filename='irregular_inclusion_sample.pt'):
    """
    Load mesh data for irregular inclusion
    Returns mesh dict with all necessary data including irregular boundary definition
    """
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
    print(f"  Inclusion center: ({mesh['center_x']}, {mesh['center_y']})")
    print(f"  Fourier modes: {mesh['n_modes']}")
    
    return mesh


def is_inside_irregular_inclusion_vectorized(x, y, center_x, center_y, R_base, a_coeffs, b_coeffs, n_modes):
    """
    Vectorized check if points (x, y) are inside the irregular inclusion
    Works with tensors of arbitrary shape
    """
    # Translate to inclusion center coordinate system
    x_local = x - center_x
    y_local = y - center_y
    
    # Convert to polar coordinates relative to inclusion center
    r = torch.sqrt(x_local**2 + y_local**2)
    theta = torch.atan2(y_local, x_local)
    
    # Compute irregular boundary radius at these angles
    r_boundary = R_base * torch.ones_like(theta)
    for n in range(1, n_modes + 1):
        r_boundary = r_boundary + R_base * (a_coeffs[n-1] * torch.cos(n * theta) + 
                                            b_coeffs[n-1] * torch.sin(n * theta))
    
    # Ensure radius stays positive and bounded
    r_boundary = torch.clamp(r_boundary, R_base * 0.5, R_base * 1.5)
    
    return r < r_boundary


def find_boundary_nodes_irregular(points, R_outer, tolerance=1e-6):
    """
    Find nodes on the outer boundary (works for any geometry)
    Returns boundary node indices and their radii from origin
    """
    radii = torch.sqrt(points[:, 0]**2 + points[:, 1]**2)
    boundary_mask = torch.abs(radii - R_outer) < tolerance
    boundary_nodes = torch.where(boundary_mask)[0]
    return boundary_nodes, radii


def check_penetration_irregular(points, U_nodes, boundary_nodes, 
                                center_x, center_y, R_base, a_coeffs, b_coeffs, n_modes):
    """
    Check if any boundary nodes penetrate the irregular inclusion after deformation
    Returns: (is_valid, max_penetration_depth, min_radius, min_node)
    """
    boundary_nodes_np = boundary_nodes.cpu().numpy()
    U_nodes_np = U_nodes.cpu().numpy()
    points_np = points.cpu().numpy()
    
    # Get deformed boundary positions
    boundary_positions = points_np[boundary_nodes_np]
    boundary_displacements = U_nodes_np[boundary_nodes_np]
    deformed_boundary = boundary_positions + boundary_displacements
    
    # Convert to torch tensors for vectorized computation
    device = points.device
    x_deformed = torch.tensor(deformed_boundary[:, 0], dtype=torch.float64, device=device)
    y_deformed = torch.tensor(deformed_boundary[:, 1], dtype=torch.float64, device=device)
    
    # Compute distance to irregular inclusion boundary for each deformed point
    x_local = x_deformed - center_x
    y_local = y_deformed - center_y
    r_actual = torch.sqrt(x_local**2 + y_local**2)
    theta = torch.atan2(y_local, x_local)
    
    # Compute irregular boundary radius at these angles
    r_boundary = R_base * torch.ones_like(theta)
    for n in range(1, n_modes + 1):
        r_boundary = r_boundary + R_base * (a_coeffs[n-1] * torch.cos(n * theta) + 
                                            b_coeffs[n-1] * torch.sin(n * theta))
    r_boundary = torch.clamp(r_boundary, R_base * 0.5, R_base * 1.5)
    
    # Clearance = actual_distance - boundary_radius
    # Positive clearance = safe, negative = penetration
    clearances = r_actual - r_boundary
    
    min_clearance = torch.min(clearances)
    min_idx = torch.argmin(clearances)
    min_node = boundary_nodes[min_idx].item()
    min_radius = r_actual[min_idx].item()
    
    is_valid = min_clearance >= 0
    penetration_depth = -min_clearance.item() if not is_valid else 0.0
    
    return is_valid, penetration_depth, min_radius, min_node


def setup_boundary_conditions_irregular(radii, center_x, center_y, R_base, 
                                       a_coeffs, b_coeffs, n_modes, n_dof, device):
    """
    Setup boundary conditions for irregular inclusion
    Fix all DOFs inside the irregular inclusion
    """
    # Get all points (reconstruct from radii and known structure)
    # We need access to points for this - let's return a function instead
    pass


def setup_boundary_conditions_irregular_with_points(points, center_x, center_y, R_base,
                                                   a_coeffs, b_coeffs, n_modes, n_dof, device):
    """
    Setup boundary conditions for irregular inclusion
    Fix all DOFs inside the irregular inclusion
    """
    # Check which nodes are inside irregular inclusion
    inside_mask = is_inside_irregular_inclusion_vectorized(
        points[:, 0], points[:, 1], center_x, center_y, R_base, a_coeffs, b_coeffs, n_modes
    )
    
    inside_nodes = torch.where(inside_mask)[0]
    
    # Create list of fixed DOFs (both x and y for nodes inside inclusion)
    fixed_dofs = []
    for node in inside_nodes:
        fixed_dofs.extend([2*node, 2*node+1])
    
    fixed_dofs = torch.tensor(fixed_dofs, dtype=torch.long, device=device)
    all_dofs = torch.arange(n_dof, device=device)
    free_dofs = torch.tensor([dof for dof in all_dofs if dof not in fixed_dofs], 
                            dtype=torch.long, device=device)
    
    print(f"\nBoundary Conditions (Irregular):")
    print(f"  Fixed nodes (inside inclusion): {len(inside_nodes)}")
    print(f"  Fixed DOFs: {len(fixed_dofs)}")
    print(f"  Free DOFs: {len(free_dofs)}")
    
    return fixed_dofs, free_dofs


def plot_deformation_irregular(points, U_nodes, boundary_nodes, F, node1, node2,
                               force_angle, force_magnitude, R_outer, 
                               center_x, center_y, R_base, a_coeffs, b_coeffs, n_modes,
                               clearance, filename='deformation_plot.png'):
    """
    Create deformation plot for irregular inclusion geometry
    """
    boundary_nodes_np = boundary_nodes.cpu().numpy()
    U_nodes_np = U_nodes.cpu().numpy()
    points_np = points.cpu().numpy()
    node1_np = node1.cpu().item()
    node2_np = node2.cpu().item()
    
    # Get boundary data
    boundary_positions = points_np[boundary_nodes_np]
    boundary_displacements = U_nodes_np[boundary_nodes_np]
    disp_magnitudes = np.sqrt(boundary_displacements[:, 0]**2 + boundary_displacements[:, 1]**2)
    
    max_mag_idx = np.argmax(disp_magnitudes)
    max_mag_node = boundary_nodes_np[max_mag_idx]
    
    # Displacement at force nodes
    disp_at_node1 = np.sqrt(U_nodes_np[node1_np, 0]**2 + U_nodes_np[node1_np, 1]**2)
    disp_at_node2 = np.sqrt(U_nodes_np[node2_np, 0]**2 + U_nodes_np[node2_np, 1]**2)
    
    # Create figure
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


def plot_displacement_magnitude_irregular(points, U_nodes, R_outer,
                                         center_x, center_y, R_base, a_coeffs, b_coeffs, n_modes,
                                         title_suffix='', filename='displacement_magnitude.png'):
    """
    Create displacement magnitude plot for irregular inclusion
    """
    points_np = points.cpu().numpy()
    U_nodes_np = U_nodes.cpu().numpy()
    
    disp_magnitude_all = np.sqrt(U_nodes_np[:, 0]**2 + U_nodes_np[:, 1]**2)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    # Scatter plot
    scatter = ax.scatter(points_np[:, 0], points_np[:, 1], 
                        c=disp_magnitude_all, cmap='hot', s=50, edgecolors='none')
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Displacement Magnitude |U|', rotation=270, labelpad=20, fontsize=12)
    
    # Plot outer boundary
    theta_plot = np.linspace(0, 2*np.pi, 200)
    ax.plot(R_outer*np.cos(theta_plot), R_outer*np.sin(theta_plot), 'k-', 
            linewidth=2, label='Outer boundary')
    
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
    
    ax.plot(x_boundary, y_boundary, 'b-', 
            linewidth=2, label='Irregular inclusion boundary')
    ax.plot([center_x], [center_y], 'ro', markersize=8, label='Inclusion center')
    
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.set_title(f'Displacement Magnitude at All Nodes{title_suffix}', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN TEST EXECUTION
# ============================================================================

# Load appropriate mesh
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
    n_dof = 2 * len(points)
    
    # Setup boundary conditions
    fixed_dofs, free_dofs = setup_boundary_conditions_irregular_with_points(
        points, center_x, center_y, R_base, a_coeffs, b_coeffs, n_modes, n_dof, device
    )
    
else:
    # Use original circular mesh loading
    mesh = load_mesh()
    points = mesh['points']
    K = mesh['K']
    R_outer = mesh['R_outer']
    R_inner = mesh['R_inner']
    device = mesh['device']
    
    # Find boundary nodes
    boundary_nodes, radii = find_boundary_nodes(points, R_outer)
    n_dof = 2 * len(points)
    
    # Setup boundary conditions
    fixed_dofs, free_dofs = setup_boundary_conditions(radii, R_inner, n_dof, device)


# ============================================================================
# TEST 1: FORCE MAGNITUDE SWEEP (Until Penetration)
# ============================================================================
print("\n" + "="*70)
print("TEST 1: Force Magnitude Sweep Until Penetration")
print("="*70)

# Test parameters
force_angle = 0.0  # Fixed angle
magnitude_start = 0.1
magnitude_increment = 0.1

print(f"\nTest parameters:")
print(f"  Fixed angle: {force_angle}°")
print(f"  Starting magnitude: {magnitude_start}")
print(f"  Increment: {magnitude_increment}")

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
    
    # Check penetration (different for irregular vs circular)
    if USE_IRREGULAR:
        is_valid, penetration_depth, min_radius, min_node = check_penetration_irregular(
            points, U_nodes, boundary_nodes, center_x, center_y, R_base, a_coeffs, b_coeffs, n_modes
        )
    else:
        is_valid, penetration_depth, min_radius, min_node = check_penetration(
            points, U_nodes, boundary_nodes, R_inner
        )
    
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
        print(f"  ✗ PENETRATION: depth = {penetration_depth:.6f}")
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
geometry_type = "Irregular" if USE_IRREGULAR else "Circular"
with open('test_results/test1_magnitude_sweep.txt', 'w') as f:
    f.write(f"TEST 1: Force Magnitude Sweep Results ({geometry_type} Inclusion)\n")
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
    
    # Check penetration (different for irregular vs circular)
    if USE_IRREGULAR:
        is_valid, penetration_depth, min_radius, min_node = check_penetration_irregular(
            points, U_nodes, boundary_nodes, center_x, center_y, R_base, a_coeffs, b_coeffs, n_modes
        )
    else:
        is_valid, penetration_depth, min_radius, min_node = check_penetration(
            points, U_nodes, boundary_nodes, R_inner
        )
    
    if not is_valid:
        print(f"  ✗ PENETRATION: depth = {penetration_depth:.6f}")
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
    
    # Save plots (use appropriate plotting function)
    deform_filename = f'test_results/deformation_plots/angle_{int(current_angle):03d}.png'
    disp_mag_filename = f'test_results/displacement_magnitude/angle_{int(current_angle):03d}.png'
    
    if USE_IRREGULAR:
        # Use irregular plotting functions
        plot_deformation_irregular(points, U_nodes, boundary_nodes, F, node1, node2,
                                  current_angle, force_magnitude, R_outer,
                                  center_x, center_y, R_base, a_coeffs, b_coeffs, n_modes,
                                  -penetration_depth, filename=deform_filename)
        
        plot_displacement_magnitude_irregular(points, U_nodes, R_outer,
                                             center_x, center_y, R_base, a_coeffs, b_coeffs, n_modes,
                                             title_suffix=f' (Angle={current_angle:.1f}°)',
                                             filename=disp_mag_filename)
    else:
        # Use original circular plotting functions
        plot_deformation_single_pair(points, U_nodes, boundary_nodes, F, node1, node2,
                                    current_angle, force_magnitude, R_outer, R_inner,
                                    -penetration_depth, filename=deform_filename)
        
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
    f.write(f"TEST 2: Angular Sweep Results ({geometry_type} Inclusion)\n")
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

print(f"\nGeometry Type: {geometry_type} Inclusion")

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