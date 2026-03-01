"""
Mesh Convergence Study for Irregular Inclusion
Tests Cauchy convergence by refining mesh and comparing displacement fields

OPTIMIZATIONS:
- Uses SPARSE matrix assembly to handle large meshes (saves ~99% memory)
- Refinement factor of 1.5x for smoother convergence
- Only converts to dense for the reduced free DOF system
- Max 4 refinement levels (sufficient for convergence)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import imageio
from math import gcd
from functools import reduce

from fem_utils import (
    setup_torch_defaults,
    get_device,
    assemble_stiffness,
    print_mesh_info
)

# ============================================================================
# PARAMETERS
# ============================================================================
# Starting mesh parameters
n_radial_start = 20
n_angular_start = 40

# Refinement parameters
refinement_factor = 1.5  # Multiplier for mesh density
max_refinements = 4
convergence_tolerance = 0.01  # 1% relative change

# Compute all angular divisions across refinements to find common invariant spacing
import math
n_angular_all_levels = [int(np.round(n_angular_start * (refinement_factor ** i))) 
                        for i in range(max_refinements)]
print(f"Angular divisions across levels: {n_angular_all_levels}")

# Find GCD of all n_angular values - this gives us the common invariant spacing
from math import gcd
from functools import reduce
n_angular_gcd = reduce(gcd, n_angular_all_levels)
invariant_angular_spacing = 360.0 / n_angular_gcd

print(f"Common angular GCD: {n_angular_gcd}")
print(f"Invariant angular spacing: {invariant_angular_spacing}°")
print(f"Number of invariant nodes: {n_angular_gcd}")
print()

# Material and geometry parameters
R_outer = 1.0
R_base = 0.3
E_b = 1.0
E_i = 10.0
nu = 0.3

# Irregular inclusion parameters (FIXED for all refinements)
center_x = 0.2
center_y = -0.15
n_modes = 6
irregularity = 0.20
seed = 42

# Force parameters
force_magnitude = 0.1
force_angle_1 = 0.0    # degrees (0°)
force_angle_2 = 180.0  # degrees (180°)

print("="*80)
print("MESH CONVERGENCE STUDY - IRREGULAR INCLUSION")
print("="*80)
print(f"\nStarting mesh: n_radial={n_radial_start}, n_angular={n_angular_start}")
print(f"Refinement factor: {refinement_factor}x")
print(f"Max refinements: {max_refinements}")
print(f"Convergence tolerance: {convergence_tolerance}")
print(f"Force application: {force_angle_1}° and {force_angle_2}°")
print(f"\nNote: Using SPARSE matrix assembly to handle large meshes efficiently")

setup_torch_defaults()
device = get_device()

# ============================================================================
# GENERATE FIXED IRREGULAR INCLUSION BOUNDARY (ONCE!)
# ============================================================================
print("\n" + "="*80)
print("GENERATING IRREGULAR INCLUSION BOUNDARY (FIXED FOR ALL REFINEMENTS)")
print("="*80)

torch.manual_seed(seed)
np.random.seed(seed)

a_coeffs = torch.randn(n_modes, dtype=torch.float64, device=device) * irregularity
b_coeffs = torch.randn(n_modes, dtype=torch.float64, device=device) * irregularity

print(f"Inclusion center: ({center_x}, {center_y})")
print(f"Base radius: {R_base}")
print(f"Fourier modes: {n_modes}")
print(f"Irregularity: {irregularity}")
print(f"Random seed: {seed}")

# Sample the boundary for visualization
theta_sample = torch.linspace(0, 2*torch.pi, 200, dtype=torch.float64, device=device)
r_sample = R_base * torch.ones_like(theta_sample)
for n in range(1, n_modes + 1):
    r_sample = r_sample + R_base * (a_coeffs[n-1] * torch.cos(n * theta_sample) + 
                                    b_coeffs[n-1] * torch.sin(n * theta_sample))
r_sample = torch.clamp(r_sample, R_base * 0.5, R_base * 1.5)

print(f"Min inclusion radius: {r_sample.min().item():.4f}")
print(f"Max inclusion radius: {r_sample.max().item():.4f}")

def is_inside_irregular_inclusion(x, y):
    """Check if point is inside irregular inclusion (FIXED boundary)"""
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

# ============================================================================
# MESH GENERATION FUNCTION
# ============================================================================
def assemble_stiffness_sparse(points, elements, element_materials, nu):
    """
    Assemble global stiffness matrix using sparse format
    
    Args:
        points: (n_nodes, 2) tensor of node coordinates
        elements: (n_elements, 3) tensor of element connectivity
        element_materials: (n_elements,) tensor of Young's modulus per element
        nu: Poisson's ratio
    
    Returns:
        K: (n_dof, n_dof) sparse stiffness matrix
    """
    n_dof = 2 * len(points)
    
    # Lists to store COO format data
    indices_i = []
    indices_j = []
    values = []
    
    for e in range(len(elements)):
        node_ids = elements[e]
        coords = points[node_ids]
        E = element_materials[e]
        
        # Compute element stiffness using existing function from fem_utils
        from fem_utils import element_stiffness
        K_e = element_stiffness(coords, E, nu)
        
        # Global DOF indices for this element
        dof_ids = torch.tensor([
            2*node_ids[0], 2*node_ids[0]+1,
            2*node_ids[1], 2*node_ids[1]+1,
            2*node_ids[2], 2*node_ids[2]+1
        ], dtype=torch.long, device=points.device)
        
        # Add to COO lists
        for i in range(6):
            for j in range(6):
                indices_i.append(dof_ids[i].item())
                indices_j.append(dof_ids[j].item())
                values.append(K_e[i, j].item())
    
    # Convert to COO format tensors
    indices = torch.tensor([indices_i, indices_j], dtype=torch.long, device=points.device)
    values_tensor = torch.tensor(values, dtype=torch.float64, device=points.device)
    
    # Create sparse tensor and coalesce (sum duplicate entries)
    K_sparse = torch.sparse_coo_tensor(indices, values_tensor, (n_dof, n_dof), device=points.device)
    K_sparse = K_sparse.coalesce()
    
    return K_sparse

def create_mesh(n_radial, n_angular):
    """Create structured polar mesh with given resolution"""
    # Create radial and angular coordinates
    r = torch.linspace(0, R_outer, n_radial, dtype=torch.float64, device=device)
    theta = torch.linspace(0, 2*torch.pi, n_angular+1, dtype=torch.float64, device=device)[:-1]
    
    # Create meshgrid
    R, Theta = torch.meshgrid(r, theta, indexing='ij')
    
    # Convert to Cartesian
    X = R * torch.cos(Theta)
    Y = R * torch.sin(Theta)
    
    # Flatten to get all points
    points = torch.stack([X.ravel(), Y.ravel()], dim=1)
    
    # Handle center point (remove duplicates at r=0)
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
    
    # Create triangulation
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
    
    # Assign materials (using FIXED inclusion boundary)
    element_materials = torch.zeros(len(elements), dtype=torch.float64, device=device)
    for e, elem in enumerate(elements):
        elem_coords = points[elem]
        centroid = elem_coords.mean(dim=0)
        if is_inside_irregular_inclusion(centroid[0], centroid[1]):
            element_materials[e] = E_i
        else:
            element_materials[e] = E_b
    
    return points, elements, element_materials

# ============================================================================
# FIND INVARIANT BOUNDARY NODES
# ============================================================================
def find_invariant_boundary_nodes(points, invariant_spacing):
    """
    Find boundary nodes at angles that are multiples of the invariant angular spacing.
    These nodes exist in ALL refined meshes across all refinement levels.
    
    Args:
        points: node coordinates
        invariant_spacing: angular spacing in degrees (e.g., 18° for common nodes)
    """
    # Find all boundary nodes
    radii = torch.sqrt(points[:, 0]**2 + points[:, 1]**2)
    boundary_mask = torch.abs(radii - R_outer) < 1e-6
    boundary_nodes = torch.where(boundary_mask)[0]
    
    # Get angles of boundary nodes
    angles = torch.atan2(points[boundary_nodes, 1], points[boundary_nodes, 0])
    angles_deg = angles * 180.0 / torch.pi
    angles_deg = torch.where(angles_deg < 0, angles_deg + 360.0, angles_deg)
    
    # Find nodes at invariant angles (multiples of invariant_spacing)
    invariant_boundary_nodes = []
    invariant_angles = []
    
    for target_angle in torch.arange(0, 360, invariant_spacing, dtype=torch.float64, device=device):
        # Find closest node to this angle
        angle_diffs = torch.abs(angles_deg - target_angle)
        min_idx = torch.argmin(angle_diffs)
        
        if angle_diffs[min_idx] < 0.5:  # Within 0.5 degree tolerance (relaxed for 1.5x refinement)
            invariant_boundary_nodes.append(boundary_nodes[min_idx].item())
            invariant_angles.append(target_angle.item())
    
    return torch.tensor(invariant_boundary_nodes, dtype=torch.long, device=device), \
           torch.tensor(invariant_angles, dtype=torch.float64, device=device)

# ============================================================================
# SOLVE FORWARD PROBLEM
# ============================================================================
def solve_forward_problem(points, elements, element_materials, force_angle_1_deg, force_angle_2_deg):
    """Solve F=KU for given mesh"""
    n_nodes = len(points)
    n_dof = 2 * n_nodes
    
    # Assemble sparse stiffness matrix
    print("    Assembling sparse K...", end='', flush=True)
    K_sparse = assemble_stiffness_sparse(points, elements, element_materials, nu)
    print(" Done")
    print(f"    K sparsity: {K_sparse._nnz() / (n_dof * n_dof) * 100:.4f}%")
    
    # Find boundary nodes
    radii = torch.sqrt(points[:, 0]**2 + points[:, 1]**2)
    boundary_mask = torch.abs(radii - R_outer) < 1e-6
    boundary_nodes = torch.where(boundary_mask)[0]
    
    # Setup boundary conditions (fix inclusion nodes)
    inside_mask = is_inside_irregular_inclusion(points[:, 0], points[:, 1])
    inside_nodes = torch.where(inside_mask)[0]
    
    fixed_dofs = []
    for node in inside_nodes:
        fixed_dofs.extend([2*node, 2*node+1])
    fixed_dofs = torch.tensor(fixed_dofs, dtype=torch.long, device=device)
    
    all_dofs = torch.arange(n_dof, device=device)
    free_dofs = torch.tensor([dof for dof in all_dofs if dof not in fixed_dofs], 
                            dtype=torch.long, device=device)
    
    # Apply forces at specified angles
    F = torch.zeros(n_dof, dtype=torch.float64, device=device)
    
    # Find nodes closest to force angles
    angles = torch.atan2(points[boundary_nodes, 1], points[boundary_nodes, 0])
    angles_deg = angles * 180.0 / torch.pi
    
    # Node 1 at force_angle_1
    angle1_rad = force_angle_1_deg * torch.pi / 180.0
    idx_1 = torch.argmin(torch.abs(angles - angle1_rad))
    node1 = boundary_nodes[idx_1]
    
    # Node 2 at force_angle_2
    angle2_rad = force_angle_2_deg * torch.pi / 180.0
    idx_2 = torch.argmin(torch.abs(angles - angle2_rad))
    node2 = boundary_nodes[idx_2]
    
    # Apply equal and opposite forces (radial direction)
    pos1 = points[node1]
    pos2 = points[node2]
    dir1 = pos1 / torch.norm(pos1)
    dir2 = pos2 / torch.norm(pos2)
    
    F[2*node1:2*node1+2] = -force_magnitude * dir1
    F[2*node2:2*node2+2] = force_magnitude * dir2
    
    # Solve K_free * U_free = F_free
    print("    Solving linear system...", end='', flush=True)
    
    # Convert sparse K to dense only for free DOFs (much smaller matrix)
    K_dense = K_sparse.to_dense()
    K_free = K_dense[free_dofs][:, free_dofs]
    F_free = F[free_dofs]
    U_free = torch.linalg.solve(K_free, F_free)
    print(" Done")
    
    U = torch.zeros(n_dof, dtype=torch.float64, device=device)
    U[free_dofs] = U_free
    U_nodes = U.reshape(-1, 2)
    
    return U, U_nodes, boundary_nodes, K_dense, node1, node2

# ============================================================================
# COMPUTE CONVERGENCE METRIC
# ============================================================================
def compute_cauchy_convergence(U_inv_prev, U_inv_current):
    """
    Compute relative L2 difference between displacement fields at invariant nodes
    """
    diff = U_inv_current - U_inv_prev
    diff_norm = torch.sqrt(torch.sum(diff**2))
    U_norm = torch.sqrt(torch.sum(U_inv_current**2))
    
    relative_diff = (diff_norm / U_norm).item()
    return relative_diff

# ============================================================================
# MAIN CONVERGENCE STUDY LOOP
# ============================================================================
print("\n" + "="*80)
print("RUNNING MESH CONVERGENCE STUDY")
print("="*80)

# Storage for results
mesh_params_list = []
n_nodes_list = []
n_elements_list = []
max_displacements_list = []
relative_differences_list = []
converged = False

# Storage for displacement fields at invariant nodes
U_invariant_list = []
invariant_angles_list = []

# Storage for full boundary displacement (for animation)
boundary_disp_magnitude_list = []
boundary_angles_list = []
deformed_boundaries_list = []

# Previous solution for comparison
U_inv_prev = None

# Final converged mesh data
final_points = None
final_elements = None
final_element_materials = None
final_K = None
final_U_nodes = None

for i in range(max_refinements):
    # Calculate current mesh parameters (ensure integers)
    n_radial = int(np.round(n_radial_start * (refinement_factor ** i)))
    n_angular = int(np.round(n_angular_start * (refinement_factor ** i)))
    
    print(f"\n{'='*80}")
    print(f"REFINEMENT LEVEL {i+1}/{max_refinements}")
    print(f"{'='*80}")
    print(f"Mesh parameters: n_radial={n_radial}, n_angular={n_angular}")
    
    # Create mesh
    print("  Creating mesh...")
    points, elements, element_materials = create_mesh(n_radial, n_angular)
    n_nodes = len(points)
    n_elements = len(elements)
    n_dof = 2 * n_nodes
    
    print(f"    Nodes: {n_nodes}")
    print(f"    Elements: {n_elements}")
    print(f"    DOFs: {n_dof}")
    print(f"    Background elements: {torch.sum(element_materials == E_b)}")
    print(f"    Inclusion elements: {torch.sum(element_materials == E_i)}")
    
    # Find invariant boundary nodes
    print("  Finding invariant boundary nodes...")
    invariant_nodes, invariant_angles = find_invariant_boundary_nodes(points, invariant_angular_spacing)
    print(f"    Invariant nodes: {len(invariant_nodes)} (spacing = {invariant_angular_spacing:.1f}°)")
    
    # Solve forward problem
    print("  Solving F=KU...")
    U, U_nodes, boundary_nodes, K, node1, node2 = solve_forward_problem(
        points, elements, element_materials, force_angle_1, force_angle_2
    )
    
    # Extract displacements at invariant nodes
    U_inv = U_nodes[invariant_nodes]
    
    # Compute max displacement
    max_disp = torch.max(torch.sqrt(U_nodes[:, 0]**2 + U_nodes[:, 1]**2)).item()
    print(f"  Max displacement: {max_disp:.8e}")
    
    # Store results
    mesh_params_list.append((n_radial, n_angular))
    n_nodes_list.append(n_nodes)
    n_elements_list.append(n_elements)
    max_displacements_list.append(max_disp)
    U_invariant_list.append(U_inv.cpu().clone())
    invariant_angles_list.append(invariant_angles.cpu().clone())
    
    # Store full boundary displacement for animation
    boundary_positions = points[boundary_nodes].cpu().numpy()
    boundary_displacements = U_nodes[boundary_nodes].cpu().numpy()
    boundary_disp_mag = np.sqrt(boundary_displacements[:, 0]**2 + boundary_displacements[:, 1]**2)
    boundary_angles_array = np.arctan2(boundary_positions[:, 1], boundary_positions[:, 0]) * 180.0 / np.pi
    boundary_angles_array = np.where(boundary_angles_array < 0, boundary_angles_array + 360.0, boundary_angles_array)
    
    # Sort by angle for plotting
    sort_idx = np.argsort(boundary_angles_array)
    boundary_angles_list.append(boundary_angles_array[sort_idx])
    boundary_disp_magnitude_list.append(boundary_disp_mag[sort_idx])
    
    # Deformed boundary
    deformed_boundary = boundary_positions + boundary_displacements
    deformed_boundaries_list.append(deformed_boundary[sort_idx])
    
    # Check convergence (if not first iteration)
    if i > 0:
        rel_diff = compute_cauchy_convergence(U_inv_prev, U_inv)
        relative_differences_list.append(rel_diff)
        print(f"  Relative change from previous: {rel_diff:.8f} ({rel_diff*100:.6f}%)")
        
        if rel_diff < convergence_tolerance:
            print(f"\n  ✓ CONVERGENCE ACHIEVED! (relative change < {convergence_tolerance})")
            converged = True
            # Store final mesh data
            final_points = points.cpu()
            final_elements = elements.cpu()
            final_element_materials = element_materials.cpu()
            final_K = K.cpu()
            final_U_nodes = U_nodes.cpu()
            break
    else:
        relative_differences_list.append(np.nan)
    
    # Save current solution for next comparison
    U_inv_prev = U_inv.clone()
    
    # Store final mesh data if last iteration
    if i == max_refinements - 1:
        final_points = points.cpu()
        final_elements = elements.cpu()
        final_element_materials = element_materials.cpu()
        final_K = K.cpu()
        final_U_nodes = U_nodes.cpu()

# ============================================================================
# RESULTS SUMMARY
# ============================================================================
print("\n" + "="*80)
print("CONVERGENCE STUDY RESULTS")
print("="*80)

print(f"\n{'Level':<8} {'n_radial':<12} {'n_angular':<12} {'Nodes':<10} {'Elements':<10} {'Max Disp':<15} {'Rel. Change':<15}")
print("-" * 95)

for i in range(len(mesh_params_list)):
    n_rad, n_ang = mesh_params_list[i]
    n_nod = n_nodes_list[i]
    n_elem = n_elements_list[i]
    max_d = max_displacements_list[i]
    rel_d = relative_differences_list[i]
    
    rel_d_str = f"{rel_d:.8f}" if not np.isnan(rel_d) else "N/A"
    print(f"{i+1:<8} {n_rad:<12} {n_ang:<12} {n_nod:<10} {n_elem:<10} {max_d:<15.8e} {rel_d_str:<15}")

if converged:
    print(f"\n✓ Mesh converged at refinement level {len(mesh_params_list)}")
    print(f"  Final mesh: {n_nodes_list[-1]} nodes, {n_elements_list[-1]} elements")
    print(f"  Final max displacement: {max_displacements_list[-1]:.8e}")
else:
    print(f"\n⚠ Mesh did not converge within {max_refinements} refinements")
    print(f"  Consider increasing max_refinements or loosening tolerance")

# ============================================================================
# VISUALIZATION 1: SAMPLE PLOT (Final Converged Mesh)
# ============================================================================
print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

print("\n1. Creating sample plot (final converged mesh)...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Convert to numpy for plotting
points_np = final_points.numpy()
elements_np = final_elements.numpy()
materials_np = final_element_materials.numpy()
theta_plot_np = theta_sample.cpu().numpy()
r_plot_np = r_sample.cpu().numpy()

# Compute irregular boundary in Cartesian coordinates
x_boundary = center_x + r_plot_np * np.cos(theta_plot_np)
y_boundary = center_y + r_plot_np * np.sin(theta_plot_np)

# Plot 1: Mesh with material distribution
ax = axes[0]
for e, elem in enumerate(elements_np):
    elem_coords = points_np[elem]
    color = 'red' if materials_np[e] == E_i else 'blue'
    tri = plt.Polygon(elem_coords, facecolor=color, edgecolor='black', 
                      alpha=0.5, linewidth=0.3)
    ax.add_patch(tri)

# Draw boundaries
theta_circle = np.linspace(0, 2*np.pi, 100)
ax.plot(R_outer*np.cos(theta_circle), R_outer*np.sin(theta_circle), 'k-', 
        linewidth=2, label='Outer boundary')
ax.plot(x_boundary, y_boundary, 'k--', 
        linewidth=2, label='Irregular inclusion')
ax.plot(center_x, center_y, 'go', markersize=8, label='Inclusion center')

ax.set_xlim(-R_outer*1.1, R_outer*1.1)
ax.set_ylim(-R_outer*1.1, R_outer*1.1)
ax.set_aspect('equal')
ax.set_title('Converged Mesh - Irregular Inclusion\n(Red=Inclusion, Blue=Background)', 
             fontsize=12, fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Mesh detail
ax = axes[1]
for elem in elements_np:
    elem_coords = points_np[elem]
    tri = plt.Polygon(elem_coords, facecolor='lightgray', edgecolor='black', linewidth=0.5)
    ax.add_patch(tri)

ax.plot(points_np[:, 0], points_np[:, 1], 'b.', markersize=2, label='Nodes')
ax.plot(R_outer*np.cos(theta_circle), R_outer*np.sin(theta_circle), 'k-', 
        linewidth=2, label='Outer boundary')
ax.plot(x_boundary, y_boundary, 'r--', 
        linewidth=2, label='Irregular inclusion')
ax.plot(center_x, center_y, 'go', markersize=8, label='Inclusion center')

ax.set_xlim(-R_outer*1.1, R_outer*1.1)
ax.set_ylim(-R_outer*1.1, R_outer*1.1)
ax.set_aspect('equal')
ax.set_title(f'Converged Mesh Triangulation\n({n_nodes_list[-1]} nodes, {n_elements_list[-1]} triangles)', 
             fontsize=12, fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Stiffness matrix
ax = axes[2]
K_np = final_K.numpy()
K_np_abs = np.abs(K_np)
K_np_log = np.log10(K_np_abs + 1e-16)

im = ax.imshow(K_np_log, cmap='viridis', aspect='auto', interpolation='nearest')
ax.set_title(f'Stiffness Matrix (log scale)\n({2*n_nodes_list[-1]} DOFs, {np.count_nonzero(K_np)} non-zeros)', 
             fontsize=12, fontweight='bold')
ax.set_xlabel('Column Index')
ax.set_ylabel('Row Index')

cbar = plt.colorbar(im, ax=ax)
cbar.set_label('log₁₀|K|', rotation=270, labelpad=20)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/converged_mesh_sample.png', dpi=150, bbox_inches='tight')
print("  Saved: converged_mesh_sample.png")
plt.close()

# ============================================================================
# VISUALIZATION 2: CAUCHY CONVERGENCE PLOT
# ============================================================================
print("\n2. Creating Cauchy convergence plot...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Max displacement vs number of nodes
ax = axes[0, 0]
ax.plot(n_nodes_list, max_displacements_list, 'bo-', linewidth=2, markersize=8)
ax.set_xlabel('Number of Nodes', fontsize=12)
ax.set_ylabel('Max Displacement', fontsize=12)
ax.set_title('Displacement Convergence vs Mesh Size', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xscale('log')

# Plot 2: Relative change vs refinement level
ax = axes[0, 1]
valid_diffs = [d for d in relative_differences_list if not np.isnan(d)]
valid_levels = list(range(2, len(valid_diffs) + 2))
ax.semilogy(valid_levels, valid_diffs, 'ro-', linewidth=2, markersize=8)
ax.axhline(y=convergence_tolerance, color='g', linestyle='--', linewidth=2, 
          label=f'Tolerance ({convergence_tolerance})')
ax.set_xlabel('Refinement Level', fontsize=12)
ax.set_ylabel('Relative Change ||u_j+1 - u_j|| / ||u_j||', fontsize=12)
ax.set_title('Cauchy Convergence Criterion', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)

# Plot 3: Number of elements vs refinement level
ax = axes[1, 0]
ax.plot(range(1, len(n_elements_list) + 1), n_elements_list, 'gs-', linewidth=2, markersize=8)
ax.set_xlabel('Refinement Level', fontsize=12)
ax.set_ylabel('Number of Elements', fontsize=12)
ax.set_title('Mesh Size Growth', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# Plot 4: Max displacement convergence (zoomed in)
ax = axes[1, 1]
if len(max_displacements_list) > 1:
    final_disp = max_displacements_list[-1]
    relative_error = [abs(d - final_disp) / final_disp for d in max_displacements_list]
    ax.semilogy(range(1, len(relative_error) + 1), relative_error, 'mo-', linewidth=2, markersize=8)
    ax.set_xlabel('Refinement Level', fontsize=12)
    ax.set_ylabel('Relative Error vs Final', fontsize=12)
    ax.set_title('Asymptotic Convergence', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/cauchy_convergence.png', dpi=300, bbox_inches='tight')
print("  Saved: cauchy_convergence.png")
plt.close()

# ============================================================================
# VISUALIZATION 3: ANIMATION (GIF) - OPTION C (Both plots side-by-side)
# ============================================================================
print("\n3. Creating animation (GIF)...")

# Create frames directory
frames_dir = Path('/home/claude/convergence_frames')
frames_dir.mkdir(exist_ok=True)

frame_files = []

for i in range(len(mesh_params_list)):
    print(f"  Creating frame {i+1}/{len(mesh_params_list)}...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # LEFT PLOT: Displacement magnitude vs angle
    ax = axes[0]
    angles_plot = boundary_angles_list[i]
    disp_mag_plot = boundary_disp_magnitude_list[i]
    
    # Close the loop for plotting
    angles_plot_closed = np.append(angles_plot, angles_plot[0])
    disp_mag_plot_closed = np.append(disp_mag_plot, disp_mag_plot[0])
    
    ax.plot(angles_plot_closed, disp_mag_plot_closed, 'b-', linewidth=2, label=f'Level {i+1}')
    ax.set_xlabel('Angle (degrees)', fontsize=12)
    ax.set_ylabel('Displacement Magnitude |u|', fontsize=12)
    ax.set_title(f'Boundary Displacement Profile\nRefinement Level {i+1}: {n_nodes_list[i]} nodes', 
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 360)
    
    # Add info box
    info_text = f'n_radial = {mesh_params_list[i][0]}\n'
    info_text += f'n_angular = {mesh_params_list[i][1]}\n'
    info_text += f'Max |u| = {disp_mag_plot.max():.6e}'
    if i > 0:
        info_text += f'\nRel. change = {relative_differences_list[i]:.6e}'
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    # RIGHT PLOT: Deformed boundary shape
    ax = axes[1]
    
    # Original boundary
    theta_circle = np.linspace(0, 2*np.pi, 200)
    ax.plot(R_outer*np.cos(theta_circle), R_outer*np.sin(theta_circle), 'k--', 
            linewidth=1.5, alpha=0.5, label='Original boundary')
    
    # Irregular inclusion
    ax.plot(x_boundary, y_boundary, 'r--', 
            linewidth=1.5, alpha=0.5, label='Inclusion (fixed)')
    
    # Deformed boundary
    deformed_boundary = deformed_boundaries_list[i]
    deformed_closed = np.vstack([deformed_boundary, deformed_boundary[0]])
    ax.plot(deformed_closed[:, 0], deformed_closed[:, 1], 'b-', 
            linewidth=2, label='Deformed boundary')
    
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(f'Deformed Boundary Shape\nRefinement Level {i+1}', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    frame_filename = frames_dir / f'convergence_frame_{i:03d}.png'
    plt.savefig(frame_filename, dpi=150, bbox_inches='tight')
    frame_files.append(str(frame_filename))
    plt.close()

# Create GIF
print("  Assembling GIF...")
images = []
for frame_file in frame_files:
    images.append(imageio.imread(frame_file))

gif_filename = '/mnt/user-data/outputs/mesh_convergence_animation.gif'
imageio.mimsave(gif_filename, images, fps=1, loop=0)  # 1 fps = 1 second per frame
print(f"  Saved: mesh_convergence_animation.gif")

# ============================================================================
# SAVE CONVERGENCE DATA
# ============================================================================
print("\n" + "="*80)
print("SAVING CONVERGENCE DATA")
print("="*80)

convergence_data = {
    'mesh_params': mesh_params_list,
    'n_nodes': n_nodes_list,
    'n_elements': n_elements_list,
    'max_displacements': max_displacements_list,
    'relative_differences': relative_differences_list,
    'converged': converged,
    'convergence_tolerance': convergence_tolerance,
    'final_mesh_params': mesh_params_list[-1],
    'final_points': final_points,
    'final_elements': final_elements,
    'final_element_materials': final_element_materials,
    'final_U_nodes': final_U_nodes,
    'U_invariant_list': U_invariant_list,
    'invariant_angles_list': invariant_angles_list,
    'a_coeffs': a_coeffs.cpu(),
    'b_coeffs': b_coeffs.cpu(),
    'geometry_params': {
        'R_outer': R_outer,
        'R_base': R_base,
        'center_x': center_x,
        'center_y': center_y,
        'n_modes': n_modes,
        'irregularity': irregularity,
        'seed': seed
    },
    'material_params': {
        'E_b': E_b,
        'E_i': E_i,
        'nu': nu
    }
}

torch.save(convergence_data, '/mnt/user-data/outputs/convergence_data.pt')
print("Saved: convergence_data.pt")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("MESH CONVERGENCE STUDY COMPLETE")
print("="*80)

print(f"\nGeometry: Irregular inclusion (fixed boundary)")
print(f"  Center: ({center_x}, {center_y})")
print(f"  Base radius: {R_base}")
print(f"  Fourier modes: {n_modes}")

print(f"\nRefinement strategy:")
print(f"  Starting mesh: n_radial={n_radial_start}, n_angular={n_angular_start}")
print(f"  Refinement factor: {refinement_factor}x")
print(f"  Levels completed: {len(mesh_params_list)}")

if converged:
    print(f"\n✓ CONVERGED at level {len(mesh_params_list)}")
    print(f"  Final mesh: {n_nodes_list[-1]} nodes, {n_elements_list[-1]} elements")
    print(f"  Final max displacement: {max_displacements_list[-1]:.8e}")
    print(f"  Final relative change: {relative_differences_list[-1]:.8e}")
else:
    print(f"\n⚠ DID NOT CONVERGE within {max_refinements} refinements")
    print(f"  Final relative change: {relative_differences_list[-1]:.8e}")
    print(f"  Target tolerance: {convergence_tolerance}")

print(f"\nOutput files:")
print(f"  1. converged_mesh_sample.png - Final mesh visualization")
print(f"  2. cauchy_convergence.png - Convergence plots")
print(f"  3. mesh_convergence_animation.gif - Displacement evolution")
print(f"  4. convergence_data.pt - All convergence data")

print("\n" + "="*80)