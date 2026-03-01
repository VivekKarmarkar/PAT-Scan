"""
Circle Mesh with Uniform Material - Analytical Validation
Tests F = KU using uniform tension analytical solution
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from fem_utils import *

# Setup
setup_torch_defaults()
device = get_device()

# ============================================================================
# PARAMETERS
# ============================================================================
R_outer = 1.0    # Radius of circle
E = 1.0          # Young's modulus (uniform throughout)
nu = 0.3         # Poisson's ratio

n_radial = 20    # Number of radial divisions
n_angular = 40   # Number of angular divisions

# Analytical solution parameter
sigma_0 = 1.0    # Applied radial stress on boundary

print("="*70)
print("Circle Mesh - Analytical Validation (PyTorch)")
print("="*70)
print("\nTest: Uniform radial tension with analytical solution")
print(f"Analytical displacement: u_r = sigma_0*(1-nu)/E * r")
print(f"Expected: u_x = {sigma_0*(1-nu)/E} * x")
print(f"Expected: u_y = {sigma_0*(1-nu)/E} * y")

# ============================================================================
# CREATE STRUCTURED POLAR GRID
# ============================================================================
print("\nCreating polar grid...")

# Create radial and angular coordinates
r = torch.linspace(0, R_outer, n_radial, dtype=torch.float64, device=device)
theta = torch.linspace(0, 2*torch.pi, n_angular+1, dtype=torch.float64, device=device)[:-1]

# Create meshgrid
R, Theta = torch.meshgrid(r, theta, indexing='ij')

# Convert to Cartesian coordinates
X = R * torch.cos(Theta)
Y = R * torch.sin(Theta)

# Flatten to get all points
points = torch.stack([X.ravel(), Y.ravel()], dim=1)

# Add center point explicitly (remove r=0 duplicates)
center_indices = torch.where(torch.sqrt(points[:, 0]**2 + points[:, 1]**2) < 1e-10)[0]
if len(center_indices) > 0:
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

n_nodes = len(points)

print(f"  Radial divisions: {n_radial}")
print(f"  Angular divisions: {n_angular}")
print(f"  Total nodes: {n_nodes}")

# ============================================================================
# CREATE TRIANGLES FROM POLAR GRID
# ============================================================================
print("\nCreating triangles from polar grid...")

elements = []

def node_idx(i, j):
    """Get node index for radial index i and angular index j"""
    if i == 0:
        return 0  # Center point
    else:
        return 1 + (i - 1) * n_angular + (j % n_angular)

# Create triangles
for i in range(n_radial - 1):
    for j in range(n_angular):
        if i == 0:
            # Triangles connected to center
            n0 = 0
            n1 = node_idx(1, j)
            n2 = node_idx(1, j + 1)
            elements.append([n0, n1, n2])
        else:
            # Quadrilaterals split into 2 triangles
            n0 = node_idx(i, j)
            n1 = node_idx(i, j + 1)
            n2 = node_idx(i + 1, j + 1)
            n3 = node_idx(i + 1, j)
            
            elements.append([n0, n1, n2])
            elements.append([n0, n2, n3])

elements = torch.tensor(elements, dtype=torch.long, device=device)
n_elements = len(elements)

print(f"  Triangles: {n_elements}")

# ============================================================================
# ASSIGN MATERIALS (Uniform)
# ============================================================================
print("\nAssigning materials...")

element_materials = torch.full((n_elements,), E, dtype=torch.float64, device=device)

print(f"  All elements have E = {E}")

n_dof = 2 * n_nodes

# ============================================================================
# ASSEMBLE GLOBAL STIFFNESS MATRIX K
# ============================================================================
print("\nAssembling stiffness matrix K...")
K = assemble_stiffness(points, elements, element_materials, nu)

print_mesh_info(n_nodes, n_elements, n_dof, K)

# ============================================================================
# COMPUTE ANALYTICAL SOLUTION (Ground Truth U)
# ============================================================================
print("\nComputing analytical solution U_true...")

# Analytical displacement: u_r = sigma_0 * (1-nu) / E * r
# In Cartesian: u_x = sigma_0 * (1-nu) / E * x
#               u_y = sigma_0 * (1-nu) / E * y

U_true = torch.zeros(n_dof, dtype=torch.float64, device=device)
displacement_factor = sigma_0 * (1 - nu) / E

for i in range(n_nodes):
    x, y = points[i]
    U_true[2*i] = displacement_factor * x      # u_x
    U_true[2*i+1] = displacement_factor * y    # u_y

print(f"  Displacement factor: {displacement_factor:.6f}")
print(f"  Max displacement (at r=R): {torch.max(torch.abs(U_true)):.6e}")

# ============================================================================
# COMPUTE FORCE VECTOR F FROM BOUNDARY TRACTIONS
# ============================================================================
print("\nComputing force vector F from boundary tractions...")

# Find boundary nodes (nodes at r = R_outer)
distances = torch.sqrt(points[:, 0]**2 + points[:, 1]**2)
boundary_mask = torch.abs(distances - R_outer) < 1e-6
boundary_nodes = torch.where(boundary_mask)[0]

print(f"  Boundary nodes: {len(boundary_nodes)}")

# Initialize force vector
F = torch.zeros(n_dof, dtype=torch.float64, device=device)

# Apply normal traction on boundary: t_n = sigma_0
# For each boundary node, compute associated arc length and apply force
# Arc length per node: ds = 2*pi*R / n_angular

thickness = 1.0  # Unit thickness
ds = 2 * torch.pi * R_outer / n_angular  # Arc length per boundary node

for node_id in boundary_nodes:
    x, y = points[node_id]
    
    # Outward normal (for circle centered at origin)
    n_x = x / R_outer
    n_y = y / R_outer
    
    # Traction: t = sigma_0 * n
    t_x = sigma_0 * n_x
    t_y = sigma_0 * n_y
    
    # Force: F = t * ds * thickness
    F[2*node_id] = t_x * ds * thickness
    F[2*node_id+1] = t_y * ds * thickness

print(f"  Applied radial traction: sigma_0 = {sigma_0}")
print(f"  Total force magnitude: {torch.norm(F):.6e}")
print(f"  Sum of F_x: {torch.sum(F[0::2]):.6e} (should be ~0 by symmetry)")
print(f"  Sum of F_y: {torch.sum(F[1::2]):.6e} (should be ~0 by symmetry)")

# ============================================================================
# SOLVE F = K * U_hat
# ============================================================================
print("\nSolving F = K * U_hat...")

# Add small regularization to handle potential rigid body modes
# (though with boundary forces, this shouldn't be necessary)
K_reg = K + 1e-10 * torch.eye(n_dof, dtype=torch.float64, device=device)

U_hat = torch.linalg.solve(K_reg, F)

print(f"  Solution complete")
print(f"  Max displacement (FEM): {torch.max(torch.abs(U_hat)):.6e}")

# ============================================================================
# COMPARE U_hat vs U_true
# ============================================================================
print("\n" + "="*70)
print("VALIDATION RESULTS")
print("="*70)

# Compute errors
error_abs = torch.norm(U_hat - U_true)
error_rel = error_abs / torch.norm(U_true)

# Nodal errors
U_hat_nodes = U_hat.reshape(n_nodes, 2)
U_true_nodes = U_true.reshape(n_nodes, 2)
nodal_errors = torch.norm(U_hat_nodes - U_true_nodes, dim=1)
max_nodal_error = torch.max(nodal_errors)
mean_nodal_error = torch.mean(nodal_errors)

print(f"\nDisplacement Comparison:")
print(f"  ||U_true||: {torch.norm(U_true):.6e}")
print(f"  ||U_hat||:  {torch.norm(U_hat):.6e}")
print(f"\nError Metrics:")
print(f"  Absolute error: {error_abs:.6e}")
print(f"  Relative error: {error_rel:.6e}")
print(f"  Max nodal error: {max_nodal_error:.6e}")
print(f"  Mean nodal error: {mean_nodal_error:.6e}")

# Check if validation passes
tolerance = 1e-2
if error_rel < tolerance:
    print(f"\n✓ VALIDATION PASSED (error < {tolerance})")
else:
    print(f"\n✗ VALIDATION FAILED (error >= {tolerance})")

# Sample point comparison
print(f"\nSample point comparisons:")
sample_indices = [0, n_nodes//4, n_nodes//2, 3*n_nodes//4, -1]
for idx in sample_indices:
    if idx < 0:
        idx = n_nodes + idx
    x, y = points[idx].cpu().numpy()
    u_true_x, u_true_y = U_true_nodes[idx].cpu().numpy()
    u_hat_x, u_hat_y = U_hat_nodes[idx].cpu().numpy()
    err = nodal_errors[idx].cpu().item()
    print(f"  Node {idx:4d} at ({x:6.3f}, {y:6.3f}):")
    print(f"    U_true: ({u_true_x:9.6f}, {u_true_y:9.6f})")
    print(f"    U_hat:  ({u_hat_x:9.6f}, {u_hat_y:9.6f})")
    print(f"    Error:  {err:.6e}")

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\nCreating visualizations...")

# Convert to numpy for plotting
points_np = points.cpu().numpy()
elements_np = elements.cpu().numpy()
U_true_nodes_np = U_true_nodes.cpu().numpy()
U_hat_nodes_np = U_hat_nodes.cpu().numpy()
nodal_errors_np = nodal_errors.cpu().numpy()

# Create figure with 4 subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 16))

# Plot 1: Mesh
ax = axes[0, 0]
for elem in elements_np:
    elem_coords = points_np[elem]
    tri = plt.Polygon(elem_coords, facecolor='lightblue', edgecolor='black', 
                      linewidth=0.3, alpha=0.7)
    ax.add_patch(tri)

ax.plot(points_np[:, 0], points_np[:, 1], 'b.', markersize=2)
theta_plot = np.linspace(0, 2*np.pi, 100)
ax.plot(R_outer*np.cos(theta_plot), R_outer*np.sin(theta_plot), 'k-', 
        linewidth=2, label='Boundary')
ax.set_xlim(-R_outer*1.1, R_outer*1.1)
ax.set_ylim(-R_outer*1.1, R_outer*1.1)
ax.set_aspect('equal')
ax.set_title(f'Mesh: {n_nodes} nodes, {n_elements} triangles', 
             fontsize=14, fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Analytical displacement magnitude
ax = axes[0, 1]
U_true_mag = np.sqrt(U_true_nodes_np[:, 0]**2 + U_true_nodes_np[:, 1]**2)
scatter = ax.scatter(points_np[:, 0], points_np[:, 1], 
                     c=U_true_mag, cmap='viridis', s=50, edgecolors='none')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('|U_true|', rotation=270, labelpad=20)
ax.plot(R_outer*np.cos(theta_plot), R_outer*np.sin(theta_plot), 'k-', linewidth=2)
ax.set_xlim(-R_outer*1.1, R_outer*1.1)
ax.set_ylim(-R_outer*1.1, R_outer*1.1)
ax.set_aspect('equal')
ax.set_title('Analytical Displacement Magnitude', fontsize=14, fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid(True, alpha=0.3)

# Plot 3: FEM displacement magnitude
ax = axes[1, 0]
U_hat_mag = np.sqrt(U_hat_nodes_np[:, 0]**2 + U_hat_nodes_np[:, 1]**2)
scatter = ax.scatter(points_np[:, 0], points_np[:, 1], 
                     c=U_hat_mag, cmap='viridis', s=50, edgecolors='none')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('|U_hat|', rotation=270, labelpad=20)
ax.plot(R_outer*np.cos(theta_plot), R_outer*np.sin(theta_plot), 'k-', linewidth=2)
ax.set_xlim(-R_outer*1.1, R_outer*1.1)
ax.set_ylim(-R_outer*1.1, R_outer*1.1)
ax.set_aspect('equal')
ax.set_title('FEM Displacement Magnitude', fontsize=14, fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid(True, alpha=0.3)

# Plot 4: Error distribution
ax = axes[1, 1]
scatter = ax.scatter(points_np[:, 0], points_np[:, 1], 
                     c=nodal_errors_np, cmap='hot', s=50, edgecolors='none')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('|U_hat - U_true|', rotation=270, labelpad=20)
ax.plot(R_outer*np.cos(theta_plot), R_outer*np.sin(theta_plot), 'k-', linewidth=2)
ax.set_xlim(-R_outer*1.1, R_outer*1.1)
ax.set_ylim(-R_outer*1.1, R_outer*1.1)
ax.set_aspect('equal')
ax.set_title(f'Error Distribution (Rel. Error = {error_rel:.2e})', 
             fontsize=14, fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid(True, alpha=0.3)

# Add text box with validation results
status = "PASSED ✓" if error_rel < tolerance else "FAILED ✗"
info_text = f'VALIDATION {status}\n\n'
info_text += f'Relative error: {error_rel:.2e}\n'
info_text += f'Max nodal error: {max_nodal_error:.2e}\n'
info_text += f'Mean nodal error: {mean_nodal_error:.2e}\n\n'
info_text += f'Mesh: {n_nodes} nodes\n'
info_text += f'DOFs: {n_dof}\n'
info_text += f'Elements: {n_elements}'

ax.text(0.98, 0.02, info_text, transform=ax.transAxes, 
        fontsize=10, verticalalignment='bottom', horizontalalignment='right',
        family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat' if error_rel < tolerance else 'lightcoral', 
                  alpha=0.9))

plt.tight_layout()
plt.savefig('circle_uniform_validation.png', dpi=150, bbox_inches='tight')
print("Saved: circle_uniform_validation.png")

# ============================================================================
# SAVE DATA
# ============================================================================
torch.save({
    'points': points.cpu(),
    'elements': elements.cpu(),
    'K': K.cpu(),
    'F': F.cpu(),
    'U_true': U_true.cpu(),
    'U_hat': U_hat.cpu(),
    'element_materials': element_materials.cpu(),
    'E': E,
    'nu': nu,
    'sigma_0': sigma_0,
    'R_outer': R_outer,
    'error_rel': error_rel.cpu(),
    'nodal_errors': nodal_errors.cpu()
}, 'circle_uniform_validation.pt')

print("\nSaved: circle_uniform_validation.pt")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Problem: Uniform circular domain with radial tension")
print(f"Analytical: u = {displacement_factor:.6f} * r")
print(f"Domain: R = {R_outer}")
print(f"Material: E = {E}, nu = {nu}")
print(f"Applied stress: sigma_0 = {sigma_0}")
print(f"\nMesh: {n_nodes} nodes, {n_elements} elements, {n_dof} DOFs")
print(f"\nValidation: {status}")
print(f"  Relative error: {error_rel:.6e}")
print(f"  Tolerance: {tolerance}")
print("="*70)