"""
Circle-in-Circle Mesh Generation
Uses structured polar grid triangulation
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
R_outer = 1.0    # Radius of outer circle (domain)
#R_inner = 0.3    # Radius of inner circle (inclusion)
R_inner = 0.1
E_b = 1.0        # Young's modulus of background
#E_i = 10.0       # Young's modulus of inclusion
E_i = 1.2
nu = 0.3         # Poisson's ratio

n_radial = 20    # Number of radial divisions
n_angular = 40   # Number of angular divisions

print("="*70)
print("Circle-in-Circle Mesh (PyTorch)")
print("="*70)

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
# ASSIGN MATERIALS
# ============================================================================
print("\nAssigning materials...")

element_materials = assign_materials_circular(points, elements, R_inner, E_b, E_i)

print(f"  Background elements: {torch.sum(element_materials == E_b)}")
print(f"  Inclusion elements: {torch.sum(element_materials == E_i)}")

n_dof = 2 * n_nodes

# ============================================================================
# ASSEMBLE GLOBAL STIFFNESS MATRIX
# ============================================================================
print("\nAssembling stiffness matrix...")
K = assemble_stiffness(points, elements, element_materials, nu)

print_mesh_info(n_nodes, n_elements, n_dof, K)

# ============================================================================
# SAVE MESH DATA
# ============================================================================
torch.save({
    'points': points.cpu(),
    'elements': elements.cpu(),
    'K': K.cpu(),
    'element_materials': element_materials.cpu(),
    'E_b': E_b,
    'E_i': E_i,
    'nu': nu,
    'R_outer': R_outer,
    'R_inner': R_inner
}, 'circle_sample.pt')

print("\nSaved to: circle_sample.pt")

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\nCreating visualization...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Convert to numpy for plotting
points_np = points.cpu().numpy()
elements_np = elements.cpu().numpy()
materials_np = element_materials.cpu().numpy()

# Plot 1: Mesh with material distribution
ax = axes[0]
for e, elem in enumerate(elements_np):
    elem_coords = points_np[elem]
    color = 'red' if materials_np[e] == E_i else 'blue'
    tri = plt.Polygon(elem_coords, facecolor=color, edgecolor='black', 
                      alpha=0.5, linewidth=0.3)
    ax.add_patch(tri)

# Draw boundaries
theta_plot = np.linspace(0, 2*np.pi, 100)
ax.plot(R_outer*np.cos(theta_plot), R_outer*np.sin(theta_plot), 'k-', 
        linewidth=2, label='Outer boundary')
ax.plot(R_inner*np.cos(theta_plot), R_inner*np.sin(theta_plot), 'k--', 
        linewidth=2, label='Inclusion boundary')

ax.set_xlim(-R_outer*1.1, R_outer*1.1)
ax.set_ylim(-R_outer*1.1, R_outer*1.1)
ax.set_aspect('equal')
ax.set_title('Circle-in-Circle Mesh\n(Red=Inclusion, Blue=Background)', 
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
ax.plot(R_outer*np.cos(theta_plot), R_outer*np.sin(theta_plot), 'k-', 
        linewidth=2, label='Outer boundary')
ax.plot(R_inner*np.cos(theta_plot), R_inner*np.sin(theta_plot), 'r--', 
        linewidth=2, label='Inclusion')

ax.set_xlim(-R_outer*1.1, R_outer*1.1)
ax.set_ylim(-R_outer*1.1, R_outer*1.1)
ax.set_aspect('equal')
ax.set_title(f'Polar Grid Triangulation\n({n_nodes} nodes, {n_elements} triangles)', 
             fontsize=12, fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Stiffness matrix
ax = axes[2]
K_np = K.cpu().numpy()
K_np_abs = np.abs(K_np)
K_np_log = np.log10(K_np_abs + 1e-16)

im = ax.imshow(K_np_log, cmap='viridis', aspect='auto', interpolation='nearest')
ax.set_title(f'Stiffness Matrix (log scale)\n({n_dof} DOFs, {np.count_nonzero(K_np)} non-zeros)', 
             fontsize=12, fontweight='bold')
ax.set_xlabel('Column Index')
ax.set_ylabel('Row Index')

cbar = plt.colorbar(im, ax=ax)
cbar.set_label('log₁₀|K|', rotation=270, labelpad=20)

plt.tight_layout()
plt.savefig('circle_sample_plot.png', dpi=150, bbox_inches='tight')
print("Saved: circle_sample_plot.png")

print("\n" + "="*70)
print("SUCCESS: Circle-in-circle mesh!")
print("="*70)
print(f"Domain: Circle with R={R_outer}, Inclusion R={R_inner}")
print(f"Materials: E_background={E_b}, E_inclusion={E_i}, nu={nu}")
print(f"Mesh: {n_nodes} nodes, {n_elements} triangles, {n_dof} DOFs")
print(f"K matrix: {K.shape}, {torch.count_nonzero(K).item()} non-zeros")
print("="*70)