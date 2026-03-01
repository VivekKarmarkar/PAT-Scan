"""
Square Domain with Circular Inclusion - Mesh Generation
Uses structured grid triangulation
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
L = 1.0          # Domain size (square side length)
R = 0.2          # Radius of circular inclusion
E_b = 1.0        # Young's modulus of background
E_i = 10.0       # Young's modulus of inclusion
nu = 0.3         # Poisson's ratio

n_grid = 40      # Grid points per side

print("="*70)
print("ULTRA-SIMPLE Structured Triangulation (PyTorch)")
print("="*70)

# ============================================================================
# CREATE STRUCTURED GRID
# ============================================================================
print("\nCreating structured grid...")

# Create grid points
x = torch.linspace(-L/2, L/2, n_grid, dtype=torch.float64, device=device)
y = torch.linspace(-L/2, L/2, n_grid, dtype=torch.float64, device=device)
X, Y = torch.meshgrid(x, y, indexing='ij')

# Flatten to get all points
points = torch.stack([X.ravel(), Y.ravel()], dim=1)
n_nodes = len(points)

print(f"  Grid: {n_grid}×{n_grid} = {n_nodes} nodes")

# ============================================================================
# CREATE TRIANGLES BY SPLITTING SQUARES
# ============================================================================
print("\nCreating triangles from grid squares...")

elements = []

# For each square in the grid, create 2 triangles
for i in range(n_grid - 1):
    for j in range(n_grid - 1):
        # Node indices for this square
        n0 = i * n_grid + j
        n1 = i * n_grid + (j + 1)
        n2 = (i + 1) * n_grid + (j + 1)
        n3 = (i + 1) * n_grid + j
        
        # Triangle 1: n0, n1, n2 (lower-right triangle)
        elements.append([n0, n1, n2])
        
        # Triangle 2: n0, n2, n3 (upper-left triangle)
        elements.append([n0, n2, n3])

elements = torch.tensor(elements, dtype=torch.long, device=device)
n_elements = len(elements)

print(f"  Squares: {(n_grid-1)}×{(n_grid-1)} = {(n_grid-1)**2}")
print(f"  Triangles: {n_elements}")

# ============================================================================
# ASSIGN MATERIALS
# ============================================================================
print("\nAssigning materials...")

element_materials = assign_materials_circular(points, elements, R, E_b, E_i)

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
    'L': L,
    'R': R
}, 'square_sample.pt')

print("\nSaved to: square_sample.pt")

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

# Circle for visualization
theta = np.linspace(0, 2*np.pi, 100)
ax.plot(R*np.cos(theta), R*np.sin(theta), 'k-', linewidth=2, label='True boundary')

ax.set_xlim(-L/2*1.1, L/2*1.1)
ax.set_ylim(-L/2*1.1, L/2*1.1)
ax.set_aspect('equal')
ax.set_title('Structured Mesh (Grid Split)\n(Red=Inclusion, Blue=Background)', 
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
ax.plot(R*np.cos(theta), R*np.sin(theta), 'r--', linewidth=2, label='True boundary')

ax.set_xlim(-L/2*1.1, L/2*1.1)
ax.set_ylim(-L/2*1.1, L/2*1.1)
ax.set_aspect('equal')
ax.set_title(f'Structured Grid Triangulation\n({n_nodes} nodes, {n_elements} triangles)', 
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
plt.savefig('square_sample_plot.png', dpi=150, bbox_inches='tight')
print("Saved: square_sample_plot.png")

print("\n" + "="*70)
print("SUCCESS: Ultra-simple structured triangulation!")
print("="*70)
print(f"Method: Split {n_grid}×{n_grid} grid squares into 2 triangles each")
print(f"Mesh: {n_nodes} nodes, {n_elements} triangles, {n_dof} DOFs")
print(f"K matrix: {K.shape}, {torch.count_nonzero(K).item()} non-zeros")
print("="*70)