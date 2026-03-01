"""
Circle with Irregular Off-Centered Inclusion Mesh Generation
Uses structured polar grid triangulation with asymmetric inclusion
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
R_base = 0.3     # Base radius of inclusion (average)
E_b = 1.0        # Young's modulus of background
E_i = 10.0       # Young's modulus of inclusion
nu = 0.3         # Poisson's ratio

n_radial = 20    # Number of radial divisions
n_angular = 40   # Number of angular divisions

# Irregular inclusion parameters
#center_x = 0.2   # X-offset of inclusion center
#center_y = -0.15 # Y-offset of inclusion center
center_x = -0.5
center_y = 0.2
n_modes = 6      # Number of Fourier modes for irregular shape
irregularity = 0.20  # Amount of irregularity (0-1, higher = more irregular)
seed = 42        # Random seed for reproducibility

print("="*70)
print("Circle with Off-Centered Irregular Inclusion (PyTorch)")
print("="*70)

# ============================================================================
# CREATE IRREGULAR INCLUSION BOUNDARY FUNCTION
# ============================================================================
print("\nGenerating irregular off-centered inclusion shape...")

torch.manual_seed(seed)
np.random.seed(seed)

# Generate random Fourier coefficients for irregular boundary
# r(theta) = R_base * (1 + sum[a_n * cos(n*theta) + b_n * sin(n*theta)])
a_coeffs = torch.randn(n_modes, dtype=torch.float64, device=device) * irregularity
b_coeffs = torch.randn(n_modes, dtype=torch.float64, device=device) * irregularity

def is_inside_irregular_inclusion(x, y):
    """
    Check if point (x, y) is inside the irregular off-centered inclusion
    """
    # Translate to inclusion center coordinate system
    x_local = x - center_x
    y_local = y - center_y
    
    # Convert to polar coordinates relative to inclusion center
    r = torch.sqrt(x_local**2 + y_local**2)
    theta = torch.atan2(y_local, x_local)
    
    # Compute irregular boundary radius at this angle
    r_boundary = R_base * torch.ones_like(theta)
    for n in range(1, n_modes + 1):
        r_boundary = r_boundary + R_base * (a_coeffs[n-1] * torch.cos(n * theta) + 
                                            b_coeffs[n-1] * torch.sin(n * theta))
    
    # Ensure radius stays positive and bounded
    r_boundary = torch.clamp(r_boundary, R_base * 0.5, R_base * 1.5)
    
    return r < r_boundary

# Sample the irregular boundary for visualization
theta_sample = torch.linspace(0, 2*torch.pi, 200, dtype=torch.float64, device=device)
r_sample = R_base * torch.ones_like(theta_sample)
for n in range(1, n_modes + 1):
    r_sample = r_sample + R_base * (a_coeffs[n-1] * torch.cos(n * theta_sample) + 
                                    b_coeffs[n-1] * torch.sin(n * theta_sample))
r_sample = torch.clamp(r_sample, R_base * 0.5, R_base * 1.5)

print(f"  Inclusion center: ({center_x}, {center_y})")
print(f"  Fourier modes: {n_modes}")
print(f"  Irregularity factor: {irregularity}")
print(f"  Min inclusion radius: {r_sample.min().item():.4f}")
print(f"  Max inclusion radius: {r_sample.max().item():.4f}")

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
# ASSIGN MATERIALS (OFF-CENTERED IRREGULAR INCLUSION)
# ============================================================================
print("\nAssigning materials to off-centered irregular inclusion...")

def assign_materials_irregular_offset(points, elements, inclusion_check_func, E_background, E_inclusion):
    """
    Assign materials based on irregular off-centered inclusion boundary
    """
    element_materials = torch.zeros(len(elements), dtype=torch.float64, device=points.device)
    
    for e, elem in enumerate(elements):
        # Get element centroid
        elem_coords = points[elem]
        centroid = elem_coords.mean(dim=0)
        
        # Check if centroid is inside irregular inclusion
        if inclusion_check_func(centroid[0], centroid[1]):
            element_materials[e] = E_inclusion
        else:
            element_materials[e] = E_background
    
    return element_materials

element_materials = assign_materials_irregular_offset(points, elements, is_inside_irregular_inclusion, E_b, E_i)

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
    'R_base': R_base,
    'center_x': center_x,
    'center_y': center_y,
    'a_coeffs': a_coeffs.cpu(),
    'b_coeffs': b_coeffs.cpu(),
    'n_modes': n_modes,
    'irregularity': irregularity,
    'seed': seed
}, 'irregular_inclusion_sample.pt')

print("\nSaved to: irregular_inclusion_sample.pt")

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\nCreating visualization...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Convert to numpy for plotting
points_np = points.cpu().numpy()
elements_np = elements.cpu().numpy()
materials_np = element_materials.cpu().numpy()
theta_plot = theta_sample.cpu().numpy()
r_plot = r_sample.cpu().numpy()

# Compute irregular boundary in Cartesian coordinates for plotting
x_boundary = center_x + r_plot * np.cos(theta_plot)
y_boundary = center_y + r_plot * np.sin(theta_plot)

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
ax.set_title('Off-Centered Irregular Inclusion\n(Red=Inclusion, Blue=Background)', 
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
plt.savefig('irregular_inclusion_plot.png', dpi=150, bbox_inches='tight')
print("Saved: irregular_inclusion_plot.png")

print("\n" + "="*70)
print("SUCCESS: Off-centered irregular inclusion mesh!")
print("="*70)
print(f"Domain: Circle with R={R_outer}")
print(f"Inclusion: Irregular shape centered at ({center_x}, {center_y})")
print(f"           Base R={R_base}, {n_modes} Fourier modes")
print(f"Materials: E_background={E_b}, E_inclusion={E_i}, nu={nu}")
print(f"Mesh: {n_nodes} nodes, {n_elements} triangles, {n_dof} DOFs")
print(f"K matrix: {K.shape}, {torch.count_nonzero(K).item()} non-zeros")
print("="*70)