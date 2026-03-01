"""
3-Component Biological Sample: Tissue with Bone and Blood Vessel
Circular domain with one circular inclusion (bone) and one irregular inclusion (blood vessel)
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
R_outer = 1.0    # Radius of outer circle (tissue sample)

# Bone parameters (circular inclusion)
bone_center_x = -0.35
bone_center_y = 0.0
bone_radius = 0.20

# Blood vessel parameters (irregular inclusion)
vessel_center_x = 0.35
vessel_center_y = 0.10
vessel_R_base = 0.18
vessel_n_modes = 6
vessel_irregularity = 0.25
vessel_seed = 42

# Material properties
E_tissue = 1.0      # Young's modulus of tissue (background)
E_bone = 20.0       # Young's modulus of bone (stiff)
E_vessel = 0.5      # Young's modulus of blood vessel (soft)
nu = 0.3            # Poisson's ratio (same for all)

# Mesh parameters
n_radial = 25       # Number of radial divisions
n_angular = 50      # Number of angular divisions

print("="*70)
print("3-Component Biological Sample: Tissue + Bone + Blood Vessel")
print("="*70)

# ============================================================================
# CREATE IRREGULAR BLOOD VESSEL BOUNDARY
# ============================================================================
print("\nGenerating irregular blood vessel shape...")

torch.manual_seed(vessel_seed)
np.random.seed(vessel_seed)

# Generate random Fourier coefficients for irregular vessel boundary
a_coeffs = torch.randn(vessel_n_modes, dtype=torch.float64, device=device) * vessel_irregularity
b_coeffs = torch.randn(vessel_n_modes, dtype=torch.float64, device=device) * vessel_irregularity

def is_inside_vessel(x, y):
    """Check if point (x, y) is inside the irregular blood vessel"""
    x_local = x - vessel_center_x
    y_local = y - vessel_center_y
    
    r = torch.sqrt(x_local**2 + y_local**2)
    theta = torch.atan2(y_local, x_local)
    
    r_boundary = vessel_R_base * torch.ones_like(theta)
    for n in range(1, vessel_n_modes + 1):
        r_boundary = r_boundary + vessel_R_base * (a_coeffs[n-1] * torch.cos(n * theta) + 
                                                    b_coeffs[n-1] * torch.sin(n * theta))
    
    r_boundary = torch.clamp(r_boundary, vessel_R_base * 0.5, vessel_R_base * 1.5)
    
    return r < r_boundary

def is_inside_bone(x, y):
    """Check if point (x, y) is inside the circular bone"""
    x_local = x - bone_center_x
    y_local = y - bone_center_y
    r = torch.sqrt(x_local**2 + y_local**2)
    return r < bone_radius

# Sample vessel boundary for visualization
theta_sample = torch.linspace(0, 2*torch.pi, 200, dtype=torch.float64, device=device)
vessel_r_sample = vessel_R_base * torch.ones_like(theta_sample)
for n in range(1, vessel_n_modes + 1):
    vessel_r_sample = vessel_r_sample + vessel_R_base * (a_coeffs[n-1] * torch.cos(n * theta_sample) + 
                                                         b_coeffs[n-1] * torch.sin(n * theta_sample))
vessel_r_sample = torch.clamp(vessel_r_sample, vessel_R_base * 0.5, vessel_R_base * 1.5)

print(f"  Bone: circular, center=({bone_center_x}, {bone_center_y}), R={bone_radius}")
print(f"  Vessel: irregular, center=({vessel_center_x}, {vessel_center_y}), base R={vessel_R_base}")
print(f"  Vessel modes: {vessel_n_modes}, irregularity: {vessel_irregularity}")

# ============================================================================
# CREATE STRUCTURED POLAR GRID
# ============================================================================
print("\nCreating polar grid...")

r = torch.linspace(0, R_outer, n_radial, dtype=torch.float64, device=device)
theta = torch.linspace(0, 2*torch.pi, n_angular+1, dtype=torch.float64, device=device)[:-1]

R, Theta = torch.meshgrid(r, theta, indexing='ij')

X = R * torch.cos(Theta)
Y = R * torch.sin(Theta)

points = torch.stack([X.ravel(), Y.ravel()], dim=1)

# Remove duplicate center points
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
n_elements = len(elements)

print(f"  Triangles: {n_elements}")

# ============================================================================
# ASSIGN MATERIALS (3 COMPONENTS)
# ============================================================================
print("\nAssigning materials to 3-component system...")

element_materials = torch.zeros(len(elements), dtype=torch.float64, device=points.device)

for e, elem in enumerate(elements):
    # Get element centroid
    elem_coords = points[elem]
    centroid = elem_coords.mean(dim=0)
    
    # Check which region the centroid is in (priority: bone > vessel > tissue)
    if is_inside_bone(centroid[0], centroid[1]):
        element_materials[e] = E_bone
    elif is_inside_vessel(centroid[0], centroid[1]):
        element_materials[e] = E_vessel
    else:
        element_materials[e] = E_tissue

n_tissue = torch.sum(element_materials == E_tissue)
n_bone = torch.sum(element_materials == E_bone)
n_vessel = torch.sum(element_materials == E_vessel)

print(f"  Tissue elements: {n_tissue}")
print(f"  Bone elements: {n_bone}")
print(f"  Vessel elements: {n_vessel}")

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
    'E_tissue': E_tissue,
    'E_bone': E_bone,
    'E_vessel': E_vessel,
    'nu': nu,
    'R_outer': R_outer,
    'bone_center': (bone_center_x, bone_center_y),
    'bone_radius': bone_radius,
    'vessel_center': (vessel_center_x, vessel_center_y),
    'vessel_R_base': vessel_R_base,
    'vessel_a_coeffs': a_coeffs.cpu(),
    'vessel_b_coeffs': b_coeffs.cpu(),
    'vessel_n_modes': vessel_n_modes,
    'vessel_irregularity': vessel_irregularity,
    'vessel_seed': vessel_seed
}, 'biological_sample_3component.pt')

print("\nSaved to: biological_sample_3component.pt")

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
vessel_r_plot = vessel_r_sample.cpu().numpy()

# Vessel boundary in Cartesian coordinates
vessel_x_boundary = vessel_center_x + vessel_r_plot * np.cos(theta_plot)
vessel_y_boundary = vessel_center_y + vessel_r_plot * np.sin(theta_plot)

# Plot 1: Mesh with material distribution
ax = axes[0]
for e, elem in enumerate(elements_np):
    elem_coords = points_np[elem]
    if materials_np[e] == E_bone:
        color = 'darkred'
    elif materials_np[e] == E_vessel:
        color = 'red'
    else:
        color = 'lightblue'
    tri = plt.Polygon(elem_coords, facecolor=color, edgecolor='black', 
                      alpha=0.6, linewidth=0.3)
    ax.add_patch(tri)

# Draw boundaries
theta_circle = np.linspace(0, 2*np.pi, 100)
ax.plot(R_outer*np.cos(theta_circle), R_outer*np.sin(theta_circle), 'k-', 
        linewidth=2, label='Tissue boundary')
ax.plot(bone_center_x + bone_radius*np.cos(theta_circle), 
        bone_center_y + bone_radius*np.sin(theta_circle), 'k--', 
        linewidth=2, label='Bone')
ax.plot(vessel_x_boundary, vessel_y_boundary, 'k:', 
        linewidth=2, label='Blood vessel')
ax.plot(bone_center_x, bone_center_y, 'ro', markersize=8)
ax.plot(vessel_center_x, vessel_center_y, 'ro', markersize=8)

ax.set_xlim(-R_outer*1.1, R_outer*1.1)
ax.set_ylim(-R_outer*1.1, R_outer*1.1)
ax.set_aspect('equal')
ax.set_title('3-Component Biological Sample\n(Dark Red=Bone, Red=Vessel, Blue=Tissue)', 
             fontsize=12, fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 2: Material properties visualization
ax = axes[1]
for e, elem in enumerate(elements_np):
    elem_coords = points_np[elem]
    E = materials_np[e]
    # Color by stiffness
    if E == E_bone:
        color = 'darkred'
    elif E == E_vessel:
        color = 'blue'
    else:
        color = 'green'
    tri = plt.Polygon(elem_coords, facecolor=color, edgecolor='black', 
                      alpha=0.6, linewidth=0.3)
    ax.add_patch(tri)

ax.plot(R_outer*np.cos(theta_circle), R_outer*np.sin(theta_circle), 'k-', linewidth=2)
ax.set_xlim(-R_outer*1.1, R_outer*1.1)
ax.set_ylim(-R_outer*1.1, R_outer*1.1)
ax.set_aspect('equal')
ax.set_title(f'Material Stiffness Distribution\nBone: E={E_bone}, Tissue: E={E_tissue}, Vessel: E={E_vessel}', 
             fontsize=12, fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('y')

# Add legend with color patches
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='darkred', alpha=0.6, label=f'Bone (E={E_bone})'),
                   Patch(facecolor='green', alpha=0.6, label=f'Tissue (E={E_tissue})'),
                   Patch(facecolor='blue', alpha=0.6, label=f'Vessel (E={E_vessel})')]
ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
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
plt.savefig('biological_sample_3component_plot.png', dpi=150, bbox_inches='tight')
print("Saved: biological_sample_3component_plot.png")

print("\n" + "="*70)
print("SUCCESS: 3-component biological sample mesh!")
print("="*70)
print(f"Domain: Circular tissue sample, R={R_outer}")
print(f"Bone: Circular inclusion at ({bone_center_x}, {bone_center_y}), R={bone_radius}")
print(f"Vessel: Irregular inclusion at ({vessel_center_x}, {vessel_center_y}), base R={vessel_R_base}")
print(f"Materials: E_tissue={E_tissue}, E_bone={E_bone}, E_vessel={E_vessel}, nu={nu}")
print(f"Mesh: {n_nodes} nodes, {n_elements} triangles, {n_dof} DOFs")
print(f"Distribution: {n_tissue} tissue, {n_bone} bone, {n_vessel} vessel elements")
print(f"K matrix: {K.shape}, {torch.count_nonzero(K).item()} non-zeros")
print("="*70)