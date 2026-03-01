"""
FEM Utilities for Mesh Creation, Loading, and Analysis
Shared functions for mesh creation, force application, solving, and visualization
"""

import torch
import numpy as np
import matplotlib.pyplot as plt


# ============================================================================
# SETUP AND CONFIGURATION
# ============================================================================

def get_device():
    """Get default device (CUDA if available, else CPU)"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup_torch_defaults():
    """Setup default torch configuration"""
    torch.set_default_dtype(torch.float64)


# ============================================================================
# ELEMENT STIFFNESS AND ASSEMBLY (Common to mesh creation)
# ============================================================================

def element_stiffness(coords, E, nu):
    """
    Compute 6x6 element stiffness matrix for a triangular element
    
    Args:
        coords: (3, 2) tensor of node coordinates
        E: Young's modulus
        nu: Poisson's ratio
    
    Returns:
        K_e: 6x6 element stiffness matrix
    """
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]
    
    # Calculate element area
    area = 0.5 * torch.abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
    area = torch.clamp(area, min=1e-10)
    
    # Shape function derivatives
    b1 = y2 - y3
    b2 = y3 - y1
    b3 = y1 - y2
    c1 = x3 - x2
    c2 = x1 - x3
    c3 = x2 - x1
    
    # Strain-displacement matrix B
    B = (1.0 / (2.0 * area)) * torch.tensor([
        [b1, 0,  b2, 0,  b3, 0 ],
        [0,  c1, 0,  c2, 0,  c3],
        [c1, b1, c2, b2, c3, b3]
    ], dtype=torch.float64, device=coords.device)
    
    # Plane stress constitutive matrix D
    D = (E / (1 - nu**2)) * torch.tensor([
        [1,  nu, 0              ],
        [nu, 1,  0              ],
        [0,  0,  (1 - nu) / 2.0 ]
    ], dtype=torch.float64, device=coords.device)
    
    # Element stiffness: K_e = area * B^T * D * B
    K_e = area * (B.T @ D @ B)
    return K_e


def assemble_stiffness(points, elements, element_materials, nu):
    """
    Assemble global stiffness matrix
    
    Args:
        points: (n_nodes, 2) tensor of node coordinates
        elements: (n_elements, 3) tensor of element connectivity
        element_materials: (n_elements,) tensor of Young's modulus per element
        nu: Poisson's ratio
    
    Returns:
        K: (n_dof, n_dof) global stiffness matrix
    """
    n_dof = 2 * len(points)
    K = torch.zeros((n_dof, n_dof), dtype=torch.float64, device=points.device)
    
    for e in range(len(elements)):
        node_ids = elements[e]
        coords = points[node_ids]
        E = element_materials[e]
        
        # Compute element stiffness
        K_e = element_stiffness(coords, E, nu)
        
        # Global DOF indices for this element
        dof_ids = torch.tensor([
            2*node_ids[0], 2*node_ids[0]+1,
            2*node_ids[1], 2*node_ids[1]+1,
            2*node_ids[2], 2*node_ids[2]+1
        ], dtype=torch.long, device=points.device)
        
        # Assemble into global matrix
        for i in range(6):
            for j in range(6):
                K[dof_ids[i], dof_ids[j]] += K_e[i, j]
    
    return K


def assign_materials_circular(points, elements, R_inclusion, E_background, E_inclusion):
    """
    Assign material properties based on circular inclusion
    
    Args:
        points: Node coordinates
        elements: Element connectivity
        R_inclusion: Radius of circular inclusion
        E_background: Young's modulus of background material
        E_inclusion: Young's modulus of inclusion material
    
    Returns:
        element_materials: Young's modulus for each element
    """
    n_elements = len(elements)
    element_materials = torch.zeros(n_elements, dtype=torch.float64, device=points.device)
    
    for i in range(n_elements):
        # Get centroid of triangle
        elem = elements[i]
        centroid = torch.mean(points[elem], dim=0)
        
        # Check if inside circle
        dist_to_center = torch.sqrt(centroid[0]**2 + centroid[1]**2)
        
        if dist_to_center <= R_inclusion:
            element_materials[i] = E_inclusion
        else:
            element_materials[i] = E_background
    
    return element_materials


def print_mesh_info(n_nodes, n_elements, n_dof, K):
    """Print mesh and stiffness matrix information"""
    print(f"\nMesh Information:")
    print(f"  Nodes: {n_nodes}")
    print(f"  Elements: {n_elements}")
    print(f"  DOFs: {n_dof}")
    print(f"\nStiffness Matrix:")
    print(f"  Shape: {K.shape}")
    print(f"  Symmetric: {torch.allclose(K, K.T, atol=1e-10)}")
    print(f"  Non-zeros: {torch.count_nonzero(K)}")
    print(f"  Sparsity: {torch.count_nonzero(K).item() / K.numel() * 100:.2f}%")


def visualize_mesh(points, elements, element_materials, E_inclusion, 
                   boundary_geometry, title, filename):
    """
    Create visualization of mesh with material distribution
    
    Args:
        points: Node coordinates
        elements: Element connectivity
        element_materials: Material properties per element
        E_inclusion: Young's modulus of inclusion (for coloring)
        boundary_geometry: Dict with geometry info for plotting boundaries
        title: Plot title
        filename: Output filename
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Convert to numpy for plotting
    points_np = points.cpu().numpy()
    elements_np = elements.cpu().numpy()
    materials_np = element_materials.cpu().numpy()
    
    # Plot 1: Mesh with material distribution
    ax = axes[0]
    for e, elem in enumerate(elements_np):
        elem_coords = points_np[elem]
        color = 'red' if materials_np[e] == E_inclusion else 'blue'
        tri = plt.Polygon(elem_coords, facecolor=color, edgecolor='black', 
                          alpha=0.5, linewidth=0.3)
        ax.add_patch(tri)
    
    # Draw boundaries
    _add_boundary_to_plot(ax, boundary_geometry)
    
    ax.set_aspect('equal')
    ax.set_title(f'{title}\n(Red=Inclusion, Blue=Background)', 
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
    _add_boundary_to_plot(ax, boundary_geometry, detail=True)
    
    ax.set_aspect('equal')
    n_nodes = len(points)
    n_elements = len(elements)
    ax.set_title(f'Triangulation Detail\n({n_nodes} nodes, {n_elements} triangles)', 
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Stiffness matrix pattern
    # This will be added by the caller if K is provided
    ax = axes[2]
    ax.set_title('Stiffness Matrix\n(Call visualize_stiffness_matrix separately)', 
                 fontsize=12, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def visualize_stiffness_matrix(K, n_dof, filename):
    """
    Visualize stiffness matrix structure
    
    Args:
        K: Stiffness matrix
        n_dof: Number of DOFs
        filename: Output filename
    """
    K_np = K.cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Use log scale to see structure better
    K_np_abs = np.abs(K_np)
    K_np_log = np.log10(K_np_abs + 1e-16)
    
    # Create visualization
    im = ax.imshow(K_np_log, cmap='viridis', aspect='auto', interpolation='nearest')
    ax.set_title(f'Stiffness Matrix (log scale)\n({n_dof} DOFs, {np.count_nonzero(K_np)} non-zeros)', 
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Column Index')
    ax.set_ylabel('Row Index')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('log₁₀|K|', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def _add_boundary_to_plot(ax, boundary_geometry, detail=False):
    """Helper function to add boundary lines to plot"""
    geom_type = boundary_geometry['type']
    
    if geom_type == 'circle':
        R_outer = boundary_geometry['R_outer']
        R_inner = boundary_geometry['R_inner']
        theta = np.linspace(0, 2*np.pi, 100)
        
        if detail:
            ax.plot(R_outer*np.cos(theta), R_outer*np.sin(theta), 'k-', 
                   linewidth=2, label='Outer boundary')
            ax.plot(R_inner*np.cos(theta), R_inner*np.sin(theta), 'r--', 
                   linewidth=2, label='Inclusion')
        else:
            ax.plot(R_outer*np.cos(theta), R_outer*np.sin(theta), 'k-', 
                   linewidth=2, label='Outer boundary')
            ax.plot(R_inner*np.cos(theta), R_inner*np.sin(theta), 'k--', 
                   linewidth=2, label='Inclusion boundary')
        
        ax.set_xlim(-R_outer*1.1, R_outer*1.1)
        ax.set_ylim(-R_outer*1.1, R_outer*1.1)
        
    elif geom_type == 'square':
        L = boundary_geometry['L']
        R = boundary_geometry['R']
        theta = np.linspace(0, 2*np.pi, 100)
        
        if detail:
            ax.plot(R*np.cos(theta), R*np.sin(theta), 'r--', 
                   linewidth=2, label='True boundary')
        else:
            ax.plot(R*np.cos(theta), R*np.sin(theta), 'k-', 
                   linewidth=2, label='True boundary')
        
        ax.set_xlim(-L/2*1.1, L/2*1.1)
        ax.set_ylim(-L/2*1.1, L/2*1.1)


# ============================================================================
# MESH LOADING (Original functions)
# ============================================================================

def load_mesh(filename='circle_sample.pt', device=None):
    """Load mesh data from file"""
    if device is None:
        device = get_device()
    
    data = torch.load(filename)
    
    mesh_data = {
        'points': data['points'].to(device),
        'elements': data['elements'].to(device),
        'K': data['K'].to(device),
        'element_materials': data['element_materials'].to(device),
        'E_b': data['E_b'],
        'E_i': data['E_i'],
        'nu': data['nu'],
        'device': device
    }
    
    # Add geometry-specific parameters if they exist
    if 'R_outer' in data:
        mesh_data['R_outer'] = data['R_outer']
        mesh_data['R_inner'] = data['R_inner']
    if 'L' in data:
        mesh_data['L'] = data['L']
        mesh_data['R'] = data['R']
    
    return mesh_data


# ============================================================================
# BOUNDARY AND FORCE FUNCTIONS (Original functions)
# ============================================================================

def find_boundary_nodes(points, R_outer, tolerance=1e-6):
    """Find nodes on the outer boundary"""
    radii = torch.sqrt(points[:, 0]**2 + points[:, 1]**2)
    boundary_mask = torch.abs(radii - R_outer) < tolerance
    boundary_nodes = torch.where(boundary_mask)[0]
    return boundary_nodes, radii


def find_node_at_angle(points, boundary_nodes, target_angle):
    """
    Find boundary node closest to target angle (in radians)
    
    Args:
        points: Node coordinates tensor
        boundary_nodes: Indices of boundary nodes
        target_angle: Target angle in radians
    
    Returns:
        Node index closest to target angle
    """
    best_node = None
    best_dist = float('inf')
    
    for node_idx in boundary_nodes:
        x, y = points[node_idx]
        node_angle = torch.atan2(y, x)
        target_normalized = torch.atan2(torch.sin(target_angle), torch.cos(target_angle))
        
        angle_diff = torch.abs(node_angle - target_normalized)
        angle_diff = torch.min(angle_diff, 2*torch.pi - angle_diff)
        
        if angle_diff < best_dist:
            best_dist = angle_diff
            best_node = node_idx
    
    return best_node


def apply_single_force_pair(points, boundary_nodes, force_magnitude, force_angle_deg, device):
    """
    Apply a single force pair at specified angle
    
    Returns:
        F: Force vector
        node1, node2: Force node indices
    """
    n_nodes = len(points)
    n_dof = 2 * n_nodes
    F = torch.zeros(n_dof, dtype=torch.float64, device=device)
    
    # Convert angle to radians
    theta_force = torch.tensor(force_angle_deg * torch.pi / 180.0, device=device)
    
    # Find nodes
    node1 = find_node_at_angle(points, boundary_nodes, theta_force)
    node2 = find_node_at_angle(points, boundary_nodes, theta_force + torch.pi)
    
    # Apply inward forces
    pos1 = points[node1]
    r1 = torch.sqrt(pos1[0]**2 + pos1[1]**2)
    force1_dir = -pos1 / r1
    
    F[2*node1] = force_magnitude * force1_dir[0]
    F[2*node1 + 1] = force_magnitude * force1_dir[1]
    
    pos2 = points[node2]
    r2 = torch.sqrt(pos2[0]**2 + pos2[1]**2)
    force2_dir = -pos2 / r2
    
    F[2*node2] = force_magnitude * force2_dir[0]
    F[2*node2 + 1] = force_magnitude * force2_dir[1]
    
    return F, node1, node2


def apply_multiple_force_pairs(points, boundary_nodes, force_magnitude, initial_angle, 
                               angular_spacing, n_force_pairs, device):
    """
    Apply multiple force pairs
    
    Returns:
        F: Force vector
        force_nodes_list: List of (node1, node2) tuples
    """
    n_nodes = len(points)
    n_dof = 2 * n_nodes
    F = torch.zeros(n_dof, dtype=torch.float64, device=device)
    
    force_nodes_list = []
    
    for i in range(n_force_pairs):
        angle_deg = initial_angle + i * angular_spacing
        angle_rad = torch.tensor(angle_deg * torch.pi / 180.0, device=device)
        
        node1 = find_node_at_angle(points, boundary_nodes, angle_rad)
        node2 = find_node_at_angle(points, boundary_nodes, angle_rad + torch.pi)
        
        # Force 1: radial INWARD
        pos1 = points[node1]
        r1 = torch.sqrt(pos1[0]**2 + pos1[1]**2)
        force1_dir = -pos1 / r1
        
        F[2*node1] += force_magnitude * force1_dir[0]
        F[2*node1 + 1] += force_magnitude * force1_dir[1]
        
        # Force 2: radial INWARD
        pos2 = points[node2]
        r2 = torch.sqrt(pos2[0]**2 + pos2[1]**2)
        force2_dir = -pos2 / r2
        
        F[2*node2] += force_magnitude * force2_dir[0]
        F[2*node2 + 1] += force_magnitude * force2_dir[1]
        
        force_nodes_list.append((node1, node2))
    
    return F, force_nodes_list


def setup_boundary_conditions(radii, R_inner, n_dof, device, tolerance=1e-6):
    """
    Setup boundary conditions: fix all inclusion nodes
    
    Returns:
        fixed_dofs: Indices of fixed DOFs
        free_dofs: Indices of free DOFs
    """
    inclusion_mask = radii <= (R_inner + tolerance)
    inclusion_nodes = torch.where(inclusion_mask)[0]
    
    fixed_dofs = []
    for node in inclusion_nodes:
        fixed_dofs.extend([2*node.item(), 2*node.item() + 1])
    
    fixed_dofs = torch.tensor(fixed_dofs, dtype=torch.long, device=device)
    
    all_dofs = torch.arange(n_dof, device=device)
    free_mask = torch.ones(n_dof, dtype=torch.bool, device=device)
    free_mask[fixed_dofs] = False
    free_dofs = all_dofs[free_mask]
    
    return fixed_dofs, free_dofs


def solve_fem(K, F, fixed_dofs, free_dofs, n_dof, device):
    """
    Solve FEM system KU = F
    
    Returns:
        U: Full displacement vector
        U_nodes: Displacement reshaped as (n_nodes, 2)
    """
    K_reduced = K[free_dofs][:, free_dofs]
    F_reduced = F[free_dofs]
    
    U_reduced = torch.linalg.solve(K_reduced, F_reduced)
    
    U = torch.zeros(n_dof, dtype=torch.float64, device=device)
    U[free_dofs] = U_reduced
    U[fixed_dofs] = 0.0
    
    n_nodes = n_dof // 2
    U_nodes = U.reshape(n_nodes, 2)
    
    return U, U_nodes


def check_penetration(points, U_nodes, boundary_nodes, R_inner):
    """
    Check if boundary penetrates inclusion
    
    Returns:
        is_valid: Boolean indicating if configuration is valid
        penetration_depth: Depth of penetration (negative if clearance)
        min_deformed_radius: Minimum radius after deformation
        min_radius_node: Node with minimum radius
    """
    boundary_nodes_np = boundary_nodes.cpu().numpy()
    U_nodes_np = U_nodes.cpu().numpy()
    points_np = points.cpu().numpy()
    
    boundary_positions = points_np[boundary_nodes_np]
    boundary_displacements = U_nodes_np[boundary_nodes_np]
    deformed_positions = boundary_positions + boundary_displacements
    deformed_radii = np.sqrt(deformed_positions[:, 0]**2 + deformed_positions[:, 1]**2)
    
    min_deformed_radius = deformed_radii.min()
    min_radius_idx = np.argmin(deformed_radii)
    min_radius_node = boundary_nodes_np[min_radius_idx]
    
    penetration_depth = R_inner - min_deformed_radius
    is_valid = penetration_depth <= 0
    
    return is_valid, penetration_depth, min_deformed_radius, min_radius_node


# ============================================================================
# PLOTTING FUNCTIONS (Original functions - keeping for backward compatibility)
# ============================================================================

def plot_deformation_single_pair(points, U_nodes, boundary_nodes, F, node1, node2,
                                force_angle, force_magnitude, R_outer, R_inner,
                                clearance, filename='deformation_plot.png'):
    """Create deformation plot for single force pair"""
    boundary_nodes_np = boundary_nodes.cpu().numpy()
    U_nodes_np = U_nodes.cpu().numpy()
    points_np = points.cpu().numpy()
    
    boundary_positions = points_np[boundary_nodes_np]
    boundary_displacements = U_nodes_np[boundary_nodes_np]
    disp_magnitudes = np.sqrt(boundary_displacements[:, 0]**2 + boundary_displacements[:, 1]**2)
    
    max_mag_idx = np.argmax(disp_magnitudes)
    max_mag_node = boundary_nodes_np[max_mag_idx]
    
    node1_np = node1.cpu().item()
    node2_np = node2.cpu().item()
    
    disp_at_node1 = np.sqrt(U_nodes_np[node1_np, 0]**2 + U_nodes_np[node1_np, 1]**2)
    disp_at_node2 = np.sqrt(U_nodes_np[node2_np, 0]**2 + U_nodes_np[node2_np, 1]**2)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    # Reference circles
    theta_plot = np.linspace(0, 2*np.pi, 200)
    ax.plot(R_outer*np.cos(theta_plot), R_outer*np.sin(theta_plot), 'k-', 
            linewidth=1.5, alpha=0.3, label='Outer boundary (reference)', zorder=1)
    ax.plot(R_inner*np.cos(theta_plot), R_inner*np.sin(theta_plot), 'r-', 
            linewidth=2, alpha=0.5, label='Inclusion (fixed)', zorder=2)
    
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
    
    # Displacement vectors
    for i in range(len(boundary_nodes_np)):
        if disp_magnitudes[i] > 1e-10:
            x0, y0 = boundary_positions[i]
            dx, dy = boundary_displacements[i]
            ax.arrow(x0, y0, dx, dy,
                    head_width=0.02, head_length=0.015, 
                    fc='black', ec='black', alpha=0.6, linewidth=1, zorder=5)
    
    # Force nodes
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
                 f'Inclusion fixed | Max disp = {disp_magnitudes.max():.3e} | ✓ VALID', 
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


def plot_deformation_multiple_pairs(points, U_nodes, boundary_nodes, F, force_nodes_list,
                                   n_force_pairs, angular_spacing, force_magnitude, 
                                   R_outer, R_inner, clearance, filename='deformation_plot.png'):
    """Create deformation plot for multiple force pairs"""
    boundary_nodes_np = boundary_nodes.cpu().numpy()
    U_nodes_np = U_nodes.cpu().numpy()
    points_np = points.cpu().numpy()
    
    boundary_positions = points_np[boundary_nodes_np]
    boundary_displacements = U_nodes_np[boundary_nodes_np]
    disp_magnitudes = np.sqrt(boundary_displacements[:, 0]**2 + boundary_displacements[:, 1]**2)
    
    max_mag_idx = np.argmax(disp_magnitudes)
    max_mag_node = boundary_nodes_np[max_mag_idx]
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    # Reference circles
    theta_plot = np.linspace(0, 2*np.pi, 200)
    ax.plot(R_outer*np.cos(theta_plot), R_outer*np.sin(theta_plot), 'k-', 
            linewidth=1.5, alpha=0.3, label='Outer boundary (reference)', zorder=1)
    ax.plot(R_inner*np.cos(theta_plot), R_inner*np.sin(theta_plot), 'r-', 
            linewidth=2, alpha=0.5, label='Inclusion (fixed)', zorder=2)
    
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
    
    # Displacement vectors
    for i in range(len(boundary_nodes_np)):
        if disp_magnitudes[i] > 1e-10:
            x0, y0 = boundary_positions[i]
            dx, dy = boundary_displacements[i]
            ax.arrow(x0, y0, dx, dy,
                    head_width=0.02, head_length=0.015, 
                    fc='black', ec='black', alpha=0.6, linewidth=1, zorder=5)
    
    # Force nodes and vectors (different colors)
    force_colors = ['red', 'orange', 'purple', 'brown', 'pink', 'olive', 'cyan', 'magenta']
    F_np = F.cpu().numpy()
    force_scale = 3.0
    
    all_force_nodes = []
    for i, (node1, node2) in enumerate(force_nodes_list):
        node1_np = node1.cpu().item()
        node2_np = node2.cpu().item()
        all_force_nodes.extend([node1_np, node2_np])
        
        pos1_np = points_np[node1_np]
        pos2_np = points_np[node2_np]
        
        color = force_colors[i % len(force_colors)]
        
        # Force nodes
        ax.scatter([pos1_np[0], pos2_np[0]], [pos1_np[1], pos2_np[1]], 
                   c=color, s=300, zorder=6, 
                   edgecolors='darkred', linewidths=3, marker='o',
                   label=f'Force pair {i+1}' if i == 0 else '')
        
        # Force vectors
        force1_np = np.array([F_np[2*node1_np], F_np[2*node1_np+1]])
        force2_np = np.array([F_np[2*node2_np], F_np[2*node2_np+1]])
        
        ax.arrow(pos1_np[0], pos1_np[1], force_scale*force1_np[0], force_scale*force1_np[1],
                 head_width=0.15, head_length=0.12, fc=color, ec='darkred', 
                 linewidth=3, zorder=7)
        ax.arrow(pos2_np[0], pos2_np[1], force_scale*force2_np[0], force_scale*force2_np[1],
                 head_width=0.15, head_length=0.12, fc=color, ec='darkred', 
                 linewidth=3, zorder=7)
    
    # Highlight max displacement node if not a force node
    if max_mag_node not in all_force_nodes:
        ax.scatter([boundary_positions[max_mag_idx, 0]], [boundary_positions[max_mag_idx, 1]], 
                   c='lime', s=300, zorder=8, label='Max displacement node', 
                   edgecolors='darkgreen', linewidths=3, marker='*')
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.set_title(f'FEM Diagnostic: {n_force_pairs} Force Pairs (spacing={angular_spacing}°)\n' + 
                 f'Inclusion fixed | Max disp = {disp_magnitudes.max():.3e} | ✓ VALID', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Info box
    info_text = f'Force pairs: {n_force_pairs}\n'
    info_text += f'Angular spacing: {angular_spacing}°\n'
    info_text += f'Force magnitude: {force_magnitude}\n'
    info_text += f'Max displacement: {disp_magnitudes.max():.3e}\n'
    info_text += f'Max disp node: {max_mag_node}\n'
    info_text += f'\nClearance: {clearance:.4f} ✓'
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()


def plot_displacement_magnitude(points, U_nodes, R_outer, R_inner, title_suffix='', 
                                filename='displacement_magnitude.png'):
    """Create displacement magnitude plot"""
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
    
    # Boundaries
    theta_plot = np.linspace(0, 2*np.pi, 200)
    ax.plot(R_outer*np.cos(theta_plot), R_outer*np.sin(theta_plot), 'k-', 
            linewidth=2, label='Outer boundary')
    ax.plot(R_inner*np.cos(theta_plot), R_inner*np.sin(theta_plot), 'b-', 
            linewidth=2, label='Inclusion boundary')
    
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