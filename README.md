# PAT-Scan

**Palpation-Assisted Tomography** — reconstructing interior material properties of tissue from boundary measurements using FEM and deep learning.

## Overview

PAT-Scan solves an inverse problem in computational mechanics: given boundary forces and measured boundary displacements on a circular tissue sample, reconstruct the interior material property distribution (elastic modulus/stiffness). It combines Finite Element Method (FEM) solvers with a U-Net neural network in a mesh-based physics-informed approach.

```
Forward problem (FEM):   forces + material properties  -->  displacements
Inverse problem (U-Net): forces + boundary displacements  -->  material properties
```

The U-Net predicts an E(x,y) material field, a differentiable FEM forward solve produces predicted displacements, and the loss is computed against measured boundary displacements. This enables end-to-end training through the FEM solver.

## Getting Started

### Prerequisites

- Python 3.10+
- PyTorch (with CUDA for GPU training)
- NumPy, SciPy, Matplotlib, scikit-image, Pillow, imageio

### Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch numpy scipy matplotlib scikit-image pillow imageio
```

### Running

Scripts must be run from their own directory (they use relative paths for mesh files, datasets, and output).

```bash
cd codefiles/working/circular_inclusion/main/

# Step 1: Generate the mesh
python create_circle_sample.py

# Step 2: Generate training dataset (angular scanning)
python angular_scanning.py

# Step 3: Train the U-Net (latest version)
python unet_train_v9.py

# Run FEM solver validation tests
python automated_tests.py
```

For the irregular inclusion variant:

```bash
cd codefiles/working/irregular_inclusion/
python create_irregular_inclusion.py
python unet_train_v9_upgraded.py
```

## Architecture

### Key Modules (`codefiles/working/circular_inclusion/main/`)

| Module | Purpose |
|--------|---------|
| `fem_utils.py` | Core FEM: element stiffness (6x6 triangular), global assembly, boundary conditions, force application, linear solve (KU=F), mesh I/O, visualization |
| `fem_utils_differentiable.py` | Batched/differentiable FEM assembly for GPU-accelerated training |
| `unet.py` | 3-level encoder-decoder with skip connections. Input: 2ch (64x64 grid), Output: 1ch (material field) |
| `unet_forward_model.py` | Connects U-Net output to FEM solver for end-to-end gradient flow |
| `unet_train_v[0-9].py` | Training script versions (v9 is latest). Loss = MSE + TV regularization |
| `create_circle_sample.py` | Structured polar grid mesh generation |
| `angular_scanning.py` | Dataset generation: sweeps force pair configurations |
| `automated_tests.py` | FEM solver validation (force magnitude sweep, angular sweep) |

### Test Cases

- **Circular inclusion** (primary): outer radius 1.0, inner radius 0.3, E_background=1.0, E_inclusion=10.0
- **Irregular inclusion**: non-circular geometry with upgraded training scripts
- **Experimental**: wavelet U-Net variants, elliptical/square/biological geometries

### Data Artifacts

- Mesh: `circle_mesh.pt`
- Datasets: `angular_scanning_dataset.pt`, `symmetric_scanning_dataset.pt`
- Checkpoints: `checkpoints/checkpoint_*.pt`
- Visualizations: animation frames (PNG) assembled into GIFs/MP4s

## Project Structure

```
codefiles/
  working/
    circular_inclusion/main/    # Primary implementation
    irregular_inclusion/        # Non-circular geometry variant
  experimental/                 # Wavelet U-Net, alternative geometries
agent_prompts/                  # Multi-phase document generation prompts
agents/                         # Generated analysis & documents
comps_stuff/                    # Comprehensive exam materials
```
