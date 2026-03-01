# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PAT-Scan (Palpation-Assisted Tomography) is a computational mechanics research project that solves an **inverse problem**: given boundary forces and measured boundary displacements on a circular tissue sample, reconstruct the interior material property distribution (elastic modulus/stiffness). It combines FEM solvers with a U-Net deep learning architecture in a mesh-based PINN approach.

## Running Code

All scripts are standalone Python files run from their respective directories. There is no build system, package manager, or test framework — scripts are executed directly.

```bash
# Activate the virtual environment first
source venv/bin/activate

# Run scripts from their directory (they use relative imports/paths)
cd codefiles/working/circular_inclusion/main/
python create_circle_sample.py      # Generate mesh (must run first)
python angular_scanning.py          # Generate training dataset
python unet_train_v9.py             # Train U-Net (latest version)
python automated_tests.py           # Run FEM solver tests

cd ../../irregular_inclusion/
python create_irregular_inclusion.py
python unet_train_v9_upgraded.py
python automated_tests_upgraded.py
```

Scripts must be run from their own directory because they use relative paths for mesh files (`circle_mesh.pt`), datasets (`angular_scanning_dataset.pt`), checkpoints, and frame output directories.

## Architecture

### Pipeline: Forward and Inverse Problems

```
Forward problem (FEM):  forces + material properties → displacements
Inverse problem (U-Net): forces + boundary displacements → material properties
```

The U-Net learns material properties by: predicting an E(x,y) field → running a differentiable FEM forward solve → comparing predicted displacements to measured ones. End-to-end training through the FEM solver.

### Key Source Modules (`codefiles/working/circular_inclusion/main/`)

- **`fem_utils.py`** — Core FEM: element stiffness matrices (6x6 triangular), global assembly, boundary conditions, force application, linear solve (KU=F), mesh I/O, visualization. Uses `torch.float64` throughout.
- **`fem_utils_differentiable.py`** — Batched/differentiable version of FEM assembly for GPU-accelerated training. `batched_element_stiffness()` operates on all elements simultaneously.
- **`unet.py`** — 3-level encoder-decoder with skip connections. Input: 2 channels (forces + boundary displacements on a 64x64 grid). Output: 1 channel (material property field, sigmoid-bounded).
- **`unet_forward_model.py` / `_differentiable.py`** — Connects U-Net output to FEM solver for end-to-end gradient flow.
- **`unet_train_v[0-9].py`** — Training script versions. v9 is latest. Each version explores different loss functions, grid search spaces, or architectural choices. Loss = MSE + TV regularization.
- **`create_circle_sample.py`** — Generates structured polar grid mesh, saves as `circle_mesh.pt`.
- **`angular_scanning.py`** — Sweeps force pair counts (1–20 pairs), runs forward FEM for each, saves dataset as `angular_scanning_dataset.pt`.
- **`automated_tests.py`** — Force magnitude sweep (until penetration) and angular sweep validation tests.

### Irregular Inclusion Variant (`codefiles/working/irregular_inclusion/`)

Mirrors the circular inclusion structure but with non-circular geometry. Files suffixed `_upgraded` adapt the circular versions. Has its own `fem_utils.py` and `fem_utils_differentiable.py` copies.

### Experimental (`codefiles/experimental/`)

Wavelet-based U-Net variants, alternative mesh geometries (ellipse, square, biological 3-component), mesh convergence studies.

## Conventions

- All FEM computation uses `torch.float64` — set via `torch.set_default_dtype(torch.float64)`.
- Plane stress assumption (2D), triangular elements with 6 DOFs per element.
- Default geometry: outer radius 1.0, inner inclusion radius 0.3, background E=1.0, inclusion E=10.0, Poisson's ratio ν=0.3.
- Grid search over hyperparameters: learning rate, TV regularization weight (`lambda_tv`), smoothing sigma, temperature, BC sharpness.
- Training scripts import everything from `fem_utils` and `unet` via wildcard (`from fem_utils import *`).
- Data artifacts saved as `.pt` (PyTorch tensors). Animation frames saved as PNGs then assembled into GIFs/MP4s.
- Output directories (`test_results/`, `angular_scanning_frames/`, `checkpoints/`, etc.) are created by scripts via `os.makedirs(..., exist_ok=True)`.

## Agent System (`agent_prompts/`)

A multi-phase document generation workflow for comprehensive exam preparation. Phase 1 agents analyze the codebase (3 thoroughness levels), Phase 2 agents generate documents, and an AI Hound agent humanizes the output. Outputs go to `agents/phase1/` and `agents/phase2/`.
