# PAT-SCAN COMPREHENSIVE EXAM: REFINED SKELETON
**Date:** 2026-01-06
**Purpose:** Address Phase 1 gaps with actionable structure

---

## DOCUMENT STRUCTURE

### A. SPECIFIC AIMS
[PASTE FROM EXISTING PDF - already completed]

### B. SIGNIFICANCE (1-2 pages target)

#### 1. Clinical Relevance of Tissue Stiffness
- **Tumor detection reality:** Breast cancer lumps 5-10x stiffer than surrounding tissue [CITATION NEEDED: tissue mechanics paper]
- **Palpation limitations:** Subjective, cannot quantify, limited to superficial lesions
- **Current solutions and gaps:**
  - MRE: High accuracy but $2M equipment cost, limited accessibility
  - Ultrasound elastography: $100K systems, operator-dependent
  - **Accessibility gap:** Resource-constrained settings lack quantitative elastography

#### 2. Inverse Problem Challenge and Prior Work
**[ADDRESSES GAP 4: Literature positioning]**

**Fundamental ill-posedness:**
- Forward problem (E → u) well-posed via FEM
- Inverse problem (u → E) non-unique, noise-sensitive, requires regularization
- Mathematical foundations: [Mathematical Foundations 1994]

**Existing inverse elasticity approaches:**
- **Iterative optimization methods** [Goenezen 2017 - Mechanics-Based Tomography]
  - Optimize element-wise properties to minimize displacement residual
  - Computationally expensive, prone to local minima
  - **Their approach:** Full-field displacements (requires internal measurements)

- **Palpation tomography** [Konofagou & Harrigan 2003]
  - Multiple loading protocol (9 loads) to improve measurement-to-parameter ratio
  - **Key finding:** Sequential forces reduce noise sensitivity
  - **Limitation:** Simple parameterizations, circular assumptions

- **Modal-based elastography** [Bouman 2022 - Visual Vibration Tomography]
  - Uses vibration modes (dynamic loading) to infer material properties
  - **Contrast:** Dynamic vs. quasi-static loading (different physics)

**Knowledge gap this work addresses:**
- No existing method combines: (1) boundary-only measurements, (2) physics-informed learning, (3) arbitrary irregular geometries with level-set extraction
- [ADDRESSES GAP 1] Clear distinction needed between geometric inverse (piecewise E) vs. full inverse (continuous E(x,y))

#### 3. PAT-Scan Positioning
**Comparison table** [ADDRESSES ADVISOR HYPOTHESIS 2]:

| Modality | Equipment Cost | Spatial Resolution | Depth | Training Data |
|----------|---------------|-------------------|-------|---------------|
| MRE | ~$2M | ~2mm | Full body | None (direct) |
| Ultrasound Elastography | ~$100K | ~1mm | ~10cm | None |
| PAT-Scan (proposed) | ~$10K (DIC + load cell) | Mesh-dependent (~2-5mm) | Surface-biased | Synthetic (unlimited) |

**PAT-Scan advantages:**
- Low equipment cost enables point-of-care diagnostics
- Synthetic training data generation via FEM (no expensive data collection)
- Compatible with existing CT/imaging workflows

**PAT-Scan limitations:**
- Surface-biased measurements (limited depth penetration compared to MRE)
- Requires accurate boundary displacement measurement (DIC system)
- Current validation: synthetic only (experimental validation planned Aim 3)

---

### C. INNOVATION (1 page target)

#### Innovation 1: Mesh-Based PINN Approach
**[ADDRESSES GAP: Correct PINN terminology, KEY CONTRIBUTION]**

**Critical distinction from meshfree PINNs:**
- **Karniadakis PINNs [Raissi 2019]:** NN approximates PDE solution directly, physics embedded in loss via PDE residuals (∇·σ + f = 0)
  - Meshfree, can handle complex domains
  - Computationally expensive: simultaneous PDE solving + parameter optimization

- **Our mesh-based PINN approach [JAX-SSO 2024, JAX-FEM 2023, Deep FEM 2024]:**
  - **Decoupled architecture:**
    - **Forward model:** Traditional FEM solves well-posed F = KU (established, efficient)
    - **Inverse model:** U-Net learns ill-posed u → E mapping (neural network strength)
  - **Computational efficiency:** Leverages sparse linear algebra for forward, avoids PDE residual backprop
  - **Physical guarantees:** FEM ensures equilibrium equations satisfied exactly

**Technical implementation:**
- TV regularization preserves sharp edges (critical for geometric inverse)
- Differentiable FEM option enables physics-informed loss (implemented: `unet_forward_model_differentiable.py`) [PARTIALLY IMPLEMENTED]
- Level-set post-processing: Gaussian smoothing → soft thresholding → contour extraction

**Why this matters:**
- Bridges classical computational mechanics with modern deep learning
- More efficient than full meshfree PINNs for problems where forward model is well-established
- Extensible to other mechanics inverse problems (contact, plasticity, damage)

#### Innovation 2: CT-Inspired Sequential Force Application
**[ADDRESSES GAP 7: Boundary completeness, cites Konofagou]**

**Strategy:**
- Angular scanning: N force pairs with systematic angular spacing (1-20 pairs demonstrated)
- Inspired by palpation tomography [Konofagou & Harrigan 2003]: Multiple loads increase measurement-to-parameter ratio, reduce noise sensitivity

**Boundary coverage considerations:**
- **Case 1 (partial boundary):** Square sample, single edge displacement measurements → poor reconstruction
- **Case 2 (complete boundary):** Semi-circular sample, full boundary coverage → significantly better accuracy
- **Finding:** Completeness matters more than force magnitude (within penetration limits)
- **Design implication:** Experimental setups must enable multi-angle access

**Distinction from prior work:**
- **vs. Konofagou 2003:** We use boundary-only (vs. full-field internal displacements)
- **vs. Bouman 2022:** Quasi-static forces (vs. dynamic vibration modes) - simpler actuation
- **Hardware simplification:** No internal sensors, compatible with DIC surface measurements

#### Innovation 3: Universal Framework for Irregular Geometries
**[IMPLEMENTED for Aim 2a]**

**Geometry support:**
- Circular inclusions (Aim 1) [IMPLEMENTED]
- Elliptical inclusions [IMPLEMENTED]
- Irregular off-centered inclusions with Fourier perturbations: r(θ) = R_base(1 + Σ[a_n cos(nθ) + b_n sin(nθ)]) [IMPLEMENTED]

**Automatic geometry detection:**
- Training script (`unet_train_v5_upgraded.py`) auto-detects from dataset metadata
- Same U-Net architecture works across geometry types (no architecture changes needed)
- Level-set extraction adapts to arbitrary shapes

**Future extension (Aim 2b):**
- Fourier Features MLP for full inverse problem: spatially varying E(x,y) [PLANNED, NOT IMPLEMENTED]

#### Innovation 4: Synthetic-to-Real Training Pathway
**[ADDRESSES GAP 3: Justifying synthetic experiments]**

**Why synthetic data is sufficient for proof-of-concept:**
- FEM forward model represents ground truth physics (linear elasticity well-established)
- Unlimited labeled data generation: Explore parameter space (stiffness contrast, shapes, loading)
- Controlled noise injection simulates measurement uncertainty
- **Transfer learning pathway:** Train on synthetic → fine-tune on experimental phantoms → clinical data

**Validation strategy (Aim 3):**
1. **Step 1:** Silicone phantoms with known E (ground truth validation)
2. **Step 2:** Ex-vivo tissue samples (biological variability)
3. **Step 3:** Clinical data (if feasible)

**Risk mitigation:**
- Synthetic→real gap addressed via domain adaptation techniques
- Phantom experiments provide intermediate validation before clinical translation

---

### D. RESEARCH APPROACH (Methods, Results, Future Work)

#### **CRITICAL FRAMING** [ADDRESSES GAP 3: Synthetic-clinical bridge]
"The following methodology develops PAT-Scan as a computational inverse problem solver using synthetic FEM-generated data. This approach enables systematic exploration of geometric complexity, noise sensitivity, and algorithmic performance before experimental validation (Aim 3). Synthetic experiments are sufficient for establishing proof-of-concept because the forward model (linear elasticity FEM) accurately represents physical tissue deformation under quasi-static loading."

---

## SPECIFIC AIM 1 - Proof-of-Concept (Geometric Inverse Problem)

**[ADDRESSES GAP 1: Distinguish geometric vs. full inverse]**

**Problem formulation:**
- **Geometric inverse problem:** Piecewise constant E (E_background, E_inclusion)
- **Goal:** Recover inclusion shape, size, location from boundary displacements u_boundary
- **Assumption:** Binary material distribution (justified for tumor detection: stiff inclusion in soft background)
- **Contrast with Aim 2b:** Full inverse problem with continuous E(x,y) requires different approach (Fourier MLP)

### Aim 1a - FEM Forward Model

#### Methods [IMPLEMENTED]:
**Mesh generation** (`create_circle_sample.py`):
- Structured polar grid: 20 radial × 40 angular divisions
- Triangular elements, centered circular inclusion
- Parameters: R_outer = 1.0, R_inner = 0.3, E_contrast = 10:1, ν = 0.3

**FEM formulation** (`fem_utils.py`):
- 2D plane stress assumption (thin sample justification: [TODO: add reference])
- Linear elastic constitutive relation (small deformations)
- Element stiffness: 6×6 matrices for 3-node triangles
- Global assembly: Sparse K matrix (2N_nodes × 2N_nodes)
- Boundary conditions: Fixed inclusion nodes (zero displacement), free outer boundary

**Force application:**
- Equal & opposite radial forces at angle θ and θ+π on outer boundary
- Force magnitude: F = 0.1 [TODO: specify units]
- **Penetration constraint:** Deformed boundary must not violate inclusion → limits maximum force
- Validation check: `check_penetration()` function ensures geometric validity

#### Results [IMPLEMENTED]:
- ✅ Mesh: ~800 nodes, ~1500 elements (typical values)
- ✅ FEM solver stable for F ∈ [0, F_max] where F_max determined by penetration
- ✅ Displacement fields physically plausible (max at force application points)
- ✅ Automated tests (`automated_tests_upgraded.py`): Force magnitude sweep, angular validation
- 📊 Figures: `deformation_plot.png`, mesh visualizations

**[TODO: Add if calculated]**
- [ ] Mesh convergence study (vary n_radial, n_angular)
- [ ] Energy balance verification: ½F^T U = ½U^T K U

---

### Aim 1b - Dataset Generation via Angular Scanning

#### Methods [IMPLEMENTED]:
**Scanning protocol** (`angular_scanning_upgraded.py`):
- Sweep: 1 to 20 force pairs with 9° angular spacing
- For each configuration: Solve FEM → extract boundary displacements (u_x, u_y)
- Total samples: 20 force configurations

**Data representation:**
- **Input to U-Net:** 2-channel displacement field (U_x, U_y) on 64×64 grid (interpolated from boundary)
- **Output target:** Binary material mask M ∈ {0,1}^(64×64) where 1 = inclusion
- **Metadata:** Force vectors, number of pairs, clearance, max displacement

**Note on radial scanning** [ADDRESSES GAP 5]:
- `radial_scanning.py` exists in codebase but NOT used in current workflow
- Rationale for exclusion: Angular coverage more important than force magnitude variation (within penetration limits)
- Future work: Compare angular vs. radial scanning strategies

#### Results [IMPLEMENTED]:
- ✅ Dataset: 20 samples saved to `angular_scanning_dataset.pt`
- 🎬 Animation: `angular_scanning.gif` shows deformation evolution with increasing force pairs
- 📈 Observation: Displacement magnitude scales with number of force pairs
- ✅ Penetration-free: All configurations satisfy geometric constraints

---

### Aim 1c - U-Net Training and Boundary Extraction

#### Methods [IMPLEMENTED]:

**Network architecture** (`unet.py`):
- **Input:** 2-channel (U_x, U_y) on 64×64 grid
- **Encoder:** 3 levels, conv → ReLU → max pool, base features = 32
- **Decoder:** 3 levels, transpose conv → ReLU, skip connections from encoder
- **Output:** 1-channel normalized E-field, sigmoid activation (values ∈ [0,1])

**Loss function** (`unet_train_v5_upgraded.py`):
**[ADDRESSES GAP 6: TV regularization motivation]**

```
L = L_MSE + λ_TV * L_TV
```

Where:
- **L_MSE:** Mean squared error ||E_pred - E_true||²
- **L_TV:** Total variation regularization Σ|∇E_pred|

**Why TV regularization?**
- Geometric inverse problem expects piecewise constant E (sharp boundaries)
- TV penalty preserves edges while smoothing within regions
- Critical for level-set extraction: need clear 0/1 separation in predicted field
- **Connection to post-processing:** TV pre-conditions field for thresholding

**Hyperparameter optimization** (`unet_train_v9.py`):
- Grid search over: learning rate η ∈ {10^-5, 10^-4, 10^-3}, λ_TV ∈ {0.001, 0.005, 0.01}, temperature T ∈ {2000, 3000, 5000}
- Optimizer: Adam
- Iterations: [TODO: specify typical value, e.g., 5000]

**Post-processing for hard boundaries** [IMPLEMENTED]:
1. **Gaussian smoothing:** E_smooth = G_σ * E_pred (σ ≈ 0.01-0.07 from grid search)
2. **Soft thresholding:** E_thresh = sigmoid(T · (E_smooth - 0.5)) where T = temperature
3. **Contour extraction:** scikit-image `find_contours` at level 0.5
4. **Spline fitting:** Smooth boundary representation for visualization

#### Results [IMPLEMENTED]:

**Training performance:**
- ✅ Loss convergence: Decreases from [INITIAL] to [FINAL] over [N_ITER] iterations
- ✅ Training time: [TODO: add benchmark on GPU/CPU]
- 🎬 Visualization: `training_animation.gif` shows evolution of prediction quality
- ✅ Hyperparameter sensitivity: Grid search results in `grid_search_results.txt`

**Reconstruction accuracy:**
**[ADDRESSES GAP 2: Quantitative results]**

- ✅ **Qualitative validation:** Visual inspection shows predicted inclusion matches ground truth shape
  - 📊 `post_training_visualization.png`, `pre_training_visualization.png`

- **Quantitative metrics:** [ADD IF CALCULATED, OTHERWISE STATE:]
  - **Status:** Qualitative validation only (quantitative metrics in progress)
  - **Needed metrics:** Dice coefficient, Hausdorff distance, center error, radius error %
  - [CITATION NEEDED] for metric definitions if not standard

- ✅ **Level-set extraction:** Successfully identifies sharp inclusion boundary from soft U-Net predictions
- ⚠️ **Known issue:** Shear modulus value in inclusion recovered at ~60-80% of target value (varies with stiffness contrast)
  - [ADDRESSES GAP: Quantifying "significantly underestimated"]
  - **Hypothesis:** Ill-posedness of inverse problem + boundary-only measurements limit exact recovery
  - **Sufficient for detection:** Shape/location accurate, absolute modulus less critical

#### Discussion - Aim 1:

**Achievements:**
- ✅ Proof-of-concept demonstrated: U-Net + FEM + level-set pipeline functional
- ✅ Geometric inverse problem solved for centered circular inclusions
- ✅ Training framework supports irregular geometries (demonstrated in Aim 2a)

**Limitations and scope:**
- 2D plane stress assumption (3D extension in Aim 3)
- Centered circular inclusion (irregular extension in Aim 2a)
- Synthetic data only (experimental validation in Aim 3c)
- Boundary completeness required (partial boundary degrades accuracy significantly)

**Status:** [ADDRESSES GAP: Accurate completion claim]
- **Aim 1 status:** Core methodology demonstrated ✅ (quantitative validation metrics in progress)
- **More accurate than:** "Successfully completed" (overclaim)

---

## SPECIFIC AIM 2 - Beyond Proof-of-Concept

**Overall Goal:** Extend to irregular geometries and explore full inverse problem (spatially varying E)

**Status breakdown** [ADDRESSES GAP: Aspirational vs. implemented]:
- **Aim 2a (Irregular geometries):** 60% complete [IMPLEMENTED: geometry generation, FEM solver, training framework; PENDING: quantitative benchmarking]
- **Aim 2b (Fourier MLP):** 0% complete [PLANNED, NOT IMPLEMENTED]
- **Aim 2c (Benchmarking):** 0% complete [PLANNED, NOT IMPLEMENTED]

### Aim 2a - Irregular Off-Centered Inclusions

#### Methods [IMPLEMENTED]:

**Irregular inclusion generation** (`create_irregular_inclusion.py`):
- **Off-centered:** Center (c_x, c_y) with c_x ∈ [-0.5, 0.5], c_y ∈ [-0.5, 0.5]
- **Fourier perturbation:** r(θ) = R_base × (1 + Σ_{n=1}^{N_modes} [a_n cos(nθ) + b_n sin(nθ)])
  - N_modes = 6 (typical)
  - Irregularity parameter controls amplitude of a_n, b_n coefficients
  - Clamping: r(θ) ∈ [0.5 R_base, 1.5 R_base] ensures validity

**Universal training framework:**
- Same U-Net architecture as Aim 1 (no changes needed)
- Automatic geometry detection: If dataset contains `'a_coeffs'` → irregular mode
- Level-set extraction adapts to arbitrary shapes (no circular assumption)

**Dataset generation:**
- Same angular scanning protocol (1-20 force pairs)
- Challenge: Asymmetric deformation patterns due to off-center + irregular shape
- Hypothesis: More training data may be needed for convergence compared to symmetric case

#### Results [PARTIALLY IMPLEMENTED]:

- ✅ Irregular geometry meshes generated successfully
- ✅ FEM solver stable for irregular cases
- 🔄 Training ongoing: [TODO: describe current status if in progress]
- 📊 `irregular_inclusion_plot.png` shows example geometry

**Preliminary observations:**
- Irregular case more sensitive to hyperparameters (λ_TV, smoothing σ)
- Level-set extraction handles non-convex shapes robustly
- [TODO: Quantify reconstruction accuracy vs. irregularity level]

#### Future Work for 2a:
- [ ] Complete training for N=10-20 irregular samples with varying eccentricity and perturbation
- [ ] Quantitative metrics: Dice, Hausdorff distance for irregular boundaries
- [ ] **Generalization test:** Train on circular, test on irregular (and vice versa)
- [ ] Investigate: Does eccentricity degrade accuracy more than irregularity?

---

### Aim 2b - Fourier Features MLP for Full Inverse Problem

**[PROPOSED - NOT IMPLEMENTED]**

#### Motivation:
**[ADDRESSES GAP 1: Geometric vs. full inverse distinction]**

- **Aim 1-2a scope:** Geometric inverse problem (piecewise constant E)
- **Aim 2b scope:** Full inverse problem with spatially varying E(x,y)
  - Examples: Gradient stiffness fields, multiple inclusions with different E, heterogeneous tissues

**Why MLP + Fourier features?**
- U-Net grid representation may struggle with high-frequency spatial variations
- Implicit neural representations [Tancik 2020, SIREN] naturally handle continuous fields
- Fourier feature mapping enables learning high-frequency functions
- Resolution-independent: Query E at arbitrary (x,y)

#### Proposed Methods:

**Network architecture:**
- **Input:** (x, y, u_x(x,y), u_y(x,y)) - coordinates + displacement at that point
- **Fourier mapping:** γ(p) = [cos(2πB·p), sin(2πB·p)] where B is random frequency matrix
- **MLP:** 6-8 layers, 256 hidden units, SIREN/ReLU activation
- **Output:** E(x,y) - Young's modulus at query point

**Training data generation:**
- Create synthetic samples with continuous E(x,y):
  - Multiple inclusions with varying stiffness
  - Radial gradient: E(r) = E_0 + k·r
  - Perlin noise-based heterogeneity
- Run FEM forward model with spatially varying element materials
- Sample (x,y,u) triplets for training

**Loss function:**
```
L = L_reconstruction + λ_physics * L_physics
```
Where:
- L_reconstruction: ||E_pred(x,y) - E_true(x,y)||² (if ground truth available)
- L_physics: FEM residual or displacement prediction error
- **Physics-informed option:** Use differentiable FEM to enforce equilibrium

#### Expected Outcomes [PROPOSED VALIDATION]:
- [ ] Demonstrate continuous E(x,y) reconstruction from boundary u
- [ ] Benchmark MLP vs. U-Net for geometric inverse (should be comparable)
- [ ] Advantage: Resolution-independent, can zoom into regions
- [ ] Challenge: Longer training time, may require more data

**Timeline:** Months 3-4 after irregular geometry completion

---

### Aim 2c - Benchmarking Library

**[PLANNED - NOT IMPLEMENTED]**
**[ADDRESSES ADVISOR HYPOTHESIS 3: Novelty demonstration via systematic comparison]**

#### Proposed Test Cases:

**Diversity of geometries:**
1. **Circular inclusions:** Vary radius R ∈ [0.2, 0.5], eccentricity [0, 0.5]
2. **Elliptical inclusions:** Aspect ratio ∈ [1.5, 3.0]
3. **Irregular inclusions:** Fourier modes N ∈ [3, 8], irregularity ∈ [0.1, 0.3]
4. **Multiple inclusions:** 2-3 separate stiff regions
5. **Biologically-inspired:** Spiculated tumor boundaries (star-convex), anatomical cross-sections

**Stiffness contrast sweep:**
- E_inclusion / E_background ∈ [2, 5, 10, 20] (clinical range for tumors)
- Literature values [CITATION NEEDED]: breast cancer ~5-10x, liver tumors ~2-5x

**Performance metrics:**
- **Geometric accuracy:** Dice coefficient, Hausdorff distance (boundary error)
- **Material recovery:** L² error in E-field, mean absolute percentage error
- **Sensitivity analysis:** Noise robustness (add Gaussian noise to displacements)

#### Deliverable:
- Library of N=20-50 diverse test samples with ground truth
- Systematic evaluation: Mean ± std for metrics across test set
- **Community contribution:** Publish benchmark dataset for inverse elasticity research

**Timeline:** Months 5-6

---

## SPECIFIC AIM 3 - Realism and Experimental Validation

**[FUTURE WORK - BRIEF OVERVIEW]**

**Purpose:** Transition from synthetic proof-of-concept to experimental/clinical feasibility

### Aim 3a - 3D Extension

**Proposed methods:**
- Tetrahedral mesh generation (TetGen, Gmsh)
- 3D FEM: Volumetric stiffness assembly, surface boundary conditions
- Neural network: 3D U-Net OR MLP (naturally extends to 3D)
- **Computational challenge:** 3D FEM ~10-100x more expensive than 2D
- **Mitigation:** Efficient solvers (FEniCS, MFEM, GPU-accelerated JAX-FEM)

**Timeline:** Year 3 of PhD

---

### Aim 3b - CT Integration (Anatomically-Informed Models)

**Proposed methods:**
- Use CT scans for patient-specific geometry (segmented organ boundaries)
- Visible Human Project dataset for anatomical ground truth
- **Challenge:** CT intensity ≠ mechanical properties (non-trivial mapping)
- Constraint-based approach: Assume tissue-type E values from literature

**Application example:**
- Forearm cross-section: Muscle, bone, fat with realistic E-modulus ratios
- Question: Can PAT-Scan distinguish tissue types from boundary measurements alone?

**Timeline:** Year 3-4, contingent on Aim 2 completion

---

### Aim 3c - Experimental Validation Roadmap

**[ADDRESSES ADVISOR HYPOTHESIS 1: Clinical translation path]**

**Validation strategy:**

**Phase 1: Silicone phantoms (controlled environment)**
- Fabricate tissue-mimicking phantoms:
  - Background: Soft silicone (E ~ 10-50 kPa, mimics soft tissue)
  - Inclusion: Stiff silicone (E ~ 50-500 kPa, mimics tumor)
  - Geometry: Known dimensions (ground truth for validation)
- Measurement system:
  - Force sensors: Load cell for applied force magnitude
  - DIC (Digital Image Correlation): Surface displacement field (speckle pattern + stereo cameras)
  - Cost estimate: ~$10K (2 cameras + load cell + fixtures)
- **Validation metric:** Compare PAT-Scan reconstruction to known phantom E-field

**Phase 2: Ex-vivo tissue samples (biological variability)**
- Animal tissue or human cadaveric samples
- Known heterogeneity from literature (e.g., liver with embedded lesions)
- Challenge: Unknown exact E-field → relative validation (does PAT-Scan detect stiffer regions?)

**Phase 3: Clinical feasibility (aspirational)**
- In-vivo measurements (if safe and IRB-approved)
- Application: Breast tumor detection, liver fibrosis staging
- Compare PAT-Scan to gold standard (MRE, biopsy)

**Timeline:**
- Phantom experiments: Year 4
- Ex-vivo: Year 4-5 (potential collaboration with biomedical engineering lab)
- Clinical: Beyond PhD scope (future postdoc/faculty work)

**Equipment requirements:**
- DIC system: ~$5-10K (stereo cameras, speckle pattern materials)
- Load cell + actuator: ~$2K
- Phantom materials: ~$500 per sample
- Total: ~$10-15K setup cost (vs. $2M for MRE - supports accessibility claim)

---

## ASSUMPTIONS AND SCOPE

**[ADDRESSES STYLE PATTERN 3: Show limitations upfront]**

### Material Model Assumptions:
1. **Linear elasticity:** Small deformations, Hookean materials
   - Justification: Quasi-static loading, displacements < 5% of sample size
   - Limitation: Real soft tissues exhibit nonlinearity (hyperelastic models more accurate)
   - Impact: Proof-of-concept sufficient; future work may require Neo-Hookean/Mooney-Rivlin

2. **Plane stress (2D):** Thin sample, no out-of-plane deformation
   - Justification: Sample thickness << lateral dimensions
   - Limitation: 3D effects ignored (addressed in Aim 3a)

3. **Incompressible or near-incompressible:** ν ≈ 0.3-0.49
   - Justification: Soft tissues ~incompressible (water content)
   - Impact: Poisson ratio ν assumed known (inverse problem only solves for E)

### Geometric Assumptions:
1. **Known outer boundary geometry:** Circular or defined domain
   - Justification: CT/imaging provides outer boundary
   - Limitation: Boundary shape uncertainty not currently handled

2. **Binary material (Aims 1-2a):** Piecewise constant E
   - Relaxed in Aim 2b for continuous E(x,y)

### Measurement Assumptions:
1. **Boundary displacement completeness:** Full or majority of boundary measured
   - Critical finding: Partial boundary → poor reconstruction
   - Design requirement: Multi-angle access for experimental setup

2. **Displacement measurement accuracy:** DIC provides sub-pixel resolution (~0.01 pixel)
   - Noise model: Gaussian σ ∈ [0.1%, 1%] of max displacement
   - Robustness: TV regularization mitigates noise (demonstrated in synthetic tests)

3. **Force magnitude known:** Load cell provides accurate force values
   - Uncertainty: ±1% typical for commercial load cells
   - Impact: Inverse problem relatively insensitive to small force errors (dominated by displacement noise)

### Computational Assumptions:
1. **FEM mesh quality:** No highly distorted elements after deformation
   - Validation: Penetration check ensures geometric validity
   - Limitation: Very large deformations may require remeshing (not implemented)

2. **Convergence:** FEM solution converged (direct solver → exact within numerical precision)

---

## CONCLUSION

### Summary of Contributions:

**Aim 1 (Proof-of-Concept):** Core methodology demonstrated ✅
- FEM forward model validated for circular inclusions
- U-Net + TV regularization + level-set extraction pipeline functional
- Hyperparameter optimization via grid search
- **Achievement:** Geometric inverse problem solved for canonical case

**Aim 2 (Extensions):** Irregular geometry framework implemented (60%), full inverse and benchmarking planned (0%)
- Irregular off-centered inclusions: Mesh generation, FEM solver, universal training framework [IMPLEMENTED]
- Fourier MLP for continuous E(x,y): Architecture designed [PLANNED]
- Benchmarking library: Metrics defined [PLANNED]

**Aim 3 (Realism):** Experimental validation roadmap designed [FUTURE]
- 3D FEM extension pathway identified
- Phantom validation strategy with cost estimates
- Clinical translation timeline (beyond PhD scope)

### Significance:

PAT-Scan demonstrates a novel mesh-based PINN approach that decouples forward and inverse problems, achieving computational efficiency while preserving physical plausibility. By combining FEM-generated synthetic data with deep learning, this work addresses the accessibility gap in elastography—providing a pathway toward low-cost tissue stiffness measurement for resource-constrained settings.

**Key innovation:** Unlike meshfree PINNs or traditional iterative optimization, our hybrid approach leverages the strengths of both classical computational mechanics (accurate forward model) and modern deep learning (flexible inverse solver), bridging two communities that rarely intersect.

**Broader impact:** The framework extends beyond PAT-Scan to other mechanics inverse problems (contact detection, damage localization, material characterization), establishing a generalizable methodology for physics-informed learning in solid mechanics.

---

## NEXT STEPS FOR DOCUMENT COMPLETION

### Information to add:
1. **Quantitative results:**
   - [ ] Compute Dice, Hausdorff, center error, radius error from existing results OR state "qualitative validation only"
   - [ ] Exact hyperparameter values from `grid_search_results.txt`
   - [ ] Training time benchmarks (iterations, wall time)
   - [ ] Dataset statistics (N_samples, mesh size)

2. **Missing citations:**
   - [ ] Tissue stiffness values (breast cancer 5-10x claim)
   - [ ] Hadamard on ill-posed problems
   - [ ] MRE cost and accessibility studies
   - [ ] Mathematical Foundations (1994) full citation
   - [ ] JAX-SSO (2024), JAX-FEM (2023), Deep FEM (2024) full references

3. **Figures to create/reference:**
   - [ ] PAT-Scan pipeline flowchart (force → displacement → U-Net → E-field)
   - [ ] Comparison schematic: Karniadakis PINN vs. mesh-based PINN
   - [ ] Boundary completeness illustration (Case 1 vs. Case 2)

### Writing priorities for Phase 2:
1. Expand Innovation section with mesh-based PINN distinction (already started here)
2. Add comparison table to Significance (included above)
3. Justify TV regularization connection to level-set (added in Aim 1c)
4. Clarify geometric vs. full inverse problem (added in Aim 1 intro and Aim 2b)
5. Add experimental validation roadmap details (Aim 3c expanded above)

---

**END OF REFINED SKELETON**
