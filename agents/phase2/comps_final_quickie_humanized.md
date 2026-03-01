# PALPATION-ASSISTED TOMOGRAPHY: PHYSICS-INFORMED NEURAL NETWORKS FOR INVERSE RECONSTRUCTION OF TISSUE STIFFNESS

**Vivek Karmarkar**
Comprehensive Exam Report
Date: January 6, 2026

---

**Examining Committee:**
- [Advisor Name] - [Department] (Chair)
- Suresh Raghavan - Biomedical Engineering
- [Committee Member 3] - [Department]
- [Committee Member 4] - [Department]

---

## A. Specific Aims

[PASTE COMPLETE SPECIFIC AIMS FROM PDF HERE]

**Summary of Aims:**

**Aim 1 - Proof-of-Concept:** Develop and validate a U-Net-based inverse solver for geometric reconstruction of circular inclusions from boundary displacement measurements. This aim establishes the core methodology combining finite element forward modeling with neural network inverse solvers.

**Aim 2 - Extensions:** Extend the framework to handle irregular, off-centered inclusions and explore advanced neural architectures for the full inverse problem with spatially varying material properties.

**Aim 3 - Realism:** Incorporate 3D modeling, CT integration, and experimental validation using tissue-mimicking phantoms to demonstrate clinical feasibility.

---

## B. Significance

### The Clinical Challenge of Tissue Stiffness Measurement

Physicians have used palpation--feeling for abnormalities by touch--to detect disease for centuries. A doctor pressing on a patient's abdomen, searching for a hard nodule in soft tissue, relies on the same mechanical intuition that guides breast self-examinations worldwide. The approach works because tumors are often remarkably stiffer than healthy tissue: breast cancer lumps exhibit stiffness values 5-10 times greater than surrounding tissue. But manual palpation hits fundamental limits. It's subjective, can't produce actual stiffness numbers, and works only for lesions close enough to the surface that fingers can reach them.

Modern elastography techniques attempt to address these limitations by generating quantitative stiffness maps. Magnetic Resonance Elastography offers excellent accuracy, but the equipment runs approximately $2 million--restricting access to major medical centers with substantial capital budgets. Ultrasound elastography brings the price down to around $100,000, making it more accessible, though the technique remains operator-dependent and difficult to standardize across clinical sites. This cost barrier matters most in resource-constrained settings, exactly where quantitative tissue characterization could make the biggest difference for diagnostic outcomes.

### The Inverse Problem Challenge

Tissue stiffness reconstruction is, at its mathematical heart, an inverse problem in continuum mechanics. The forward direction is well-understood: given material properties and applied forces, computing displacement fields through finite element methods has been validated extensively over decades. The inverse direction--inferring what materials are present from observed displacements--is fundamentally ill-posed. Solutions aren't unique, and small measurement errors can produce wildly different reconstructions.

Traditional approaches split into two camps. Iterative optimization methods, like those developed by Goenezen and colleagues for mechanics-based tomography, adjust element-wise material properties to minimize discrepancies between measured and predicted displacements. These methods are rigorous but computationally demanding and prone to getting stuck in local minima. They also typically need full-field displacement data from inside the sample, requiring specialized interior imaging.

The key insight for this work came from Konofagou and Harrigan's 2003 palpation tomography paper. They showed that multiple loading configurations can dramatically improve the measurement-to-parameter ratio, making reconstructions less sensitive to noise. Applying nine distinct force patterns yielded significantly better results than single-force measurements. Their approach, though, was limited to simple geometric parameterizations and struggled with the irregular inclusion shapes that characterize actual tumors.

More recently, Katie Bouman's Visual Vibration Tomography demonstrated that material properties can be inferred from observed motion patterns using vibration modes as the interrogation mechanism. This dynamic elastography approach extracts sub-pixel motion from video, avoiding direct contact forces entirely. Our work takes a different path: quasi-static loading, which simplifies both the actuation hardware and the forward model physics while still capturing the essential stiffness contrast information.

What knowledge gap does PAT-Scan fill? It sits at the intersection of three requirements that no existing method satisfies simultaneously: boundary-only displacement measurements (compatible with surface imaging like Digital Image Correlation), physics-informed learning that maintains mechanical plausibility, and handling arbitrary irregular geometries without restrictive shape assumptions.

### Positioning PAT-Scan Against Existing Modalities

Where does PAT-Scan fit among established elastography techniques? The tradeoffs involve cost, resolution, depth, and data requirements:

| Modality | Equipment Cost | Spatial Resolution | Depth Penetration | Training Data Needs |
|----------|---------------|-------------------|-------------------|---------------------|
| MRE | ~$2M | ~2mm | Full body | None (direct measurement) |
| Ultrasound Elastography | ~$100K | ~1mm | ~10cm | None |
| PAT-Scan (proposed) | ~$10K | Mesh-dependent (~2-5mm) | Surface-biased | Synthetic (unlimited) |

The primary advantage is equipment accessibility. A complete measurement system needs only a Digital Image Correlation setup (stereo cameras with speckle pattern capability) and calibrated force sensors--equipment found in many mechanical engineering laboratories already. Total hardware cost around $10,000 represents a 200-fold reduction compared to MRE, potentially enabling point-of-care diagnostics where current elastography is simply too expensive.

The approach does carry limitations worth acknowledging. Measurements concentrate at the surface where forces are applied, limiting depth penetration compared to MRE's full-body capability. PAT-Scan demands accurate boundary displacement measurement, requiring careful DIC calibration. And critically, current validation exists only on synthetic finite element data--experimental validation with physical phantoms remains an essential next step (Aim 3).

Yet the synthetic data pathway itself offers a real advantage. Unlike clinical imaging modalities requiring expensive data collection campaigns, PAT-Scan's training data can be generated indefinitely through FEM simulation. This enables systematic exploration of stiffness contrasts, geometric complexity, and noise characteristics without patient recruitment or imaging time.

### Potential Impact

Successful development of PAT-Scan would contribute on two fronts. Scientifically, the work establishes methodology for inverse problems in computational mechanics by demonstrating that mesh-based physics-informed neural networks can efficiently decouple forward and inverse problem solving. This framework extends beyond tissue mechanics to contact detection, damage localization, and material characterization problems across solid mechanics.

From a healthcare perspective, PAT-Scan provides a pathway toward accessible quantitative tissue stiffness measurement, particularly relevant for resource-constrained settings. Compatibility with existing CT and imaging workflows suggests potential integration into standard diagnostic pipelines without building entirely new infrastructure.

---

## C. Innovation

### Innovation 1: Mesh-Based Physics-Informed Neural Networks

The most significant innovation here lies in the hybrid computational architecture bridging classical finite element methods with deep learning. This differs fundamentally from both traditional inverse problem solvers and the recently popularized meshfree physics-informed neural networks.

Meshfree PINNs, as developed by Karniadakis and colleagues, embed PDEs directly into the neural network training loss through automatic differentiation. The network learns to approximate the solution field while satisfying governing equations--equilibrium, constitutive relations, boundary conditions--at collocation points throughout the domain. This meshfree approach handles complex geometries elegantly and solves forward problems without traditional discretization.

But meshfree PINNs run into computational challenges on inverse problems. The network must simultaneously solve the forward PDE and identify unknown material parameters--a coupled optimization that can be expensive and prone to convergence difficulties. Each training iteration requires computing PDE residuals through automatic differentiation, adding substantial overhead.

Our mesh-based approach follows a different philosophy: decouple the forward and inverse problems, letting each computational method do what it does best. The forward model uses traditional finite element analysis for the well-posed linear elasticity problem F = KU. Forces F are applied, the stiffness matrix K assembled from known element properties, displacements U computed via direct sparse solver. This FEM component represents established computational mechanics, runs efficiently, and guarantees exact satisfaction of equilibrium within numerical precision.

The inverse model employs a U-Net neural network to learn the ill-posed mapping from measured boundary displacements to material property fields. The network processes displacement data as a 2-channel image (x and y displacement) and outputs a material property map. Training proceeds without explicit PDE residuals in the loss function. Instead, physics is encoded implicitly through FEM-generated training data and explicitly through Total Variation regularization that preserves sharp material interfaces.

Why does this decoupled architecture work well? Computational efficiency improves dramatically because FEM forward solves leverage sparse linear algebra rather than backpropagating through PDE residuals. Physical guarantees are stronger because the forward model exactly satisfies equilibrium, whereas meshfree PINNs satisfy PDEs only approximately at collocation points. Modularity allows updating either component independently--improved FEM solvers or advanced neural architectures can be substituted without restructuring the entire framework.

Recent work corroborates this hybrid approach. JAX-SSO (2024) showed that differentiable FEM combined with neural networks enables efficient structural optimization. JAX-FEM (2023) developed GPU-accelerated differentiable solvers for inverse design and mechanistic data science. Deep FEM (2024) explicitly integrates traditional finite element discretization with physics-informed neural networks for improved accuracy.

Our implementation includes an option for fully differentiable physics-informed training through the `unet_forward_model_differentiable.py` module, which assembles the FEM stiffness matrix from predicted material properties and computes displacement residuals. This enables incorporating physics directly into the loss function when desired while maintaining the option for purely data-driven training.

### Innovation 2: CT-Inspired Sequential Force Application

The measurement strategy draws inspiration from Computed Tomography's principle of interrogating samples from multiple angles to reconstruct internal structure. Rather than applying a single force and attempting reconstruction, we systematically apply force pairs at varying angular positions around the boundary, collecting displacement data after each loading configuration.

This builds directly on the multiple loading protocol from Konofagou and Harrigan's palpation tomography work--their demonstration that nine distinct loads improve the measurement-to-parameter ratio and reduce noise sensitivity. Our angular scanning strategy extends this by sweeping through 1 to 20 force pair configurations with systematic angular spacing.

A critical finding emerged during development: boundary displacement completeness matters more than force magnitude variation. Comparing partial boundary coverage (square sample with single edge measurements) versus complete boundary coverage (semi-circular sample with full boundary access), reconstruction accuracy improved dramatically with complete coverage--even at lower force magnitudes. This observation has important practical implications: measurement systems must enable multi-angle access to the sample boundary, potentially requiring specialized fixtures or rotating stages.

The sequential loading approach distinguishes PAT-Scan from prior elastography work in several ways. Compared to Konofagou's method, we use boundary-only measurements rather than full-field internal displacements, simplifying hardware requirements to surface imaging. Compared to Bouman's Visual Vibration Tomography, we employ quasi-static forces rather than dynamic vibration modes, eliminating high-speed video and modal analysis requirements. The quasi-static regime also simplifies the forward model to linear elasticity without dynamic terms.

Practically speaking, force application requires only simple actuation--calibrated pushers or compressed air jets directed radially inward at the boundary. This hardware is substantially simpler than vibration exciters or ultrasound transducers.

### Innovation 3: Universal Framework for Irregular Geometries

Geometric irregularity represents a fundamental challenge for inverse problems in tissue mechanics. Real tumors have spiculated, non-convex boundaries that defy simple parametric descriptions. The framework developed here handles arbitrary inclusion shapes without geometry-specific algorithm modifications.

Mesh generation supports three geometric classes. Circular inclusions serve as the canonical test case for proof-of-concept (Aim 1). Elliptical inclusions introduce eccentricity and orientation parameters. Irregular inclusions employ Fourier mode decomposition: the boundary radius varies with angle theta according to r(theta) = R_base * (1 + sum[a_n*cos(n*theta) + b_n*sin(n*theta)]), where N_modes typically equals 6 and coefficients are randomly sampled within bounds. Inclusions can be positioned off-center, creating asymmetric deformation patterns.

The training framework automatically detects geometry type from dataset metadata. If Fourier coefficients are present, irregular geometry mode activates; if only radius is specified, circular mode is assumed. Remarkably, the same U-Net architecture functions across all geometry types without modification. The network learns to recognize geometric features directly from displacement field patterns, adapting its internal representations to handle circular symmetry or irregular complexity as the data demands.

Level-set post-processing extracts crisp boundaries from soft neural network predictions. The U-Net outputs continuous material property fields with values between 0 and 1. Gaussian smoothing suppresses high-frequency noise, soft thresholding via sigmoid sharpens material transitions, and contour extraction identifies the 0.5 level set as the interface. This pipeline works for arbitrary topologies--convex or non-convex, simply connected or multiply connected--because level-set methods make no geometric assumptions.

### Innovation 4: Synthetic-to-Real Training Pathway

A common criticism of machine learning in mechanics is the requirement for large labeled datasets. Clinical tissue measurements are expensive to collect and difficult to annotate with ground truth material properties. PAT-Scan sidesteps this by using the finite element forward model as a synthetic data generator.

The FEM solver accepts arbitrary geometric descriptions and material distributions, solves for displacement fields under specified loading, and outputs perfectly labeled training pairs: displacement input, material property target. This enables unlimited data generation with controlled parameter sweeps across stiffness contrasts, inclusion sizes, shapes, and loading configurations. Noise can be injected to simulate measurement uncertainty, and datasets can be expanded systematically until training converges.

This synthetic training approach is defensible for proof-of-concept because the forward model--linear elasticity via FEM--accurately represents the underlying physics of quasi-static tissue deformation. The constitutive relations (Hooke's law for small deformations) and equilibrium equations are well-established and experimentally validated. Synthetic FEM data therefore captures the physical relationships the neural network must learn.

The pathway from synthetic to real follows staged validation. First, training and testing occur entirely on FEM-generated data to establish that the inverse solver works in principle. Second, silicone tissue-mimicking phantoms with known stiffness values provide intermediate validation--controlled laboratory conditions but with real measurement noise from DIC and force sensors. Third, ex-vivo tissue samples introduce biological variability while maintaining some ground truth through mechanical testing of excised specimens. Fourth (and this is speculative), clinical measurements on patients could provide ultimate validation, though that lies beyond the current PhD scope.

Transfer learning bridges the synthetic-to-real gap. A network pre-trained on millions of synthetic examples develops internal representations of displacement-to-stiffness mappings. Fine-tuning on a small dataset of real measurements adapts the network to measurement-specific noise characteristics and geometric variations not captured in simulation. This data-efficient adaptation strategy has proven successful in computer vision and shows promise for mechanics inverse problems.

---

## D. Research Approach

### Methodological Framing

PAT-Scan development proceeds as a computational inverse problem solver using FEM-generated synthetic data. This approach enables systematic exploration of geometric complexity, noise sensitivity, and algorithmic performance before experimental validation. Synthetic experiments suffice for proof-of-concept because the forward model--linear elasticity FEM--accurately represents physical tissue deformation under quasi-static loading, and the training objective is to learn this physics-based mapping from displacement observations to material property distributions.

---

## Specific Aim 1: Proof-of-Concept for Geometric Inverse Problem

### Problem Formulation

Aim 1 tackles the geometric inverse problem: recovering the shape, size, and location of a stiff inclusion embedded in soft background material from boundary displacement measurements. The material distribution is piecewise constant--two distinct Young's modulus values (E_background and E_inclusion) separated by a sharp interface. This binary material assumption fits tumor detection scenarios where a stiff nodule sits within compliant surrounding tissue.

This differs from the full inverse problem (Aim 2b) where material properties vary continuously as E(x,y). The geometric inverse problem is more tractable because it reduces the unknown from a continuous field to finite geometric parameters describing the inclusion boundary. Still, recovering this boundary from boundary-only displacements remains challenging given the ill-posed nature of the problem.

### Aim 1a: FEM Forward Model Development

Finite element modeling provides the foundation for both training data generation and physics validation. Mesh generation follows a structured polar grid strategy, dividing the circular domain into radial and angular segments (typically 20 radial layers and 40 angular divisions). This structured approach ensures consistent element quality and simplifies material assignment based on radial distance.

The domain consists of an outer circle with radius R_outer = 1.0 containing a concentric circular inclusion of radius R_inner = 0.3. Material properties are E_background = 1.0 and E_inclusion = 10.0, representing a 10:1 stiffness contrast typical of soft tissue-tumor systems. Poisson's ratio is set to nu = 0.3. All nodes within the inclusion boundary are fixed at zero displacement, mimicking a rigid core.

The FEM formulation employs plane stress assumption, appropriate for thin samples where out-of-plane stresses are negligible. Each triangular element contributes a 6x6 stiffness matrix relating nodal displacements to nodal forces, computed from element geometry and material properties. These element matrices assemble into a global sparse stiffness matrix K of dimension 2N_nodes x 2N_nodes (each node has x and y displacement degrees of freedom).

Force application follows a paired strategy: equal and opposite radial forces at boundary nodes located at angle theta and theta + pi. This balanced loading prevents rigid body motion while creating distinct deformation patterns. Force magnitude is constrained by geometric validity--the deformed boundary must not penetrate the fixed inclusion. A penetration checking function evaluates this constraint and determines maximum allowable force for each configuration.

The FEM solution proceeds through standard direct sparse linear algebra. Boundary conditions modify rows of K and force vector F to enforce zero displacement at fixed nodes. The resulting linear system KU = F is solved using sparse Cholesky decomposition. Solution time runs on the order of seconds for typical mesh sizes (800 nodes, 1500 elements), making forward solves computationally inexpensive.

Validation confirmed expected behaviors. Displacement fields are largest at force application points and decay with distance, matching physical intuition. The stiff inclusion experiences minimal deformation while soft background deforms substantially. Automated testing scripts sweep force magnitudes and angular configurations, verifying solution stability and penetration constraint satisfaction across the operating range.

### Aim 1b: Dataset Generation via Angular Scanning

The angular scanning protocol systematically explores force configurations for training data. Starting from a single force pair (at 0 degrees and 180 degrees), the number of pairs increases incrementally to 20, with angular spacing of 9 degrees between successive pairs. For each configuration, the FEM solver computes equilibrium displacements, and values at boundary nodes are extracted.

Data representation for neural network training requires converting irregular boundary node positions to a regular grid. Displacements are interpolated onto a 64x64 pixel grid covering the domain, creating two-channel images (x and y displacement). The ground truth target is a binary material mask on the same grid--pixels labeled 1 inside the inclusion, 0 outside.

This scanning generates 20 training samples per geometry, each representing distinct interrogation of the sample. The dataset saves with metadata including force vectors, force pair counts, clearance distances, and maximum displacements. This metadata enables analyzing how reconstruction accuracy depends on force configuration richness.

Animating the scanning process reveals how deformation patterns evolve. With a single force pair, displacement localizes near force application points. As more pairs are added, the deformation pattern becomes more uniform around the boundary, providing richer information about the inclusion's influence. This progressive information accumulation mirrors CT's principle of improving reconstruction through multiple projection angles.

### Aim 1c: U-Net Training and Boundary Extraction

The neural network follows the U-Net design originally developed for biomedical image segmentation--well-suited for geometric inverse problems. The network accepts 2-channel input (displacement fields) and produces 1-channel output (normalized material property field). The encoder pathway has three levels of convolution-ReLU-pooling operations, progressively downsampling while increasing feature channels (base 32 features, doubling at each level). The decoder upsamples through transposed convolutions while incorporating skip connections from corresponding encoder levels, combining high-level semantic information with fine-grained spatial details.

The training loss combines mean squared error with Total Variation regularization:

L = ||E_pred - E_true||^2 + lambda_TV * sum|grad(E_pred)|

The MSE term ensures predicted material fields match ground truth. The TV term penalizes spatial gradients--crucial for geometric inverse problems because the expected solution is piecewise constant with sharp boundaries. Total Variation preserves edges while smoothing within regions, pre-conditioning predictions for level-set extraction.

Why does TV regularization matter so much for post-processing? Without it, neural networks produce soft, blurred transitions between materials. MSE loss alone might achieve low error, but the resulting field lacks the sharp 0/1 separation needed for clean boundary extraction. TV regularization guides the network toward piecewise constant solutions, making subsequent thresholding robust.

Hyperparameter optimization proceeded through grid search over learning rate (10^-5, 10^-4, 10^-3), TV weight lambda_TV (0.001, 0.005, 0.01), and thresholding temperature T (2000, 3000, 5000). Training used Adam optimizer for 5000 iterations per configuration, with best hyperparameters selected based on reconstruction accuracy on held-out validation geometries. Results indicated learning rate 10^-4, TV weight 0.005, and temperature 3000 provided the most consistent performance across stiffness contrasts.

Post-processing transforms soft predictions into hard geometric boundaries through three steps. First, Gaussian smoothing with sigma around 0.03 suppresses high-frequency noise. Second, soft thresholding via sigmoid (E_thresh = sigmoid(T * (E_smooth - 0.5))) sharpens transitions, with temperature T controlling steepness. Third, contour extraction identifies the 0.5 level set, yielding polygon or spline representation of the inclusion boundary.

Training convergence was monitored through loss curves and visual inspection. Combined loss typically decreased from initial values around 0.1 to final values near 0.01 over 5000 iterations--an order of magnitude improvement. Animation of training evolution shows the network initially predicting uniform fields, then gradually developing circular features that sharpen and localize to match ground truth.

Reconstruction accuracy was evaluated qualitatively through visual comparison of predicted and true boundaries. Level-set extraction successfully identified sharp inclusion boundaries from soft U-Net outputs. Predicted shapes closely matched ground truth circles, with center location errors typically under 5% of inclusion radius.

Quantitative metrics--Dice coefficient for overlap, Hausdorff distance for maximum boundary error, radius estimation error--would provide more rigorous validation. These remain to be computed from existing results. Current status is qualitative validation demonstrating proof-of-concept, with quantitative benchmarking as ongoing work.

One observation deserves mention: recovered stiffness values within inclusions tend to be 60-80% of target values rather than exact matches. This underestimation likely reflects fundamental ill-posedness combined with boundary-only measurements. Displacement data constrain inclusion geometry strongly but provide weaker constraints on absolute stiffness magnitude. For tumor detection, though, accurate shape and location may matter more clinically than exact stiffness quantification.

### Discussion: Aim 1 Achievements and Limitations

Aim 1 demonstrated proof-of-concept for the geometric inverse problem with centered circular inclusions. The integrated pipeline--FEM forward model, angular scanning, U-Net training with TV regularization, level-set boundary extraction--functions as designed and produces physically plausible reconstructions.

What was achieved: the mesh-based PINN framework architecture, validation that boundary-only measurements contain sufficient information for inclusion reconstruction, demonstration that TV regularization works well for geometric inverse problems, and hyperparameter optimization through systematic grid search.

What limitations emerged: the 2D plane stress assumption limits applicability to thin samples (3D extension in Aim 3); restriction to centered circular inclusions is artificial (irregular geometry in Aim 2a); validation occurs entirely on synthetic data (experimental validation in Aim 3c). And critically, boundary displacement completeness emerged as a hard requirement--partial boundary coverage significantly degrades reconstruction accuracy, imposing design constraints on experimental systems.

Current status is best characterized as "core methodology demonstrated" rather than "fully completed." The algorithmic framework is functional and validated on canonical test cases. Quantitative performance metrics and broader geometric validation remain ongoing.

---

## Specific Aim 2: Extension to Irregular Geometries and Advanced Architectures

### Overall Goal and Status

Aim 2 extends PAT-Scan beyond proof-of-concept to handle realistic geometric complexity and explore alternative neural architectures. The aim subdivides into three components with different completion status:

**Aim 2a** (Irregular off-centered inclusions): 60% complete. Geometry generation, FEM solver adaptation, and universal training framework are implemented and functional. Quantitative benchmarking and systematic geometric variation studies remain pending.

**Aim 2b** (Fourier Features MLP for continuous material fields): 0% complete. Planned but not yet implemented.

**Aim 2c** (Benchmarking library): 0% complete. Planned but not implemented.

### Aim 2a: Irregular Off-Centered Inclusions

Real tumors have irregular boundaries that challenge reconstruction algorithms. Aim 2a introduces geometric complexity through Fourier mode perturbations applied to a circular base shape. Boundary radius varies with angle: r(theta) = R_base * (1 + sum[a_n*cos(n*theta) + b_n*sin(n*theta)]), where mode count N_modes typically equals 6 and Fourier coefficients are randomly sampled. An irregularity parameter controls perturbation amplitude, and clamping ensures r(theta) stays between 0.5 and 1.5 times base radius to prevent self-intersection or extreme aspect ratios.

The inclusion center can also be displaced from domain center by offsets c_x and c_y, ranging from -0.5 to 0.5 in normalized coordinates. This off-centering creates asymmetric deformation patterns, providing a more realistic robustness test.

Mesh generation for irregular geometries required generalizing material assignment logic. Rather than checking if points fall within a circle (simple radial threshold), the algorithm transforms to local coordinates centered on the inclusion, computes angular position theta, evaluates the Fourier series for boundary radius at that angle, and compares point distance to this angle-dependent threshold.

The FEM solver required no modifications--the same linear elasticity formulation applies regardless of inclusion shape. Force application and boundary condition enforcement proceed identically to circular cases. This solver generality underscores an advantage of FEM-based forward modeling: geometric complexity is absorbed into mesh generation while the solver remains unchanged.

The training framework incorporates automatic geometry detection by inspecting dataset metadata. If fields named 'a_coeffs' and 'b_coeffs' (Fourier coefficients) are present, irregular geometry mode activates. Otherwise, circular mode is assumed. Remarkably, the same U-Net architecture trained on circular cases also functions on irregular cases without modification. The network extracts geometric features from displacement patterns rather than relying on geometric assumptions.

Preliminary training on irregular geometries shows promise but reveals increased sensitivity to hyperparameters compared to circular cases. Optimal TV weight and smoothing parameters differ, suggesting irregular boundaries require more careful regularization tuning. Level-set extraction handles non-convex shapes successfully, demonstrating that post-processing generalizes beyond simple circular topology.

Future work for Aim 2a includes systematic training across 10-20 irregular samples with varying eccentricity and Fourier mode amplitudes, computing quantitative accuracy metrics (Dice coefficient, Hausdorff distance), and investigating generalization: does training on circular geometries transfer to irregular ones, and vice versa? The hypothesis is that training on diverse geometries produces a more robust inverse solver than circular cases alone.

### Aim 2b: Fourier Features MLP for Full Inverse Problem (Planned)

Aims 1 and 2a address geometric inverse problems where material properties are piecewise constant. Aim 2b extends to the full inverse problem with spatially varying Young's modulus E(x,y). Continuous fields present challenges for grid-based U-Net representations, which may struggle to capture high-frequency spatial variations.

Implicit neural representations offer an alternative. The network maps continuous coordinates (x,y) to material property E(x,y), enabling resolution-independent querying. Fourier feature mapping--embedding input coordinates as gamma(p) = [cos(2*pi*B*p), sin(2*pi*B*p)] where B is a random frequency matrix--allows MLPs to learn high-frequency functions that standard ReLU networks find difficult.

The proposed architecture accepts four inputs: x-coordinate, y-coordinate, x-displacement u_x(x,y), and y-displacement u_y(x,y) at the query location. Fourier feature embedding expands these to high-dimensional frequency space, then a 6-8 layer MLP with 256 hidden units maps to single output: Young's modulus E(x,y).

Training data generation requires synthetic samples with continuous E(x,y) distributions--multiple inclusions with different stiffness, radial gradient fields, Perlin noise heterogeneity. For each sample, the FEM solver assigns spatially varying element properties, solves for displacements, and samples (x,y,u) triplets.

The loss function could incorporate both data-driven and physics-informed components:

L = ||E_pred(x,y) - E_true(x,y)||^2 + lambda_physics * (displacement prediction error)

The physics term would use differentiable FEM to assemble K from predicted E-field, solve for predicted displacements, and penalize deviation from measurements. This ensures predicted material fields are not only close to ground truth but consistent with observed mechanical behavior.

Expected outcomes include demonstrating continuous E(x,y) reconstruction from boundary displacements, benchmarking MLP versus U-Net on geometric inverse problems as controlled comparison, and quantifying resolution independence advantages of implicit representations. Anticipated challenges include longer training times and potential need for larger datasets compared to geometric inverse cases.

### Aim 2c: Benchmarking Library (Planned)

Systematic validation requires diverse test cases spanning geometric and material complexity. The proposed library would include 20-50 samples: circular inclusions with varying radius and eccentricity, elliptical inclusions with aspect ratios 1.5 to 3.0, irregular Fourier-perturbed inclusions with varying mode counts and amplitudes, multiple-inclusion configurations (2-3 separate stiff regions), and biologically-inspired geometries like spiculated tumor boundaries.

Stiffness contrast would be systematically varied across clinically relevant ranges: E_inclusion/E_background in {2, 5, 10, 20}, corresponding to literature values for various pathologies. Breast cancer shows approximately 5-10x stiffness increase; liver tumors may show 2-5x.

Performance metrics would assess geometric accuracy (Dice coefficient, Hausdorff distance) and material recovery accuracy (L2 error in E-field, mean absolute percentage error). Noise robustness testing would add Gaussian noise to synthetic displacements at levels corresponding to DIC uncertainty (0.1-1% of maximum displacement).

The benchmark dataset would be published for community use, enabling validation of alternative inverse elastography methods on standardized test cases. Statistical analysis would report mean and standard deviation for each metric, providing confidence intervals for reconstruction performance.

This component is planned for months 5-6, following completion of irregular geometry training (Aim 2a).

---

## Specific Aim 3: Incorporating Realism and Experimental Validation (Future Work)

Aim 3 represents the transition from computational proof-of-concept to experimental and clinical feasibility. This is planned for later PhD stages (years 3-4) and outlined here to provide context for the research trajectory.

### Aim 3a: 3D Extension

Extending to three dimensions requires generalizing the FEM solver to tetrahedral elements and volumetric domains. Mesh generation tools like TetGen or Gmsh create unstructured tetrahedral meshes from geometric descriptions. 3D stiffness assembly follows the same principles as 2D but with larger element matrices (12x12 for 4-node tetrahedra with 3 displacement components per node).

Computational costs scale dramatically--3D FEM problems are typically 10-100x more expensive than 2D equivalents. Efficient solvers become critical: FEniCS for automated assembly, MFEM for high-performance computing, or JAX-FEM for GPU-accelerated differentiable simulation.

For neural networks, either 3D U-Net architectures or implicit MLP representations could work. The MLP approach may be particularly attractive for 3D because it avoids memory explosion from volumetric convolutions on fine grids.

### Aim 3b: CT Integration for Anatomically-Informed Models

Medical imaging provides patient-specific geometry that could constrain reconstructions. CT scans yield outer boundary geometry after segmentation, importable into the FEM meshing pipeline. The Visible Human Project dataset offers anatomically accurate cross-sections as realistic test geometries.

A key challenge: CT intensity doesn't directly translate to mechanical properties. Hounsfield units correlate with density, not stiffness. A hybrid approach might constrain tissue-type regions (muscle, fat, bone) to literature-reported E-modulus ranges while allowing PAT-Scan to refine local variations within those bounds.

An example application: forearm cross-section reconstruction, where muscle, bone, and fat have distinct stiffness values. Can PAT-Scan distinguish these tissue types from boundary displacements alone, or does the inverse problem require anatomical priors?

### Aim 3c: Experimental Validation Roadmap

The pathway from simulation to experimental reality follows three phases of increasing complexity and biological realism.

**Phase 1:** Silicone tissue-mimicking phantoms fabricated in controlled conditions. Background material would be soft silicone (Young's modulus 10-50 kPa, mimicking soft tissue) with embedded stiff inclusions at 50-500 kPa (mimicking tumors). Inclusions fabricated with known dimensions and positions provide ground truth.

The measurement system combines calibrated force sensors (load cell) with Digital Image Correlation for surface displacement measurement. DIC requires speckle patterns on phantom surfaces and stereo camera imaging to extract sub-pixel displacement fields. Total hardware cost is estimated at $10,000 (cameras, lenses, calibration equipment, load cell, actuation)--substantially less than clinical elastography systems.

Validation applies the angular scanning protocol to phantoms, measures boundary displacements via DIC, runs the trained PAT-Scan network, and compares predicted inclusion geometry to known fabricated geometry. Success would be Dice coefficient >0.8 and boundary error <10% of inclusion size.

**Phase 2:** Biological variability through ex-vivo tissue. Animal tissue or human cadaveric specimens could be tested, with mechanical testing after imaging providing partial ground truth. The challenge: exact spatial E distribution is unknown, so validation becomes relative--does PAT-Scan correctly identify stiffer versus softer regions?

**Phase 3:** The most speculative component--in-vivo clinical measurements. This requires IRB approval, careful safety protocols for force application, and clinical collaboration. Applications might include breast tumor detection or liver fibrosis staging. Validation would compare PAT-Scan reconstructions to clinical gold standards (MRE, biopsy).

Equipment requirements for experimental validation are modest. A complete DIC setup costs $5,000-10,000. Precision load cell and actuator add approximately $2,000. Phantom materials run around $500 per sample. Total laboratory setup of $10,000-15,000 is 200x less than MRE equipment.

Timeline projections place phantom experiments in year 4, following Aims 1 and 2 completion. Ex-vivo tissue testing could occur in years 4-5, potentially through collaboration with biomedical engineering or medical school laboratories. Clinical feasibility studies likely lie beyond the PhD, representing future postdoctoral or faculty research.

---

## Assumptions and Scope

### Material Model Assumptions

The current implementation assumes linear elasticity with small deformations and Hookean constitutive relations. This is justified when displacements remain below approximately 5% of sample dimensions and materials don't exhibit significant nonlinearity. For many quasi-static palpation scenarios on soft tissues, linear elasticity provides reasonable accuracy.

Real soft tissues do exhibit nonlinear stress-strain behavior, particularly at larger deformations. Hyperelastic models (Neo-Hookean, Mooney-Rivlin) would provide more accurate representations. Extending to nonlinear constitutive laws is feasible--the FEM solver would require iterative Newton-Raphson solution rather than direct linear solve--but adds computational cost. For proof-of-concept, linear elasticity suffices.

Plane stress assumption restricts applicability to thin samples where out-of-plane stresses are negligible compared to in-plane. Appropriate for 2D proof-of-concept but must be relaxed for realistic 3D applications (Aim 3a).

Poisson's ratio is assumed known and spatially constant (nu around 0.3-0.49). Soft tissues are nearly incompressible due to high water content, justifying values near 0.5. The inverse problem currently solves only for Young's modulus E, not Poisson ratio. Simultaneous identification of both would require additional measurement information or constitute an even more ill-posed problem.

### Geometric Assumptions

Outer boundary geometry is assumed known, either from direct measurement or medical imaging. For canonical circular domains in Aims 1-2, this is trivially satisfied. For anatomically realistic geometries (Aim 3b), CT or MRI provides outer boundary segmentation.

Material distribution in Aims 1-2a is restricted to binary or piecewise constant (geometric inverse problem). This relaxes in Aim 2b where continuous E(x,y) is considered (full inverse problem).

### Measurement Assumptions

Boundary displacement completeness emerged as a critical requirement. Reconstruction accuracy degrades substantially with only partial boundary data. Practical systems must enable access to a majority of the boundary, potentially requiring multi-angle camera positions or sample rotation.

Displacement measurement accuracy is assumed within DIC capabilities--typically sub-pixel resolution (0.01-0.05 pixels). In physical units, this translates to micrometer-scale accuracy for typical camera setups. Noise modeling uses additive Gaussian noise at 0.1-1% of maximum displacement, corresponding to realistic DIC uncertainty.

Force magnitude is assumed measurable via load cells with +/-1% accuracy, standard for commercial sensors. The inverse problem appears relatively insensitive to small force magnitude errors--displacement pattern matters more than absolute magnitude--but this sensitivity hasn't been rigorously quantified.

### Computational Assumptions

FEM mesh quality is maintained through structured generation algorithms producing well-conditioned elements. Highly distorted elements after deformation could degrade solution accuracy, but penetration constraints prevent deformations large enough to cause significant distortion.

The FEM solution is assumed converged, guaranteed for direct sparse solvers up to numerical precision (typically 10^-12 relative error for double precision). No iterative solver tolerance parameters require tuning.

---

## Conclusion

### Summary of Progress

This comprehensive exam presents Palpation-Assisted Tomography (PAT-Scan), a computational framework for reconstructing tissue stiffness from boundary displacement measurements using physics-informed neural networks. The work demonstrates a novel mesh-based PINN architecture that decouples forward and inverse problems, achieving computational efficiency while maintaining physical rigor.

Aim 1 established proof-of-concept methodology. The FEM forward model generates high-fidelity synthetic training data for circular inclusion geometries. The U-Net inverse solver, trained with Total Variation regularization, reconstructs inclusion boundaries from boundary displacements with qualitative accuracy confirmed through visual validation. Level-set post-processing extracts crisp geometric boundaries from soft network predictions. Hyperparameter optimization identified robust training configurations.

Aim 2a extended the framework to irregular off-centered inclusions using Fourier mode perturbations. Geometry generation, FEM adaptation, and universal training infrastructure are implemented. Preliminary results show the same neural architecture generalizes across geometric complexity without modification. Quantitative benchmarking remains ongoing.

Aim 2b and 2c remain planned, representing natural extensions to continuous material fields and systematic benchmarking.

Aim 3 provides a roadmap for experimental validation, 3D extension, and clinical translation, with equipment specifications and cost estimates supporting the accessibility claim.

### Significance and Broader Impact

The central innovation bridges classical computational mechanics with deep learning through a hybrid architecture that plays to each approach's strengths. Unlike meshfree PINNs that solve PDEs during training, the mesh-based approach uses established FEM solvers for forward physics and reserves neural networks for the ill-posed inverse mapping. This decoupling improves efficiency, strengthens physical guarantees, and enables modular development.

From a healthcare perspective, PAT-Scan addresses the accessibility gap in quantitative elastography. Equipment requirements--Digital Image Correlation and load cells--total approximately $10,000, representing 200-fold cost reduction compared to MRE and 10-fold compared to ultrasound elastography. This positions PAT-Scan as a potential solution for resource-constrained settings.

The framework extends beyond tissue mechanics to other solid mechanics inverse problems: contact detection, damage localization, material characterization. By demonstrating that boundary-only measurements suffice for internal structure reconstruction when combined with physics-informed learning, this work provides methodology applicable across computational mechanics.

Current limitations include restriction to 2D synthetic validation, underestimation of absolute stiffness values (60-80% recovery), and requirement for boundary displacement completeness. These motivate the experimental validation outlined in Aim 3.

The research establishes that geometric inverse problems in elasticity can be solved efficiently using mesh-based PINNs, that irregular geometries are handled without geometric assumptions through level-set methods, and that synthetic FEM-generated training data provides sufficient physics fidelity for proof-of-concept development. Experimental validation with physical phantoms represents the critical next step.

---

## References

1. Konofagou, E. E., & Harrigan, T. P. (2003). Palpation Tomography: A New Technique for Modulus Estimation in Elastography. IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control.

2. Goenezen, S., et al. (2017). Mechanics-Based Tomography: A Preliminary Feasibility Study. PLOS ONE.

3. Bouman, K. L., et al. (2022). Visual Vibration Tomography: Estimating Interior Material Properties from Monocular Video. ACM Transactions on Graphics.

4. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems Involving Nonlinear Partial Differential Equations. Journal of Computational Physics, 378, 686-707.

5. Wu, G., et al. (2024). JAX-SSO: A Differentiable Finite Element Analysis Solver for Structural Optimization with Seamless Integration with Neural Networks. arXiv:2407.20026.

6. Xue, T., et al. (2023). JAX-FEM: A Differentiable GPU-Accelerated 3D Finite Element Solver for Automatic Inverse Design and Mechanistic Data Science. Computer Physics Communications, 291, 108802.

7. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. Medical Image Computing and Computer-Assisted Intervention (MICCAI), 234-241.

8. Tancik, M., et al. (2020). Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains. Advances in Neural Information Processing Systems (NeurIPS), 33.

9. [Mathematical Foundations reference - 1994 linear 3D elasticity paper - full citation needed]

10. [Tissue stiffness values reference - breast cancer 5-10x stiffness contrast claim - citation needed]

11. [MRE cost and accessibility reference - citation needed]

---

**Document Status:**
- Skeleton: Complete with all Phase 1 gaps addressed
- Final draft: Complete prose version (this document)
- Quantitative metrics: Pending calculation from existing results
- Citations: Core references included, some specific values need sourcing
- Figures: Referenced but not embedded (exist in project directories)

**Files Generated:**
- `/home/vivekkarmarkar/Python Files/PAT-Scan-Copy/agents/phase2/comps_skeleton_refined_quickie.md`
- `/home/vivekkarmarkar/Python Files/PAT-Scan-Copy/agents/phase2/comps_final_quickie.md`
- `/home/vivekkarmarkar/Python Files/PAT-Scan-Copy/agents/phase2/comps_final_quickie_humanized.md`
