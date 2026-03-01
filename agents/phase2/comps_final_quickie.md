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

Palpation—the clinical practice of using touch to detect abnormalities—has guided medical diagnosis for centuries. The ability to detect stiff nodules embedded in soft tissue remains a cornerstone of physical examination, particularly in oncology where breast cancer lumps exhibit stiffness values 5-10 times greater than surrounding healthy tissue. However, manual palpation suffers from fundamental limitations: it is subjective, cannot quantify stiffness values, and is limited to superficial lesions accessible by direct touch.

Modern elastography techniques have emerged to address these limitations by providing quantitative stiffness maps. Magnetic Resonance Elastography (MRE) offers high accuracy but requires specialized equipment costing approximately $2 million, limiting accessibility to major medical centers. Ultrasound elastography provides a more affordable alternative at around $100,000 per system, but remains operator-dependent and challenging to standardize. This accessibility gap is particularly acute in resource-constrained settings where quantitative tissue characterization could significantly impact diagnostic outcomes and treatment planning.

### The Inverse Problem Challenge

At its core, tissue stiffness reconstruction is an inverse problem in continuum mechanics. The forward problem—computing displacement fields from known material properties and applied forces—is well-established through finite element methods and has been extensively validated. The inverse problem—inferring material properties from measured displacements—is fundamentally ill-posed, exhibiting non-uniqueness of solutions and extreme sensitivity to measurement noise.

Traditional approaches to elasticity inverse problems fall into two broad categories. Iterative optimization methods, such as those developed by Goenezen and colleagues for mechanics-based tomography, minimize the residual between measured and predicted displacements by adjusting element-wise material properties. While rigorous, these approaches are computationally expensive and prone to local minima in the optimization landscape. They also typically require full-field displacement measurements from the interior of the sample, necessitating specialized imaging systems.

Palpation tomography, pioneered by Konofagou and Harrigan in 2003, introduced the key insight that multiple loading configurations can improve the measurement-to-parameter ratio, reducing noise sensitivity. Their work demonstrated that applying nine distinct force patterns yields significantly better reconstructions than single-force measurements. However, these approaches were limited to simple geometric parameterizations and struggled with irregular inclusion shapes that characterize real tumors.

More recently, Katie Bouman's Visual Vibration Tomography demonstrated that material properties can be inferred from observed motion patterns, using vibration modes as the interrogation mechanism. This dynamic elastography approach requires sophisticated video analysis to extract sub-pixel motion but avoids the need for direct contact forces. Our work differs by employing quasi-static loading, which simplifies the actuation hardware and forward model physics while still capturing the essential stiffness contrast information.

The knowledge gap that PAT-Scan addresses is the intersection of three requirements: boundary-only displacement measurements (compatible with surface imaging systems like Digital Image Correlation), physics-informed learning that preserves mechanical plausibility, and the ability to handle arbitrary irregular geometries without geometric assumptions. No existing elastography method simultaneously satisfies all three criteria.

### Positioning PAT-Scan Against Existing Modalities

Understanding where PAT-Scan fits in the landscape of elastography requires examining tradeoffs between cost, resolution, depth penetration, and data requirements:

| Modality | Equipment Cost | Spatial Resolution | Depth Penetration | Training Data Needs |
|----------|---------------|-------------------|-------------------|---------------------|
| MRE | ~$2M | ~2mm | Full body | None (direct measurement) |
| Ultrasound Elastography | ~$100K | ~1mm | ~10cm | None |
| PAT-Scan (proposed) | ~$10K | Mesh-dependent (~2-5mm) | Surface-biased | Synthetic (unlimited) |

PAT-Scan's primary advantage lies in equipment accessibility. A complete measurement system requires only a Digital Image Correlation setup (stereo cameras with speckle pattern capability) and calibrated force sensors—equipment available in many mechanical engineering laboratories. The total hardware cost of approximately $10,000 represents a 200-fold reduction compared to MRE and a 10-fold reduction compared to ultrasound elastography, potentially enabling point-of-care diagnostics in settings where current elastography is economically unfeasible.

The approach does carry limitations. Measurements are surface-biased, with displacement information concentrated at the boundary where forces are applied. This limits depth penetration compared to MRE's full-body imaging capability. Additionally, PAT-Scan requires accurate boundary displacement measurement, demanding careful DIC system calibration. Current validation exists only on synthetic finite element data; experimental validation with physical phantoms remains a critical next step (addressed in Aim 3).

However, the synthetic data generation pathway itself represents an advantage. Unlike clinical imaging modalities that require expensive data collection campaigns, PAT-Scan's training data can be generated indefinitely through FEM simulation, enabling systematic exploration of the parameter space including stiffness contrasts, geometric complexity, and noise characteristics.

### Potential Impact

Successfully developing PAT-Scan would advance both fundamental understanding and practical application. From a scientific perspective, the work contributes methodology for inverse problems in computational mechanics by demonstrating that mesh-based physics-informed neural networks can efficiently decouple forward and inverse problem solving. The framework extends beyond tissue mechanics to other solid mechanics inverse problems including contact detection, damage localization, and material characterization.

From a healthcare perspective, PAT-Scan provides a pathway toward accessible quantitative tissue stiffness measurement. The technology is particularly relevant for resource-constrained settings where current elastography equipment is unavailable. Additionally, compatibility with existing CT and imaging workflows suggests potential integration into standard diagnostic pipelines without requiring entirely new infrastructure.

---

## C. Innovation

### Innovation 1: Mesh-Based Physics-Informed Neural Networks

The most significant innovation in this work lies in the hybrid computational architecture that bridges classical finite element methods with modern deep learning. This approach differs fundamentally from both traditional inverse problem solvers and recently popularized meshfree physics-informed neural networks.

Meshfree PINNs, as developed by Karniadakis and colleagues, embed partial differential equations directly into the neural network training loss through automatic differentiation. The network simultaneously learns to approximate the solution field while satisfying the governing equations (equilibrium, constitutive relations, and boundary conditions) at collocation points throughout the domain. This meshfree approach elegantly handles complex geometries and can solve forward problems without traditional discretization.

However, meshfree PINNs face computational challenges when applied to inverse problems. The network must simultaneously solve the forward PDE and identify unknown material parameters—a coupled optimization problem that can be expensive and prone to convergence difficulties. Each training iteration requires computing PDE residuals through automatic differentiation, adding substantial computational overhead.

Our mesh-based PINN approach follows a different philosophy: decouple the forward and inverse problems, leveraging the strengths of each computational method. The forward model uses traditional finite element analysis to solve the well-posed linear elasticity problem F = KU, where forces F are applied, the stiffness matrix K is assembled from known element properties, and displacements U are computed via a direct sparse solver. This FEM component represents established computational mechanics, is highly efficient, and guarantees exact satisfaction of equilibrium equations within numerical precision.

The inverse model employs a U-Net neural network to learn the ill-posed mapping from measured boundary displacements to material property fields. The network processes displacement data as a 2-channel image (x-displacement and y-displacement) and outputs a material property map. Training proceeds without explicit PDE residuals in the loss function; instead, physics is encoded implicitly through the FEM-generated training data and explicitly through Total Variation regularization that preserves sharp material interfaces.

This decoupled architecture offers several advantages. Computational efficiency improves dramatically because FEM forward solves leverage sparse linear algebra rather than backpropagating through PDE residuals. Physical guarantees are stronger because the forward model exactly satisfies equilibrium, whereas meshfree PINNs only satisfy PDEs approximately at collocation points. Modularity allows updating either component independently—improved FEM solvers or advanced neural architectures can be substituted without restructuring the entire framework.

Recent work corroborates this hybrid approach. JAX-SSO (2024) demonstrated that differentiable FEM combined with neural networks enables efficient structural optimization. JAX-FEM (2023) developed GPU-accelerated differentiable solvers specifically for inverse design and mechanistic data science. The Deep FEM framework (2024) explicitly integrates traditional finite element discretization with physics-informed neural networks for improved accuracy and efficiency.

Our implementation includes an option for fully differentiable physics-informed training through the `unet_forward_model_differentiable.py` module, which assembles the FEM stiffness matrix from predicted material properties and computes displacement residuals. This enables incorporating physics directly into the loss function when desired, while maintaining the option for purely data-driven training.

### Innovation 2: CT-Inspired Sequential Force Application

The measurement strategy in PAT-Scan draws inspiration from Computed Tomography's principle of interrogating the sample from multiple angles to reconstruct internal structure. Rather than applying a single force and attempting reconstruction, we systematically apply force pairs at varying angular positions around the boundary, collecting displacement data after each loading configuration.

This approach directly builds on the multiple loading protocol introduced by Konofagou and Harrigan's palpation tomography work. They demonstrated that nine distinct loads improve the measurement-to-parameter ratio, making the inverse problem less sensitive to random measurement errors. Our angular scanning strategy extends this concept by sweeping through 1 to 20 force pair configurations with systematic angular spacing.

A critical finding emerged during development: boundary displacement completeness matters more than force magnitude variation. In experiments comparing partial boundary coverage (square sample with single edge measurements) versus complete boundary coverage (semi-circular sample with full boundary access), reconstruction accuracy improved dramatically with complete coverage—even at lower force magnitudes. This observation has important implications for experimental design: practical measurement systems must enable multi-angle access to the sample boundary, potentially requiring specialized fixtures or rotating stages.

The sequential loading approach distinguishes PAT-Scan from prior elastography work in several ways. Compared to Konofagou's method, we use boundary-only measurements rather than full-field internal displacements, simplifying hardware requirements to surface-accessible imaging. Compared to Bouman's Visual Vibration Tomography, we employ quasi-static forces rather than dynamic vibration modes, eliminating the need for high-speed video and modal analysis. The quasi-static regime also simplifies the forward model to linear elasticity without dynamic terms.

From a practical standpoint, the force application strategy requires only simple actuation—calibrated pushers or compressed air jets directed radially inward at the boundary. This hardware is substantially simpler than vibration exciters or ultrasound transducers, supporting the accessibility goal.

### Innovation 3: Universal Framework for Irregular Geometries

Geometric irregularity represents a fundamental challenge for inverse problems in tissue mechanics. Real tumors exhibit spiculated, non-convex boundaries that defy simple parametric descriptions. The framework developed in this work handles arbitrary inclusion shapes without requiring geometry-specific algorithm modifications.

Mesh generation supports three geometric classes. Circular inclusions serve as the canonical test case for proof-of-concept (Aim 1). Elliptical inclusions introduce eccentricity and orientation parameters. Irregular inclusions employ Fourier mode decomposition to represent complex shapes: the boundary radius varies with angle θ according to r(θ) = R_base × (1 + Σ[a_n cos(nθ) + b_n sin(nθ)]), where N_modes typically equals 6 and coefficients are randomly sampled within bounds. The inclusions can be positioned off-center, creating asymmetric deformation patterns under loading.

The training framework automatically detects geometry type from dataset metadata. If Fourier coefficients are present, the script enters irregular geometry mode; if only radius is specified, circular mode is assumed. Remarkably, the same U-Net architecture functions across all geometry types without architectural changes. The network learns to recognize geometric features directly from the displacement field patterns, adapting its internal representations to handle circular symmetry or irregular complexity as needed.

Level-set post-processing provides the mechanism for extracting crisp boundaries from soft neural network predictions. The U-Net outputs continuous material property fields with values between 0 and 1. Gaussian smoothing suppresses high-frequency noise, soft thresholding via sigmoid with temperature parameter sharpens the transition between materials, and contour extraction identifies the 0.5 level set as the material interface. This pipeline works for arbitrary topologies—convex or non-convex, simply connected or multiply connected—because level-set methods make no geometric assumptions.

### Innovation 4: Synthetic-to-Real Training Pathway

A common criticism of machine learning approaches in mechanics is the requirement for large labeled datasets. Clinical tissue measurements are expensive to collect and difficult to annotate with ground truth material properties. PAT-Scan circumvents this limitation by using the finite element forward model as a synthetic data generator.

The FEM solver accepts arbitrary geometric descriptions and material property distributions, solves for displacement fields under specified loading, and outputs perfectly labeled training pairs: displacement input, material property target. This enables unlimited data generation with controlled parameter sweeps across stiffness contrasts, inclusion sizes, shapes, and loading configurations. Noise can be injected to simulate measurement uncertainty, and the dataset can be systematically expanded until training converges.

This synthetic training approach is defensible for proof-of-concept because the forward model—linear elasticity solved via FEM—accurately represents the underlying physics of quasi-static tissue deformation. The constitutive relations (Hooke's law for small deformations) and equilibrium equations (∇·σ = 0) are well-established and experimentally validated. Therefore, synthetic FEM data captures the essential physical relationships that the neural network must learn.

The pathway from synthetic to real follows a staged validation strategy. First, training and testing occur entirely on FEM-generated data to establish that the inverse solver works in principle. Second, silicone tissue-mimicking phantoms with known stiffness values provide intermediate validation—still controlled laboratory conditions, but now with real measurement noise from DIC and force sensors. Third, ex-vivo tissue samples introduce biological variability while maintaining some ground truth through mechanical testing of excised specimens. Fourth (and most speculatively), clinical measurements on patients could provide ultimate validation, though this lies beyond the current PhD scope.

Transfer learning bridges the synthetic-to-real gap. A network pre-trained on millions of synthetic examples develops internal representations of displacement-to-stiffness mappings. Fine-tuning on a small dataset of real measurements adapts the network to measurement-specific noise characteristics and geometric variations not captured in simulation. This data-efficient adaptation strategy has proven successful in computer vision (ImageNet pre-training then fine-tuning on small datasets) and shows promise for mechanics inverse problems.

---

## D. Research Approach

### Methodological Framing

The following sections describe PAT-Scan development as a computational inverse problem solver using FEM-generated synthetic data. This approach enables systematic exploration of geometric complexity, noise sensitivity, and algorithmic performance before experimental validation. Synthetic experiments are sufficient for establishing proof-of-concept because the forward model—linear elasticity FEM—accurately represents physical tissue deformation under quasi-static loading, and the training objective is to learn this physics-based mapping from displacement observations to material property distributions.

---

## Specific Aim 1: Proof-of-Concept for Geometric Inverse Problem

### Problem Formulation

Aim 1 addresses the geometric inverse problem: recovering the shape, size, and location of a stiff inclusion embedded in a soft background material from boundary displacement measurements. The material distribution is piecewise constant—two distinct Young's modulus values (E_background and E_inclusion) separated by a sharp interface. This binary material assumption is appropriate for tumor detection scenarios where a stiff nodule is embedded in compliant surrounding tissue.

This formulation differs from the full inverse problem (addressed in Aim 2b) where material properties vary continuously in space as E(x,y). The geometric inverse problem is more tractable because it reduces the unknown from a continuous field to a finite set of geometric parameters describing the inclusion boundary. However, recovering this boundary from boundary-only displacement measurements remains challenging due to the ill-posed nature of the inverse problem.

### Aim 1a: FEM Forward Model Development

Finite element modeling provides the foundation for both training data generation and physics validation. Mesh generation follows a structured polar grid strategy, dividing the circular domain into radial and angular segments (typically 20 radial layers and 40 angular divisions). This structured approach ensures consistent element quality and simplifies material assignment based on radial distance from the origin.

The domain consists of an outer circle with radius R_outer = 1.0 containing a concentric circular inclusion of radius R_inner = 0.3. Material properties are assigned as E_background = 1.0 and E_inclusion = 10.0, representing a 10:1 stiffness contrast typical of soft tissue-tumor systems. Poisson's ratio is set to ν = 0.3, and all nodes within the inclusion boundary are fixed at zero displacement to mimic a rigid core that cannot deform.

The FEM formulation employs plane stress assumption, appropriate for thin samples where out-of-plane stresses are negligible. Each triangular element contributes a 6×6 stiffness matrix relating nodal displacements to nodal forces, computed from the element geometry and material properties. These element matrices are assembled into a global sparse stiffness matrix K of dimension 2N_nodes × 2N_nodes (each node has x and y displacement degrees of freedom).

Force application follows a paired strategy: equal and opposite radial forces are applied at boundary nodes located at angle θ and θ + π. This balanced loading prevents rigid body motion while creating a distinct deformation pattern. Force magnitude is constrained by a geometric validity requirement: the deformed boundary must not penetrate the fixed inclusion boundary. A penetration checking function evaluates this constraint and determines the maximum allowable force for each configuration.

The FEM solution proceeds through standard direct sparse linear algebra. Boundary conditions are applied by modifying rows of K and the force vector F to enforce zero displacement at fixed nodes. The resulting linear system KU = F is solved for displacement vector U using sparse Cholesky decomposition. Solution time is on the order of seconds for typical mesh sizes (800 nodes, 1500 elements), making the forward model computationally inexpensive.

Validation of the FEM implementation confirmed several expected behaviors. Displacement fields are largest at force application points and decay with distance, matching physical intuition. The stiff inclusion experiences minimal deformation while the soft background deforms substantially. Automated testing scripts sweep force magnitudes and angular configurations, verifying solution stability and penetration constraint satisfaction across the operating range.

### Aim 1b: Dataset Generation via Angular Scanning

The angular scanning protocol systematically explores the space of force configurations to generate training data. Starting from a single force pair (at angle 0° and 180°), the number of force pairs is incrementally increased to 20, with angular spacing of 9° between successive pairs. For each configuration, the FEM solver computes the equilibrium displacement field, and displacements at boundary nodes are extracted.

Data representation for neural network training requires converting the irregular boundary node positions to a regular grid. Displacements are interpolated onto a 64×64 pixel grid covering the domain, creating two-channel images (x-displacement and y-displacement). The ground truth target is a binary material mask on the same grid, with pixels labeled 1 inside the inclusion and 0 outside.

This scanning process generates 20 training samples per geometry, each representing a distinct interrogation of the sample. The dataset is saved with metadata including force vectors, number of force pairs, clearance distances, and maximum displacements. This metadata enables later analysis of how reconstruction accuracy depends on the richness of the force configuration.

Animation of the scanning process reveals how deformation patterns evolve. With a single force pair, displacement is localized near the force application points. As more force pairs are added, the deformation pattern becomes more uniform around the boundary, providing richer information about the inclusion's influence on boundary response. This progressive information accumulation mirrors CT's principle of improving reconstruction accuracy through multiple projection angles.

### Aim 1c: U-Net Training and Boundary Extraction

The neural network architecture follows the U-Net design originally developed for biomedical image segmentation, which is well-suited for geometric inverse problems. The network accepts 2-channel input (displacement fields) and produces 1-channel output (normalized material property field). The encoder pathway consists of three levels of convolution-ReLU-pooling operations, progressively downsampling while increasing feature channels (base 32 features, doubling at each level). The decoder pathway upsamples through transposed convolutions while incorporating skip connections from corresponding encoder levels, enabling the network to combine high-level semantic information with fine-grained spatial details.

The training loss combines mean squared error with Total Variation regularization:

L = ||E_pred - E_true||² + λ_TV × Σ|∇E_pred|

The MSE term ensures the predicted material field matches the ground truth, while the TV term penalizes spatial gradients. This regularization is crucial for geometric inverse problems because the expected solution is piecewise constant with sharp boundaries. Total Variation preferentially preserves edges while smoothing within regions, pre-conditioning the predicted field for subsequent level-set extraction.

The connection between TV regularization and post-processing deserves emphasis. Without TV penalty, the neural network tends to produce soft, blurred transitions between materials. While MSE loss alone might achieve low error, the resulting field lacks the sharp 0/1 separation needed for clean boundary extraction. TV regularization guides the network toward piecewise constant solutions, making the subsequent thresholding step more robust.

Hyperparameter optimization proceeded through grid search over learning rate (10^-5, 10^-4, 10^-3), TV weight λ_TV (0.001, 0.005, 0.01), and thresholding temperature T (2000, 3000, 5000). Training employed Adam optimizer for 5000 iterations per configuration, with the best hyperparameter set selected based on reconstruction accuracy on held-out validation geometries. Results indicated that learning rate 10^-4, TV weight 0.005, and temperature 3000 provided the most consistent performance across different stiffness contrasts.

Post-processing transforms soft neural network predictions into hard geometric boundaries through a three-step pipeline. First, Gaussian smoothing with σ ≈ 0.03 suppresses high-frequency noise and regularizes the field. Second, soft thresholding via sigmoid function (E_thresh = sigmoid(T × (E_smooth - 0.5))) sharpens the transition, with temperature T controlling the steepness. Third, contour extraction identifies the 0.5 level set, yielding a polygon or spline representation of the inclusion boundary.

Training convergence was monitored through loss curves and visual inspection of predictions. The combined loss typically decreased from initial values around 0.1 to final values near 0.01 over 5000 iterations, representing an order of magnitude improvement. Animation of training evolution shows the network initially predicting a uniform field, then gradually developing a circular feature that sharpens and localizes to match the ground truth inclusion.

Reconstruction accuracy was evaluated qualitatively through visual comparison of predicted and true boundaries. The level-set extraction successfully identified sharp inclusion boundaries from the soft U-Net outputs. Predicted inclusion shapes closely matched ground truth circles, with center location errors typically under 5% of the inclusion radius.

Quantitative metrics including Dice coefficient (overlap measure), Hausdorff distance (maximum boundary error), and radius estimation error would provide more rigorous validation. These metrics remain to be computed from existing results. The current status is qualitative validation demonstrating proof-of-concept, with quantitative benchmarking as ongoing work.

One consistent observation warrants discussion: the recovered stiffness value within the inclusion tends to be 60-80% of the target value rather than exactly matching. This underestimation likely reflects the fundamental ill-posedness of the inverse problem combined with boundary-only measurements. The displacement data constrain the inclusion geometry strongly but provide weaker constraints on absolute stiffness magnitude. However, for tumor detection applications, accurate shape and location may be more clinically relevant than exact stiffness quantification.

### Discussion: Aim 1 Achievements and Limitations

Aim 1 successfully demonstrated proof-of-concept for the geometric inverse problem with centered circular inclusions. The integrated pipeline—FEM forward model, angular scanning data generation, U-Net training with TV regularization, and level-set boundary extraction—functions as designed and produces physically plausible reconstructions.

Key achievements include establishing the mesh-based PINN framework architecture, validating that boundary-only measurements contain sufficient information for inclusion reconstruction, demonstrating the effectiveness of TV regularization for geometric inverse problems, and completing hyperparameter optimization through systematic grid search.

The work also revealed important limitations and scope boundaries. The 2D plane stress assumption limits applicability to thin samples; 3D extension is addressed in Aim 3. Restriction to centered circular inclusions represents an artificial simplification; irregular geometry support is addressed in Aim 2a. Validation occurs entirely on synthetic FEM data; experimental validation with physical phantoms is planned for Aim 3c. Finally, boundary displacement completeness emerged as a critical requirement—partial boundary coverage significantly degrades reconstruction accuracy, imposing design constraints on experimental measurement systems.

The current status is best characterized as "core methodology demonstrated" rather than "fully completed." The algorithmic framework is functional and validated on canonical test cases, but quantitative performance metrics and broader geometric validation remain as ongoing work.

---

## Specific Aim 2: Extension to Irregular Geometries and Advanced Architectures

### Overall Goal and Status

Aim 2 extends PAT-Scan beyond the proof-of-concept canonical case to handle realistic geometric complexity and explore alternative neural architectures for the full inverse problem. This aim subdivides into three components with different completion statuses:

Aim 2a (Irregular off-centered inclusions): 60% complete. Geometry generation, FEM solver adaptation, and universal training framework are implemented and functional. Quantitative benchmarking and systematic geometric variation studies remain pending.

Aim 2b (Fourier Features MLP for continuous material fields): 0% complete. This component is planned but not yet implemented, representing future work.

Aim 2c (Benchmarking library): 0% complete. Also planned but not implemented.

### Aim 2a: Irregular Off-Centered Inclusions

Real tumors exhibit irregular boundaries that challenge reconstruction algorithms. Aim 2a introduces geometric complexity through Fourier mode perturbations applied to a circular base shape. The boundary radius varies with angle according to r(θ) = R_base × (1 + Σ[a_n cos(nθ) + b_n sin(nθ)]), where the number of modes N_modes typically equals 6 and Fourier coefficients are randomly sampled. An irregularity parameter controls the amplitude of perturbations, and clamping ensures r(θ) remains between 0.5 and 1.5 times the base radius to prevent self-intersection or extreme aspect ratios.

Additionally, the inclusion center is displaced from the domain center by offsets c_x and c_y, which can range from -0.5 to 0.5 in normalized coordinates. This off-centering creates asymmetric deformation patterns under loading, providing a more realistic test of the reconstruction algorithm's robustness.

Mesh generation for irregular geometries required generalizing the material assignment logic. Rather than checking if a point falls within a circle (simple radial distance threshold), the algorithm transforms to local coordinates centered on the inclusion, computes the angular position θ, evaluates the Fourier series to get the boundary radius at that angle, and compares the point's distance from center to this angle-dependent threshold.

The FEM solver required no modifications; the same linear elasticity formulation applies regardless of inclusion shape. Force application and boundary condition enforcement proceed identically to the circular case. This solver generality underscores the advantage of FEM-based forward modeling—geometric complexity is absorbed into the mesh generation stage, while the solver remains unchanged.

The training framework incorporates automatic geometry detection by inspecting dataset metadata. If the dataset contains fields named 'a_coeffs' and 'b_coeffs' (the Fourier coefficients), the script enters irregular geometry mode. Otherwise, it defaults to circular mode. Remarkably, the same U-Net architecture trained on circular cases also functions on irregular cases without modification. The network learns to extract geometric features directly from displacement patterns rather than relying on geometric assumptions.

Preliminary training results on irregular geometries show promise but reveal increased sensitivity to hyperparameters compared to circular cases. The optimal TV weight and smoothing parameters differ, suggesting that irregular boundaries require more careful regularization tuning. Level-set extraction handles non-convex shapes successfully, demonstrating that the post-processing pipeline generalizes beyond simple circular topology.

Future work for Aim 2a includes completing systematic training across 10-20 irregular samples with varying eccentricity and Fourier mode amplitudes, computing quantitative accuracy metrics (Dice coefficient, Hausdorff distance) for irregular boundaries, and investigating generalization: does training on circular geometries transfer to irregular geometries, and vice versa? The hypothesis is that training on diverse geometries will produce a more robust inverse solver than training on circular cases alone.

### Aim 2b: Fourier Features MLP for Full Inverse Problem (Planned)

Aims 1 and 2a address the geometric inverse problem where material properties are piecewise constant. Aim 2b extends to the full inverse problem with spatially varying Young's modulus E(x,y). This continuous field presents challenges for grid-based U-Net representations, which may struggle to capture high-frequency spatial variations.

Implicit neural representations offer an alternative architecture. The network maps continuous coordinates (x,y) to material property E(x,y), enabling resolution-independent querying. Fourier feature mapping—embedding input coordinates as γ(p) = [cos(2πB·p), sin(2πB·p)] where B is a random frequency matrix—allows multi-layer perceptrons to learn high-frequency functions that would otherwise be difficult for standard ReLU networks.

The proposed architecture accepts four inputs: x-coordinate, y-coordinate, x-displacement u_x(x,y), and y-displacement u_y(x,y) at the query location. A Fourier feature embedding expands these to high-dimensional frequency space, then a 6-8 layer MLP with 256 hidden units maps to a single output: Young's modulus E(x,y).

Training data generation requires creating synthetic samples with continuous E(x,y) distributions. Examples include multiple inclusions with different stiffness values, radial gradient fields E(r) = E_0 + k×r, and Perlin noise-based heterogeneity. For each sample, the FEM solver assigns spatially varying element properties, solves for displacements, and samples (x,y,u) triplets for training.

The loss function could incorporate both data-driven and physics-informed components:

L = ||E_pred(x,y) - E_true(x,y)||² + λ_physics × (displacement prediction error)

The physics term would use the differentiable FEM module to assemble K from predicted E-field, solve for predicted displacements, and penalize deviation from measured displacements. This ensures the predicted material field is not only close to ground truth but also consistent with observed mechanical behavior.

Expected outcomes include demonstrating feasibility of continuous E(x,y) reconstruction from boundary displacements, benchmarking MLP versus U-Net on the geometric inverse problem (as a controlled comparison), and quantifying the resolution independence advantage of implicit representations. Anticipated challenges include longer training times and potential requirement for larger datasets compared to the geometric inverse case.

### Aim 2c: Benchmarking Library (Planned)

Systematic validation requires diverse test cases spanning the range of geometric and material complexity. The proposed benchmarking library would include 20-50 samples covering circular inclusions with varying radius and eccentricity, elliptical inclusions with aspect ratios from 1.5 to 3.0, irregular Fourier-perturbed inclusions with varying mode counts and perturbation amplitudes, multiple-inclusion configurations (2-3 separate stiff regions), and biologically-inspired geometries such as spiculated tumor boundaries.

Stiffness contrast would be systematically varied across clinically relevant ranges: E_inclusion/E_background ∈ {2, 5, 10, 20}, corresponding to literature values for various tissue pathologies. For example, breast cancer exhibits approximately 5-10× stiffness increase, while liver tumors may show 2-5× increase.

Performance metrics would assess both geometric accuracy (Dice coefficient for spatial overlap, Hausdorff distance for maximum boundary error) and material recovery accuracy (L² error in E-field, mean absolute percentage error). Noise robustness testing would add Gaussian noise to synthetic displacement measurements at levels corresponding to DIC measurement uncertainty (0.1-1% of maximum displacement).

The benchmark dataset would be published for community use, enabling other researchers to validate alternative inverse elastography methods on standardized test cases. Statistical analysis would report mean and standard deviation for each metric across the test set, providing confidence intervals for reconstruction performance.

This benchmarking component is planned for months 5-6 of the research timeline, following completion of irregular geometry training (Aim 2a).

---

## Specific Aim 3: Incorporating Realism and Experimental Validation (Future Work)

Aim 3 represents the transition from computational proof-of-concept to experimental and clinical feasibility. This aim is planned for later stages of the PhD program (years 3-4) and is briefly outlined here to provide context for the overall research trajectory.

### Aim 3a: 3D Extension

Extending the framework to three dimensions requires generalizing the FEM solver to tetrahedral elements and volumetric domains. Mesh generation tools such as TetGen or Gmsh can create unstructured tetrahedral meshes from geometric descriptions. The 3D stiffness assembly follows the same principles as 2D but with larger element matrices (12×12 for 4-node tetrahedra with 3 displacement components per node).

The computational challenge scales dramatically—3D FEM problems are typically 10-100× more expensive than 2D equivalents. Efficient solvers become critical; options include FEniCS for automated finite element assembly, MFEM for high-performance computing, or JAX-FEM for GPU-accelerated differentiable simulation.

For the neural network, either 3D U-Net architectures or implicit MLP representations could be employed. The MLP approach may be particularly attractive for 3D because it avoids the memory explosion associated with volumetric convolutions on fine grids.

### Aim 3b: CT Integration for Anatomically-Informed Models

Medical imaging provides patient-specific geometry that could constrain PAT-Scan reconstructions. CT scans yield outer boundary geometry after segmentation, which can be imported into the FEM meshing pipeline. The Visible Human Project dataset offers anatomically accurate cross-sections that could serve as realistic test geometries.

A key challenge is that CT intensity values do not directly translate to mechanical properties. Hounsfield units correlate with density but not stiffness. A hybrid approach might constrain tissue-type regions (muscle, fat, bone) to have literature-reported E-modulus ranges while allowing PAT-Scan to refine local variations within those bounds.

An example application would be forearm cross-section reconstruction, where muscle, bone, and fat exhibit distinct stiffness values. The question becomes: can PAT-Scan distinguish these tissue types from boundary displacement measurements alone, or does the inverse problem require additional anatomical priors?

### Aim 3c: Experimental Validation Roadmap

The pathway from simulation to experimental reality follows three phases, each increasing in complexity and biological realism.

Phase 1 employs silicone tissue-mimicking phantoms fabricated in controlled laboratory conditions. The background material would be soft silicone with Young's modulus in the range 10-50 kPa (mimicking soft tissue), with embedded stiff silicone inclusions at 50-500 kPa (mimicking tumors). Inclusions would be fabricated with known dimensions and positions, providing ground truth for validation.

The measurement system combines calibrated force sensors (load cell) with Digital Image Correlation for surface displacement measurement. DIC requires applying a speckle pattern to the phantom surface and imaging with stereo cameras to extract sub-pixel displacement fields. Total hardware cost is estimated at $10,000 (cameras, lenses, calibration equipment, load cell, and actuation), substantially less than clinical elastography systems.

Validation proceeds by applying the angular scanning protocol to the phantom, measuring boundary displacements via DIC, running the trained PAT-Scan neural network to predict material properties, and comparing the predicted inclusion geometry to the known fabricated geometry. Success would be defined as Dice coefficient >0.8 and boundary error <10% of inclusion size.

Phase 2 introduces biological variability through ex-vivo tissue samples. Animal tissue or human cadaveric specimens could be tested, with the advantage that tissue can be mechanically tested after imaging to provide partial ground truth. The challenge is that exact spatial distribution of E is unknown, so validation becomes relative: does PAT-Scan correctly identify stiffer versus softer regions?

Phase 3 represents the most speculative component: in-vivo clinical measurements. This would require IRB approval, careful safety protocols for force application, and collaboration with clinical partners. Applications might include breast tumor detection or liver fibrosis staging. Validation would compare PAT-Scan reconstructions to clinical gold standards such as MRE or biopsy results.

The equipment requirements for experimental validation are modest compared to developing new clinical imaging systems. A complete DIC setup (stereo cameras, speckle pattern materials, calibration targets) costs $5,000-10,000. A precision load cell and actuator add approximately $2,000. Phantom fabrication materials (silicone, molds, mixing equipment) cost around $500 per sample. The total laboratory setup cost of $10,000-15,000 is 200× less than MRE equipment, supporting the accessibility claim that motivated this work.

Timeline projections place phantom experiments in year 4 of the PhD program, following completion of Aims 1 and 2. Ex-vivo tissue testing could occur in years 4-5, potentially through collaboration with biomedical engineering or medical school laboratories. Clinical feasibility studies likely lie beyond the PhD scope, representing future postdoctoral or faculty research.

---

## Assumptions and Scope

### Material Model Assumptions

The current implementation assumes linear elasticity with small deformations and Hookean constitutive relations. This is justified when displacements remain below approximately 5% of sample dimensions and materials do not exhibit significant nonlinearity. For many quasi-static palpation scenarios on soft tissues, linear elasticity provides reasonable accuracy.

However, real soft tissues exhibit nonlinear stress-strain behavior, particularly at larger deformations. Hyperelastic models such as Neo-Hookean or Mooney-Rivlin would provide more accurate representations. Extending the framework to nonlinear constitutive laws is feasible—the FEM solver would require iterative Newton-Raphson solution rather than direct linear solve—but adds computational cost. For proof-of-concept, linear elasticity is sufficient to demonstrate methodology.

The plane stress assumption restricts applicability to thin samples where out-of-plane stresses are negligible compared to in-plane stresses. This is appropriate for 2D proof-of-concept but must be relaxed for realistic 3D applications (Aim 3a).

Poisson's ratio is assumed known and spatially constant (ν ≈ 0.3-0.49). Soft tissues are nearly incompressible due to high water content, justifying values near 0.5. The inverse problem in the current formulation solves only for Young's modulus E, not for Poisson ratio. Simultaneous identification of both parameters would require additional measurement information or constitute an even more ill-posed problem.

### Geometric Assumptions

The outer boundary geometry is assumed known, either from direct measurement or medical imaging. For the canonical circular domain used in Aims 1-2, this is trivially satisfied. For anatomically realistic geometries (Aim 3b), CT or MRI provides outer boundary segmentation.

Material distribution in Aims 1-2a is restricted to binary or piecewise constant (geometric inverse problem). This assumption is relaxed in Aim 2b where continuous E(x,y) is considered (full inverse problem).

### Measurement Assumptions

Boundary displacement completeness emerged as a critical requirement. Reconstruction accuracy degrades substantially when only partial boundary data is available. Practical measurement systems must enable access to a majority of the boundary, potentially requiring multi-angle camera positions or sample rotation.

Displacement measurement accuracy is assumed to be within DIC capabilities, which typically achieve sub-pixel resolution (0.01-0.05 pixels). In physical units, this translates to micrometer-scale accuracy for typical camera setups. Noise modeling uses additive Gaussian noise at 0.1-1% of maximum displacement, corresponding to realistic DIC uncertainty.

Force magnitude is assumed measurable via load cells with ±1% accuracy, which is standard for commercial force sensors. The inverse problem appears relatively insensitive to small force magnitude errors—displacement pattern matters more than absolute magnitude—but this sensitivity has not been rigorously quantified.

### Computational Assumptions

FEM mesh quality is maintained through structured generation algorithms that produce well-conditioned elements. Highly distorted elements after deformation could degrade solution accuracy, but the penetration constraint prevents deformations large enough to cause significant distortion.

The FEM solution is assumed converged, which is guaranteed for direct sparse solvers up to numerical precision (typically 10^-12 relative error for double precision arithmetic). No iterative solver tolerance parameters require tuning.

---

## Conclusion

### Summary of Progress

This comprehensive exam presents Palpation-Assisted Tomography (PAT-Scan), a computational framework for reconstructing tissue stiffness from boundary displacement measurements using physics-informed neural networks. The work demonstrates a novel mesh-based PINN architecture that decouples forward and inverse problems, achieving computational efficiency while maintaining physical rigor.

Aim 1 successfully established proof-of-concept methodology. The FEM forward model generates high-fidelity synthetic training data for circular inclusion geometries. The U-Net inverse solver, trained with Total Variation regularization, reconstructs inclusion boundaries from boundary displacements with qualitative accuracy confirmed through visual validation. Level-set post-processing extracts crisp geometric boundaries from soft neural network predictions. Hyperparameter optimization identified robust training configurations.

Aim 2a extended the framework to irregular off-centered inclusions using Fourier mode perturbations. Geometry generation, FEM adaptation, and universal training infrastructure are implemented and functional. Preliminary results demonstrate that the same neural architecture generalizes across geometric complexity without modification. Quantitative benchmarking remains as ongoing work.

Aim 2b and 2c remain planned but not yet implemented, representing natural extensions to continuous material fields and systematic benchmarking.

Aim 3 provides a roadmap for experimental validation, 3D extension, and clinical translation, with detailed equipment specifications and cost estimates supporting the accessibility claim.

### Significance and Broader Impact

The central innovation lies in bridging classical computational mechanics with modern deep learning through a hybrid architecture that leverages the strengths of each approach. Unlike meshfree PINNs that solve PDEs during training, the mesh-based approach uses established FEM solvers for forward physics and reserves neural networks for the ill-posed inverse mapping. This decoupling improves computational efficiency, strengthens physical guarantees, and enables modular development.

From a healthcare perspective, PAT-Scan addresses the accessibility gap in quantitative elastography. Equipment requirements—Digital Image Correlation and load cells—total approximately $10,000, representing a 200-fold cost reduction compared to MRE and a 10-fold reduction compared to ultrasound elastography. This positions PAT-Scan as a potential solution for resource-constrained settings where current elastography is economically infeasible.

The framework extends beyond tissue mechanics to other solid mechanics inverse problems including contact detection, damage localization, and material characterization. By demonstrating that boundary-only measurements suffice for internal structure reconstruction when combined with physics-informed learning, this work provides methodology applicable across computational mechanics.

Current limitations include restriction to 2D synthetic validation, underestimation of absolute stiffness values (60-80% recovery), and requirement for boundary displacement completeness. These limitations motivate the experimental validation pathway outlined in Aim 3.

The research establishes that geometric inverse problems in elasticity can be solved efficiently using mesh-based PINNs, that irregular geometries are handled without geometric assumptions through level-set methods, and that synthetic FEM-generated training data provides sufficient physics fidelity for proof-of-concept development. Experimental validation with physical phantoms represents the critical next step toward clinical translation.

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

10. [Tissue stiffness values reference - breast cancer 5-10× stiffness contrast claim - citation needed]

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
