# PALPATION-ASSISTED TOMOGRAPHY: A MESH-BASED PHYSICS-INFORMED NEURAL NETWORK FRAMEWORK FOR INVERSE RECONSTRUCTION OF TISSUE STIFFNESS FROM BOUNDARY DISPLACEMENT MEASUREMENTS

**Vivek Karmarkar**

Comprehensive Examination Report
Ph.D. Program in Mechanical Engineering

January 2026

---

**Examining Committee:**
- [Advisor Name], Department of Mechanical Engineering (Chair)
- Suresh Raghavan, Department of Biomedical Engineering
- [Committee Member 3], Department of [X]
- [Committee Member 4], Department of [X]

---

## A. Specific Aims

Physicians have detected tumors through palpation for centuries, using the elevated stiffness of malignant tissue to identify abnormalities by touch. This practice encodes a biomechanical principle that remains clinically relevant: pathological tissues often exhibit Young's modulus values five to ten times greater than surrounding healthy tissue, creating detectable mechanical signatures under applied load (Samani et al., 2007). Yet manual palpation remains subjective, limited to superficial lesions, and incapable of quantifying the stiffness values that could inform treatment decisions. Converting palpation's intuitive mechanical assessment into quantitative diagnostic data motivates this research.

Modern elastography addresses this need through sophisticated imaging systems. Magnetic Resonance Elastography provides gold-standard quantitative stiffness maps but requires equipment costing roughly two million dollars, restricting access to major medical centers. Ultrasound elastography offers a more affordable alternative at approximately one hundred thousand dollars per system, yet remains operator-dependent and challenging to standardize. This creates a significant accessibility gap: patients in resource-constrained settings currently lack access to quantitative tissue stiffness measurement.

The core technical challenge is solving an inverse problem in continuum mechanics. The forward problem---computing displacement fields from known material properties and applied forces---is well-established through finite element methods. The inverse problem---inferring material properties from measured displacements---is ill-posed in the sense of Hadamard, exhibiting non-uniqueness of solutions and extreme sensitivity to measurement noise. When measurements are restricted to the boundary rather than the full internal displacement field, the inverse problem becomes even more severely underdetermined.

This research develops Palpation-Assisted Tomography (PAT-Scan), a computational framework that reconstructs tissue stiffness from boundary displacement measurements using a mesh-based physics-informed neural network architecture. The approach decouples the well-posed forward problem (solved exactly via finite element analysis) from the ill-posed inverse problem (addressed through a learned neural network mapping), achieving computational efficiency while maintaining physical rigor. The central hypothesis is that multiple boundary displacement measurements from different force configurations contain sufficient information to constrain the geometric inverse problem of inclusion localization, enabling reconstruction accuracy adequate for tumor detection at a fraction of current equipment costs.

**Aim 1: Establish proof-of-concept for the geometric inverse problem with circular inclusions.**

Aim 1a develops a finite element forward model for two-dimensional circular domains with plane stress formulation, structured polar grid meshing, and boundary force application protocols. Aim 1b generates training datasets through CT-inspired sequential angular scanning, applying force pairs at varying angles to probe the sample from multiple directions. Aim 1c trains a U-Net neural network with Total Variation regularization to learn the inverse mapping from boundary displacements to material property fields, followed by level-set post-processing to extract crisp inclusion boundaries.

**Aim 2: Extend the framework to realistic geometric complexity and advanced neural architectures.**

Aim 2a handles irregular, off-centered inclusions using Fourier mode boundary representations, demonstrating that the same neural network architecture generalizes across geometric complexity without modification. Aim 2b develops an implicit neural representation using Fourier feature-augmented multilayer perceptrons for continuous material field reconstruction. Aim 2c establishes a standardized benchmarking library for quantitative validation.

**Aim 3: Incorporate three-dimensional modeling, anatomical constraints from medical imaging, and experimental validation using tissue-mimicking phantoms.**

Aim 3a extends the finite element formulation to tetrahedral meshes and volumetric domains. Aim 3b integrates patient-specific geometry from CT segmentation. Aim 3c validates the framework experimentally using silicone phantoms with known stiffness values, measured via Digital Image Correlation. The experimental validation pathway requires approximately ten thousand dollars in equipment---a two-hundred-fold cost reduction compared to MRE.

---

## B. Significance

### The Clinical Challenge of Tissue Stiffness Measurement

Physicians have detected tumors through palpation for centuries, using a biomechanical principle that remains clinically relevant: pathological tissue often exhibits markedly elevated stiffness compared to healthy surrounding tissue. When a clinician presses on a suspected mass and feels resistance, they detect the increased Young's modulus that characterizes many malignancies. Breast cancer lumps, for example, exhibit stiffness values five to ten times greater than surrounding healthy breast tissue (Samani et al., 2007; Sarvazyan et al., 1998). This stiffness contrast creates a mechanical fingerprint distinguishing pathological from healthy tissue.

Yet manual palpation suffers from serious limitations. The technique is inherently subjective---different clinicians may interpret the same finding differently. It cannot quantify stiffness values, only provide qualitative assessment. Most critically, palpation is limited to superficial lesions accessible by direct touch; deeper pathology escapes detection entirely. These limitations motivate elastography: imaging modalities that quantify tissue stiffness non-invasively.

The current landscape of quantitative elastography presents a stark accessibility trade-off. Magnetic Resonance Elastography represents the gold standard, providing high-resolution three-dimensional stiffness maps with full-body imaging capability. But MRE requires specialized equipment including vibration exciters and modified MRI sequences, with total system costs approaching two million dollars (Mariappan et al., 2010). This expense restricts MRE to major medical centers, excluding the majority of clinical settings worldwide.

Ultrasound elastography offers a more affordable alternative at approximately one hundred thousand dollars per system, with the advantage of real-time imaging and integration with existing ultrasound workflows (Sigrist et al., 2017). But ultrasound elastography remains operator-dependent, requiring significant training to obtain reproducible measurements. Standardization across institutions and operators remains an ongoing challenge.

This accessibility gap has tangible clinical consequences. In resource-constrained settings---community health centers, developing regions, point-of-care screening scenarios---quantitative tissue stiffness measurement is simply unavailable. Patients in these settings cannot benefit from the diagnostic information that elastography provides to patients at major medical centers. Democratizing access to quantitative stiffness measurement could improve diagnostic outcomes for a large underserved population.

### The Inverse Problem in Continuum Mechanics

From a computational mechanics perspective, tissue stiffness reconstruction is an inverse problem. The forward problem is well-established: given material properties and applied forces, compute the resulting displacement field. For well-posed boundary conditions, this forward problem admits a unique solution and is routinely solved via finite element methods. Matrix form:

**F** = **K** **U**

where **F** is the applied force vector, **K** is the global stiffness matrix assembled from element contributions, and **U** is the displacement vector. Given **K** and **F**, solving for **U** is standard linear algebra.

The inverse problem reverses this relationship: given measured displacements **U** and applied forces **F**, determine the material properties encoded in **K**. This inverse problem is ill-posed in Hadamard's sense---solutions may not exist, may not be unique, and do not depend continuously on the data (Barbone and Oberai, 2004). Small measurement errors can produce large changes in inferred material properties. Regularization is essential to stabilize the inverse problem and select physically meaningful solutions from the space of possibilities consistent with noisy measurements.

The ill-posedness becomes more severe when measurements are restricted to the boundary rather than the full interior displacement field. Boundary-only measurements provide less information about interior material distribution, exacerbating non-uniqueness. Yet boundary measurements are also much easier to obtain experimentally---surface displacements can be measured via optical methods like Digital Image Correlation without requiring internal probes or specialized imaging hardware.

### Prior Approaches and Their Limitations

Traditional approaches to elastography inverse problems fall into several categories, each with characteristic strengths and limitations.

**Iterative optimization methods** minimize the residual between measured and predicted displacements by adjusting element-wise material properties. Goenezen and colleagues developed a rigorous mechanics-based tomography framework using iterative optimization with regularization (Goenezen et al., 2011, 2017). Their approach achieves accurate reconstructions when full-field internal displacement data is available and has been validated on tissue-mimicking phantoms. But the method is computationally expensive, requiring many forward solves per reconstruction, and can converge to local minima in the optimization landscape. Most critically for our context, these methods typically require full-field internal displacement measurements, necessitating specialized imaging systems.

**Palpation tomography**, pioneered by Konofagou and Harrigan (2003), introduced a key insight relevant to our work: applying multiple loading configurations significantly improves reconstruction accuracy. Their experiments demonstrated that nine distinct force patterns yield reconstructions far superior to single-force measurements, because multiple loadings provide redundant constraints that reduce noise sensitivity. This multiple-loading principle directly inspires our angular scanning protocol. But palpation tomography was restricted to simple geometric parameterizations and struggled with the irregular inclusion shapes characteristic of real tumors.

**Visual Vibration Tomography**, developed by Katie Bouman and colleagues, takes a different approach by inferring material properties from observed vibration modes (Bouman et al., 2022). By analyzing sub-pixel motion in video captured with standard cameras, this method recovers interior stiffness distributions from surface motion patterns. The approach avoids physical contact and requires only commodity video equipment. But it requires dynamic excitation and sophisticated video analysis to extract modal content, adding complexity to the measurement process. Our work differs by employing quasi-static loading, which simplifies the forward model physics to linear elastostatics and reduces hardware requirements to static force application and displacement measurement.

**Physics-informed neural networks (PINNs)**, introduced by Raissi, Perdikaris, and Karniadakis (2019), embed governing partial differential equations directly into neural network training through automatic differentiation. The network learns to approximate solution fields while simultaneously satisfying the governing equations at collocation points throughout the domain. This approach has achieved success across many physics domains and has been applied to tissue elasticity reconstruction (Karniadakis et al., 2020, 2022). Yet when applied to inverse problems, the coupled optimization of solving the forward PDE while simultaneously identifying unknown parameters creates computational challenges. Each training iteration requires computing PDE residuals through automatic differentiation at thousands of points, adding substantial computational overhead.

### The Gap PAT-Scan Fills

Examining existing approaches reveals a gap at the intersection of four requirements: (1) compatibility with boundary-only displacement measurements, enabling surface-accessible imaging; (2) physics-informed learning that maintains mechanical plausibility; (3) ability to handle arbitrary irregular geometries without geometric assumptions; and (4) low equipment cost enabling broad accessibility.

No existing method satisfies all four criteria simultaneously. MRE provides physics-based imaging but requires million-dollar equipment. Ultrasound elastography reduces cost but still exceeds one hundred thousand dollars and requires internal data. Visual Vibration Tomography achieves low cost but requires dynamic excitation. Mechanics-based tomography is rigorous but computationally expensive and requires internal measurements. Meshfree PINNs provide physics-informed learning but face computational challenges for inverse problems and typically assume full-field data.

PAT-Scan targets this gap. By combining mesh-based physics (finite element analysis) with neural network inverse solving, we achieve physics-informed reconstruction from boundary-only measurements at equipment cost around ten thousand dollars. The framework handles irregular geometries through universal post-processing without requiring geometry-specific algorithm modifications.

| Modality | Equipment Cost | Data Requirements | Geometry Handling | Physics-Informed |
|----------|---------------|-------------------|-------------------|------------------|
| MRE | ~$2M | Full-field internal | Arbitrary | Yes |
| Ultrasound Elastography | ~$100K | Internal + surface | Arbitrary | Partial |
| Visual Vibration Tomography | ~$5K | Surface (dynamic) | Arbitrary | Yes |
| Mechanics-Based Tomography | Variable | Full-field internal | Limited | Yes |
| **PAT-Scan** | **~$10K** | **Boundary only** | **Arbitrary** | **Yes** |

### Potential Impact

If PAT-Scan delivers on its promise, the implications extend beyond methodology to healthcare access itself.

From a scientific perspective, this work demonstrates that mesh-based physics-informed architectures can efficiently decouple forward and inverse problem solving, avoiding the computational expense of coupled optimization while maintaining physical rigor. The framework provides a template applicable to other inverse problems in solid mechanics, including contact detection, damage localization, and material characterization. The methodology bridges computational mechanics and machine learning communities, demonstrating how established numerical methods and modern deep learning can work together effectively.

From a healthcare perspective, PAT-Scan provides a pathway toward accessible quantitative tissue stiffness measurement. Equipment costs of approximately ten thousand dollars---a two-hundred-fold reduction compared to MRE---could enable deployment in settings currently underserved by elastography. The approach is particularly relevant for point-of-care diagnostics, community health centers, and resource-constrained regions. Compatibility with existing CT and imaging workflows suggests potential integration into standard diagnostic pipelines without requiring entirely new infrastructure.

---

## C. Innovation

### Innovation 1: Mesh-Based Physics-Informed Neural Network Architecture

The central methodological innovation in this work is a hybrid computational architecture that bridges classical finite element methods with modern deep learning. This mesh-based PINN approach differs from both traditional inverse problem solvers and recently popularized meshfree physics-informed neural networks.

Consider the conceptual distinction. Meshfree PINNs, as developed by Karniadakis and colleagues (Raissi et al., 2019), embed partial differential equations directly into the neural network training loss. The network simultaneously learns to approximate the solution field while satisfying governing equations---equilibrium, constitutive relations, boundary conditions---at collocation points sampled throughout the domain. This approach avoids traditional mesh discretization and can handle complex geometries naturally.

But when applied to inverse problems, meshfree PINNs face computational challenges. The network must simultaneously solve the forward PDE and identify unknown material parameters---a coupled optimization problem. Each training iteration requires computing PDE residuals through automatic differentiation at thousands of collocation points, backpropagating gradients through both the network and the physics constraints. For large-scale inverse problems, this becomes computationally prohibitive.

Our mesh-based approach follows a different philosophy: decouple the forward and inverse problems, allowing each computational method to handle what it does best.

The **forward model** uses traditional finite element analysis to solve the well-posed linear elasticity problem:

**F** = **K**(**E**) **U**

where the stiffness matrix **K** depends on the material property distribution **E**. Given known material properties and applied forces, we solve for displacements via direct sparse linear algebra. This FEM component represents established computational mechanics---highly efficient, well-validated, and guaranteeing exact satisfaction of equilibrium equations within numerical precision. No approximate satisfaction at collocation points; the variational formulation ensures equilibrium is satisfied in an integral sense over each element.

The **inverse model** employs a U-Net convolutional neural network to learn the ill-posed mapping from boundary displacements to material property fields. The network processes displacement data as a two-channel image (x-displacement and y-displacement on a regular grid) and outputs a material property map. Training proceeds without explicit PDE residuals in the loss function. Instead, physics is encoded implicitly through the FEM-generated training data and explicitly through Total Variation regularization that preserves sharp material interfaces characteristic of tumor-in-tissue scenarios.

This decoupled architecture offers concrete advantages:

First, computational efficiency improves by orders of magnitude. The forward FEM solve uses sparse linear algebra with complexity scaling linearly with problem size, rather than backpropagating through PDE residuals at every training iteration. Training requires only data-driven optimization of network parameters, with physics encoded in the training data rather than computed on-the-fly.

The physical guarantees are also stronger. The FEM forward model exactly satisfies equilibrium within numerical precision, whereas meshfree PINNs only satisfy PDEs approximately at collocation points. This difference becomes significant for stiff problems where approximate satisfaction may introduce spurious oscillations.

And the modular design enables independent updates. The forward solver and inverse network can be improved separately. More sophisticated FEM solvers (nonlinear constitutive models, contact mechanics) can be substituted without restructuring the neural network. Advanced network architectures can be explored without modifying the forward model.

Recent work in the computational mechanics community corroborates this hybrid approach. JAX-FEM (Xue et al., 2023) developed GPU-accelerated differentiable finite element solvers specifically for inverse design, demonstrating that automatic differentiation through assembled stiffness matrices is computationally tractable. JAX-SSO (Wu et al., 2024) applied differentiable FEM to structural optimization with neural network integration. The Deep FEM framework (Li et al., 2024) explicitly integrates finite element discretization with physics-informed loss functions, providing theoretical justification for the accuracy improvements of mesh-based over meshfree approaches.

Our implementation includes an option for fully differentiable physics-informed training through a differentiable FEM module that assembles the stiffness matrix from predicted material properties and computes displacement residuals. This enables incorporating physics directly into the loss function when desired, while maintaining the option for purely data-driven training.

### Innovation 2: Boundary-Only Reconstruction Capability

A defining characteristic of PAT-Scan is its reliance on boundary-only displacement measurements. This design choice requires careful framing, because at first glance, restricting to boundary data appears to discard valuable information about interior deformation.

The key insight is that boundary measurements, while individually less informative than internal measurements, can provide sufficient constraint when collected from multiple loading configurations. Each force configuration creates a distinct deformation pattern, and the boundary displacement response encodes information about how internal material properties influence mechanical behavior. A stiff inclusion resists deformation, creating a characteristic "shadow" in the boundary displacement field that differs from the response of homogeneous material.

This principle echoes Computed Tomography, where individual X-ray projections contain limited spatial information, but many projections from different angles provide sufficient constraint for unique three-dimensional reconstruction. Similarly, individual boundary displacement measurements provide limited constraint on internal stiffness distribution, but systematic scanning through multiple force configurations accumulates sufficient information for geometric reconstruction.

Konofagou and Harrigan's palpation tomography work (2003) demonstrated this principle experimentally, showing that nine distinct force patterns improve reconstruction accuracy compared to single-force measurements. Our angular scanning protocol extends this insight, systematically applying up to twenty force pair configurations with nine-degree angular spacing.

The trade-off inherent in boundary-only measurement is explicit:

| We Trade Away | We Gain |
|---------------|---------|
| Deep internal stiffness detail | 200-fold equipment cost reduction |
| Sub-millimeter spatial resolution | Surface-accessible measurement via DIC |
| Full volumetric mapping capability | Potential for real-time inference |
| Absolute stiffness precision | Sufficient geometric accuracy for detection |

For many clinical applications---tumor detection, fibrosis staging---geometric accuracy matters more than absolute stiffness quantification. Correctly identifying that an inclusion exists, where it is located, and approximately how large it is may be diagnostically sufficient, even if the precise stiffness value carries some uncertainty.

### Innovation 3: Universal Framework for Irregular Geometries

Real tumors exhibit irregular, spiculated boundaries that challenge reconstruction algorithms assuming simple geometric parameterizations. The PAT-Scan framework handles arbitrary inclusion shapes through flexible boundary representation combined with universal post-processing.

For training data generation, we represent irregular boundaries using Fourier mode decomposition. The boundary radius varies with angle according to:

r(theta) = R_base * (1 + sum_{n=1}^{N} [a_n cos(n theta) + b_n sin(n theta)])

where N_modes typically equals six, and Fourier coefficients are randomly sampled to generate diverse training geometries. Clamping ensures the radius remains between fifty and one-hundred-fifty percent of the base radius, preventing self-intersection. The inclusion center can be offset from the domain center, introducing asymmetry.

This representation provides smooth, parameterizable boundaries compatible with finite element meshing, with controllable complexity (more modes enable more irregularity) and no assumptions about convexity or connectivity.

The same U-Net architecture---without any modification---functions across geometry types. The network learns to extract geometric features from displacement patterns rather than relying on geometric assumptions. The encoder pathway captures hierarchical spatial features at multiple scales, while skip connections preserve fine-grained boundary information. This feature-based learning naturally generalizes; the network learns "what displacement patterns indicate a stiff inclusion" rather than "what circular inclusions look like."

Level-set post-processing provides the mechanism for extracting crisp boundaries from soft neural network predictions. After Gaussian smoothing and sigmoid thresholding, contour extraction identifies the 0.5 level set as the material interface. Level-set methods (Osher and Sethian, 1988) define boundaries implicitly as zero-crossings of continuous functions, handling arbitrary topology---convex, non-convex, multiply connected---without geometric assumptions.

### Innovation 4: Synthetic-to-Real Training Pathway

A common criticism of machine learning approaches in mechanics is the requirement for large labeled datasets. Clinical tissue measurements are expensive to collect and difficult to annotate with ground truth material properties. PAT-Scan addresses this limitation by using the finite element forward model as a synthetic data generator.

The FEM solver accepts arbitrary geometric descriptions and material property distributions, solves for displacement fields under specified loading, and outputs perfectly labeled training pairs. This enables unlimited data generation with controlled parameter sweeps across stiffness contrasts, inclusion sizes, shapes, and loading configurations. The dataset can be systematically expanded until training converges.

This synthetic training approach is defensible because the forward model---linear elasticity solved via finite element methods---accurately represents the underlying physics of quasi-static tissue deformation. Hooke's law for small deformations and equilibrium equations are well-established and experimentally validated. Synthetic FEM data therefore captures the essential physical relationships the neural network must learn.

The pathway from synthetic to real validation follows a staged strategy. First, training and testing occur entirely on FEM-generated data to establish that the inverse solver works in principle. Second, silicone tissue-mimicking phantoms with known stiffness values provide intermediate validation---still controlled laboratory conditions, but real measurement noise from DIC and force sensors. Third, ex-vivo tissue samples introduce biological variability while maintaining partial ground truth through mechanical testing. Fourth, clinical measurements could provide ultimate validation, though this lies beyond current PhD scope.

Transfer learning bridges the synthetic-to-real gap. A network pre-trained on millions of synthetic examples develops internal representations of displacement-to-stiffness mappings. Fine-tuning on a small dataset of real measurements adapts the network to measurement-specific noise characteristics and geometric variations not captured in simulation. This data-efficient adaptation strategy has proven successful across computer vision and shows promise for mechanics inverse problems.

---

## D. Research Approach

PAT-Scan development proceeds as a computational inverse problem solver validated on finite element-generated synthetic data. This simulation-first approach enables systematic exploration of geometric complexity, noise sensitivity, and algorithmic performance before experimental validation. Synthetic experiments establish proof-of-concept because the forward model---linear elasticity solved via FEM---accurately represents the physical phenomena, and the training objective is to learn this physics-based mapping from displacement observations to material property distributions.

---

### Aim 1: Proof-of-Concept for the Geometric Inverse Problem

#### Problem Formulation

Aim 1 addresses the geometric inverse problem: recovering the shape, size, and location of a stiff inclusion embedded in soft background material from boundary displacement measurements. The material distribution is piecewise constant---two distinct Young's modulus values separated by a sharp interface. This binary material assumption matches tumor detection scenarios where a stiff nodule is embedded in compliant surrounding tissue.

The geometric inverse problem differs from the full inverse problem (Aim 2b) where material properties vary continuously as E(x,y). The geometric formulation reduces the unknown from a continuous field to a finite set of parameters describing the inclusion boundary, making the problem more tractable while remaining clinically relevant for detection applications.

#### Aim 1a: FEM Forward Model Development

**Mesh Generation.** The computational domain is a circular region of normalized radius R_outer = 1.0 containing a concentric circular inclusion of radius R_inner = 0.3. Mesh generation follows a structured polar grid strategy, dividing the domain into n_radial = 20 radial layers and n_angular = 40 angular divisions. This structured approach yields 761 nodes and 1520 triangular elements, ensuring consistent element quality while simplifying material assignment based on radial distance.

**Material Properties.** Background material is assigned E_background = 1.0, and inclusion material is assigned E_inclusion = 10.0, representing a ten-to-one stiffness contrast characteristic of breast tumor-in-tissue scenarios (Samani et al., 2007). Poisson's ratio is set to nu = 0.3, and all nodes within the inclusion boundary are treated as fixed at zero displacement, modeling a rigid core that cannot deform.

**Finite Element Formulation.** The analysis employs plane stress assumption, appropriate for thin samples where out-of-plane stresses are negligible. The constitutive matrix for plane stress is:

D = [E/(1-nu^2)] * [[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]]

Each triangular element contributes a 6x6 element stiffness matrix K_e computed from the element geometry and material properties. The strain-displacement matrix B is derived from linear shape function derivatives, and element stiffness follows from:

K_e = Area * B^T * D * B

Element matrices are assembled into a global sparse stiffness matrix K of dimension 2N_nodes x 2N_nodes, where each node has two displacement degrees of freedom.

**Boundary Conditions and Force Application.** Boundary conditions fix all degrees of freedom for nodes inside the inclusion (zero displacement). Force application follows a paired strategy: equal and opposite radial inward forces are applied at boundary nodes located at angle theta and theta + pi. This balanced loading prevents rigid body motion while creating distinct deformation patterns. A penetration checking function ensures the deformed outer boundary does not violate the inclusion boundary constraint, limiting force magnitude to physically valid configurations.

**Solution Procedure.** The FEM system is solved via direct linear algebra using PyTorch's torch.linalg.solve(). After applying boundary conditions by modifying relevant rows of K and the force vector F, the reduced linear system is solved for free-dof displacements. Solution time is on the order of seconds for the 761-node mesh, making the forward model computationally inexpensive.

**Validation.** The FEM implementation was validated through automated testing scripts that sweep force magnitudes and angular configurations. Displacement fields exhibit expected behavior: largest magnitudes at force application points, decay with distance, minimal deformation within the stiff inclusion, and satisfaction of the penetration constraint across the operating range.

#### Aim 1b: Dataset Generation via Angular Scanning

The angular scanning protocol systematically explores the space of force configurations, drawing inspiration from Computed Tomography's principle of interrogating the sample from multiple angles.

**Scanning Protocol.** Starting from a single force pair at angle 0 degrees and 180 degrees, the number of force pairs increases incrementally to twenty, with angular spacing of 360/40 = 9 degrees between successive pairs. For each configuration, the FEM solver computes the equilibrium displacement field, and displacements at boundary nodes are extracted.

**Data Representation.** Displacement data is interpolated from irregular boundary node positions onto a regular 64x64 pixel grid covering the domain, creating two-channel images representing x-displacement and y-displacement components. The ground truth target is a binary material mask on the same grid, with pixels labeled 1 inside the inclusion and 0 outside.

**Dataset Structure.** Each geometry yields twenty training samples, one per force configuration, representing distinct interrogations of the sample. The dataset is saved with metadata including force vectors, number of force pairs, clearance distances, and maximum displacements for later analysis.

**Observation: Boundary Completeness.** One finding during development surprised us: boundary displacement completeness matters more than force magnitude variation. We had expected force magnitude to dominate, but the data told a different story. When comparing partial boundary coverage to complete boundary coverage, reconstruction accuracy improved with complete coverage---even at lower force magnitudes. This observation has important implications for experimental design: practical measurement systems must enable multi-angle access to the sample boundary.

#### Aim 1c: U-Net Training and Boundary Extraction

**Architecture.** The neural network follows the U-Net design originally developed for biomedical image segmentation (Ronneberger et al., 2015). The network accepts two-channel input (displacement fields on 64x64 grid) and produces one-channel output (normalized material property field). The encoder pathway consists of three levels of convolution-ReLU-pooling operations, progressively downsampling while increasing feature channels from a base of 32 to 64 to 128 features. The bottleneck contains 256 features. The decoder pathway upsamples through transposed convolutions while incorporating skip connections from corresponding encoder levels, enabling the network to combine high-level semantic information with fine-grained spatial details. The final layer is a 1x1 convolution followed by sigmoid activation, producing output values between 0 and 1.

**Loss Function.** The training loss combines mean squared error with Total Variation regularization:

L = MSE(E_pred, E_true) + lambda_TV * TV(E_pred)

where TV(E) = sum |nabla E| penalizes spatial gradients in the predicted material field. The MSE term ensures the predicted field matches ground truth. The TV term is important for geometric inverse problems because the expected solution is piecewise constant with sharp boundaries. Total Variation preferentially preserves edges while smoothing within regions, pre-conditioning the predicted field for subsequent level-set extraction.

**Hyperparameter Optimization.** Training parameters were optimized via grid search over learning rate ([1e-4, 5e-4, 1e-3, 5e-3, 1e-2]), TV weight lambda_TV ([0.001, 0.005, 0.01, 0.02, 0.05]), smoothing sigma ([0.01, 0.03, 0.05, 0.07, 0.1]), and thresholding temperature ([100, 500, 1000, 2000, 5000]). Grid search used 200 iterations per configuration to identify promising parameter regions, with full training of 5000 iterations for the best configuration. Optimization used the Adam optimizer.

**Post-Processing Pipeline.** Raw neural network outputs require post-processing to extract hard geometric boundaries:

1. Gaussian smoothing with sigma approximately 0.03 suppresses high-frequency noise
2. Soft thresholding via sigmoid: E_thresh = sigmoid(T * (E_smooth - 0.5)), where temperature T controls transition sharpness
3. Contour extraction at the 0.5 level set yields a polygon or spline representation of the inclusion boundary

**Training Convergence.** Loss typically decreased from initial values around 0.1 to final values near 0.01 over 5000 iterations, representing an order of magnitude improvement. Visual inspection of training evolution shows the network initially predicting uniform fields, then gradually developing circular features that sharpen and localize to match ground truth inclusions.

**Reconstruction Accuracy.** Qualitative evaluation demonstrates that level-set extraction successfully identifies sharp inclusion boundaries from soft U-Net outputs. Predicted inclusion shapes closely match ground truth circles, with center location errors typically under five percent of the inclusion radius. Quantitative metrics including Dice coefficient and Hausdorff distance are available in the codebase for systematic evaluation.

**Stiffness Underestimation.** One consistent observation warrants discussion: the recovered stiffness value within the inclusion tends to be sixty to eighty percent of the target value. This underestimation reflects the fundamental ill-posedness of boundary-only inverse problems. Displacement patterns strongly constrain inclusion geometry---location, size, shape---but provide weaker constraints on absolute stiffness magnitude. A scaling ambiguity exists: a smaller inclusion with higher stiffness can produce boundary effects similar to a larger inclusion with lower stiffness. Without additional constraints such as material property bounds or anatomical priors, perfect stiffness recovery is theoretically difficult. For clinical applications emphasizing tumor detection rather than precise stiffness quantification, geometric accuracy is likely the more relevant performance metric.

#### Aim 1 Discussion

Aim 1 successfully demonstrated proof-of-concept for the geometric inverse problem with centered circular inclusions. The integrated pipeline---FEM forward model, angular scanning data generation, U-Net training with TV regularization, and level-set boundary extraction---functions as designed and produces physically plausible reconstructions.

**What worked:** The mesh-based PINN framework architecture is established and validated. Boundary-only measurements contain sufficient information for inclusion reconstruction when multiple loading configurations are used. TV regularization effectively promotes piecewise-constant solutions appropriate for geometric inverse problems. Hyperparameter optimization via grid search identified robust training configurations. Level-set post-processing produces crisp geometric boundaries from soft network predictions.

**What remains limited:** The current implementation is restricted to two-dimensional plane stress formulations, limiting applicability to thin samples. Aim 1 addressed only centered circular inclusions, an artificial simplification. Validation occurred entirely on synthetic FEM data; experimental validation remains essential. Boundary displacement completeness emerged as a critical requirement---partial boundary coverage significantly degrades reconstruction accuracy.

**Status:** The core methodology is demonstrated and validated on canonical test cases. The framework is ready for extension to irregular geometries (Aim 2) and experimental validation (Aim 3).

---

### Aim 2: Extension to Irregular Geometries and Advanced Architectures

#### Aim 2a: Irregular Off-Centered Inclusions (60% Complete)

Real tumors exhibit irregular, spiculated boundaries that deviate substantially from circular or elliptical idealization. Aim 2a extends the framework to handle such geometric complexity.

**Fourier Mode Boundary Representation.** Irregular boundaries are represented via Fourier mode decomposition:

r(theta) = R_base * (1 + sum_{n=1}^{N_modes} [a_n cos(n theta) + b_n sin(n theta)])

where N_modes = 6 provides sufficient flexibility for moderately irregular shapes, and Fourier coefficients a_n, b_n are randomly sampled within bounds controlled by an irregularity parameter. Clamping ensures 0.5*R_base < r(theta) < 1.5*R_base to prevent self-intersection or extreme aspect ratios.

**Off-Center Positioning.** The inclusion center is displaced from the domain center by offsets c_x and c_y, which can range over normalized coordinates. Off-centering creates asymmetric deformation patterns under loading, testing the reconstruction algorithm's robustness beyond symmetric configurations.

**Implementation.** Mesh generation for irregular geometries generalizes the material assignment logic. Rather than comparing radial distance to a fixed threshold, the algorithm transforms to local coordinates centered on the inclusion, computes the angular position, evaluates the Fourier series at that angle, and compares point distance to the angle-dependent boundary radius. The FEM solver requires no modification---the same linear elasticity formulation applies regardless of inclusion shape. This solver generality demonstrates a key advantage of FEM-based forward modeling: geometric complexity is absorbed entirely into mesh generation, while the solver remains unchanged.

**Automatic Geometry Detection.** The training framework inspects dataset metadata to detect geometry type. If fields named 'a_coeffs' and 'b_coeffs' (the Fourier coefficients) are present, the script enters irregular geometry mode; otherwise it defaults to circular mode. This automatic detection enables a single training script to handle both geometry types without manual configuration.

**U-Net Generalization.** The same U-Net architecture trained on circular cases also functions on irregular cases without modification. The network learns to extract geometric features directly from displacement patterns rather than relying on hardcoded geometric assumptions. Preliminary results on irregular geometries show promise, though with increased sensitivity to hyperparameters compared to circular cases.

**Remaining Work.** Systematic training across ten to twenty irregular samples with varying eccentricity and Fourier mode amplitudes is ongoing. Quantitative accuracy metrics for irregular boundaries will be computed using Dice coefficient and Hausdorff distance. Transfer learning experiments will test whether training on circular geometries provides useful initialization for irregular geometry learning.

#### Aim 2b: Fourier Features MLP for Continuous Material Fields (Planned)

Aims 1 and 2a address the geometric inverse problem where material properties are piecewise constant. Aim 2b extends to the full inverse problem with spatially varying Young's modulus E(x,y).

**Motivation.** Continuous material fields present challenges for grid-based U-Net representations, which may struggle to capture high-frequency spatial variations. Implicit neural representations offer an alternative: the network maps continuous coordinates (x,y) directly to material property E(x,y), enabling resolution-independent querying.

**Fourier Feature Embedding.** Standard multilayer perceptrons with ReLU activations struggle to learn high-frequency functions, a phenomenon known as spectral bias. Fourier feature mapping addresses this limitation by embedding input coordinates as gamma(p) = [cos(2*pi*B*p), sin(2*pi*B*p)] where B is a random frequency matrix (Tancik et al., 2020). This embedding allows the network to represent high-frequency content.

**Proposed Architecture.** The network accepts four inputs: x-coordinate, y-coordinate, and displacement components u_x(x,y), u_y(x,y) at the query location. Fourier embedding expands these to high-dimensional frequency space. A six-to-eight layer MLP with 256 hidden units maps to a single output: predicted Young's modulus E(x,y).

**Physics-Informed Training.** The loss function can incorporate physics through the differentiable FEM module: assemble the stiffness matrix from predicted E(x,y), solve for predicted displacements, and penalize deviation from measured displacements. This ensures the predicted material field is consistent with observed mechanical behavior.

#### Aim 2c: Benchmarking Library (Planned)

Systematic validation requires standardized test cases spanning geometric and material complexity. The benchmarking library will include twenty to fifty samples covering circular, elliptical, and irregular inclusions with stiffness contrasts from two-fold to twenty-fold. Performance metrics will include Dice coefficient for spatial overlap, Hausdorff distance for boundary error, and L2 error in material fields. Noise robustness testing will add Gaussian noise to synthetic displacement measurements at levels corresponding to realistic DIC uncertainty. The benchmark dataset will be released publicly for community validation.

---

### Aim 3: Incorporating Realism and Experimental Validation

Aim 3 represents the transition from computational proof-of-concept to experimental feasibility. This aim is planned for later stages of the PhD program and is outlined here to establish the pathway from simulation to validation.

#### Aim 3a: 3D Extension

Extending the framework to three dimensions requires tetrahedral finite element formulations (four-node elements with twelve DOF per element) and volumetric meshes generated using tools such as TetGen or Gmsh. The computational cost scales significantly---three-dimensional problems are typically ten to one hundred times more expensive than two-dimensional equivalents. GPU-accelerated solvers such as JAX-FEM provide a pathway to tractable three-dimensional inverse problems. For the neural network, either 3D U-Net architectures or implicit MLP representations could be employed, with MLPs particularly attractive for 3D to avoid the memory explosion of volumetric convolutions.

#### Aim 3b: CT Integration

Medical imaging provides patient-specific geometry that could constrain PAT-Scan reconstructions. CT scans yield outer boundary geometry after segmentation, importable into the FEM meshing pipeline. A key challenge is that CT intensity (Hounsfield units) correlates with density rather than stiffness. A hybrid approach might constrain tissue-type regions to have literature-reported stiffness ranges while allowing PAT-Scan to refine local variations.

#### Aim 3c: Experimental Validation Roadmap

The pathway from simulation to experimental validation follows three phases:

**Phase 1: Silicone Phantoms.** Background material is soft silicone with Young's modulus in the range 10-50 kPa (mimicking soft tissue), with embedded stiff silicone inclusions at 50-500 kPa (mimicking tumors). Inclusions are fabricated with known dimensions and positions, providing ground truth. The measurement system combines Digital Image Correlation for surface displacement measurement with calibrated load cells for force measurement. DIC requires applying a speckle pattern to the phantom surface and imaging with stereo cameras to extract sub-pixel displacement fields. Total hardware cost is estimated at ten thousand dollars.

**Phase 2: Ex-Vivo Tissue.** Animal tissue or human cadaveric specimens introduce biological variability while maintaining partial ground truth through post-imaging mechanical testing.

**Phase 3: Clinical Feasibility.** Clinical measurements would require IRB approval, safety protocols for force application, and clinical collaboration. This phase likely extends beyond PhD scope.

**Timeline:** Phantom experiments are targeted for year four of the PhD program, following completion of Aims 1 and 2.

---

## E. Assumptions and Scope

**Material Model.** The current implementation assumes linear elasticity with Hooke's law for small deformations. This is justified when displacements remain below approximately five percent of sample dimensions and materials do not exhibit significant nonlinearity. For proof-of-concept, linear elasticity is sufficient; extension to hyperelastic models represents future work. The plane stress assumption restricts applicability to thin samples. Poisson's ratio is assumed known and spatially constant.

**Geometry.** The outer boundary geometry is assumed known from measurement or imaging. Material distribution is restricted to piecewise constant in Aims 1 and 2a, with continuous fields addressed in Aim 2b.

**Measurements.** Boundary displacement completeness is critical---reconstruction accuracy degrades substantially with partial boundary coverage. Displacement measurement accuracy is assumed within DIC capabilities (sub-pixel resolution, micrometer scale). Force magnitude accuracy is assumed within one percent, achievable with standard load cells.

**Computation.** FEM mesh quality is maintained through structured generation. FEM convergence is guaranteed for direct solve. All computations use double precision arithmetic.

---

## F. Conclusion

### Summary of Progress

This comprehensive exam presents Palpation-Assisted Tomography (PAT-Scan), a computational framework for reconstructing tissue stiffness from boundary displacement measurements using physics-informed neural networks. The work demonstrates a mesh-based architecture that decouples forward and inverse problems, achieving computational efficiency while maintaining physical rigor.

Aim 1 successfully established proof-of-concept. The FEM forward model generates high-fidelity synthetic training data. The U-Net inverse solver, trained with Total Variation regularization, reconstructs inclusion boundaries from boundary displacements. Level-set post-processing extracts crisp geometric boundaries. Hyperparameter optimization identified robust training configurations.

Aim 2a extended the framework to irregular off-centered inclusions. Fourier mode boundary representation enables diverse geometric training data. Automatic geometry detection and universal U-Net architecture demonstrate generalization across geometric complexity.

### Significance and Broader Impact

The central innovation bridges classical computational mechanics with modern deep learning through a hybrid architecture that uses the strengths of each approach. Unlike meshfree PINNs that solve PDEs during training, the mesh-based approach uses established FEM solvers for forward physics while reserving neural networks for the ill-posed inverse mapping.

From a healthcare perspective, PAT-Scan addresses the accessibility gap in quantitative elastography. Equipment requirements total approximately ten thousand dollars---a two-hundred-fold cost reduction compared to MRE. This positions PAT-Scan as a potential solution for resource-constrained settings.

### Current Limitations

The framework is currently limited to two-dimensional synthetic validation. Stiffness underestimation (sixty to eighty percent recovery) reflects the fundamental ill-posedness of boundary-only inverse problems. Boundary completeness is required for accurate reconstruction.

### Next Steps

Immediate priorities include computing quantitative metrics from existing results, completing Aim 2a benchmarking, and preparing for experimental phantom validation. The research trajectory leads from simulation through phantom validation toward clinical feasibility studies.

---

## References

1. Barbone, P. E., & Oberai, A. A. (2004). Elastic modulus imaging: on the uniqueness and nonuniqueness of the elastography inverse problem in two dimensions. *Inverse Problems*, 20(1), 283.

2. Bouman, K. L., et al. (2022). Visual Vibration Tomography: Estimating Interior Material Properties from Monocular Video. *ACM Transactions on Graphics* (SIGGRAPH).

3. Goenezen, S., et al. (2011). Linear and Nonlinear Elastic Modulus Imaging: An Application to Breast Cancer Diagnosis. *IEEE Transactions on Medical Imaging*.

4. Goenezen, S., et al. (2017). Mechanics-Based Tomography: A Preliminary Feasibility Study. *PLOS ONE*.

5. Hughes, T. J. R. (2000). *The Finite Element Method: Linear Static and Dynamic Finite Element Analysis*. Dover Publications.

6. Konofagou, E. E., & Harrigan, T. P. (2003). Palpation Tomography: A New Technique for Modulus Estimation in Elastography. *IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control*.

7. Li, X., et al. (2024). The Deep Finite Element Method: A Deep Learning Framework Integrating Physics-Informed Neural Networks with the Finite Element Method. *Journal of Computational Physics*.

8. Mariappan, Y. K., et al. (2010). Magnetic Resonance Elastography: A Review. *Clinical Anatomy*, 23(5), 497-511.

9. Oberai, A. A., et al. (2003). Solution of inverse problems in elasticity imaging using the adjoint method. *Inverse Problems*, 19(2), 297.

10. Osher, S., & Sethian, J. A. (1988). Fronts propagating with curvature-dependent speed: Algorithms based on Hamilton-Jacobi formulations. *Journal of Computational Physics*, 79(1), 12-49.

11. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems. *Journal of Computational Physics*, 378, 686-707.

12. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, 234-241.

13. Rudin, L. I., Osher, S., & Fatemi, E. (1992). Nonlinear total variation based noise removal algorithms. *Physica D*, 60(1-4), 259-268.

14. Samani, A., Zubovits, J., & Plewes, D. (2007). Elastic moduli of normal and pathological human breast tissues: an inversion-technique-based investigation of 169 samples. *Physics in Medicine and Biology*, 52(6), 1565-1576.

15. Sarvazyan, A. P., et al. (1998). Biophysical bases of elasticity imaging. *Acoustical Imaging*, 23, 223-240.

16. Sigrist, R. M. S., et al. (2017). Ultrasound Elastography: Review of Techniques and Clinical Applications. *Theranostics*, 7(5), 1303-1329.

17. Sutton, M. A., Orteu, J. J., & Schreier, H. (2009). *Image Correlation for Shape, Motion and Deformation Measurements*. Springer.

18. Tancik, M., et al. (2020). Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains. *Advances in Neural Information Processing Systems (NeurIPS)*, 33.

19. Wu, G., et al. (2024). JAX-SSO: A Differentiable Finite Element Analysis Solver for Structural Optimization with Seamless Integration with Neural Networks. arXiv:2407.20026.

20. Xue, T., et al. (2023). JAX-FEM: A Differentiable GPU-Accelerated 3D Finite Element Solver for Automatic Inverse Design and Mechanistic Data Science. *Computer Physics Communications*, 291, 108802.
