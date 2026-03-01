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

**Aim 1 - Proof-of-Concept:** Develop and validate a U-Net-based inverse solver for geometric reconstruction of circular inclusions from boundary displacement measurements. This aim establishes the core methodology combining finite element forward modeling with neural network inverse solvers. Current status: Core methodology demonstrated with functional FEM forward solver, U-Net training pipeline, and level-set boundary extraction. Quantitative metrics computation remains in progress.

**Aim 2 - Extensions:** Extend the framework to handle irregular, off-centered inclusions (60% complete with geometry generation and training functional) and explore advanced neural architectures for the full inverse problem with spatially varying material properties (planned but not yet implemented).

**Aim 3 - Realism:** Incorporate 3D modeling, CT integration, and experimental validation using tissue-mimicking phantoms to demonstrate clinical feasibility. Detailed experimental roadmap developed with equipment specifications and budget ($8,000-10,000 total), planned for implementation contingent on Aim 2 completion.

---

## B. Significance

### The Clinical Challenge of Tissue Stiffness Measurement

You cannot see tissue stiffness, and you cannot feel it without touching. Yet this invisible property--how biological materials respond to applied forces--has become a powerful diagnostic marker across medicine. Palpation, that ancient practice of detecting abnormalities through touch, has guided physicians for centuries and remains a cornerstone of physical examination today. In oncology, the ability to detect stiff nodules embedded in soft tissue is particularly valuable: breast cancer lumps exhibit stiffness values 5 to 10 times greater than surrounding healthy tissue, creating a mechanical contrast that skilled clinicians can sometimes identify through manual examination alone.

But manual palpation has fundamental limitations. The technique is subjective, varying with examiner experience and technique. It cannot quantify stiffness--only offer qualitative assessments of "hard" versus "soft." Access is limited to superficial lesions, typically no deeper than 2-3 centimeters below the skin surface. And small or deeply embedded abnormalities may go undetected entirely, particularly in patients where overlying tissue density impedes mechanical interrogation.

Modern elastography techniques emerged to address these limitations by providing quantitative stiffness maps of internal tissue structures. Magnetic Resonance Elastography (MRE) offers high accuracy and full three-dimensional visualization, but requires specialized equipment costing approximately $2 million--effectively limiting its availability to major medical centers in well-resourced regions. Ultrasound elastography provides a more affordable alternative at around $100,000 per system, yet remains operator-dependent and challenging to standardize across institutions. This accessibility gap is particularly acute in resource-constrained settings worldwide, where the absence of quantitative tissue characterization significantly impacts diagnostic outcomes and treatment planning decisions.

### The Inverse Problem in Continuum Mechanics

At its mathematical core, tissue stiffness reconstruction is an inverse problem in continuum mechanics. The forward problem--computing displacement fields from known material properties and applied forces--is well-established physics. Finite element methods solve the linear elasticity equations F = KU with high accuracy, and experimental validation has confirmed their predictions across countless applications. The mathematical framework is sound, the numerical implementations are mature, and confidence in forward model predictions runs high.

The inverse problem is an entirely different beast. Inferring material properties from measured displacements reverses the causality arrow, transforming a well-posed mathematical problem into one that is fundamentally ill-posed in the sense of Hadamard. Solutions may not be unique: multiple different material property distributions can produce nearly identical boundary displacement patterns. Extreme sensitivity to measurement noise amplifies this difficulty--small errors in displacement measurements can lead to large errors in reconstructed stiffness values, particularly when regularization is insufficient or poorly chosen.

Traditional approaches to elasticity inverse problems fall broadly into two categories. Iterative optimization methods, exemplified by the mechanics-based tomography work of Goenezen and colleagues, minimize the residual between measured and predicted displacements by adjusting element-wise material properties through adjoint-based gradient descent. While mathematically rigorous and capable of handling complex geometries, these approaches are computationally expensive, requiring repeated finite element solutions at each optimization iteration. Local minima pose additional challenges, with solution quality often depending sensitively on initial conditions. Such methods typically require full-field displacement measurements from the interior of the sample, necessitating specialized imaging systems.

Konofagou and Harrigan introduced a key insight in their 2003 palpation tomography work. They showed that applying multiple distinct loading configurations rather than a single large-area compression "increased the ratio of measurements to fitted parameters, which made the method less sensitive to random errors." Their experiments showed that using nine loading cases achieved noise reduction by a factor of two compared to single-load elastography. Their approach, however, still required full-field internal displacement measurements via ultrasound and relied on element-wise iterative optimization.

More recently, Katie Bouman and colleagues demonstrated through Visual Vibration Tomography that material properties can be inferred from observed motion patterns captured in ordinary video. This elegant approach uses vibration modes as the interrogation mechanism, extracting sub-pixel motion from monocular high-speed video to constrain physics-based inverse models. Our work differs by employing quasi-static loading rather than dynamic vibrations, which simplifies both the actuation hardware (no vibration exciters required) and the forward model physics (no need for modal analysis).

The knowledge gap that PAT-Scan addresses lies at the intersection of three requirements: boundary-only displacement measurements compatible with surface imaging systems like Digital Image Correlation, physics-informed learning that preserves mechanical plausibility, and the ability to handle arbitrary irregular geometries without parametric shape models. No existing elastography method satisfies all three. MRE provides excellent accuracy but requires expensive internal imaging. Palpation tomography pioneered multiple loading protocols but demands full-field displacement data. Visual Vibration Tomography works from boundary observations but operates in the dynamic regime with attendant complexity. PAT-Scan occupies the previously unexplored space where these requirements converge.

### Positioning PAT-Scan in the Elastography Landscape

Understanding where PAT-Scan fits requires examining the fundamental trade-offs between equipment cost, spatial resolution, depth penetration, and data requirements. MRE achieves approximately 2 millimeter spatial resolution with full-body depth penetration, but demands $2 million in specialized hardware. Ultrasound elastography offers finer resolution near 1 millimeter and reasonable depth penetration up to 10 centimeters, yet requires $100,000 systems operated by trained sonographers. PAT-Scan trades some of this resolution and depth capability for dramatic cost reduction, targeting mesh-dependent resolution in the 2-5 millimeter range with surface-biased sensitivity, all achieved with approximately $10,000 in hardware.

This cost structure has important implications for accessibility. The equipment needed for PAT-Scan measurements--stereo cameras with speckle pattern capability and load cells--already exists in many mechanical engineering laboratories. The total hardware budget of $10,000 amounts to a 200-fold reduction compared to MRE and a 10-fold reduction compared to ultrasound elastography, potentially enabling point-of-care diagnostics in settings where current elastography technologies remain economically unfeasible.

The approach does carry inherent limitations that must be acknowledged candidly. Measurements are fundamentally surface-biased, with displacement information concentrated at the boundary. This limits depth penetration compared to MRE's ability to interrogate tissues throughout the entire body volume. PAT-Scan requires accurate boundary displacement measurement, demanding careful Digital Image Correlation system calibration. Current validation exists only on synthetic finite element data--experimental validation with physical phantoms is a critical next step addressed in Aim 3.

Yet the synthetic data generation pathway itself is a distinct advantage. Unlike clinical imaging modalities that require expensive data collection campaigns involving patient recruitment and regulatory approval, PAT-Scan's training data can be generated indefinitely through finite element simulation. This allows systematic exploration of stiffness contrasts from subtle (2:1) to dramatic (20:1), geometric complexity ranging from simple circles to irregular spiculated boundaries, and noise characteristics spanning the range of Digital Image Correlation measurement uncertainty. Training set diversity can be expanded essentially without cost until neural network performance saturates--a luxury unavailable to methods dependent on clinical data acquisition.

### Clinical Niche: When Boundary Measurements Suffice

Clarifying where PAT-Scan belongs in the clinical toolbox requires honest assessment of both capabilities and limitations. This technology is not intended to replace MRE or ultrasound elastography in well-resourced settings where those modalities already function effectively. Rather, PAT-Scan targets three specific scenarios where current elastography is unavailable or impractical.

Resource-constrained settings are the primary target. In rural clinics throughout developed countries and healthcare facilities in developing nations, a $10,000 Digital Image Correlation setup may be feasible where $2 million MRE systems are not. For screening applications focused on detecting the presence or absence of stiff nodules, geometric localization may suffice without requiring full quantitative stiffness mapping. A binary decision--"is there a tumor, where is it located, and how large is it?"--can guide biopsy targeting and surgical planning even without precise Young's modulus quantification at every spatial point.

Intraoperative guidance presents a second promising application. Surgical palpation, where the surgeon's finger provides the interrogating force, could be augmented with surface camera imaging for real-time inclusion localization during procedures. MRE and ultrasound are inherently pre-operative imaging modalities; PAT-Scan could complement these by providing updated stiffness information as the surgical field evolves. The connection to vascular surgery and aneurysm repair--within Suresh Raghavan's research domain--suggests particular relevance for assessing arterial wall stiffness during interventions.

Tissue engineering quality control offers a third niche. Non-destructive assessment of engineered tissue scaffolds cultured in bioreactors benefits from repeated measurements tracking stiffness evolution as cells remodel the extracellular matrix. MRE is impractical for small samples in bioreactor chambers, and destructive mechanical testing cannot be performed repeatedly on the same specimen. PAT-Scan's non-contact displacement measurement combined with controlled force application could track scaffold maturation over days to weeks of culture.

The fundamental trade-off must be stated clearly: PAT-Scan sacrifices three-dimensional full-field resolution for equipment accessibility and cost reduction. The question worth asking is not "Is PAT-Scan better than MRE?" but rather "Is PAT-Scan adequate for applications where MRE is unavailable?" This framing positions the technology as a complement to existing methods rather than a replacement.

### Potential Impact on Science and Healthcare

Successfully developing PAT-Scan would advance both fundamental understanding in computational mechanics and practical capabilities in clinical medicine. From a scientific perspective, the work contributes methodology for inverse problems by showing that mesh-based physics-informed neural networks can efficiently decouple forward and inverse problem solving. The framework addresses a central challenge: how to use the complementary strengths of classical numerical methods and modern deep learning without sacrificing the rigor of either.

The mesh-based PINN architecture is a distinct paradigm from the meshfree physics-informed neural networks popularized by Karniadakis and colleagues. Where meshfree PINNs embed partial differential equations directly into the neural network loss function through automatic differentiation, the mesh-based approach delegates the well-posed forward problem to established finite element solvers while reserving neural networks for the ill-posed inverse mapping. This decoupling improves computational efficiency by using sparse linear algebra rather than backpropagating through PDE residuals, strengthens physical guarantees by exactly satisfying equilibrium to numerical precision, and allows modular development where improvements to either component can proceed independently.

Beyond tissue mechanics, this framework extends naturally to other solid mechanics inverse problems. Structural health monitoring could use the same principles for damage localization in composite materials. Geophysics applications might use surface seismic measurements to image subsurface inclusions. Materials science could apply the approach to void detection in additively manufactured parts. The unifying principle--systematic interrogation from multiple angles combined with physics-informed learning--applies broadly across domains where direct interior access is difficult or impossible.

From a healthcare perspective, PAT-Scan provides a pathway toward accessible quantitative tissue stiffness measurement in settings where current technology cannot reach. The 200-fold cost reduction compared to MRE is not incremental but potentially transformative. Whether this potential translates to clinical impact depends critically on experimental validation demonstrating that synthetic training transfers to real tissue measurements--the central objective of Aim 3.

---

## C. Innovation

### Innovation Framework

PAT-Scan introduces several methodological innovations that distinguish it from prior work in inverse elastography. We reformulate the traditional continuous material field estimation problem as a geometric segmentation task for systems with discrete material regions, dramatically reducing dimensionality while maintaining clinical relevance. We employ a hybrid physics-machine learning architecture that decouples finite element forward solving from neural network inverse learning. And we implement a systematic interrogation strategy inspired by computed tomography, applying forces sequentially at varied angular positions while incorporating universal geometry handling that adapts automatically to circular, elliptical, or irregular inclusion shapes.

These innovations combine to achieve computational cost reductions of 10 to 100 times compared to iterative optimization methods while maintaining physical plausibility through exact finite element constraint satisfaction.

### Innovation 1: Geometric Inverse Problem Reformulation

Traditional elasticity inverse methods, including the mechanics-based tomography approach of Goenezen and colleagues and the adjoint-based techniques of Oberai and coworkers, formulate the problem as continuous material field estimation. The goal is to recover spatially-varying Young's modulus E(x,y) at every element throughout the domain--a high-dimensional optimization problem requiring careful regularization to prevent overfitting. With typical finite element meshes containing 1,000 to 10,000 elements, the unknown parameter space has corresponding dimensionality.

PAT-Scan reformulates the problem for systems with piecewise constant material properties: a stiff inclusion embedded in a soft background. Rather than solving for continuous E(x,y), the geometric inverse problem seeks to identify the inclusion boundary--the curve separating the two material regions. This reformulation reduces the unknown from a continuous field requiring thousands of parameters to a geometric boundary curve requiring perhaps 100 parameters, achieving dramatic dimensionality reduction while preserving the information most relevant for clinical decision-making.

The key insight is that many clinically important applications are fundamentally segmentation tasks rather than continuous field estimation problems. Tumor detection asks "Is there a stiff nodule present? Where is it located? How large is it?" These questions concern geometric localization--shape, size, and position--not precise Young's modulus quantification at every spatial point. Similarly, tissue engineering assessment seeks to identify which scaffold regions have been remodeled into stiffer tissue. In each case, the inclusion-background distinction constitutes the diagnostic feature, making binary segmentation the appropriate problem formulation.

This reformulation lets us draw on U-Net's proven strength in medical image segmentation. Originally developed by Ronneberger and colleagues for cell detection in microscopy images, U-Net has become the architecture of choice for pixel-wise classification in biomedical imaging. By framing stiffness reconstruction as segmentation rather than regression, we access this mature body of knowledge while dramatically simplifying the inverse problem.

The distinction from Goenezen's mechanics-based tomography deserves attention, as this is the closest prior work. Goenezen solves for continuous E(x,y) at every element using iterative adjoint optimization--a general formulation applicable to arbitrary material heterogeneity. PAT-Scan reframes the problem specifically for two-component systems as geometric segmentation. Goenezen's approach is computationally intensive with risk of local minima; our learned U-Net mapping achieves fast single-forward-pass inference after training.

When should one use the geometric inverse formulation versus the full continuous inverse problem? The answer depends on the physical system. For discrete pathologies--tumor nodules with distinct boundaries, manufactured defects with sharp edges--geometric segmentation is appropriate and computationally efficient. For gradual spatial variations--fibrosis progression showing smooth stiffness gradients, compositionally-graded materials--the full inverse problem becomes necessary. PAT-Scan's contribution is recognizing when reformulation to the geometric problem suffices and exploiting that structure.

### Innovation 2: Mesh-Based Physics-Informed Neural Network Architecture

The term "physics-informed neural network" encompasses two distinct computational paradigms that differ fundamentally in how they couple physics constraints with neural network learning. Clarifying this distinction is essential for understanding PAT-Scan's methodological contribution.

Meshfree physics-informed neural networks, as developed by Raissi, Karniadakis, and colleagues, embed partial differential equations directly into the neural network training loss through automatic differentiation. The network learns to approximate the solution field while simultaneously satisfying governing equations at collocation points sampled throughout the domain. The loss function combines data-driven terms with physics-driven terms penalizing PDE residuals: L = L_data + lambda times L_PDE. This approach handles complex geometries without traditional discretization and has achieved impressive results across fluid mechanics, solid mechanics, and heat transfer.

Meshfree PINNs face computational challenges, though, when applied to inverse problems in structural mechanics. The network must simultaneously solve the forward PDE and identify unknown material parameters--a coupled optimization that can be expensive and difficult to converge. Each training iteration requires computing PDE residuals through automatic differentiation of network outputs, adding substantial computational overhead. Physical accuracy is approximate, depending on how well the loss minimization succeeds; typical residuals achieve magnitudes of 10^-3 to 10^-5 rather than the 10^-12 precision available from direct solvers.

Our mesh-based PINN approach follows a different philosophy: decouple the forward and inverse problems, using each computational method for its appropriate task. The forward model uses traditional finite element analysis to solve the well-posed linear elasticity problem F = KU. Forces F are applied at boundary nodes, the stiffness matrix K is assembled from known element properties, and displacements U are computed via direct sparse Cholesky decomposition. This FEM component is established computational mechanics--highly efficient through sparse linear algebra, guaranteed to satisfy equilibrium to numerical precision, and requiring only seconds of computation for typical mesh sizes.

The inverse model employs a U-Net neural network to learn the ill-posed mapping from measured boundary displacements to material property fields. The network processes displacement data as a two-channel image and produces a one-channel material property map. Training proceeds without explicit PDE residuals in the loss function. Instead, physics is encoded implicitly through the FEM-generated training data, which inherently satisfies mechanical equilibrium, and explicitly through Total Variation regularization that preserves sharp material interfaces.

This decoupled architecture offers concrete advantages. Computational efficiency improves dramatically because finite element forward solves use sparse linear algebra rather than backpropagating through PDE residuals. In practice, this translates to FEM solution times of seconds versus minutes to hours for meshfree PINN training--a 10 to 100 times speedup that compounds over the thousands of forward solves needed to generate training datasets.

Physical guarantees strengthen because the forward model exactly satisfies equilibrium equations within numerical precision, whereas meshfree PINNs satisfy equilibrium only approximately. For applications where mechanical equilibrium must hold strictly--structural analysis, safety-critical systems--this exactness is valuable.

Modularity allows independent improvement of either component. Advances in finite element methods--adaptive mesh refinement, nonlinear material models--can be incorporated without modifying the neural network. Conversely, advances in deep learning--transformer architectures, diffusion models--can be substituted for U-Net without changing the FEM component.

Recent work corroborates this hybrid approach. Wu and colleagues developed JAX-SSO in 2024, showing that differentiable finite element analysis combined with neural networks allows efficient structural optimization. Xue and coworkers created JAX-FEM in 2023, a GPU-accelerated differentiable 3D finite element solver for automatic inverse design. These efforts recognize the value in combining classical discretization methods with modern differentiable programming.

Our implementation includes an option for fully differentiable physics-informed training through the `unet_forward_model_differentiable.py` module. This component assembles the FEM stiffness matrix from predicted material properties, solves for predicted displacements, and computes residuals against measured displacements. This allows physics-informed training when desired while maintaining flexibility for purely data-driven learning when the implicit physics encoding suffices.

I want to frame this decoupling as an intentional design choice rather than a compromise. We are not "settling for mesh-based because meshfree is too difficult." We are strategically separating the well-posed forward problem (best handled by mature FEM solvers) from the ill-posed inverse problem (where neural networks excel). This separation exploits the best capabilities of each approach.

### Innovation 3: CT-Inspired Angular Scanning with Universal Geometry Handling

The measurement strategy in PAT-Scan draws explicit inspiration from computed tomography: interrogating the sample from multiple angles reveals interior structure from boundary projections. Rather than applying a single force configuration and attempting reconstruction from limited data, we systematically apply force pairs at varying angular positions, collecting displacement measurements after each loading. This builds directly on the multiple loading insight pioneered by Konofagou and Harrigan.

Their key contribution showed that applying multiple distinct loads rather than a single compression "increased the ratio of measurements to fitted parameters, which made the method less sensitive to random errors." Palpation tomography experiments achieved noise reduction by a factor of two using nine loading cases compared to single-load elastography. PAT-Scan extends this concept through systematic angular scanning, sweeping from 1 to 20 force pair configurations with controlled angular spacing.

Our angular scanning protocol reveals progressive information accumulation as force pair count increases. With a single force pair applied at opposing boundary points, displacement concentrates near the force application locations, providing limited interrogation of the inclusion. Adding force pairs at 90-degree intervals increases boundary coverage. As the number grows to 10 and eventually 20, deformation becomes increasingly uniform around the boundary, with the displacement pattern encoding richer information about the inclusion's shape, size, and position. Animation of the scanning process makes this evolution visually apparent: sparse angular sampling yields localized deformation, while dense angular sampling produces near-complete boundary interrogation reminiscent of CT's many-angle projections.

A critical finding emerged during development: boundary displacement completeness matters more than force magnitude variation. Experiments comparing partial boundary coverage versus complete boundary coverage showed dramatic reconstruction accuracy improvements with complete coverage, even when applied forces were smaller. This observation suggests that practical measurement systems must enable multi-angle access to the sample boundary--through rotating stages, mirror systems, or multiple synchronized cameras.

The decision to exclude radial force scanning--varying force magnitude at fixed angular positions--is an intentional design choice grounded in linear elasticity physics. Force scaling by a constant factor yields linearly scaled displacement response without providing new geometric information. Only angular diversity creates genuinely new interrogation configurations that constrain the inverse problem.

Universal geometry handling is an innovation that deserves emphasis. The same U-Net architecture functions across circular, elliptical, and irregular inclusion geometries without architectural modifications. Training scripts automatically detect geometry type from dataset metadata and proceed with identical network architecture and training procedures. The network learns to recognize geometric features directly from displacement field patterns.

Level-set post-processing complements this universality by extracting boundaries from soft neural network predictions without geometric assumptions. The three-step pipeline--Gaussian smoothing, soft thresholding via sigmoid, and contour extraction--works for arbitrary topologies. Convex or non-convex shapes, smooth or spiculated boundaries all yield to the same post-processing procedure because level-set methods make no prior assumptions about boundary structure.

From a practical hardware perspective, the force application strategy requires only simple actuation: calibrated pushers or compressed air jets directed radially inward. This is substantially simpler than the vibration exciters needed for Visual Vibration Tomography or the ultrasound transducers required for ultrasound elastography.

### Innovation 4: Synthetic-to-Real Training Pathway

A common criticism of machine learning approaches in computational mechanics centers on data requirements. Clinical tissue measurements are expensive to collect, requiring patient recruitment, imaging protocol development, and regulatory approval. Annotating such data with ground truth material properties is even more challenging--how does one know the true Young's modulus distribution when that is precisely what we seek to measure?

PAT-Scan circumvents this data bottleneck by using the finite element forward model as a synthetic data generator. The FEM solver accepts arbitrary geometric descriptions and material property distributions, solves the linear elasticity equations, and outputs perfectly labeled training pairs. This allows unlimited data generation with controlled parameter sweeps across stiffness contrasts, inclusion sizes, shapes, loading configurations, and noise levels.

This synthetic training approach is defensible for proof-of-concept development because the forward model accurately represents the underlying physics of quasi-static tissue deformation. Linear elasticity has been validated experimentally across countless materials. The equilibrium equations derive from fundamental force balance, not empirical approximation. Finite element discretization has mature mathematical foundations with well-understood convergence properties. Therefore, synthetic FEM data captures the essential physical relationships the neural network must learn--with fidelity limited only by the small-deformation linear elasticity assumptions.

The pathway from synthetic to real validation follows a staged strategy. First, training and testing occur entirely on FEM-generated data to establish that the inverse solver works in principle. This stage is currently complete for circular geometries and in progress for irregular shapes. Second, silicone tissue-mimicking phantoms provide intermediate validation in controlled laboratory conditions. Real measurement noise enters the system, but geometric ground truth remains available. Third, ex-vivo tissue samples introduce biological variability while maintaining some level of ground truth. Fourth, clinical measurements on patients would provide ultimate validation, though this lies well beyond PhD scope.

Transfer learning bridges the synthetic-to-real gap efficiently. A neural network pre-trained on millions of synthetic examples develops internal representations of displacement-to-stiffness mappings. Fine-tuning on a small dataset of real measurements adapts the network to measurement-specific noise and systematic biases. This data-efficient adaptation has proven successful across computer vision, and the same principle applies to mechanics inverse problems.

---

## D. Research Approach

### Methodological Framing

The research approach develops PAT-Scan as a computational inverse problem solver using finite element-generated synthetic data. This methodology allows systematic exploration of geometric complexity, noise sensitivity, and algorithmic performance before committing resources to experimental validation. The approach is scientifically defensible because the forward model--linear elasticity solved via finite elements--accurately represents quasi-static tissue deformation under small-strain conditions.

Each aim is presented following a consistent structure: methods describe the technical approach, results summarize current achievements with honest assessment of completion status, and discussion contextualizes findings. Clear status indicators (completed, in progress, or planned) maintain transparency about what has been accomplished versus what remains.

---

## Specific Aim 1: Proof-of-Concept for Geometric Inverse Problem

### Problem Formulation

Aim 1 addresses the geometric inverse problem: recovering the shape, size, and location of a stiff inclusion embedded in a soft background when displacement measurements are available only at the boundary. The material distribution is assumed piecewise constant--two distinct Young's modulus values separated by a sharp interface. This models clinically relevant scenarios where a stiff tumor nodule is embedded in compliant surrounding tissue.

This formulation differs from the full inverse problem where material properties vary continuously as E(x,y). The geometric inverse problem is more tractable because it reduces the unknown from a continuous field to a finite set of geometric parameters. But recovering this boundary from boundary-only measurements remains challenging due to the ill-posed nature of elasticity inverse problems.

The canonical test case employs a circular domain with outer radius R_outer = 1.0 containing a concentric circular inclusion of radius R_inner = 0.3. Material properties are E_background = 1.0 and E_inclusion = 10.0, a 10:1 stiffness contrast typical of soft tissue-tumor systems. Poisson's ratio is set to 0.3 for both materials, and all nodes within the inclusion are fixed at zero displacement to mimic a rigid core.

### Aim 1a: FEM Forward Model Development

Finite element modeling provides the computational foundation for both training data generation and physics validation. The mesh generation strategy employs a structured polar grid, dividing the circular domain into 20 radial layers and 40 angular segments. This structured approach ensures consistent element quality and simplifies material assignment through geometric tests based on radial distance. The resulting mesh contains approximately 800 nodes and 1,500 triangular elements.

The finite element formulation employs plane stress assumption, appropriate for thin samples. Each triangular element contributes a 6x6 stiffness matrix relating the displacements of its three nodes to the forces acting on those nodes. Assembly into the global sparse stiffness matrix K proceeds through the standard direct stiffness method.

Force application follows a paired strategy to prevent rigid body motion. Equal and opposite radial forces are applied at boundary nodes at angle theta and theta + pi, pushing inward toward the inclusion. This balanced loading ensures force and moment equilibrium, eliminating the need for additional constraints. The magnitude of applied forces is constrained by a geometric validity requirement: the deformed boundary must not penetrate the fixed inclusion. A penetration checking function evaluates the minimum distance between deformed boundary nodes and the inclusion, determining the maximum allowable force magnitude.

Solution of the linear system KU = F proceeds through direct sparse linear algebra. Boundary conditions are enforced by modifying the stiffness matrix to impose zero displacement at inclusion nodes. The sparse Cholesky solver computes displacements with numerical precision near 10^-12 in double precision. Solution times remain on the order of seconds even for the largest meshes considered.

Validation confirmed expected physical behaviors. Displacement fields achieve maximum values at force application points and decay with distance. The stiff inclusion experiences minimal deformation while the soft background deforms substantially, correctly reflecting the 10:1 stiffness contrast. Automated testing scripts verified solution stability and penetration constraint satisfaction across the operating range.

### Aim 1b: Dataset Generation via Angular Scanning

The angular scanning protocol systematically explores force configurations to generate training data with varying richness of boundary interrogation. Beginning from a single force pair at 0 and 180 degrees, the number of pairs is incrementally increased up to 20, with angular spacing ensuring uniform coverage. For each configuration, the finite element solver computes the equilibrium displacement field, and displacements at boundary nodes are extracted.

Converting irregular boundary node positions to the regular grid format required for neural network training demands interpolation. Displacement components are interpolated onto a 64x64 pixel grid covering the square domain. The ground truth target is a binary material mask, with pixels labeled 1 inside the inclusion and 0 outside. This transforms the mechanical inverse problem into an image segmentation task.

The scanning process generates 20 training samples per geometric configuration. Animation of the process provides visual insight into how deformation patterns evolve with increasing force pair count. With a single pair, displacement localizes near the force application points. As additional pairs are added, the deformation pattern becomes more symmetric and provides richer sampling of the inclusion's mechanical influence. By 20 force pairs, deformation appears nearly uniform around the boundary. This progressive information accumulation mirrors computed tomography's principle.

### Aim 1c: U-Net Training and Boundary Extraction

The neural network architecture follows the U-Net design originally developed for biomedical image segmentation. The network accepts the two-channel displacement field (64x64 pixels) and produces a one-channel material property field. The encoder pathway consists of three levels of convolution-ReLU-pooling operations, progressively downsampling while increasing feature channels. The decoder pathway reverses this through transposed convolutions, with skip connections linking encoder and decoder levels.

The training loss combines mean squared error with Total Variation regularization: L = ||E_pred - E_true||^2 + lambda_TV times the sum of gradient magnitudes. The MSE term provides supervised learning; the TV term penalizes spatial gradients, implementing preference for piecewise constant solutions. This regularization is crucial because the expected solution has sharp transitions between inclusion and background with uniform values within each region.

The connection between Total Variation regularization and the level-set post-processing deserves attention. Without TV penalty, neural networks tend to produce soft, gradually varying transitions. While MSE alone might achieve low loss values, the resulting smooth fields lack the sharp separation needed for clean boundary extraction. TV regularization guides the network toward naturally sharp transitions, pre-conditioning the output for robust level-set extraction.

Hyperparameter optimization proceeded through systematic grid search over learning rate, TV weight, and thresholding temperature. Results indicated that learning rate 10^-4, TV weight 0.005, and temperature 3000 provided consistent reconstruction quality across different configurations.

The post-processing pipeline transforms soft predictions into hard geometric boundaries through three steps. Gaussian smoothing suppresses noise. Soft thresholding via sigmoid sharpens the transition between materials, with temperature controlling steepness. Contour extraction identifies the 0.5 level set, yielding a polygon representation of the inclusion boundary.

Training convergence was monitored through loss curves and visual inspection. The combined loss typically decreased from initial values around 0.1 to final values near 0.01 over 5,000 iterations. Animation shows the network initially predicting a nearly uniform material field, then gradually developing a circular feature that progressively sharpens to match ground truth.

Reconstruction accuracy has been evaluated qualitatively through visual comparison of predicted and true boundaries. The level-set extraction successfully identifies sharp inclusion boundaries from soft outputs across test cases. Predicted shapes closely match ground truth in size and position. Quantitative metrics including Dice coefficient and Hausdorff distance remain to be computed from existing checkpoint files--the current status is qualitative validation demonstrating proof-of-concept, with rigorous quantitative benchmarking ongoing.

One consistent observation warrants discussion: the recovered stiffness value within the inclusion tends to be 60-80% of the target rather than exactly matching. This systematic underestimation likely reflects fundamental physics of boundary-only measurement rather than algorithmic deficiency. Displacement measurements strongly constrain geometric parameters--where is the inclusion, how large--because these directly affect boundary response patterns. Absolute stiffness magnitude is more weakly constrained because a stiffer smaller inclusion can produce similar boundary displacements to a softer larger one.

For clinical tumor detection, this underestimation may be acceptable. The diagnostic questions concern geometric localization rather than precise Young's modulus quantification. Relative stiffness contrast matters more than absolute values, particularly given tissue-to-tissue variability in mechanical properties.

### Discussion: Aim 1 Achievements and Limitations

Aim 1 successfully demonstrated proof-of-concept for the geometric inverse problem on centered circular inclusions. The integrated pipeline--finite element forward model, angular scanning dataset generation, U-Net training with TV regularization, and level-set boundary extraction--functions as designed and produces physically plausible reconstructions.

Key achievements include establishing the mesh-based PINN framework with clear separation between forward solving and inverse mapping. The approach validates that boundary-only displacement measurements contain sufficient information for inclusion reconstruction, dramatically simplifying hardware requirements. TV regularization is effective for geometric inverse problems, producing naturally sharp transitions. Hyperparameter optimization identified training configurations that generalize across stiffness contrasts.

The work also revealed important limitations. The two-dimensional plane stress assumption limits current applicability, though 3D extension in Aim 3a is a natural generalization. Restriction to centered circular inclusions is an artificial simplification; irregular geometry support in Aim 2a relaxes this constraint. All validation currently occurs on synthetic data--experimental validation with physical phantoms will provide the critical test.

Boundary displacement completeness emerged as a critical requirement. Experiments with partial coverage showed dramatically degraded accuracy compared to complete coverage, even when the completeness case used smaller forces. Practical measurement systems must provide multi-angle access to the sample boundary.

The current status is best characterized as "core methodology demonstrated" rather than "fully completed and validated." The foundation is solid; comprehensive validation continues.

---

## Specific Aim 2: Extension to Irregular Geometries and Advanced Architectures

### Overall Goal and Status

Aim 2 extends PAT-Scan beyond canonical circular inclusions to handle realistic geometric complexity and explore alternative neural architectures. Aim 2a, addressing irregular off-centered inclusions, is approximately 60% complete with geometry generation, solver adaptation, and universal training framework implemented. Aim 2b, proposing Fourier feature MLPs for continuous material field reconstruction, is planned but not implemented. Aim 2c, developing a comprehensive benchmarking library, is similarly planned.

### Aim 2a: Irregular Off-Centered Inclusions

Real tumors exhibit geometric features that challenge reconstruction algorithms: spiculated boundaries, non-convex shapes, and off-centered positions that break convenient symmetries. Aim 2a introduces this complexity through Fourier mode perturbations applied to a circular base shape, creating inclusions whose boundary radius varies with angle according to r(theta) = R_base times (1 + sum of Fourier terms). The number of modes typically equals 6, and coefficients are randomly sampled to generate diverse shapes. An irregularity parameter controls perturbation amplitude, with clamping to prevent self-intersection.

Additionally, the inclusion center can be displaced by offsets ranging from -0.5 to 0.5 in normalized coordinates. This creates fundamentally asymmetric deformation patterns, providing a more demanding test of the algorithm's ability to handle geometric complexity.

Mesh generation required generalizing the material assignment logic. The algorithm transforms each element centroid to local coordinates centered on the inclusion, computes the angular position, evaluates the Fourier series to determine the boundary radius at that angle, and compares the element's distance to this threshold.

Remarkably, the finite element solver itself required no changes. The same linear elasticity formulation, element stiffness computation, global assembly, and sparse solution apply regardless of inclusion shape. Geometric complexity is entirely absorbed into mesh generation, validating a key design principle of the decoupled architecture.

The training framework incorporates automatic geometry detection by inspecting dataset metadata. The same U-Net architecture functions without modification on irregular cases--no architectural changes, no geometry-specific hyperparameters. The network learns to extract geometric features directly from displacement patterns.

Preliminary results on irregular geometries show promise while revealing increased sensitivity to hyperparameters. The optimal TV weight and smoothing parameters differ from values that worked for circles, suggesting irregular boundaries require more careful regularization tuning. Level-set extraction handles non-convex shapes successfully.

Future work includes systematic training across 10-20 irregular samples, computing quantitative accuracy metrics, and investigating transfer learning: does a network trained on circular geometries generalize to irregular ones without retraining?

### Aim 2b: Fourier Features MLP for Full Inverse Problem (Planned)

While Aims 1 and 2a address geometric segmentation with piecewise constant properties, many applications require reconstructing continuous E(x,y). Fibrosis progression shows gradual stiffness increases. Compositionally graded materials exhibit smoothly varying properties. For these problems, the geometric approach falls short.

Implicit neural representations offer an alternative to grid-based U-Net. Rather than discretizing onto a fixed grid, implicit representations parameterize a continuous function E(x,y) that can be queried at arbitrary locations. Fourier feature mapping addresses MLPs' difficulty learning high-frequency variations by embedding coordinates as periodic functions.

The proposed architecture accepts four inputs per query point: coordinates and displacements at that location. Fourier embedding expands these to high dimensions, then an MLP maps to Young's modulus. Training data generation requires creating synthetic samples with continuous E(x,y) distributions.

Expected outcomes include demonstrating feasibility of continuous reconstruction and benchmarking MLPs against U-Net. Anticipated challenges include longer training times and potentially larger dataset requirements.

### Aim 2c: Benchmarking Library (Planned)

Systematic validation requires diverse test cases spanning geometric and material complexity. The proposed library would include 20-50 samples covering circular, elliptical, irregular, and multiple-inclusion configurations. Stiffness contrast would vary across clinically relevant ranges. Noise robustness testing would add Gaussian noise at various levels.

The benchmark dataset would be published for community use, enabling fair comparison between inverse elastography methods that currently use different test problems and metrics.

---

## Specific Aim 3: Incorporating Realism and Experimental Validation (Future Work)

Aim 3 represents the transition from computational proof-of-concept to experimental feasibility. This aim is planned for PhD years 3-4 and is outlined here to demonstrate a viable path to validation exists.

### Aim 3a: 3D Extension

Extension to three dimensions requires generalizing the solver to tetrahedral elements and volumetric domains. Mesh generation tools like TetGen or Gmsh can create unstructured tetrahedral meshes. The computational challenge scales significantly--3D problems typically cost 10 to 100 times more than equivalent 2D problems.

For the neural network, either 3D U-Net architectures or implicit MLP representations could be employed. The MLP approach may be particularly attractive because it avoids memory explosion associated with volumetric convolutions.

### Aim 3b: CT Integration for Anatomically-Informed Models

Medical imaging provides patient-specific geometry that could constrain reconstructions. CT scans yield outer boundary geometry after segmentation. A hybrid approach might constrain tissue-type regions identified from CT to have mechanical properties within literature-reported ranges while allowing PAT-Scan to refine local variations.

### Aim 3c: Experimental Validation Roadmap

The pathway from simulation to experimental validation follows three phases. Phase 1 employs silicone tissue-mimicking phantoms. The background material would be soft silicone with Young's modulus 20-40 kPa, while embedded inclusions would use stiffer silicone achieving 5-10 times stiffness contrast.

The measurement system combines calibrated force application with Digital Image Correlation displacement measurement. A load cell with 0.1-10 N range provides reproducible force application. Stereo DIC captures surface displacement at sub-pixel resolution. Total hardware cost is estimated at $8,000-10,000.

Success metrics would include Dice coefficient exceeding 0.75, center localization error below 15% of inclusion radius, and Hausdorff boundary error below 10% of inclusion size.

Phase 2 introduces biological variability through ex-vivo tissue samples. Phase 3, clinical measurements on patients, would require IRB approval and lies beyond PhD scope.

---

## Assumptions and Scope

### Material Model Assumptions

The current implementation assumes linear elasticity with small deformations. This approximation is justified when strains remain below 5%, conditions satisfied in many quasi-static palpation scenarios. Gentle compression of breast tissue during examination typically induces 2-4% strain. Linear elasticity provides reasonable accuracy in these regimes.

Real soft tissues exhibit nonlinear stress-strain behavior at larger deformations. Extending to nonlinear constitutive laws is technically feasible but adds computational cost by factors of 10 to 100.

### Geometric Assumptions

Outer boundary geometry is assumed known from direct measurement or imaging. Material distribution in Aims 1-2a is restricted to piecewise constant, though Aim 2b relaxes this.

### Measurement Assumptions

Boundary displacement completeness emerged as critical. Reconstruction accuracy degrades substantially with partial boundary data. Practical systems must enable access to a substantial fraction of the boundary.

Displacement measurement accuracy is assumed within Digital Image Correlation capabilities, typically 0.01-0.05 pixel resolution. Systematic errors from calibration are not currently modeled but will enter during experimental validation.

### Computational Assumptions

Finite element mesh quality is maintained through structured generation algorithms. The FEM solution is assumed converged to numerical precision, guaranteed for direct sparse solvers.

---

## Conclusion

### Summary of Progress

This comprehensive exam presents Palpation-Assisted Tomography, a computational framework for reconstructing tissue stiffness from boundary displacement measurements using physics-informed neural networks. The work demonstrates a mesh-based PINN architecture that decouples forward and inverse problems, achieving computational efficiency through sparse linear algebra while maintaining physical rigor.

Aim 1 established proof-of-concept methodology for the geometric inverse problem. The finite element forward model generates high-fidelity synthetic training data. The U-Net inverse solver reconstructs inclusion boundaries with qualitative accuracy confirmed through visual comparison. Level-set post-processing successfully extracts crisp boundaries from soft predictions.

Aim 2a extended the framework to irregular off-centered inclusions using Fourier perturbations. Preliminary results show that the same neural network architecture generalizes across geometric complexity without modification--a genuine empirical finding about the representational capacity of displacement-based learned mappings.

Aim 3 provides a detailed roadmap for experimental validation with equipment costing $8,000-10,000--a 200-fold reduction compared to MRE.

### Central Innovations

Three methodological innovations distinguish PAT-Scan. Problem reformulation from continuous field estimation to geometric segmentation reduces dimensionality while maintaining clinical relevance. The mesh-based PINN architecture delegates the well-posed forward problem to finite element solvers while reserving neural networks for the ill-posed inverse mapping. Systematic interrogation inspired by CT applies forces at varied angular positions, with universal geometry handling that functions across different inclusion shapes.

### Broader Impact

The framework extends naturally to other solid mechanics inverse problems. Structural health monitoring could use the same principles for damage localization. Geophysics applications might employ surface measurements to image subsurface inclusions. Materials science could apply the approach to quality control in additive manufacturing.

### Limitations and Path Forward

Current limitations include restriction to two-dimensional geometries and validation only on synthetic data. The gap between synthetic proof-of-concept and experimental validation is the most critical limitation to address.

The path forward proceeds through experimental validation with silicone phantoms. Success in Phase 1--achieving Dice coefficients exceeding 0.75 on physical specimens with realistic measurement noise--would validate that synthetic training transfers to real measurements.

The research establishes that boundary measurements combined with systematic interrogation and physics-informed learning can achieve accessible elastography. The algorithmic foundation is solid. The experimental validation will determine whether this foundation supports clinical translation.

---

## References

1. Konofagou, E. E., & Harrigan, T. P. (2003). Palpation Tomography: A New Technique for Modulus Estimation in Elastography. *IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control*, 50(11), 1465-1477.

2. Goenezen, S., Barbone, P., & Oberai, A. A. (2017). Mechanics-Based Tomography: A Preliminary Feasibility Study. *PLOS ONE*, 12(7), e0181804.

3. Bouman, K. L., Ye, V., Dabov, A., Veeraraghavan, A., Baraniuk, R. G., & Kemelmacher-Shlizerman, I. (2022). Visual Vibration Tomography: Estimating Interior Material Properties from Monocular Video. *ACM Transactions on Graphics (SIGGRAPH)*, 41(4), Article 71.

4. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems Involving Nonlinear Partial Differential Equations. *Journal of Computational Physics*, 378, 686-707.

5. Wu, G., Xiao, M., & Wolfe, K. C. (2024). JAX-SSO: A Differentiable Finite Element Analysis Solver for Structural Optimization with Seamless Integration with Neural Networks. *arXiv preprint arXiv:2407.20026*.

6. Xue, T., Adriaenssens, S., & Mao, S. (2023). JAX-FEM: A Differentiable GPU-Accelerated 3D Finite Element Solver for Automatic Inverse Design and Mechanistic Data Science. *Computer Physics Communications*, 291, 108802.

7. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. In *Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, 234-241. Springer.

8. Tancik, M., Srinivasan, P. P., Mildenhall, B., Fridovich-Keil, S., Raghavan, N., Singhal, U., Ramamoorthi, R., Barron, J. T., & Ng, R. (2020). Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains. In *Advances in Neural Information Processing Systems (NeurIPS)*, 33, 7537-7547.

9. Rudin, L. I., Osher, S., & Fatemi, E. (1992). Nonlinear Total Variation Based Noise Removal Algorithms. *Physica D: Nonlinear Phenomena*, 60(1-4), 259-268.

10. [Hughes, T. J. R. FEM textbook - full citation needed]

11. [Zienkiewicz FEM reference - full citation needed]

12. [Mathematical Foundations 1994 - Linear 3D elasticity, ill-posedness theory - full citation needed]

13. [Tissue stiffness values reference - Breast cancer 5-10x stiffness contrast - citation needed]

14. [MRE cost reference - Equipment ~$2M - citation needed]

15. [Osher-Sethian level-set methods reference - full citation needed]

16. [Deep FEM 2024 citation - from PINNs literature folder - full citation needed]

17. [Karniadakis et al. PINNs elasticity 2022 - full citation needed]

18. [Oberai et al. inverse elasticity foundations - full citation needed]

---

**Document Status:**
- Final markdown draft: Complete with fluent academic prose
- Humanized to remove AI signatures while preserving technical quality
- Citations: Core references included with placeholders for remaining sources
- Target length: 12-15 pages achieved
- Ready for LaTeX conversion and final polishing

**Files Generated:**
- `/home/vivekkarmarkar/Python Files/PAT-Scan-Copy/agents/phase2/comps_final_good_humanized.md`

---

**END OF HUMANIZED DRAFT**
