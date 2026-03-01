# Palpation-Assisted Tomography

---

## A. Specific Aims (already done)

---

## B. Significance

### The Clinical Challenge: From Qualitative Touch to Quantitative Maps

Physicians have relied on the sense of touch for medical diagnosis throughout recorded history—palpation represents one of the oldest forms of physical examination. The practice encodes a biomechanical principle: pathological tissue often exhibits markedly elevated stiffness compared to healthy surrounding tissue. Haptic perception provides unique information about physical properties (stiffness, texture, dynamic behavior) that visual perception alone cannot capture (Chase & Follmer, 2019). This explains why clinicians have historically depended on palpation: touch reveals material properties invisible to the eye. When a clinician presses on a suspected breast tumor and feels resistance, they sense the increased stiffness that characterizes many malignancies. Breast cancer lumps, for example, exhibit stiffness values 5 to 10 times greater than surrounding healthy breast tissue (Levental et al., 2009).

Yet manual palpation suffers from serious limitations that restrict its clinical utility:

- **Subjectivity**: Different clinicians may interpret the same finding differently, depending on experience and tactile sensitivity
- **Superficiality**: Detection is limited to lesions within a few centimeters of the surface; deeper pathology escapes detection entirely
- **Non-quantitative**: Palpation provides only qualitative assessment (something feels hard) without numerical stiffness values that could inform treatment planning or monitor therapeutic response

These limitations create a clear clinical need: objective, quantitative tissue mechanical characterization that maintains palpation's conceptual simplicity while overcoming its practical constraints.

### Historical Evolution: From Scalar Measurements to Spatial Maps

The progression from qualitative palpation to quantitative elastography followed a century-long arc of technological development:

**Phase 1: Early Quantification Attempts (1912-1979)**

The first attempts to quantify tissue mechanics focused on measuring scalar stiffness values at specific locations rather than reconstructing spatial maps:

- **Schade (1912)** pioneered apparatus for measuring elastic properties of connective tissue, establishing the foundation for quantitative biomechanics
- **Kirk & Kvorning (1949)** conducted systematic quantitative measurements of elastic properties of skin and subcutaneous tissue using Schade's apparatus, comparing young versus old individuals and demonstrating age-dependent changes in tissue mechanics
- **Sapuntsov et al. (1979)** assessed rheological properties of soft tissues in human limbs to distinguish healthy versus diseased tissue in patients with lymphatic circulation disorders, representing early quantitative tissue mechanics work that preceded the formal field of elastography

These early efforts shared a common characteristic: they quantified **scalar stiffness measures** at specific locations rather than spatial distributions. The leap from scalar measurements to spatial maps required significant algorithmic and computational advances that did not emerge from simply extending the earlier methods.

**Phase 2: Transition to Spatial Maps (Post-WWII to Present)**

The concept of spatially resolved stiffness maps emerged from advances in medical ultrasound imaging following World War II (Newman & Rozycki, 1998). During WWI and WWII, sonar technology was developed for submarine detection using piezoelectric transducers; after the war, military veterans and researchers adapted this technology for medical diagnostics. Key pioneers included George Ludwig (US Navy, 1948) who conducted early tissue experiments, John Wild (British Royal Army Medical Corps veteran) who developed ultrasound probes, and Ian Donald whose landmark 1958 Lancet paper established obstetric ultrasound (Donald, MacVicar & Brown, 1958). This transition represented more than technological progress—it demanded a fundamental reconceptualization of the problem.

Building on ultrasound imaging advances, the formal field of **elastography** emerged in the late 1980s. Lerner and Parker (1987, 1988) presented the first sonoelasticity images using vibration-based methods, and Ophir et al. (1991) introduced compression-based strain imaging and coined the term "elastography" (Ormachea & Parker, 2020). This shift from scalar measurements to spatial maps required both new conceptual frameworks and technical capabilities:

- The algorithmic challenge of reconstructing two-dimensional or three-dimensional property fields from limited measurements required sophisticated inverse problem solvers
- Computational tools became available to implement these algorithms only in recent decades
- The result is a modern elastography landscape emphasizing wave-based methods (MRE, ultrasound elastography) that probe tissue at multiple frequencies or from multiple directions

**Current Elastography Landscape and the Accessibility Gap**

Modern elastography techniques provide remarkable imaging capabilities but at significant cost:

- **Magnetic Resonance Elastography (MRE)** (Muthupillai et al., 1995): ~$1-3M equipment (MRI system with MRE capability), full-body volumetric imaging, high spatial resolution (mm-scale), gold standard for elastography
- **Ultrasound Elastography** (Ophir et al., 1991): ~$50-200K (mid-to-high range systems with elastography), operator-dependent, limited depth penetration (~10 cm), real-time imaging capability

While these techniques excel in well-resourced settings, a significant accessibility gap remains: resource-constrained healthcare settings lack access to quantitative stiffness measurement despite its diagnostic value. The cost barrier limits clinical adoption in exactly those settings where low-cost screening tools could have the greatest public health impact.

**We propose PAT Scan** as a methodology inspired by the clinical practice of palpation but envisioned to be compatible with minimal equipment requirements. The essence of PAT Scan is developing an algorithm that tackles a mathematical challenge known as an inverse problem, potentially enabling tissue stiffness assessment at a fraction of current equipment costs.

### The Inverse Problem Challenge: From Well-Posed Forward to Ill-Posed Inverse

The mathematical foundation of elastography rests on the relationship between material properties, applied forces, and resulting deformations. In the forward direction, this relationship is well-established:

**Forward Problem (Well-Posed):**
- **Given**: Material properties E(x, y), geometry, applied forces F
- **Compute**: Displacement field U throughout the domain
- **Governing equation**: Linearized Elasticity matrix form, F = KU, where K is the global stiffness matrix assembled from material properties
- **Solution method**: Finite Element Method (FEM) with direct linear solvers
- **Computational status**: Exact solution (within discretization error), computationally efficient, stable

The forward problem is straightforward: we know what we have (materials, forces), and we calculate what happens (displacements). The inverse problem flips this relationship—and the difficulty increases dramatically.

**Inverse Problem (Ill-Posed):**
- **Given**: Measured displacements U, applied forces F
- **Compute**: Material property distribution E(x, y) throughout the domain
- **Mathematical challenges**:
  - **Non-uniqueness**: Multiple material distributions may produce similar boundary displacements
  - **Noise sensitivity**: Small measurement errors can lead to large reconstruction errors
  - **Underdetermined**: Boundary measurements provide far fewer constraints than unknown material property values (especially for sparse boundary-only data)
- **Required additional information**: Regularization (smoothness constraints, sparsity priors), physics constraints, structural priors (e.g., piecewise constant distributions)

The boundary-only measurement constraint makes the inverse problem particularly challenging.

This is where our reformulation of the problem as geometric inverse problem becomes crucial: clinicians might be aware of the materials characterizing the inclusion and background tissue and might involve scenarios with discrete inclusions in background tissue rather than arbitrary continuous property fields - by exploiting structural knowledge, we convert the problem to a purely geometric inverse problem focused on extracting the location, size and shape of the inclusion while dramatically reducing the effective dimensionality of the problem.

### Prior Work: Positioning PAT Scan in the Landscape

To understand where PAT Scan fits within the broader elastography landscape, we must examine how existing approaches tackle the inverse problem—and where they encounter limitations that motivate our alternative formulation.

#### Mechanics-Based Tomography (MBT): Iterative Optimization Approach

**Palpation Tomography (Konofagou & Harrigan, 2003)** introduced the concept of using boundary displacement measurements from surface loading to reconstruct interior stiffness distributions, drawing inspiration from the clinical practice of sequential finger loading during manual palpation. **Mechanics-Based Tomography (Mei et al., 2017; based on earlier work by Goenezen et al., 2011)** extended this approach with rigorous mathematical formulation and demonstrated feasibility on simulated tissue-mimicking scenarios.

The MBT framework minimizes a data-fitting objective:

min ||U_measured(E) - U_predicted(E)||²

where U_predicted is computed by solving the forward FEM problem for a candidate material distribution E. The optimization uses adjoint-based gradient computation (Plessix, 2006) to iteratively update the material property field until predicted displacements match measurements.

**Strengths of MBT:**
- Mathematically rigorous inverse problem formulation
- Successfully demonstrated on silicone phantoms with known ground truth
- Makes minimal assumptions about material property structure

**Core Limitations:**

**1. Structural Insight Gap:**

MBT treats the inverse problem as continuous field optimization over infinite-dimensional space: E(x, y) ∈ ℝ₊ for all points (x, y) in the domain. However, many clinically relevant scenarios involve **piecewise structures**: a discrete pathological inclusion embedded in relatively homogeneous background tissue. For such cases:

**2. Force Application Strategy Gap:**

**Critically, the MBT literature does not explore optimal force application strategies.** Mei et al. (2017) explicitly state this limitation in their paper—the authors focus on the inverse problem formulation and optimization but do not investigate how the choice, number, spacing, or sequencing of applied forces affects reconstruction quality. This represents a significant methodological gap: even a perfectly formulated inverse solver will fail if the input data (boundary displacements from specific force configurations) does not contain sufficient information to constrain the solution.

PAT Scan directly addresses this gap by developing a **CT-inspired force application strategy** with analytical criteria for angular spacing derived from Boussinesq decay analysis. This systematic approach to data acquisition is a primary novel contribution absent from prior palpation elastography work.

**Returning to Structural Insight:** For piecewise material distributions:
- Inclusion stiffness E_inclusion = α × E_background for some multiplicative contrast factor α
- The material distribution is characterized by an inclusion boundary (a one-dimensional curve in 2D) plus two scalar values (E_background, α)

**Our Key Recognition:** This is fundamentally a **segmentation problem** rather than continuous field reconstruction:
- Identify the inclusion boundary (binary segmentation: background versus inclusion)
- Estimate stiffness contrast (parameter estimation: α value)

This reformulation offers significant advantages:
- **Dimensionality reduction**: From E(x, y) everywhere → binary mask M(x, y) + two scalars
- **Natural architecture alignment**: U-Net (Ronneberger et al., 2015)—with ~88,000 citations making it one of the most influential neural network architectures ever published—was specifically designed for biomedical image segmentation, and our insight is that tissue inclusion identification is fundamentally a segmentation problem
- **Workflow standardization**: Optimization routines from standard Deep Learning libraries
- **Robustness to initialization**: Neural networks trained end-to-end are less sensitive to initial guesses than iterative optimizers

Metaphorically, we shift from "learning continuous dials" (MBT adjusting E(x, y) values everywhere) to "flipping binary switches" (U-Net identifying inclusion boundaries).

#### Visual Vibration Tomography (Bouman et al., 2022): Dynamic Excitation and Optical Measurement

Bouman and colleagues demonstrated a remarkable capability: estimating heterogeneous material properties (Young's modulus and density) of 3D objects from monocular video of their surface vibrations. The approach identifies image-space vibration modes from sub-pixel motion tracking, then solves an inverse problem to recover voxelized material property distributions that would produce the observed modal behavior.

**Key methodological elements:**
- **Domain**: Dynamic vibration response (elastodynamics) rather than quasi-static loading
- **Measurement**: Monocular camera capturing sub-pixel surface displacements via phase-based motion analysis (Wadhwa et al., 2013)
- **Physics**: Modal decomposition using finite element eigenanalysis (Ku = ω²Mu)
- **Inverse formulation**: Optimization to find material properties that minimize ||Ku - ω²Mu||² subject to constraints

**Strengths:**
- Non-contact measurement (optical rather than mechanical transducers)
- Video-based data acquisition using ubiquitous cameras
- Successfully demonstrated on both simulated and real objects (drum heads, Jello cubes)

**Relationship to PAT Scan:**

Visual Vibration Tomography and PAT Scan occupy complementary domains:
- **Physics regime**: VVT uses dynamic vibration (elastodynamics), PAT Scan uses quasi-static palpation forces (static elasticity)
- **Excitation**: VVT requires vibration induction (mechanical shaker or impact), PAT Scan uses controlled static force pairs
- **Model reduction**: VVT exploits modal decomposition of vibration modes, which inspires our 2-component material model and use of U-Net as an appropriate surrogate

The VVT framework demonstrates that physics-constrained inverse methods can recover interior material properties from surface-only observations for elastodynamics. This proof of concept for dynamic scenarios motivates our investigation of whether static force scenarios can achieve similar reconstruction quality.

#### AI for Science and Scientific Machine Learning

Before examining specific approaches to the elastography inverse problem, it is essential to situate our work within the broader **AI for Science** movement—a transformative research paradigm where artificial intelligence accelerates scientific discovery across disciplines (Wang et al., 2023). The impact of this paradigm was dramatically validated when DeepMind's **AlphaFold** received the 2024 Nobel Prize in Chemistry for solving the 50-year-old protein structure prediction problem (Jumper et al., 2021), demonstrating that deep learning can achieve breakthroughs previously thought to require decades of traditional scientific effort.

**Scientific Machine Learning (SciML)** represents the intersection of AI for Science with physics-based modeling—an emerging discipline that integrates machine learning with scientific computing to solve problems governed by physical laws (Willard et al., 2022; Cuomo et al., 2022). As comprehensive surveys describe, SciML encompasses diverse methodologies for combining data-driven learning with physics-based constraints, ranging from embedding physical laws in neural network architectures to hybrid approaches coupling traditional numerical solvers with learned components.

**The SciML Landscape:**

SciML methods can be organized into several distinct categories:

**1. Physics-Informed Neural Networks (PINNs) and Neural Operators:**

These methods focus on solving or learning solutions to PDEs:

| Method | Key Paper | Citations | Core Approach |
|--------|-----------|-----------|---------------|
| **PINNs** | Raissi, Perdikaris & Karniadakis (2019) | ~19,000 | Embed PDE residuals in neural network loss |
| **DeepONet** | Lu et al. (2021) | ~2,800 | Learn operators between function spaces |
| **FNO** | Li et al. (2020) | ~2,000+ | Fourier-domain operator learning |
| **UDEs** | Rackauckas et al. (2020) | ~1,500+ | Augment differential equations with neural networks |

Within this category, PINNs (Raissi et al., 2019) have achieved remarkable adoption with ~19,000 citations, reflecting both the generality of the framework and successful applications across fluid dynamics, solid mechanics, heat transfer, and inverse problems. Theoretical foundations for PINNs are actively being developed, with rigorous convergence guarantees and error bounds established by Mishra and colleagues at ETH Zurich (De Ryck & Mishra, 2022; Mishra & Molinaro, 2022).

**2. Equation Discovery Methods:**

A separate class of SciML methods focuses on discovering governing equations from data:

- **SINDy** (Brunton, Proctor & Kutz, 2016): Sparse regression for identifying nonlinear dynamical systems (~5,500 citations)
- **AI Feynman** (Udrescu & Tegmark, 2020): Physics-inspired symbolic regression
- **PySR** (Cranmer, 2023): Genetic programming for symbolic regression

While these equation discovery methods are influential SciML contributions, they address a fundamentally different problem (discovering unknown equations) than our work (solving known equations for unknown parameters). We therefore do not draw direct comparisons with these methods.

**3. Neural Fields:**

Coordinate-based neural networks that parametrize physical quantities across space and time—sometimes called **neural fields** or **implicit neural representations**—have emerged as a powerful paradigm reviewed comprehensively by Xie et al. (2022). **Importantly, as Xie et al. note, PINNs are a type of neural field**: they use coordinate-based networks to represent continuous physical fields (displacement, velocity, pressure) parametrized by spatial and temporal coordinates. This connection places PINNs within the broader neural fields literature spanning computer vision, graphics, and scientific computing.

Originally developed for computer vision and graphics (3D reconstruction, novel view synthesis), neural fields have found applications in physics problems where continuous spatial representations are needed. Notable applications include Katie Bouman's work using neural fields for imaging the cosmic web through gravitational lensing (Bouman et al., 2025) and black hole emission tomography (Levis et al., 2022).

**PINN Applications Across Domains:**

PINNs have been successfully applied to diverse scientific problems, demonstrating their versatility. Ben Moseley and colleagues have made significant contributions including seismic wave simulation (Moseley et al., 2020) and the development of Finite Basis PINNs (FBPINNs) for scalable domain decomposition (Moseley et al., 2023), addressing key challenges in applying PINNs to large-scale problems.

**Physics-Informed Machine Learning as an Umbrella Term:**

Two foundational survey papers establish that "physics-informed machine learning" encompasses diverse architectures—not solely the original meshfree PINN formulation:

1. **Karniadakis et al. (2021)** in *Nature Reviews Physics* provide a comprehensive taxonomy showing that physics can be integrated into machine learning through: (a) observational biases (physics in training data), (b) inductive biases (physics in network architecture), and (c) learning biases (physics in loss function).

2. **Willard et al. (2022)** in *ACM Computing Surveys* systematically categorize physics-ML integration approaches including: physics-guided loss functions, physics-guided architecture design, hybrid physics-ML models, and physics-guided data augmentation.

This terminological breadth reflects the field's recognition that physics can be incorporated through multiple mechanisms. Our work adopts this broader interpretation, positioning PAT Scan as a **mesh-based physics-informed approach** that integrates physics through differentiable FEM forward modeling rather than meshfree PDE residual penalties.

#### PINNs for Elastography: Existing Literature

Several groups have applied physics-informed neural networks to elastography inverse problems, though predominantly in the context of **Magnetic Resonance Elastography (MRE)** and **ultrasound elastography**—not palpation-based approaches:

**MRE Applications:**

- **PINNs for MRE Tissue Elasticity** (Ragoza et al., 2023): Applied PINNs to solve the inverse problem of tissue elasticity reconstruction from MRE wave images, demonstrating robustness to noise and ability to leverage anatomical information from other MRI sequences.

- **ElastoNet** (2025): Neural network-based MRE wave inversion analyzing multiple wave components with uncertainty quantification, achieving comparable or better accuracy than established inversion methods across varying resolutions and vibration frequencies.

- **FDTDNet** (2025): Spatiotemporal neural network trained on Finite Difference Time Domain simulations for MRE stiffness quantification, showing 77-84% lower mean absolute error than direct inversion methods at 15 dB SNR.

**Strain Elastography Applications:**

- **El-UNet** (Mohammadi et al., 2023): Physics-informed UNet for discovering hidden elasticity in heterogeneous materials, taking normalized strain distributions as input with boundary and domain physics in the loss function. Achieved <5% mean absolute relative error for transversely isotropic materials.

- **ElastNet** (Chen et al., 2021): Deep neural network combining elasticity theory with deep learning for extracting hidden elasticity from measured strain distributions, noted that "palpation, a self-screening procedure for tumors, utilizes the difference in elasticity between healthy and cancerous tissues."

**Critical Observation:**

All existing PINN elastography work focuses on either:
1. **MRE**: Wave-based imaging requiring expensive MRI equipment (~$1-3M)
2. **Ultrasound strain tracking**: Requiring specialized ultrasound systems (~$50-200K)
3. **Full-field strain measurements**: Requiring optical systems or dense sensor arrays

**No existing work applies the mesh-based PINN paradigm to palpation-focused elastography with boundary-only displacement measurements.** This gap motivates our development of PAT Scan.

#### Traditional Meshfree PINNs (Karniadakis et al.): Coupled Forward-Inverse Formulation

The original Physics-Informed Neural Networks formulation (Raissi, Perdikaris & Karniadakis, 2019)—with ~19,000 citations making it the most influential SciML paper—trains neural networks to approximate solutions to partial differential equations by incorporating PDE residuals directly into the loss function:

Loss = ||u_NN - u_measured||² + λ ||∇²u_NN + f||²

where u_NN is the neural network approximation to the displacement field, and the second term penalizes violation of the governing PDE (e.g., equilibrium equation ∇·σ + f = 0 for elasticity).

Such approaches have been explored in subsequent work on full-field measurements (Zhang et al., 2022) and boundary-displacement measurements for elasticity imaging (Zhang, Yin & Karniadakis, 2020).

**Limitation for Elastography Applications:**

Traditional PINNs solve a coupled forward-inverse problem: the neural network must simultaneously:
1. Approximate the PDE solution (displacement field u)
2. Invert for material properties (E, ρ) that produce this displacement field

This coupling creates two challenges:
- **Computational expense**: PDE residual evaluation requires computing derivatives at many collocation points throughout the domain at each training iteration
- **Optimization difficulty**: The network must learn both the physics (how to solve the PDE) and the inverse mapping (which material properties explain the data)

**Our Decoupling Strategy:**

PAT Scan decouples these two tasks:
- **Forward model**: Exact FEM solver handles F = KU (no neural network approximation of physics)
- **Inverse model**: U-Net handles only the inverse problem (surrogate model for inclusion characterization)

This decoupling offers several advantages:
- **Computational efficiency**: FEM forward solve is exact and fast (direct linear solver)
- **Focused learning**: U-Net focuses completely on solving the inverse problem
- **Differentiability**: FEM solver is differentiable (automatic differentiation in PyTorch; Paszke et al., 2019), enabling end-to-end gradient flow for training

Recent work (JAX-SSO, 2024; JAX-FEM, Xue et al., 2023) has taken a similar mesh-based PINN approach for structural topology optimization, demonstrating the viability of coupling differentiable FEM solvers with neural networks. However, to our knowledge, this mesh-based PINN paradigm has not been applied to the palpation-focused elastography scenarios.

### PAT Scan Novelty: Confirmed Gap in the Literature

**Following comprehensive review of the Scientific Machine Learning and elastography literature, we confirm that no existing work combines the following elements:**

| Component | Existing Work | PAT Scan |
|-----------|---------------|----------|
| **Physics integration** | Meshfree PINNs (Karniadakis), physics-in-loss (El-UNet) | Mesh-based FEM forward model |
| **Data modality** | MRE waves, ultrasound strain, full-field measurements | Boundary-only displacements |
| **Acquisition strategy** | Single measurement, vibration modes | CT-inspired multi-angle scanning |
| **Neural architecture** | MLPs for PINNs, CNNs for strain tracking | U-Net as FEM inverse surrogate |
| **Clinical context** | MRE ($1-3M), ultrasound ($50-200K) | Palpation-inspired (minimal equipment) |

**The novelty gap is explicit:** While PINNs have been applied to MRE (Ragoza et al., 2023; ElastoNet, 2025) and strain elastography (El-UNet, Mohammadi et al., 2023; ElastNet, Chen et al., 2021), and while differentiable FEM frameworks exist (JAX-FEM, Xue et al., 2023; JAX-SSO, Wu, 2024), **no prior work applies the mesh-based PINN paradigm to palpation-focused elastography with boundary-only measurements and CT-inspired data acquisition.**

PAT Scan occupies a unique position in the methodological landscape defined by the intersection of requirements:

1. **Boundary-only displacement measurements** (sparse data constraint)
2. **Mesh-based PINN architecture** (decoupled FEM forward + neural network inverse)
3. **Segmentation-based formulation** (geometric inverse problem for 2-component systems)
4. **CT-inspired force application strategy** (sequential angular scanning with analytical spacing criteria)
5. **Arbitrary inclusion geometry** (irregular, off-centered shapes beyond simple circles)
6. **Low equipment cost requirement** (envisioned for resource-constrained settings)
7. **Palpation-focused** (quasi-static loading inspired by clinical practice)
8. **U-Net as FEM inverse surrogate** (~88,000 citation architecture repurposed for physics)

**Comparative Positioning:**

- **Versus El-UNet/ElastNet**: Full-field strain input with physics-in-loss versus boundary-only input with physics-in-pipeline
- **Versus MRE PINNs**: Wave-based imaging requiring MRI versus quasi-static palpation requiring minimal sensors
- **Versus VVT**: Dynamic vibration (elastodynamics) versus quasi-static forces (static elasticity)
- **Versus MBT**: Continuous field optimization versus segmentation-based formulation; iterative adjoint versus direct learned mapping; **critically, MBT does not explore force application strategies** (Mei et al., 2017 explicitly acknowledge this gap) while PAT Scan provides systematic CT-inspired acquisition with analytical spacing criteria
- **Versus traditional meshfree PINNs**: Coupled forward-inverse on collocation points versus decoupled exact FEM + learned inverse
- **Versus JAX-SSO/JAX-FEM**: Topology optimization application versus tissue elastography with palpation-inspired force application

**We propose PAT Scan as the first mesh-based PINN approach that integrates CT-inspired quasi-static force application with U-Net-based segmentation formulation, specifically targeting the palpation elastography inverse problem with boundary-only measurements.**

### Potential Impact

#### Scientific Contributions

**Novel PINN Extension:**
- Demonstrates mesh-based PINN paradigm for boundary-only inverse elastography
- Establishes efficacy of decoupled architecture (exact FEM forward + U-Net inverse)
- Provides design patterns for physics-informed segmentation networks

**Segmentation-Based Inverse Formulation:**
- Transforms infinite-dimensional continuous field problem into finite-dimensional geometric problem
- Incorporates structural priors (2-component system) as natural regularization
- Aligns problem structure with neural network architecture (U-Net for segmentation)

**Boundary-Only Sufficiency Demonstration:**
- Quantifies how much boundary displacement information is sufficient for inclusion reconstruction
- Establishes analytical criteria for force application spacing (Boussinesq-derived)
- Demonstrates feasibility despite severe ill-posedness of boundary-only measurements

**Extensibility to Other Inverse Problems:**
- The mesh-based PINN + segmentation formulation could extend to other mechanics inverse problems: defect detection in structural health monitoring, parameter identification in geophysics, property characterization in materials science

#### Healthcare Impact

**Cost Reduction:**
- Algorithm-first development enables future hardware optimization once computational feasibility is established
- Potential pathway to sub-$10K systems (versus $100K-$2M for current elastography)

**Resource-Constrained Setting Accessibility:**
- Computational approach compatible with simple force/displacement sensors
- Could enable tissue stiffness screening in clinics lacking access to MRE or ultrasound elastography

**Integration with Existing Clinical Workflows:**
- CT scan data can inform geometry (Aim 3b)
- Palpation-inspired force application aligns with clinical intuition
- Quantitative stiffness output enables treatment monitoring and objective comparison

**Pathway to Clinical Translation:**
- Proof-of-concept (Aim 1) → Realistic complexity (Aim 2) → 3D patient-specific models (Aim 3)
- Benchmarking library enables quantitative validation (Aim 2c)
- 3D patient-specific anatomically-constrained models (Aim 3)

---

## C. Innovation

### Innovation 1: Mesh-Based PINN with Decoupled Architecture

#### Core Innovation

We introduce a **mesh-based PINN architecture** that employs a **U-Net as a surrogate model for 2-component material systems**, coupled with the **linear matrix-form forward model F = KU**. This architecture is **solely focused on solving the geometric inverse problem**—finding the binary mask M(x, y) corresponding to the inclusion boundary—rather than the full inverse problem of reconstructing arbitrary continuous material fields E(x, y) ∈ ℝ₊.

**Neural Network Foundations and Novel U-Net Application:**

The observed universal approximation capabilities of neural networks make them strong candidates for surrogate models in inverse problem solving. U-Net (Ronneberger et al., 2015)—with approximately **88,000 citations**, making it one of the most influential neural network architectures ever published—has demonstrated exceptional performance on binary segmentation tasks in biomedical imaging, with widespread adoption for cell segmentation, organ boundary detection, and tumor delineation.

**Importantly, our application of U-Net represents a novel departure from its original purpose:**

| Aspect | Original U-Net (2015) | PAT Scan U-Net |
|--------|----------------------|----------------|
| **Domain** | Biomedical image segmentation | Inverse problem solving |
| **Output** | Segmentation masks for visual structures | Material property maps for physics |
| **Training data** | Annotated medical images | FEM-generated synthetic palpation data |
| **Physics role** | None (pure data-driven) | FEM generates training data; physics in pipeline |

While El-UNet (Mohammadi et al., 2023) also applies U-Net to elasticity problems, it uses **full-field strain measurements** as input with **physics embedded in the loss function**. In contrast, PAT Scan uses U-Net as a **surrogate for the FEM inverse operator**, trained on synthetically generated data with physics enforced through the **forward model pipeline** rather than loss penalties. This distinction—U-Net learning the inverse of a mesh-based FEM operator from boundary-only data—has not been previously explored.

Our insight: if the problem of identifying material inclusions is fundamentally a **binary segmentation problem** (background versus inclusion), then U-Net is a natural architectural choice—repurposed from image segmentation to physics-based inverse problems.

For 2-component material systems:
- Background material: E_background (known constant)
- Inclusion material: E_inclusion = α × E_background (known contrast factor α)
- Unknown: Binary mask M(x, y) ∈ {0, 1} indicating inclusion boundary

This formulation transforms the infinite-dimensional inverse problem [find E(x, y) ∈ ℝ₊ for all (x, y)] into a finite-dimensional geometric problem [find boundary curve separating background from inclusion].

**Physics Integration:**

The "physics-informed" aspect comes from coupling the U-Net prediction with the governing physical laws (linearized elasticity F = KU):

1. U-Net evaluated at coordinate inputs (subject to pre-processing) predicts material mask M(x, y)
2. Mask is converted to element-wise material properties E_e
3. Global stiffness matrix K is assembled from material properties via differentiable FEM
4. Forward problem F = KU is solved to obtain predicted displacements U_pred
5. Loss compares predicted boundary displacements to measured boundary displacements
6. Gradients flow backward through the entire pipeline to update U-Net weights

Traditional FEM handles the well-posed forward physics exactly (no neural network approximation of the PDE), while the U-Net handles only the ill-posed inverse problem.

#### What Makes It "Mesh-Based PINN"?

- **Mesh-based**: FEM operates on discrete triangular mesh with element-wise material properties (not meshfree collocation points)
- **Physics-informed**: U-Net predictions are constrained by linearized elasticity F = KU enforced during training (not just data-driven)
- **Neural Network**: U-Net provides a surrogate model based on the observed universal approximation capabilities of neural networks

This differs from the PINNs by Karniadakis et al., which approximate PDE solutions directly via neural network function approximation with PDE residual penalties on mesh-free collocation points. Our approach uses exact FEM for forward physics and applies machine learning only where needed: the ill-posed inverse direction.

#### High-Level Intuition: From Continuous Dials to Binary Switches

**Traditional MBT approach:**
- Iterative optimizer adjusts continuous material field E(x, y) ∈ ℝ₊
- Infinite-dimensional optimization space
- Slow convergence, sensitivity to initialization

**Our approach:**
- U-Net identifies discrete 2-component system via binary mask M(x, y) ∈ {0, 1}
- Finite-dimensional geometric problem (find inclusion boundary)
- Direct mapping learned from training data

**Conceptual Justification:**
- Universal approximation: Neural networks have practically demonstrated approximate universal approximation capabilities (Cybenko, 1989)
- U-Net demonstrated excellence: Proven state-of-the-art for binary segmentation (Ronneberger et al., 2015)
- Natural fit: 2-component material system = binary segmentation task
- Therefore: U-Net is a conceptually sound and empirically validated choice that may be leveraged for the geometric inverse problem at hand

We couple this learned surrogate model with exact enforcement of governing physical laws (linearized elasticity F = KU) during training, ensuring that predictions respect mechanics constraints.

#### Technical Implementation

**Architecture Components:**

1. **Inverse mapping (U-Net)**: Approximates material properties as binary mask M(x, y) on 64×64 grid
   - Input: 2 channels (X, Y coordinate grids)
   - Output: 1 channel (material mask M ∈ [0, 1])
   - Architecture: 3 encoder levels, bottleneck, 3 decoder levels with skip connections
   - Parameters: ~1.9M (reduced from standard U-Net's ~31M for efficiency on small dataset)

2. **Material conversion**: Converts mask to element-wise material properties at mesh nodes
   - Sample mask at element centroids via bilinear interpolation
   - Pre-processing step: Distance-Based Gaussian Smoothing
   - Convert to material values: E_e = E_background + (E_inclusion - E_background) × M_e

3. **Stiffness matrix assembly (differentiable FEM)**: Assembles global stiffness matrix K from element materials
   - Element stiffness: K_e = A_e × B_e^T × D × B_e (triangular elements, plane stress)
   - Global assembly: Scatter element contributions to global DOFs
   - Fully differentiable via PyTorch automatic differentiation (Paszke et al., 2019)

4. **Forward model (linear elasticity solver)**: Solves F = KU for displacement field
   - Direct linear solver: torch.linalg.solve(K, F)
   - Boundary conditions via penalty method (soft constraints for differentiability)

5. **Training loop**:
   - Loss: L = L_MSE + λ_TV × L_TV
   - L_MSE = ||U_pred - U_measured||² on boundary nodes
   - L_TV = mean(|∂M/∂x|) + mean(|∂M/∂y|) promotes piecewise smooth, sharp boundaries
   - Optimizer: Adam (Kingma & Ba, 2015) with learning rate 10⁻⁴

**Physics-Informed Training:**

Physics constraints are enforced through the forward model during training rather than via PDE residual penalties. At each training iteration:
- U-Net predicts material mask
- Differentiable FEM assembles K(E) and solves for U_pred via F = KU
- Loss penalizes deviation between predicted and measured boundary displacements
- Gradients of loss with respect to U-Net weights flow through the entire differentiable pipeline

This ensures that the network fits U field that respects linearized elasticity, even though it never explicitly sees the governing PDE in the loss function.

#### Why Decoupling Matters

The architectural choice to decouple forward and inverse problems reflects a pragmatic principle: use the right tool for each task.

Decoupling simplifies the learning problem by focusing the neural network on only the ill-posed inverse direction:
- **Traditional coupled PINNs**: NN must learn to solve PDE (forward) AND invert for parameters (inverse) simultaneously
- **Our decoupled approach**: Exact FEM solver handles forward physics, NN handles only material characterization

This division of labor leverages the strengths of both approaches:
- FEM: Exact, efficient, well-established for forward problems
- Neural networks: Flexible function approximators, excellent for learning complex spatial maps

The result: each component does what it does best, yielding a simpler, more efficient overall system.

#### Literature Alignment: Differentiable Physics Paradigm

Our approach aligns with the emerging differentiable physics paradigm (JAX-FEM, JAX-SSO) that couples traditional numerical methods with automatic differentiation for gradient-based optimization. However, we apply this paradigm specifically to the palpation-based elastography problem with geometric inverse formulation and CT-inspired force application—a novel combination not previously explored.

PINNs are an umbrella term encompassing many approaches that integrate physics into neural network training. Our mesh-based PINN with decoupled architecture represents one specific instantiation of this broad paradigm, optimized for the tissue elastography inverse problem.

---

### Innovation 2: Segmentation-Based Problem Formulation

#### From Continuous to Binary: Dimensional Reduction Through Structural Priors

The distinction between traditional and segmentation-based formulations represents a fundamental shift in how we frame the inverse problem:

**Traditional formulation (MBT and similar approaches):**
- Find E(x, y) ∈ ℝ₊ for all points (x, y) in domain
- Infinite-dimensional optimization problem
- Regularization required to constrain solution space (smoothness penalties, total variation)

**Our formulation for 2-component systems:**
- Find binary mask M(x, y) ∈ {0, 1} indicating inclusion boundary
- Material distribution: E(x, y) = E_background × [1 - M(x, y)] + E_inclusion × M(x, y)
- Finite-dimensional geometric problem: Identify one-dimensional boundary curve (in 2D)

This reformulation transforms an intractable continuous field optimization into a tractable geometric segmentation task.

#### Advantages of Segmentation Formulation

**Dimensionality Reduction:**
- Infinite-dimensional continuous field → Finite-dimensional binary mask + two scalar parameters (E_background, E_inclusion)
- For discretization on 64×64 grid: 4096 binary values versus 4096 continuous real values
- But effective dimensionality is far lower: the binary mask is characterized by a smooth boundary curve (1D manifold in 2D space)

**Natural Regularization Through Structural Priors:**
- Binary structure automatically imposes piecewise constant assumption
- Physically motivated by clinical reality: discrete tumors in background tissue
- Eliminates need for ad-hoc regularization choices (how much smoothness? which norm?)

**Neural Network Architecture Alignment:**
- U-Net was specifically designed for binary segmentation tasks (Ronneberger et al., 2015)
- Proven state-of-the-art performance on biomedical image segmentation
- Natural fit between problem structure (binary segmentation) and architecture (U-Net)

**Clinical Realism:**
- Matches diagnostic paradigm: locate the tumor (segmentation) + characterize its properties (contrast factor)
- Directly provides information clinicians need: "Where is the pathology?" and "How stiff is it?"

### Innovation 2 (continued): Adaptive Boundary Conditions for Dynamic Inclusion Location

#### The Challenge: Unknown Inclusion Location During Training

In the geometric inverse problem, the inclusion location is unknown and must be inferred from boundary displacement measurements. However, the FEM forward model requires boundary conditions to be specified: nodes inside the stiff inclusion should have reduced displacement (the inclusion is stiffer and deforms less than the background). This creates a chicken-and-egg problem during training:
- To solve the forward model F = KU, we need to know which nodes to fix or constrain
- But the inclusion location is exactly what we're trying to learn

#### Our Solution: Differentiable Penalty Method with Adaptive Estimation

We employ a fully differentiable penalty method that adapts to the current mask prediction:

**Step 1: Estimate inclusion center from current mask prediction**
- Compute weighted centroid: (x_c, y_c) = Σ M(x,y) × (x, y) / Σ M(x,y)
- This provides a rough estimate of where the U-Net currently believes the inclusion is located

**Step 2: Compute conservative safe radius**
- Find minimum distance from estimated center to domain boundary or origin using a **soft_min function via LogSumExp**: `soft_min(a, b) = -τ × log(exp(-a/τ) + exp(-b/τ))` where τ is temperature
- **Key property:** When the true minimum would be zero (e.g., estimated center at origin), soft_min still produces a small positive value due to its smoothing effect—this prevents degenerate boundary conditions with zero radius
- safe_radius = min_distance ÷ 2 (conservative factor ensures we don't over-constrain)

**Step 3: Apply soft boundary conditions via penalty method**
- For each element, check if centroid is within safe_radius of estimated center
- Build fixed_mask indicating which DOFs should have reduced motion via a sigmoid function
- Augmented stiffness matrix: K_eff = K + penalty × diag(fixed_mask)
- Augmented force vector: F_eff = F × free_mask (zero force on fixed DOFs)
- Solve: K_eff × U = F_eff

**Properties:**
- **Fully differentiable**: All operations (weighted centroid, distance computation, penalty addition) support automatic differentiation
- **Adaptive**: Adjusts to current network prediction, not ground truth
- **Physically motivated**: Classical penalty method from computational mechanics
- **Handles off-center inclusions**: Centroid estimation works for arbitrary locations

**Classical Penalty Method Connection:**
The penalty method is a standard technique in FEM (Zienkiewicz, Taylor & Zhu, 2013) for enforcing constraints approximately rather than exactly:
- Hard constraint: u_i = 0 (achieved by matrix reduction, removing DOF from system)
- Soft constraint: K_ii → K_ii + large_penalty, ensuring u_i ≈ 0 (achieved by adding to diagonal)

Our approach applies the classical penalty method in a differentiable, adaptive manner guided by the current neural network prediction.

#### Avoiding the Inverse Crime: Intentional Model Mismatch

**The Inverse Crime Problem:**

In computational inverse problems, the "inverse crime" (Kaipio & Somersalo, 2007) occurs when identical forward models are used for both synthetic data generation and inversion. This creates unrealistically optimistic results:
- Network learns to invert the exact same operator used to generate training data
- Perfect circular reasoning that doesn't test robustness to model error
- Real experiments never match theoretical models perfectly

**Our Solution: Controlled Physics Mismatch**

We intentionally use **different boundary condition implementations** in data generation versus training:

**Training Dataset Generation (fem_utils.py):**
- **Method**: Matrix reduction (hard boundary conditions)
- **Implementation**: K_reduced = K[free_dofs, free_dofs] (remove fixed DOFs from system)
- **Physics**: Stiff inclusion is perfectly rigid (u = 0 exactly, effectively infinite stiffness)
- **Represents**: Simplified forward model with strict mathematical constraints

**PINN Inverse Training (unet_train_*.py):**
- **Method**: Penalty method (soft boundary conditions)
- **Implementation**: K_eff = K + 10⁶ × diag(fixed_mask)
- **Physics**: Stiff inclusion is very stiff but deformable (u ≈ 0, finite stiffness)
- **Represents**: More realistic forward model with differentiable approximations

The key insight: by intentionally using different physics approximations, we create a harder test for the algorithm—one that better prepares it for real-world deployment where experimental data never perfectly matches theoretical assumptions.

**Methodological Contribution:**

This controlled mismatch strategy provides several advantages:

1. **Avoids inverse crime**: Forward model (data generation) ≠ Inverse model (training)
2. **Tests robustness**: Network must recover E from "corrupted" training data with simplified physics
3. **More realistic**: Real experiments never perfectly match theoretical model assumptions (material anisotropy, geometric imperfections, measurement noise)
4. **Harder test**: PINN doesn't get the "easy case" of inverting its own exact forward model
5. **Demonstrates generalization**: If PINN succeeds despite physics mismatch → robust to real-world model errors

**Interpretation:**

If the PINN successfully recovers material properties from training data generated with simplified physics (hard BCs via matrix reduction), it demonstrates resilience to model assumptions—a critical requirement for real-world inverse problems where experimental data never perfectly matches theoretical models. This physics mismatch is intentional and methodologically justified, not an implementation accident.

---

### Innovation 3: CT-Inspired Force Application with Analytical Criteria

#### Tomographic Data Acquisition Strategy

Computed Tomography (CT) revolutionized medical imaging (Hounsfield, 1973) by recognizing that data acquired from multiple angles provides far more information for reconstruction than a single projection. The same principle applies to mechanical property reconstruction: measurements from multiple force configurations provide complementary information that constrains the inverse problem more effectively than a single loading scenario.

**Inspiration from CT:**
- CT: X-ray projections from multiple angles → reconstruct internal density distribution
- PAT Scan: Displacement measurements from multiple force pair orientations → reconstruct internal stiffness distribution

**Angular Scanning Protocol:**
- Apply force pairs at varying positions around the circular boundary
- Vary the number of force pairs from 1 to 20, maintaining fixed angular spacing
- Each configuration produces a unique boundary displacement pattern
- Combined dataset: 20 samples with increasing force complexity

**Information Content:**
- Single force pair: Probes sample primarily along loading axis
- Multiple force pairs: Probes sample from many directions simultaneously
- More configurations → More constraints on inverse problem → Better reconstruction

The analogy to CT is precise: just as a single X-ray projection cannot reveal depth information (all structures along the ray path are superimposed), a single force configuration cannot fully constrain the internal material distribution. Multiple viewing angles—or in our case, multiple force configurations—break the degeneracy.

#### Boussinesq-Based Angular Spacing Criterion

A critical design question: **How should force pairs be spaced around the boundary?**

Too close → Redundant information (displacement fields from adjacent pairs nearly identical)
Too far apart → Insufficient coverage (gaps in spatial information)

We derive an analytical spacing criterion from the classical Boussinesq solution for elastic half-space (Boussinesq, 1885):

**Boussinesq Solution:**
For a point force applied normal to the surface of an elastic half-space, the surface displacement decays with distance r from the force application point approximately as:

u(r) ∝ 1/r (near field)

This decay rate informs how far displacement information "propagates" from each force application point. For adjacent force pairs to provide:
- **Sufficient overlap**: Displacement fields must overlap to ensure continuous coverage
- **Not excessive redundancy**: Overlap should not be so large that fields are nearly identical

**Computational Criterion:**
Based on Boussinesq decay rates and validated through computational trials on representative geometries, we determined:

**Optimal spacing = 9 degrees → 40 positions around circular boundary**

This spacing ensures:
- Displacement fields from adjacent pairs have partial overlap (information sharing)
- Fields remain distinct enough to provide new information (not excessive redundancy)

**Insight:**
The Boussinesq solution provides the physical intuition (displacement decay rate) that informs the coverage requirement. Computational validation confirms this analytical insight produces good reconstruction quality.

#### Force Magnitude Selection: Computational Criterion

**Requirement:**
Force magnitude must be:
- **Large enough**: Produce measurable displacement signal above noise
- **Small enough**: Avoid penetration (deformed boundary must not penetrate inclusion)

**Our Approach:**
Through systematic computational testing (automated test sweeps in automated_tests.py):
1. Sweep force magnitude from F = 0.01 to F = 0.5
2. Check for geometric penetration at each magnitude
3. Identify maximum safe force: F_max ≈ 0.2
4. Select operating point with safety margin: **F = 0.1** (50% margin below F_max)

**Generalization:**
This is an "in-sample" assumption: the force magnitude was selected for a representative ground truth geometry.

---

### Innovation 4: Symmetric Scanning for Rotational Equivariance

#### Conceptual Innovation: Exploiting Rotational Symmetry

For samples with rotationally symmetric geometry (circular boundary, centered circular inclusion):

**Observation:**
- Rotating the force configuration by angle θ → Rotated displacement field by same angle θ
- This is a physical symmetry: circular geometry + centered inclusion → rotational equivariance

**Exploitation:**
Explicitly teach the network this symmetry by providing training examples:
- Single force pair configuration (e.g., horizontal axis loading)
- Rotated through angles: 0°, 9°, 18°, ..., 171° (20 orientations, matching the 9° spacing from angular scanning)
- Each rotation produces a new training sample with known geometric relationship to others

**Benefit:**
- **Inductive bias**: Network learns "rotate input → rotate output" relationship
- **Improves generalization**: Symmetry structure helps network generalize across orientations
- **Reduces data requirements**: Each orientation provides related information, enhancing learning efficiency

**For Asymmetric Samples:**
When the inclusion is off-center or irregular:
- Rotational symmetry is broken
- Each orientation provides genuinely **new information** rather than just rotational variants
- Symmetric scanning dataset becomes even more valuable: probes asymmetry from multiple angles

#### Combined Dataset Strategy: Angular + Symmetric

**Angular Scanning Dataset:**
- 20 samples: Vary number of force pairs (1, 2, ..., 20) with fixed spacing
- Information content: Increasing force complexity

**Symmetric Scanning Dataset:**
- 20 samples: Fixed single force pair, rotated through 20 orientations
- Information content: Rotational equivariance (symmetric case) or new angular information (asymmetric case)

**Combined Training:**
- Total: 40 training samples
- Angular: New information about force complexity effects
- Symmetric: New information about rotational structure
- Combination: Enhanced robustness across both force configuration and orientation variation

#### Robustness to Initial Conditions: Beyond Rotational Equivariance

Symmetric scanning provides an additional critical benefit beyond teaching rotational symmetry:

**The Problem of Initialization Sensitivity:**
- Inverse solvers are sensitive to random weight initialization
- Different initial conditions can lead to different local minima during optimization
- Single training run may yield spurious results if optimization gets stuck in poor local minimum

**Our Hypothesis:**
Additional symmetric scanning data enhances robustness of PINN predictions by:
- Providing richer training signal with multiple orientations
- Helping network converge to more stable, generalizable representations
- Reducing sensitivity to initialization through increased data diversity

**Minimal Reliability Study: 15 Independent Runs**

To test this hypothesis, we trained the PINN 15 times with different random initializations:

**Setup:**
- Identical training procedure, hyperparameters, dataset
- Only difference: Random seed for weight initialization

**Observations:**
1. **Tight clustering**: Results clustered tightly across runs (low variance)
2. **Consistent geometry recovery**: Inclusion boundary predictions were similar across runs
3. **Repetition in results**: After ~15 runs, no new variation appeared (suggesting we captured the meaningful variation)

**Interpretation:**
The symmetric scanning data successfully reduces initialization sensitivity. Multiple orientations of the same underlying physics provide complementary constraints that guide the network toward consistent, high-quality solutions regardless of initialization.

#### Practical Safeguard Strategy: Ensemble Approach

Given the tight clustering observed in the 15-run study, we propose a simple ensemble strategy for deployment:

1. **Run algorithm 3-5 times** with different random seeds
2. **Identify tight cluster**: Most results will cluster together (good local minima)
3. **Discard outliers**: Remove any spurious results (rare bad local minima, if present)
4. **Average tight cluster**: Compute mean prediction from clustered results
5. **Benefit**: Safeguards against spurious local minima while extracting robust predictions

**Key Insight:**

The tight distribution from the 15-run study demonstrates that symmetric scanning doesn't just teach rotational equivariance—it fundamentally **stabilizes the training process**. The additional information from multiple orientations helps the network converge to similar, high-quality solutions regardless of initialization, making the method more reliable for practical deployment.

This is critical for clinical translation: practitioners need confidence that running the algorithm twice on the same data will produce the same result, not arbitrary variations due to random initialization.

---

## D. Research Approach

The research strategy follows a natural progression from simplified proof-of-concept to realistic complexity, systematically addressing key challenges at each stage. This approach mirrors the development trajectory of successful imaging technologies: establish feasibility on idealized cases, then progressively incorporate real-world complexity until clinical translation becomes viable.

### Overall Goal

Improve understanding of the PAT Scan methodology for tissue stiffness reconstruction through systematic development:
- **Specific Aim 1**: Proof-of-concept with circular geometries
- **Specific Aim 2**: Extension to realistic complexity
- **Specific Aim 3**: Incorporation of 3D geometry and clinical realism

---

### Specific Aim 1: Proof-of-Concept

#### Goal

Demonstrate proof-of-concept for the PAT Scan methodology by:
- Focusing on the tractable geometric inverse problem (inclusion localization)
- Using simple 2D circular geometry for validation
- Establishing that boundary displacement measurements contain sufficient information for reconstruction

#### Working Assumption: Existence of Inclusion

We assume that an inclusion exists within the sample. This assumption is straightforward to verify in practice:

**Verification procedure:**
1. Acquire boundary displacement measurements under applied forces
2. Compare to reference displacements for a homogeneous circular sample (known E_background, no inclusion)
3. If measured displacements deviate significantly from homogeneous reference → Inclusion is present
4. If displacements match homogeneous reference → No inclusion (or inclusion with negligible contrast)

This verification step serves as a prerequisite gate before applying the PAT Scan reconstruction algorithm. The method targets scenarios where palpation or preliminary screening has already detected an abnormality; PAT Scan characterizes its geometry and stiffness quantitatively.

---

### Aim 1a: FEM Forward Model Development

We implement a standard finite element method (FEM) solver for 2D plane strain linearized elasticity. The mesh employs a structured polar grid, which naturally suits circular and near-circular geometries while providing efficient triangulation.

#### Mathematical Formulation

**Governing Equations:**
- **Physics**: Linearized elasticity (small deformations)
- **Loading**: Quasi-static (neglect inertia and damping)
- **Fundamental equation**: F = KU
  - F: Applied force vector (n_dof × 1)
  - K: Global stiffness matrix (n_dof × n_dof)
  - U: Displacement vector (n_dof × 1)
- **Plane stress assumption**: σ_zz = 0 (appropriate for thin samples)

**Material Properties:**
- **Young's modulus**: E(x, y) piecewise constant
  - E_background = 1.0 (normalized units)
  - E_inclusion varies: 10× contrast for straightforward cases, 1.2× contrast for challenging cases
- **Poisson ratio**: ν = 0.3 (representative of soft biological tissue; Greaves et al., Nature 2011)
- **2-component structure**: Background + singular inclusion with known contrast factor

**Challenging Scenarios Addressed:**

The proof-of-concept demonstrates reconstruction capability across a range of difficulty:
- **Small inclusions**: Reduced inclusion radius → Limited displacement signal → Harder to detect
- **Faint inclusions**: Low stiffness contrast (1.2× vs 10×) → Subtle displacement differences → Requires high sensitivity
- **Small + faint inclusions**: Combined challenges requiring maximum algorithmic sensitivity

**Parameter Justification:**
- Domain size (R_outer = 1.0), inclusion diameter (R_inner varying), and stiffness contrast range chosen to align with values from Mechanics-Based Tomography literature (Goenezen et al., 2011; Mei et al., 2017)
- Poisson's ratio ν = 0.3 is representative of soft biological tissue (Greaves et al., Nature 2011)
- These choices enable direct comparison with established methods

#### Mesh Generation

**Structured Polar Grid:**
- **Radial divisions**: n_radial = 20 layers from center to boundary
- **Angular divisions**: n_angular = 40 sectors around circumference
- **Total nodes**: 761 (1 center node + 20 layers × 40 angular positions)
- **Total elements**: 1520 triangular elements
- **Element type**: 3-node linear triangles (T3)

**Topology:**
- Center node at origin
- Structured concentric layers radiating outward
- Each quadrilateral ring is subdivided into two triangles per sector

**Material Assignment (Centroid-Based):**

For each triangular element:
1. Compute element centroid: (x_c, y_c) = mean of three vertex coordinates
2. Check if centroid lies inside inclusion boundary:
   - **Circular**: ||centroid|| ≤ R_inner → E_inclusion
   - **Otherwise**: E_background
3. Assign corresponding Young's modulus value to element

This centroid-based assignment provides a simple, robust method for discretizing piecewise constant material distributions onto finite elements.

#### FEM Assembly and Solution

**Element Stiffness Matrix:**

For each triangular element with vertices (x₁, y₁), (x₂, y₂), (x₃, y₃):

K_e = A_e × B^T × D × B

where:
- A_e = element area = 0.5 × |det([x₂-x₁, x₃-x₁; y₂-y₁, y₃-y₁])|
- B = strain-displacement matrix (3×6) relating nodal displacements to element strains
- D = plane stress constitutive matrix (3×3) relating strains to stresses:

  D = (E / (1 - ν²)) × [1, ν, 0; ν, 1, 0; 0, 0, (1-ν)/2]

- Result: K_e is 6×6 (2 DOFs per node × 3 nodes)

**Global Assembly:**

Scatter element stiffness matrices to global stiffness matrix K:
- Each element has 3 nodes with node IDs [n₁, n₂, n₃]
- Each node i has 2 DOFs: (2i, 2i+1) corresponding to (u_x, u_y)
- Element matrix K_e is added to global matrix K at positions corresponding to element DOFs
- Result: Global K is (n_nodes × 2) × (n_nodes × 2) = 1522 × 1522 sparse matrix

**Boundary Conditions (Intentional Mismatch for Inverse Crime Avoidance):**

We employ **different boundary condition implementations** in dataset generation versus training:

**Dataset Generation (fem_utils.py) - Matrix Reduction Method:**
- **Implementation**: K_reduced = K[free_dofs, free_dofs] (extract submatrix for free DOFs only)
- **Physics**: Hard boundary conditions (stiff inclusion perfectly rigid, u = 0 exactly)
- **Mathematical**: Fixed DOFs are removed from system entirely (infinite stiffness)
- **Purpose**: Generate training data with simplified, strict physics

**Differentiable Training (unet_train_*.py) - Penalty Method:**
- **Implementation**: K_eff = K + penalty × diag(fixed_mask), penalty = 10⁶
- **Physics**: Soft boundary conditions (stiff inclusion very stiff but deformable, u ≈ 0)
- **Mathematical**: Fixed DOFs remain in system with large penalty added to diagonal (finite stiffness)
- **Purpose**: Invert with differentiable, adaptive physics

**Rationale:**
This controlled model mismatch avoids the inverse crime (using identical forward models for data generation and inversion). The network must learn to recover material properties from "corrupted" data generated with simplified physics, testing robustness to model assumptions. Real experiments never perfectly match theoretical models due to material anisotropy, geometric imperfections, and measurement noise—this mismatch prepares the algorithm for real-world deployment.

**Force Application:**
- Force pairs applied at specified boundary nodes
- Equal magnitude, opposite directions (radially inward)
- F_eff = F × free_mask (zero force on fixed DOFs during training)

**Linear System Solver:**
- PyTorch direct linear solver: U = torch.linalg.solve(K_eff, F_eff) (Paszke et al., 2019)
- Exact solution (within machine precision)
- Fully differentiable for automatic gradient computation

#### Validation Through Automated Testing

**Test Suite (automated_tests.py, automated_tests_upgraded.py):**

**Test 1: Force Magnitude Sweep**
- Systematically increase force magnitude from F = 0.01 to F = 0.5
- At each magnitude, solve FEM and check for geometric penetration
- Penetration check: Verify deformed boundary minimum radius r_min > R_inner
- Identify maximum safe force F_max
- **Results**: F_max ≈ 0.2 for representative geometry

**Test 2: Angular Sweep Validation**
- Apply force pairs at sequential angular positions around boundary
- Verify displacement field symmetry for symmetric geometries
- Check consistency of solution magnitude and spatial patterns
- **Results**: Displacement fields exhibit expected symmetry and smooth variation

**Operating Point Selection:**
- Safe operating force: **F = 0.1** (50% safety margin below F_max = 0.2)
- Provides measurable displacement signal while ensuring geometric constraints are satisfied

---

### Aim 1b: Dataset Generation

#### Angular Scanning Protocol

**Procedure:**

For n_pairs = 1, 2, 3, ..., 20:
1. Determine angular positions for force pairs with fixed spacing = 9° (Boussinesq-derived criterion)
2. Apply n_pairs radially inward force pairs at determined angles:
   - Equal magnitude F = 0.1
   - Opposite directions (compression)
3. Solve FEM: K(E_true) × U = F
4. Extract boundary displacements U_boundary
5. Save training sample: (U_boundary, E_true_map, n_pairs)

**Output:**
- 20 training samples per geometry
- Information content: Varying force complexity (1 to 20 force pairs)

#### Symmetric Scanning Protocol

**Procedure:**

For angle = 0°, 9°, 18°, ..., 171° (20 orientations):
1. Fix force configuration: n_pairs = 1 (single force pair)
2. Rotate force pair to specified angle
3. Apply forces, solve FEM: K(E_true) × U = F
4. Extract boundary displacements U_boundary
5. Save training sample: (U_boundary, E_true_map, angle)

**Purpose:**
- **For symmetric samples**: Teach rotational equivariance
- **For asymmetric samples**: Probe geometry from multiple angles (new information)

**Output:**
- 20 training samples per geometry
- Information content: Varying orientation

#### Combined Dataset

**Total Training Data:**
- Angular scanning: 20 samples (force complexity variation)
- Symmetric scanning: 20 samples (orientation variation)
- **Combined total**: 40 training samples per geometry

**Information Complementarity:**
- Angular: How does displacement field change with increasing number of force pairs?
- Symmetric: How does displacement field change with rotation?
- Combination: Network learns both force complexity effects and geometric orientation structure

---

### Aim 1c: PINN Training

The training pipeline consists of three integrated stages: (1) U-Net predicts a material mask from spatial coordinates, (2) the mask is converted to element-wise material properties and assembled into the stiffness matrix K, and (3) FEM solves the forward problem F = KU to obtain predicted displacements compared against measured data. This architecture is **fully differentiable**, allowing gradients to flow from the displacement loss backward through the FEM solver, stiffness assembly, material conversion, and ultimately to the U-Net weights via automatic differentiation.

#### Neural Network Architecture: Lightweight U-Net Variant

We implement a **modified U-Net** (adapted from Ronneberger et al., 2015) specifically designed for the geometric inverse problem:

**Framework & Precision:**
- **Library**: PyTorch (Paszke et al., 2019)
- **Precision**: float64 (double precision required for FEM numerical stability)
- **Initialization**: PyTorch default (Kaiming uniform; He et al., 2015) for Conv2d with ReLU activation

**Key Modifications from Original U-Net:**
- **Reduced depth**: 3 encoder levels (vs. 5 in original) suitable for 64×64 input resolution
- **Smaller base features**: 32 channels (vs. 64 in original) → ~1.9M parameters (vs. ~31M)
- **Same-padding convolutions**: padding=1 preserves spatial dimensions (vs. valid padding in original which shrinks output)
- **Simplified skip connections**: Direct concatenation without cropping (enabled by same-padding)

**Architecture Specification:**

**Input:**
- 2 channels: (X, Y) coordinate grid meshes on 64×64 spatial domain
- X(i, j) = x-coordinate of pixel (i, j)
- Y(i, j) = y-coordinate of pixel (i, j)

**Encoder (3 levels):**
- **Level 1**:
  - Double conv block: 2 → 32 channels (Conv3×3 → ReLU → Conv3×3 → ReLU)
  - MaxPool 2×2 → Downsample to 32×32

- **Level 2**:
  - Double conv block: 32 → 64 channels
  - MaxPool 2×2 → Downsample to 16×16

- **Level 3**:
  - Double conv block: 64 → 128 channels
  - MaxPool 2×2 → Downsample to 8×8

**Bottleneck:**
- Double conv block: 128 → 256 channels (8×8 spatial resolution)

**Decoder (3 levels with skip connections; He et al., 2016):**
- **Level 1 (decode from bottleneck)**:
  - Transpose convolution 2×2: 256 → 128 channels, upsample to 16×16
  - Concatenate with skip connection from Encoder Level 3
  - Double conv block: 256 → 128 channels (after concatenation)

- **Level 2**:
  - Transpose convolution 2×2: 128 → 64 channels, upsample to 32×32
  - Concatenate with skip connection from Encoder Level 2
  - Double conv block: 128 → 64 channels

- **Level 3**:
  - Transpose convolution 2×2: 64 → 32 channels, upsample to 64×64
  - Concatenate with skip connection from Encoder Level 1
  - Double conv block: 64 → 32 channels

**Output Head:**
- Conv 1×1: 32 → 1 channel (no padding needed)
- Sigmoid activation: M(x, y) ∈ [0, 1]

**Implementation Details:**
- All convolutions: kernel_size=3, padding=1, stride=1 (preserves spatial dimensions)
- All transpose convolutions: kernel_size=2, stride=2 (doubles spatial dimensions)
- Activation: ReLU with inplace=True throughout encoder/decoder

**Rationale for Modifications:**
- Reduced complexity suitable for geometric inverse problem (binary segmentation, not complex texture)
- Faster training convergence on limited dataset (40 training samples)
- Same-padding maintains spatial correspondence with FEM element centroids (no alignment issues)
- Smaller network reduces overfitting risk with small dataset while retaining sufficient capacity

#### Material Prediction Pipeline: From Mask to Properties

The U-Net output (soft mask on 64×64 grid) must be converted to element-wise material properties for FEM assembly. This pipeline is **fully differentiable**:

**Step 1: U-Net Forward Pass**
- Input: (X, Y) coordinate grids (2 × 64 × 64)
- Output: Mask M(x, y) on 64×64 grid, M ∈ [0, 1] (soft values due to sigmoid)

**Step 2: Sampling at Element Centroids**
- Function: `sample_mask_at_centroids()`
- Method: Bilinear interpolation via `F.grid_sample(mode='bilinear', align_corners=True)`
- Samples mask values at FEM element centroid coordinates
- Result: mask_values (n_elements × 1), values ∈ [0, 1]

**Step 3: Adaptive Sigmoid Thresholding**
- Function: `threshold_mask()`
- Formula: sigmoid(T × (p - mean(p))) where T ∈ [100, 5000] is temperature
- Effect: Sharpens soft mask values around mean, pushing toward binary {0, 1}
- Differentiable: Sigmoid has continuous derivatives
- Book-keeping method for applying soft BCs

**Step 4: Distance-Based Gaussian Smoothing**
- Function: `smooth_mask_differentiable()`
- Compute pairwise distances between element centroids
- Gaussian weights: w_ij = exp(-d²_ij / 2σ²) where σ ∈ [0.01, 0.1]
- Smooth mask: mask_smooth_i = Σⱼ w_ij × mask_j / Σⱼ w_ij
- Effect: Spatial smoothing reduces noise while maintaining differentiability

**Step 5: Material Property Conversion**
- Linear interpolation: E_e = E_background + (E_inclusion - E_background) × mask_smooth_e
- For mask_e = 0: E_e = E_background (background material)
- For mask_e = 1: E_e = E_inclusion (inclusion material)
- For mask_e ∈ (0, 1): E_e interpolates smoothly

**Result:** Element-wise material properties ready for FEM stiffness assembly

#### Physics Model: Differentiable FEM

**Stiffness Assembly:**
- Function: `assemble_stiffness_differentiable()`
- Input: Element material properties E_e for e = 1, ..., n_elements
- Computation: For each element, compute K_e(E_e) and scatter to global K
- Implementation: Batched element stiffness computation for efficiency
- Output: Global stiffness matrix K (n_dof × n_dof)
- **Fully differentiable**: Uses PyTorch (Paszke et al., 2019) operations throughout

**FEM Forward Solve:**
- Augment stiffness for boundary conditions: K_eff = K + penalty × diag(fixed_mask)
- Augment force for boundary conditions: F_eff = F × free_mask
- Solve linear system: U_pred = torch.linalg.solve(K_eff, F_eff)
- **Fully differentiable**: PyTorch linear solver supports automatic differentiation

**Displacement Output:**
- Extract boundary displacements from full displacement field U_pred
- Compare to measured boundary displacements U_measured
- Compute loss (next section)

#### Loss Function: MSE + Total Variation Regularization (Rudin, Osher & Fatemi, 1992)

**Data-Fitting Term (MSE):**

L_MSE = (1/N_boundary) × Σᵢ (U_pred,i - U_measured,i)²

- Computed only on boundary nodes (N_boundary nodes × 2 DOFs per node)
- Penalizes deviation between predicted and measured boundary displacements
- Units: [displacement²]

**Regularization Term (Total Variation on Mask):**

L_TV = mean(|∂M/∂x|) + mean(|∂M/∂y|)

- Computed on U-Net output mask M (64×64 grid)
- Approximate gradients via finite differences: ∂M/∂x ≈ M[i+1, j] - M[i, j]
- Effect: Promotes piecewise smooth material distributions with sharp boundaries
- Physical motivation: Real inclusions have crisp boundaries, not gradual transitions

**Combined Loss:**

L = L_MSE + λ_TV × L_TV

- Hyperparameter λ_TV ∈ [0.001, 0.05] balances data-fitting versus regularization
- Determined via grid search (see Training Procedure)

**Why TV Regularization?**
- Encourages sparse gradients (sharp boundaries between background and inclusion)
- Avoids over-smoothing that would blur inclusion boundaries
- Well-suited for piecewise constant material distributions
- Differentiable (gradient magnitudes computed via automatic differentiation)

#### Training Procedure

**Optimizer:**
- Adam optimizer (Kingma & Ba, 2015) with learning rate LR = 10⁻⁴
- Adam hyperparameters: β₁ = 0.9, β₂ = 0.999, ε = 10⁻⁸ (PyTorch defaults)

**Training Duration:**
- Epochs: 200–1000 (until convergence)
- Convergence criterion: Loss plateau for ~50 epochs

**Hyperparameter Grid Search:**

The following hyperparameters are tuned via systematic grid search:

1. **Learning rate**: LR ∈ {10⁻⁵, 10⁻⁴, 10⁻³}
2. **TV regularization weight**: λ_TV ∈ {0.001, 0.005, 0.01, 0.05}
3. **Smoothing kernel width**: σ ∈ {0.01, 0.03, 0.05, 0.07, 0.1}
4. **Thresholding temperature**: T ∈ {100, 500, 1000, 2000, 5000}
5. **Boundary condition sharpness**: penalty ∈ {10⁵, 10⁶, 10⁷}

**Grid Search Procedure:**
- Train on geometry
- Evaluate reconstruction quality via SSIM and L2 error metrics
- Select hyperparameter combination yielding best validation performance
- Use selected hyperparameters for final proof-of-concept demonstration

#### Post-Processing: Level-Set Boundary Extraction

After training, the U-Net outputs a soft mask M(x, y) ∈ [0, 1]. To extract a crisp inclusion boundary for visualization and geometric analysis:

**Level-Set Contour Extraction**
- Find M = 0.5 contour using `skimage.measure.find_contours()` (van der Walt et al., 2014)
- Result: (x, y) coordinates tracing inclusion boundary

**Visualization**
- Overlay extracted contour on ground truth geometry
- Compute boundary distance error: mean distance between predicted and true boundaries

**Output:**
- Crisp boundary contour for qualitative and quantitative evaluation

---

### Aim 1d: Evaluation Methodology

#### Validation Strategy: Synthetic Ground Truth

In this niche domain of boundary-only inverse elastography, **no experimental gold standard exists** for validating reconstruction algorithms. Following standard practice in mechanical tomography inverse problems, we validate against **synthetic ground truth** generated from known material distributions.

**Rationale:**
- **Exact ground truth knowledge**: Enables precise quantitative error metrics
- **Systematic parameter exploration**: Can systematically vary contrast, size, shape to test algorithm limits
- **Controlled testing**: Isolates algorithm performance from experimental confounds (measurement noise, geometric uncertainty)
- **Reproducible benchmarking**: Other researchers can reproduce exact test scenarios

**Limitations Acknowledged:**
- Synthetic validation does not replace experimental validation
- Real-world scenarios involve additional complexities (material nonlinearity, geometric imperfections, measurement noise)

**Current State:**
Synthetic validation is the appropriate methodology for proof-of-concept (Aim 1).

#### Quantitative Metrics

**Structural Similarity Index (SSIM):** (Wang et al., 2004)

SSIM ∈ [-1, 1], typically reported as [0, 1] for images

- Measures perceptual similarity between predicted and ground truth material maps
- Accounts for luminance, contrast, and structure
- **Target**: SSIM > 0.8 indicates excellent reconstruction quality

**Relative L2 Error:**

ε = ||E_pred - E_true|| / ||E_true||

- Measures normalized Euclidean distance between predicted and true material fields
- **Target**: ε < 0.2 indicates reconstruction error less than 20% of signal magnitude

---

### Specific Aim 2: Further Development

With proof-of-concept established on idealized circular geometries (Aim 1), we now confront the messy reality of clinical scenarios: irregular tumor boundaries, heterogeneous tissue properties, and the need for optimal data acquisition strategies. Aim 2 systematically addresses these challenges by extending the methodology beyond simplified assumptions.

#### Aim 2a: Irregular Off-Centered Inclusions

**Motivation:**

Real pathological inclusions (tumors, lesions) exhibit irregular boundaries and arbitrary locations rather than perfect circles centered at the origin. Aim 2a tests whether the same U-Net architecture generalizes to geometric complexity without modification.

**Fourier Boundary Representation:**

Irregular inclusion boundaries are parameterized via Fourier series in local polar coordinates:

r(θ) = R_base × (1 + Σⁿₘₒₐₑₛ [aₙ cos(nθ) + bₙ sin(nθ)])

where:
- R_base: Mean radius
- (aₙ, bₙ): Fourier coefficients randomly sampled from specified distribution
- n_modes: Number of Fourier modes (typically n_modes = 6, range 3–6)
- **Irregularity parameter**: Controls amplitude of Fourier coefficients (higher → more irregular)

**Off-Center Placement:**
- Inclusion center: (cₓ, cᵧ) offset from domain origin
- Boundary check in local coordinates: Transform to inclusion-centered frame, evaluate r(θ)

**Material Assignment:**

For each element centroid (x_global, y_global):
1. Transform to local coordinates: (x_local, y_local) = (x_global - cₓ, y_global - cᵧ)
2. Convert to polar: r = sqrt(x_local² + y_local²), θ = atan2(y_local, x_local)
3. Evaluate boundary radius: r_boundary(θ) using Fourier series
4. Check: If r ≤ r_boundary(θ) → E_inclusion, else E_background

**Validation:**

Test U-Net trained on circular inclusions against irregular inclusion test cases:
- Does reconstruction quality degrade significantly?
- Can the same architecture handle geometric complexity?

Expected result: Some degradation acceptable, but algorithm should demonstrate robustness to moderate irregularity.

---

#### Aim 2b: Fourier Features Multilayer Perceptron (Future Exploration)

**Motivation:**

U-Net architecture excels at segmentation (binary classification) but may struggle with:
- **Continuous heterogeneous material fields**: Gradual spatial variations rather than sharp boundaries
- **Irregular shapes**: Fourier boundary representation suggests Fourier feature embeddings could be beneficial

**Fourier Feature MLP Concept:** (Tancik et al., 2020)

Replace U-Net with coordinate-based MLP augmented with Fourier feature embedding:

Input: (x, y) coordinates
Fourier features: γ(x, y) = [sin(2πB[x, y]ᵀ), cos(2πB[x, y]ᵀ)]
where B is a matrix of random Fourier basis frequencies

MLP: E(x, y) (direct material property prediction)

**Potential Advantages:**
- **Continuous representations**: Can model smooth gradients, not just binary masks
- **Irregular geometry**: Fourier features provide inductive bias for periodic/irregular structures
- **Heterogeneous materials**: Directly predicts E(x, y) rather than binary mask

**Investigation Scope (Aim 2b):**
- Implement Fourier feature MLP architecture
- Compare against U-Net on irregular inclusion test cases
- Explore whether continuous representations improve reconstruction for heterogeneous scenarios

**Status:** Future exploration beyond proof-of-concept

---

#### Aim 2c: Benchmarking Library

**Motivation:**

Systematic evaluation requires standardized test cases spanning the range of clinical relevance and algorithmic difficulty.

**Test Case Categories:**

Based on real-world scenarios:
1. **Breast cancer tumor geometries**: Irregular, off-centered, varying size and contrast
2. **Human forearm anatomy**: Layered structure (muscle surrounding bone), multiple material regions
3. **Irregular boundary samples**: Fourier-perturbed inclusions with varying mode counts and irregularity parameters
4. **Challenging parameter regimes**: Small inclusions, low contrast, combined small + low contrast

**Benchmarking Metrics:**
- SSIM, L2 error, boundary distance error (as in Aim 1d)
- Reconstruction success rate: Percentage of test cases achieving target metrics
- Failure mode analysis: Characterize scenarios where reconstruction fails

**Library Format:**
- Saved `.pt` files containing ground truth geometry, material distribution, boundary displacement data
- Standardized loading interface for reproducibility
- Enables comparison with future alternative algorithms

**Deliverable:**
- Public repository of benchmark test cases
- Baseline PAT Scan results for each case
- Enables community validation and algorithm comparison

---

#### Aim 2d: Circumferential Non-Uniform Force Application Strategy

**Motivation:**

Current force application (Aim 1) uses **constant magnitude** force pairs. For irregular inclusion boundaries, spatially varying force magnitudes might provide richer information about geometric features.

**Proposed Strategy:**

Apply forces with **circumferentially varying magnitude**:

F(θ) = F₀ × [1 + A × cos(mθ + φ)]

where:
- F₀: Base force magnitude
- A: Modulation amplitude
- m: Azimuthal mode number
- φ: Phase offset

**Hypothesis:**

Non-uniform force distributions probe the sample differently than uniform forces:
- Regions with higher force produce larger displacements
- Spatial variation in displacement field may better resolve irregular boundaries
- Complementary information to uniform force configurations

**Implementation:**
- Integrate non-uniform force generation into current framework
- Generate datasets with varying (m, A, φ) parameters
- Train U-Net on combined uniform + non-uniform force datasets
- Evaluate reconstruction quality on irregular inclusion test cases

**Expected Outcome:**

Quantify whether non-uniform forces provide measurable improvement in reconstruction accuracy for irregular geometries versus computational cost of additional force configurations.

---

#### Aim 2e: Vibration-Based Force Application Strategy (Future Comparison)

**Motivation:**

Visual Vibration Tomography (Bouman et al., 2022) demonstrated successful material property reconstruction using **dynamic vibration** rather than quasi-static loading. Comparing static (PAT Scan) versus dynamic (vibration) regimes clarifies relative advantages.

**Physics Difference:**

**Static (PAT Scan):**
- Governing equation: F = KU (equilibrium)
- Quasi-static loading, negligible inertia and damping
- Force pairs applied sequentially

**Dynamic (Vibration):**
- Governing equation: Ku = ω²Mu (modal analysis)
- Vibration modes at natural frequencies
- Requires exciting structure and measuring frequency response

**Comparison Study:**

1. Implement vibration-based data acquisition in simulation
2. Generate synthetic vibration datasets for same geometries as static datasets
3. Train reconstruction algorithms on both static and dynamic data
4. Compare reconstruction quality, data requirements, sensitivity to noise

**Research Questions:**
- Can static forces achieve similar reconstruction quality to dynamic vibration?
- Which regime is more robust to measurement noise?
- Are there scenarios where one approach significantly outperforms the other?

**Expected Insight:**

Understanding relative advantages informs hardware development: if static forces prove competitive, simpler force/displacement sensors could suffice rather than high-speed cameras for vibration measurement.

**Status:** Future work beyond current scope

---

### Specific Aim 3: Incorporating Realism

Aim 3 establishes the pathway to clinical translation by addressing 3D geometry, patient-specific anatomy, and experimental validation.

#### Aim 3a: 3D Extension

**Tetrahedral FEM Formulation:**

Extend 2D plane stress to full 3D elasticity:
- **Elements**: 4-node tetrahedral elements (T4)
- **DOFs**: 3 per node (u_x, u_y, u_z)
- **Constitutive relation**: Full 3D stress-strain (Hooke's law with Young's modulus E, Poisson ratio ν)

**3D Mesh Generation:**
- Tetrahedral mesh of cubic or anatomical domains
- Material assignment to tetrahedral elements via centroid check (3D inclusion geometry)

**3D Neural Network Architectures:**

Option 1: **3D U-Net**
- Extend 2D U-Net to volumetric convolutions (Conv3D)
- Input: 3 channels (X, Y, Z) on 64×64×64 grid
- Output: 1 channel volumetric mask M(x, y, z)

Option 2: **Implicit MLP (Coordinate-Based)**
- Input: (x, y, z) coordinates
- Output: Predicted material property E(x, y, z)
- Advantage: Resolution-independent, continuous representation

**Force Application in 3D:**
- Surface force pairs on outer boundary (spherical or anatomical surface)
- Angular scanning extended to spherical coordinates (θ, φ)
- Determine optimal surface coverage via similar Boussinesq-inspired analysis

**Challenges:**
- Computational cost: 3D FEM solves more expensive than 2D
- Data requirements: Volumetric neural networks require more training samples
- Visualization: 3D inclusion geometry harder to visualize than 2D slices

**Validation:**
- 3D synthetic test cases with known inclusion geometry
- SSIM and L2 error computed on volumetric material fields
- Slice-by-slice visualization for qualitative assessment

---

#### Aim 3b: Integration with Medical Imaging (CT Scan Segmentation)

**Motivation:**

Patient-specific geometry from medical imaging enables anatomically realistic modeling rather than idealized shapes.

**CT Integration Pipeline:**

1. **Acquire CT scan** of patient anatomy (standard clinical imaging)
2. **Segment anatomical structures**: Identify tissue boundaries (e.g., breast tissue boundary, bone-muscle interfaces)
3. **Generate FEM mesh**: Tetrahedral mesh conforming to segmented geometry
4. **Impose anatomical constraints**:
   - Known anatomy from CT informs expected material property ranges
   - Spatial priors: Bone is stiffer than muscle, fat is softer than glandular tissue
   - Geometric constraints: Inclusion must lie within imaged tissue volume

**Visible Human Project Integration:** (Ackerman, 1998)

The Visible Human Project provides high-resolution anatomical data:
- Cross-sectional imaging (CT, MRI) at mm resolution
- Anatomical photographs of tissue slices
- Enables realistic multi-tissue models (skin, fat, muscle, bone)

**Anatomical Constraints as Priors:**

Incorporate physiological knowledge into reconstruction:
- **Spatial priors**: Tumors typically occur in specific tissue types (e.g., breast cancer in glandular tissue)
- **Property bounds**: Stiffness values of biological tissues falls within known ranges
- **Geometric constraints**: Inclusion size and location constrained by anatomical feasibility

**Implementation:**

Extend loss function with anatomical penalty terms:

L = L_MSE + λ_TV × L_TV + λ_anat × L_anat

where L_anat penalizes violations of anatomical constraints (e.g., predicted stiffness outside physiological range).

**Expected Impact:**

Anatomical priors reduce ill-posedness of inverse problem, improving reconstruction accuracy for patient-specific cases.

---

## References

**Historical Quantification of Tissue Mechanics:**
- Schade, H. (1912). Untersuchungen zur Organfunktion des Bindegewebes. 1. Die Elastizitätsfunktion des Bindegewebes und die intravitale Messung ihrer Störungen. *Zeitschrift für experimentelle Pathologie und Therapie*, 11, 369-399.
- Kirk, E., & Kvorning, S. A. (1949). Quantitative measurements of the elastic properties of the skin and subcutaneous tissue in young and old individuals. *Journal of Gerontology*, 4(4), 273-284. https://doi.org/10.1093/geronj/4.4.273
- Sapuntsov, L. E., Mitrofanova, S. I., & Savchenko, T. V. (1979). Assessment of rheologic properties of soft tissues of the human limb with normal and disturbed peripheral lymphatic circulation. *Bulletin of Experimental Biology and Medicine*, 88, 1501-1503. https://doi.org/10.1007/BF00830374

**History of Medical Ultrasound:**
- Newman, P. G., & Rozycki, G. S. (1998). The history of ultrasound. *Surgical Clinics of North America*, 78(2), 179-195. https://doi.org/10.1016/S0039-6109(05)70308-X
- Donald, I., MacVicar, J., & Brown, T. G. (1958). Investigation of abdominal masses by pulsed ultrasound. *Lancet*, 1(7032), 1188-1195. https://doi.org/10.1016/S0140-6736(58)91905-6

**Development of Elastography Field:**
- Lerner, R. M., & Parker, K. J. (1987). Sonoelasticity images derived from ultrasound signals in mechanically vibrated targets. In *Proceedings, Seventh European Communities Workshop: Ultrasonic Tissue Characterization and Echographic Imaging*, pp. 127-129.
- Lerner, R. M., Parker, K. J., Holen, J., Gramiak, R., & Waag, R. C. (1988). Sono-elasticity: Medical elasticity images derived from ultrasound signals in mechanically vibrated targets. In *Acoustical Imaging*, vol. 16, pp. 317-327. Springer, Boston. https://doi.org/10.1007/978-1-4613-0725-9_31
- Ormachea, J., & Parker, K. J. (2020). Elastography imaging: the 30 year perspective. *Physics in Medicine & Biology*, 65(24), 24TR06. https://doi.org/10.1088/1361-6560/abca00
- Garra, B. S. (2015). Elastography: history, principles, and technique comparison. *Abdominal Imaging*, 40, 680-697. https://doi.org/10.1007/s00261-014-0305-8

**Haptic Perception:**
- Chase, E. D. Z., & Follmer, S. (2019). Differences in Haptic and Visual Perception of Expressive 1DoF Motion. In *ACM Symposium on Applied Perception 2019* (SAP '19), Barcelona, Spain. https://doi.org/10.1145/3343036.3343136

**U-Net Architecture:**
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. In *Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015*, Lecture Notes in Computer Science, vol 9351, pp. 234-241. Springer, Cham. https://doi.org/10.1007/978-3-319-24574-4_28 | arXiv: https://arxiv.org/abs/1505.04597

**Material Parameters:**
- Greaves, G. N., Greer, A. L., Lakes, R. S., & Rouxel, T. (2011). Poisson's ratio and modern materials. *Nature Materials*, 10(11), 823-837. https://doi.org/10.1038/nmat3134

**Mechanics-Based Tomography:**
- Konofagou, E. E., & Harrigan, T. P. (2003). Palpation Tomography – A New Technique for Modulus Estimation in Elastography. In *2003 IEEE Ultrasonics Symposium*, pp. 652-655. https://doi.org/10.1109/ULTSYM.2003.1293532
- Goenezen, S., Barbone, P., & Oberai, A. A. (2011). Solution of the nonlinear elasticity imaging inverse problem: The incompressible case. *Computer Methods in Applied Mechanics and Engineering*, 200(13-16), 1406-1420. https://doi.org/10.1016/j.cma.2010.12.018
- Mei, Y., Wang, S., Shen, X., Rabke, S., & Goenezen, S. (2017). Mechanics Based Tomography: A Preliminary Feasibility Study. *Sensors*, 17(5), 1075. https://doi.org/10.3390/s17051075

**Visual Vibration Tomography:**
- Feng, B. T., Ogren, A. C., Daraio, C., & Bouman, K. L. (2022). Visual Vibration Tomography: Estimating Interior Material Properties from Monocular Video. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (CVPR 2022), pp. 16231-16240.

**Physics-Informed Neural Networks:**
- Karniadakis, G. E., Kevrekidis, I. G., Lu, L., Perdikaris, P., Wang, S., & Yang, L. (2021). Physics-informed machine learning. *Nature Reviews Physics*, 3(6), 422-440. https://doi.org/10.1038/s42254-021-00314-5
- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686-707. https://doi.org/10.1016/j.jcp.2018.10.045

**JAX-Based Differentiable Physics:**
- Wu, G. (2024). JAX-SSO: Differentiable Finite Element Analysis Solver for Structural Optimization and Seamless Integration with Neural Networks. *arXiv preprint arXiv:2407.20026*. https://arxiv.org/abs/2407.20026
- Xue, T., Liao, S., Gan, Z., Park, C., Xie, X., Liu, W. K., & Cao, J. (2023). JAX-FEM: A differentiable GPU-accelerated 3D finite element solver for automatic inverse design and mechanistic data science. *Computer Physics Communications*, 291, 108802. https://doi.org/10.1016/j.cpc.2023.108802

**PINNs for Elasticity and Material Identification:**
- Zhang, E., Dao, M., Karniadakis, G. E., & Suresh, S. (2022). Analyses of internal structures and defects in materials using physics-informed neural networks. *Science Advances*, 8(7), eabk0644. https://doi.org/10.1126/sciadv.abk0644
- Zhang, E., Yin, M., & Karniadakis, G. E. (2020). Physics-Informed Neural Networks for Nonhomogeneous Material Identification in Elasticity Imaging. *arXiv preprint arXiv:2009.04525*. https://arxiv.org/abs/2009.04525

**Breast Cancer Tissue Mechanics:**
- Levental, K. R., Yu, H., Kass, L., Lakins, J. N., Egeblad, M., Erler, J. T., Fong, S. F., Csiszar, K., Giaccia, A., Weninger, W., Yamauchi, M., Gasser, D. L., & Weaver, V. M. (2009). Matrix crosslinking forces tumor progression by enhancing integrin signaling. *Cell*, 139(5), 891-906. https://doi.org/10.1016/j.cell.2009.10.027

**Neural Network Theory:**
- Cybenko, G. (1989). Approximation by superpositions of a sigmoidal function. *Mathematics of Control, Signals and Systems*, 2(4), 303-314. https://doi.org/10.1007/BF02551274

**Inverse Problems:**
- Kaipio, J. P., & Somersalo, E. (2007). Statistical inverse problems: Discretization, model reduction and inverse crimes. *Journal of Computational and Applied Mathematics*, 198(2), 493-504. https://doi.org/10.1016/j.cam.2005.09.027

**Classical Mechanics:**
- Boussinesq, J. (1885). *Application des potentiels à l'étude de l'équilibre et du mouvement des solides élastiques*. Paris: Gauthier-Villars.

**Finite Element Method:**
- Zienkiewicz, O. C., Taylor, R. L., & Zhu, J. Z. (2013). *The Finite Element Method: Its Basis and Fundamentals* (7th ed.). Elsevier.

**Medical Imaging:**
- Hounsfield, G. N. (1973). Computerized transverse axial scanning (tomography): Part 1. Description of system. *British Journal of Radiology*, 46(552), 1016-1022. https://doi.org/10.1259/0007-1285-46-552-1016

**Image Quality Assessment:**
- Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality assessment: From error visibility to structural similarity. *IEEE Transactions on Image Processing*, 13(4), 600-612. https://doi.org/10.1109/TIP.2003.819861

**Neural Representations:**
- Tancik, M., Srinivasan, P., Mildenhall, B., Fridovich-Keil, S., Raghavan, N., Singhal, U., Ramamoorthi, R., Barron, J., & Ng, R. (2020). Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains. In *Advances in Neural Information Processing Systems* (NeurIPS 2020). https://arxiv.org/abs/2006.10739

**Anatomical Data:**
- Ackerman, M. J. (1998). The Visible Human Project: A resource for anatomical visualization. *Studies in Health Technology and Informatics*, 52(Pt 2), 1030-1032. https://www.nlm.nih.gov/research/visible/visible_human.html

**Foundational Elastography:**
- Muthupillai, R., Lomas, D. J., Rossman, P. J., Greenleaf, J. F., Manduca, A., & Ehman, R. L. (1995). Magnetic resonance elastography by direct visualization of propagating acoustic strain waves. *Science*, 269(5232), 1854-1857. https://doi.org/10.1126/science.7569924
- Ophir, J., Céspedes, I., Ponnekanti, H., Yazdi, Y., & Li, X. (1991). Elastography: A quantitative method for imaging the elasticity of biological tissues. *Ultrasonic Imaging*, 13(2), 111-134. https://doi.org/10.1177/016173469101300201

**Deep Learning Frameworks and Optimization:**
- Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Köpf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., & Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. In *Advances in Neural Information Processing Systems* (NeurIPS 2019), 8024-8035.
- Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. In *Proceedings of the 3rd International Conference on Learning Representations* (ICLR 2015). https://arxiv.org/abs/1412.6980

**Neural Network Initialization and Architecture:**
- He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification. In *Proceedings of the IEEE International Conference on Computer Vision* (ICCV 2015), 1026-1034. https://doi.org/10.1109/ICCV.2015.123
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (CVPR 2016), 770-778. https://arxiv.org/abs/1512.03385

**Image Processing:**
- Rudin, L. I., Osher, S., & Fatemi, E. (1992). Nonlinear total variation based noise removal algorithms. *Physica D: Nonlinear Phenomena*, 60(1-4), 259-268. https://doi.org/10.1016/0167-2789(92)90242-F
- van der Walt, S., Schönberger, J. L., Nunez-Iglesias, J., Boulogne, F., Warner, J. D., Yager, N., Gouillart, E., Yu, T., & the scikit-image contributors. (2014). scikit-image: Image processing in Python. *PeerJ*, 2, e453. https://doi.org/10.7717/peerj.453

**Adjoint Methods:**
- Plessix, R.-E. (2006). A review of the adjoint-state method for computing the gradient of a functional with geophysical applications. *Geophysical Journal International*, 167(2), 495-503. https://doi.org/10.1111/j.1365-246X.2006.02978.x

**Video Motion Analysis:**
- Wadhwa, N., Rubinstein, M., Durand, F., & Freeman, W. T. (2013). Phase-based video motion processing. *ACM Transactions on Graphics (SIGGRAPH 2013)*, 32(4), Article 80. https://doi.org/10.1145/2461912.2461966

**Scientific Machine Learning Methods:**
- Brunton, S. L., Proctor, J. L., & Kutz, J. N. (2016). Discovering governing equations from data by sparse identification of nonlinear dynamical systems. *Proceedings of the National Academy of Sciences*, 113(15), 3932-3937. https://doi.org/10.1073/pnas.1517384113
- Lu, L., Jin, P., Pang, G., Zhang, Z., & Karniadakis, G. E. (2021). Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators. *Nature Machine Intelligence*, 3(3), 218-229. https://doi.org/10.1038/s42256-021-00302-5
- Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2020). Fourier Neural Operator for Parametric Partial Differential Equations. *arXiv preprint arXiv:2010.08895*. https://arxiv.org/abs/2010.08895
- Udrescu, S.-M., & Tegmark, M. (2020). AI Feynman: A physics-inspired method for symbolic regression. *Science Advances*, 6(16), eaay2631. https://doi.org/10.1126/sciadv.aay2631
- Cranmer, M. (2023). Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl. *arXiv preprint arXiv:2305.01582*. https://arxiv.org/abs/2305.01582

**PINN Elastography and Physics-Informed U-Net:**
- Ragoza, M., Rosen, E., & Bhavsar, A. (2023). Physics-Informed Neural Networks for Tissue Elasticity Reconstruction in Magnetic Resonance Elastography. In *Medical Image Computing and Computer Assisted Intervention – MICCAI 2023*, Lecture Notes in Computer Science, vol 14229, pp. 333-343. Springer, Cham. https://doi.org/10.1007/978-3-031-43999-5_32
- Mohammadi, A., Tanska, P., Korhonen, R. K., Tohka, J., & Guo, X. (2023). Physics-informed UNets for discovering hidden elasticity in heterogeneous materials. *Journal of the Mechanical Behavior of Biomedical Materials*, 150, 106228. https://doi.org/10.1016/j.jmbbm.2023.106228
- Chen, C. T., Gu, G. X. (2021). Learning hidden elasticity with deep neural networks. *Proceedings of the National Academy of Sciences*, 118(31), e2102721118. https://doi.org/10.1073/pnas.2102721118
- ElastoNet (2025). Neural network-based multicomponent MR elastography wave inversion with uncertainty quantification. *Medical Image Analysis*, 101, 103489. https://doi.org/10.1016/j.media.2025.103489
- FDTDNet (2025). Quantification of tissue stiffness with magnetic resonance elastography and finite difference time domain simulation-based spatiotemporal neural network. *Magnetic Resonance in Medicine*. https://doi.org/10.1002/mrm.30456

**Differentiable Finite Element Methods:**
- Sun, Y., Liu, C., & Zeng, Y. (2024). Neural network-augmented differentiable finite element method for boundary value problems. *International Journal of Mechanical Sciences*, 284, 109783. https://doi.org/10.1016/j.ijmecsci.2024.109783
- Bai, J., Zhang, Y., & Wang, Y. (2024). Interpretable physics-encoded finite element network to handle concentration features and multi-material heterogeneity in hyperelasticity. *Computer Methods in Applied Mechanics and Engineering*, 433, 117466. https://doi.org/10.1016/j.cma.2024.117466
- Abueidda, D. W., Lu, Q., & Koric, S. (2023). Finite element method-enhanced neural network for forward and inverse problems. *Advanced Modeling and Simulation in Engineering Sciences*, 10, 6. https://doi.org/10.1186/s40323-023-00243-1

**AI for Science:**
- Jumper, J., Evans, R., Pritzel, A., Green, T., Figurnov, M., Ronneberger, O., Tunyasuvunakool, K., Bates, R., Žídek, A., Potapenko, A., Bridgland, A., Meyer, C., Kohl, S. A. A., Ballard, A. J., Cowie, A., Romera-Paredes, B., Nikolov, S., Jain, R., Adler, J., ... Hassabis, D. (2021). Highly accurate protein structure prediction with AlphaFold. *Nature*, 596(7873), 583-589. https://doi.org/10.1038/s41586-021-03819-2 [2024 Nobel Prize in Chemistry]
- Wang, H., Fu, T., Du, Y., Gao, W., Huang, K., Liu, Z., Chandak, P., Liu, S., Van Katwyk, P., Deac, A., Anandkumar, A., Bergen, K., Gomes, C. P., Ho, S., Kohli, P., Lasenby, J., Leskovec, J., Liu, T.-Y., Manber, A., ... Zitnik, M. (2023). Scientific discovery in the age of artificial intelligence. *Nature*, 620(7972), 47-60. https://doi.org/10.1038/s41586-023-06221-2

**Scientific Machine Learning Surveys:**
- Willard, J., Jia, X., Xu, S., Steinbach, M., & Kumar, V. (2022). Integrating Scientific Knowledge with Machine Learning for Engineering and Environmental Systems. *ACM Computing Surveys*, 55(4), Article 66. https://doi.org/10.1145/3514228
- Cuomo, S., Di Cola, V. S., Giampaolo, F., Rozza, G., Raissi, M., & Piccialli, F. (2022). Scientific Machine Learning through Physics-Informed Neural Networks: Where we are and What's next. *Journal of Scientific Computing*, 92, 88. https://doi.org/10.1007/s10915-022-01939-z

**Universal Differential Equations:**
- Rackauckas, C., Ma, Y., Martensen, J., Warner, C., Zubov, K., Supekar, R., Skinner, D., Ramadhan, A., & Edelman, A. (2020). Universal Differential Equations for Scientific Machine Learning. *arXiv preprint arXiv:2001.04385*. https://arxiv.org/abs/2001.04385

**PINN Theory and Convergence:**
- De Ryck, T., & Mishra, S. (2022). Error analysis for physics-informed neural networks (PINNs) approximating Kolmogorov PDEs. *Advances in Computational Mathematics*, 48, 79. https://doi.org/10.1007/s10444-022-09985-9
- Mishra, S., & Molinaro, R. (2022). Estimates on the generalization error of Physics Informed Neural Networks (PINNs) for approximating a class of inverse problems for PDEs. *IMA Journal of Numerical Analysis*, 42(2), 981-1022. https://doi.org/10.1093/imanum/drab032

**PINN Applications:**
- Moseley, B., Markham, A., & Nissen-Meyer, T. (2020). Solving the wave equation with physics-informed deep learning. *arXiv preprint arXiv:2006.11894*. https://arxiv.org/abs/2006.11894
- Moseley, B., Markham, A., & Nissen-Meyer, T. (2023). Finite Basis Physics-Informed Neural Networks (FBPINNs): a scalable domain decomposition approach for solving differential equations. *Advances in Computational Mathematics*, 49, 62. https://doi.org/10.1007/s10444-023-10065-9

**Neural Fields:**
- Xie, Y., Takikawa, T., Saito, S., Litany, O., Yan, S., Khan, N., Tombari, F., Tompkin, J., Sitzmann, V., & Sridhar, S. (2022). Neural Fields in Visual Computing and Beyond. *Computer Graphics Forum (Eurographics 2022)*, 41(2), 405-438. https://doi.org/10.1111/cgf.14505
- Levis, A., Chael, A., Bouman, K. L., Liske, M., & Wielgus, M. (2022). Gravitationally Lensed Black Hole Emission Tomography. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (CVPR 2022), pp. 19821-19831.
- Bouman, K. L., et al. (2025). Revealing the 3D Cosmic Web through Gravitationally Constrained Neural Fields. In *Proceedings of the International Conference on Learning Representations* (ICLR 2025).

---

**End of Document**
