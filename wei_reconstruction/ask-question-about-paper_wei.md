journal_metadata:
  paper: wei
  skill: ask-question-about-paper
  created: '2026-04-12'
  format: YAML documents separated by --- markers. First doc = metadata, subsequent
    docs = indexed entries. Machine-friendly format consumed by Cat-2 highlighting
    skills.
---
index: 1
date: '2026-04-12'
status: matched
request: what optimizer does the Wei paper use?
excerpt: The optimization problem is solved by the limited Broyden-Fletcher-Goldfarb-Shanno(L-BFGS)
  method which requires the objective function and the gradients of the objective
  function with respect to the optimization variables. The gradients of the objective
  function with respect to the optimization variables are obtained by the adjoint
  method which has been thoroughly discussed in [27].
location:
  page_pdf: 6
  page_printed: 131
  section: Inverse problem (continued from page 5, right before "Gradient computation"
    subsection)
  surrounding_context: |-
    First paragraph on page 6, immediately above the "Gradient computation"
    subsection heading. L-BFGS also appears visually in the Fig. 5 flowchart
    below on the same page, inside the "Is Pi minimized? -> No -> Update
    optimization variables" loop, labeled "L-BFGS".
  before: |-
    (page break — this paragraph continues from the Inverse problem
    subsection on page 5)
  after: |-
    The flow chart of the procedure to solve the inverse problem is shown
    in Fig. 5.
  bbox: null
comments: |-
  The paper uses L-BFGS (limited-memory BFGS), a quasi-Newton optimizer.
  This directly answers the question. Notably, the gradients fed to L-BFGS
  are computed via the ADJOINT METHOD (hand-derived, not autodiff) — this
  is the key architectural difference from PAT-Scan's approach, which uses
  Adam + autodiff via PyTorch's loss.backward(). The adjoint gradient
  derivation occupies pages 6-7 (Eqs. 16-26) and is the most
  labor-intensive part of the Wei paper's methodology.

  The raw.txt renders the en-dash in "Broyden-Fletcher-Goldfarb-Shanno" as
  a plain hyphen; the original PDF uses en-dashes. Recorded as-is from the
  text layer.
---
index: 2
date: '2026-04-12'
status: matched
request: did they use experimental data in the Wei paper?
excerpt: Due to the limited availability of experimental datasets, this paper primarily
  focuses on numerical verification. It should be noted that the implicit inverse
  method has demonstrated successful applications in reconstructing the nonhomogeneous
  shear modulus distribution of soft solids using experimental datasets [38]. However,
  considering the superior performance of the explicit inverse solver in numerical
  validation, we are confident that it can achieve even better results when applied
  to the experimental datasets.
location:
  page_pdf: 19
  page_printed: 144
  section: Discussion
  surrounding_context: |-
    Third and final paragraph of the Discussion subsection, immediately
    after the paragraph acknowledging limitations of the explicit method.
    This is the last paragraph before the page ends (continues on page 20
    with "future research, we plan to explore...").
  before: |-
    ensuring the selection of the most suitable approach for achieving
    meaningful and reliable results.
  after: |-
    In future research, we plan to explore the integration of optical
    measurement techniques with the proposed explicit inverse approach
    (continues on page 20).
  bbox: null
comments: |-
  The answer is NO — they did NOT use experimental data. The paper
  "primarily focuses on numerical verification," meaning all results come
  from simulated/synthetic forward-problem data with known ground-truth
  shear modulus distributions, not from real tissue measurements. They
  explicitly cite "limited availability of experimental datasets" as the
  reason and note that ref [38] (Goenezen et al., 2021) validated the
  IMPLICIT method on experimental data, but their OWN explicit method
  has only been tested on synthetic numerical examples.
---
index: 3
date: '2026-04-12'
status: matched
request: did they use a linear or nonlinear Physics model in Wei paper?
excerpt: |-
  Given the low elastic modulus of soft tissues, this study will consider finite deformation in the analysis. The strong form of the nonlinear elastic problem is stated as follows:
  ...
  We assume the soft tissue of interest obeys the incompressible and neo-Hookean law:
location:
  page_pdf: 2
  page_printed: 127
  section: Methods / Forward problem
  surrounding_context: |-
    First two paragraphs of the "Forward problem" subsection under "Methods",
    immediately after the section heading. The nonlinear model is established
    here as the foundation — Eq. (1) gives the strong form (Div(P)=0), Eq. (2)
    defines the neo-Hookean constitutive law S = mu(I - C^{-1}) + pC^{-1}.
    The weak form follows on lines 65+ as "the nonlinear elastic forward
    problem."
  before: |-
    Methods
    Forward problem
  after: |-
    S = mu(I - C^{-1}) + pC^{-1}  (Eq. 2)
    where C = F^T F is the Cauchy-Green tensor. p is the hydrostatic pressure.
  bbox: null
comments: |-
  NONLINEAR. The paper uses a neo-Hookean hyperelastic constitutive model
  with finite (large) deformation — this is explicitly nonlinear. The key
  phrase "this study will consider finite deformation" on line 41 rules out
  linear elasticity (which assumes infinitesimal/small strain). The
  neo-Hookean law (Eq. 2) is a specific nonlinear strain-energy function.

  This is a significant contrast with PAT-Scan, which uses linear plane-stress
  elasticity. The Wei paper's use of nonlinear mechanics is more physically
  realistic for soft tissue (large deformations are common in biological
  materials) but also makes the inverse problem harder and the hand-derived
  adjoint gradients (pages 6-7) more complex.

  The paper also says "incompressible" alongside neo-Hookean, meaning they
  enforce a volumetric constraint (hydrostatic pressure p as a Lagrange
  multiplier). This adds a mixed formulation (displacement + pressure) to
  the FEM, visible in the weak form on line 65: "Find u^h, p^h in M^h x P^h."
---
index: 4
date: '2026-04-12'
status: matched
request: how fast is the algorithm in the Wei paper?
excerpt: Note that it takes only 52 iterations for convergence. ... Since the total
  number of optimization variables is remarkably reduced in the explicit inverse method,
  the total number of minimization iterations is reduced. The total number of minimization
  iterations for the explicit inverse method is approximately 10 times less than that
  required for the implicit inverse method.
location:
  page_pdf: 8
  page_printed: 133
  section: 'Results / Case 1: A layered rectangular structure'
  surrounding_context: |-
    Middle of the Case 1 results discussion, after describing the 56
    optimization variables (50 geometric + 6 shear moduli) and before
    comparing reconstruction quality. This is the first time the paper
    quantifies speed, using the layered-structure example.
  before: |-
    each layer varies and finally reach to the target shear modulus
    distribution (Sample 1).
  after: |-
    Furthermore, the shear modulus reconstruction obtained through the
    implicit (continues on page 9)
  bbox: null
comments: |-
  The paper reports speed ONLY in iteration counts, NEVER in wall-clock
  time (seconds/minutes). No "CPU time", "runtime", or "elapsed time"
  appears anywhere in the 22-page paper. This is a significant omission
  — iteration counts are not directly comparable to actual speed because:
  (1) each iteration's cost depends on problem size and FEM solve cost,
  (2) explicit and implicit methods may have different per-iteration costs,
  (3) hardware/implementation details are never disclosed.

  The specific numbers available across the paper:
  - Layered case (page 8): 52 iterations for explicit, ~10x less than implicit
  - Inclusion case (page 17-18): 328 iterations for explicit vs 4354 for
    implicit (13.3x speedup in iterations)

  Without wall-clock timing, "how fast" cannot be answered in absolute terms
  (e.g., "runs in 5 minutes"). The paper only answers "how many fewer
  iterations" — which is a relative measure, not an absolute one.
---
index: 5
date: '2026-04-12'
status: matched
request: what does optimize geometric parameters mean? what are geometric parameters?
excerpt: For the layered structure, ri = r1i(t1), r2i(t2), ..., rni(tn) is the geometric
  parameter vector representing the distance between the i th interface and C0 at
  the location ti. For the structure with inclusions embedded, the geometric parameter
  vector ri = r1i, r2i, ..., rni = xi0, yi0, di1, ..., din-2 is consisting of the
  coordinates of the center and distances between the control points to the center
  of i th component.
location:
  page_pdf: 5
  page_printed: 130
  section: Methods / Inverse problem
  surrounding_context: |-
    In the Inverse problem subsection, immediately after the objective
    function (Eq. 14) is introduced. This passage defines the geometric
    parameter vector r_i for BOTH problem types (layered + inclusion).
    It appears between the objective function definition and the
    regularization discussion.
  before: |-
    (Eq. 14 — the objective function to be minimized, involving computed
    vs measured displacement fields)
  after: |-
    In Eq. (14), the discrepancy between the computed displacement field
    and measured displacement field on partial surface is minimized in
    L-2 norm.
  bbox: null
comments: |-
  The paper defines "geometric parameters" explicitly in two cases:

  1. LAYERED STRUCTURES: the geometric parameter vector r_i contains
     the DISTANCES from each B-spline control point to the reference
     boundary C0. These distances define the shape of the interface
     between neighboring material layers. Optimizing them = moving the
     layer boundaries.

  2. INCLUSION STRUCTURES: the geometric parameter vector r_i contains
     the COORDINATES OF THE CENTER (x0, y0) plus the DISTANCES from each
     control point to that center (d1, d2, ...). These define the shape
     and position of each embedded inclusion. Optimizing them = moving
     and reshaping the inclusions.

  In both cases, "optimize geometric parameters" means: adjust the
  positions of B-spline control points that define region boundaries,
  so the computed displacement field (from FEM forward solve) matches
  the measured displacement field. The optimization variables are
  control-point coordinates, NOT per-pixel material properties.

  A concrete example on page 17 (line 10): "Each inclusion was
  represented by 16 geometric parameters and 1 material property" —
  so for one inclusion, there are 16 numbers describing its shape/position
  + 1 number for its stiffness.

  Additional context on page 8 (line 37-38): "we utilize 10 control
  points to interpolate each interface. Thus, the total number of
  optimization variables is 56 including 50 geometric variables and
  6 shear moduli."
---
index: 6
date: '2026-04-12'
status: matched
request: what geometric parameters are HARDCODED by the practitioner prior to optimization
  in the Wei paper?
excerpt: Considering the uncertainty regarding the total number of layers, we initially
  assume a configuration with six layers for the inverse problem, as depicted in Fig.
  7(a). For the explicit inverse scheme, we utilize 10 control points to interpolate
  each interface. Thus, the total number of optimization variables is 56 including
  50 geometric variables and 6 shear moduli.
location:
  page_pdf: 8
  page_printed: 133
  section: 'Results / Case 1: A layered rectangular structure'
  surrounding_context: |-
    Second paragraph of Case 1 results, setting up the explicit inverse
    problem BEFORE showing any optimization results. Fig. 7(a) shows
    the initial guess configuration that follows from these choices.
  before: |-
    The shear modulus values of each layer in the structure are of a
    similar order to that of skin tissue [36].
  after: |-
    From Fig. 7, it can be seen that both the shape of the interfaces
    between the neighboring layers and the shear modulus values of each
    layer varies and finally reach to the target shear modulus distribution.
  bbox: null
comments: |-
  This passage answers the question directly. Two geometric parameters
  are hardcoded by the practitioner BEFORE the optimizer runs:

  (1) NUMBER OF LAYERS: "we initially assume a configuration with six
      layers" — the word "assume" is the key verb; the practitioner
      chose 6, the optimizer does not change this number.

  (2) NUMBER OF CONTROL POINTS PER INTERFACE: "we utilize 10 control
      points to interpolate each interface" — "we utilize" shows
      practitioner agency; the optimizer moves these 10 points but
      cannot add an 11th or remove one.

  Everything ELSE — the 50 geometric variable VALUES (control point
  positions) and the 6 shear modulus VALUES — is what the optimizer
  actually solves for. The hardcoded parameters define the STRUCTURE
  of the optimization problem (how many variables, how they're
  organized); the optimized parameters fill in the VALUES within that
  structure.

  This excerpt is from Case 1. The other cases hardcode different
  numbers: Case 2 (page 11) uses 4 layers / 14 control points;
  Case 3 (page 17) uses 4 inclusions / 16 geometric parameters each.
---
index: 7
date: '2026-04-12'
status: matched
request: do they deal with a 2-component system in the Wei paper?
excerpt: The target shear modulus distributions for one(the homogenous case), two,
  and three layers are depicted in Fig. 6(a), (b), and (c), respectively.
location:
  page_pdf: 8
  page_printed: 133
  section: 'Results / Case 1: A layered rectangular structure'
  surrounding_context: |-
    First paragraph of Case 1 results, describing the three target
    configurations (Samples 1, 2, 3) for the layered rectangular
    structure. Fig. 6 shows the three targets visually.
  before: |-
    we change the location of the concentrated force and incorporate
    multiple surface displacement datasets to solve the inverse problem.
  after: |-
    The shear modulus values of each layer in the structure are of a
    similar order to that of skin tissue [36].
  bbox: null
comments: |-
  YES — the paper deals with a 2-component system in multiple places:

  (1) Case 1, Sample 2 (this excerpt, page 8): "two ... layers" in
      Fig. 6(b) — a bilayer rectangular structure with two distinct
      shear modulus values.

  (2) Case 2 (page 11/14): the "bilayer ring structure" (Fig. 17
      caption on page 14: "Target shear modulus distribution of a
      bilayer ring structure") — inner layer 0.129 MPa, outer layer
      0.386 MPa.

  (3) Case 3, Sample 1 (page 15/17): background tissue + one
      embedded inclusion = 2 material components (1 MPa background +
      5 MPa inclusion).

  The excerpt from page 8 is the most concise because it lists all
  three target configurations in one sentence — "one(the homogeneous
  case), two, and three layers" — making Sample 2 (the 2-component
  case) visible as part of a progression.
---
index: 8
date: '2026-04-12'
status: null
request: what geometric parameters are used for the FIRST EXAMPLE of a 22-component
  system in the Wei paper?
comments: |-
  The paper does not contain a 22-component system. No passage in any of
  the 22 raw.txt files mentions "22" in connection with components, layers,
  regions, or inclusions.

  The largest systems studied are:
  - 3 components (Case 1, Sample 3): a 3-layer rectangular structure
  - 2 components (Case 2): a bilayer ring structure
  - 2 components (Case 3, Sample 2): background + 2 inclusions (= 3 regions)

  The paper never exceeds 3 distinct material regions in any numerical
  example.
---
index: 9
date: '2026-04-12'
status: matched
request: what geometric parameters are used for the FIRST TWO-COMPONENT system in
  the Wei paper?
excerpt: Considering the uncertainty regarding the total number of layers, we initially
  assume a configuration with six layers for the inverse problem, as depicted in Fig.
  7(a). For the explicit inverse scheme, we utilize 10 control points to interpolate
  each interface. Thus, the total number of optimization variables is 56 including
  50 geometric variables and 6 shear moduli.
location:
  page_pdf: 8
  page_printed: 133
  section: 'Results / Case 1: A layered rectangular structure'
  surrounding_context: |-
    Second paragraph of Case 1 results. This setup applies to ALL
    three samples in Case 1, including Sample 2 (the first
    two-component/bilayer system in the paper).
  before: |-
    The target shear modulus distributions for one(the homogenous case),
    two, and three layers are depicted in Fig. 6(a), (b), and (c),
    respectively.
  after: |-
    From Fig. 7, it can be seen that both the shape of the interfaces
    between the neighboring layers and the shear modulus values of each
    layer varies and finally reach to the target shear modulus distribution.
  bbox: null
comments: |-
  The first two-component system in the paper is Case 1, Sample 2 — the
  bilayer rectangular structure (Fig. 6(b), with two layers of different
  shear modulus values).

  The geometric parameters hardcoded for this sample are:
  - 6 layers ASSUMED (even though the target only has 2 — the optimizer
    is expected to collapse the redundant layers)
  - 10 control points per interface
  - Total: 50 geometric variables + 6 shear moduli = 56 optimization variables

  IMPORTANT: these geometric parameters are the SAME for all three
  samples in Case 1 (the 1-component, 2-component, and 3-component
  targets). The paper uses ONE geometric setup for the entire Case 1
  and lets the optimizer find different solutions for different targets.
  The practitioner did NOT change the geometric parameters between
  Sample 1, Sample 2, and Sample 3 within Case 1.

  This is notable because it means the 6-layer / 10-control-point
  configuration was chosen to be "general enough" for all three
  targets within Case 1, rather than being tuned to the specific
  number of target layers.
