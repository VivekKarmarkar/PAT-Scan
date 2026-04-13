journal_metadata:
  paper: wei
  skill: find-evidence-in-paper
  created: '2026-04-12'
  format: YAML documents separated by --- markers. First doc = metadata, subsequent
    docs = indexed entries. Machine-friendly format consumed by Cat-2 highlighting
    skills.
---
index: 1
date: '2026-04-12'
status: matched
request: that they haven't provided publicly available data in the Wei paper
excerpt: Data will be made available on request.
location:
  page_pdf: 21
  page_printed: 146
  section: Data availability
  surrounding_context: |-
    Standalone one-line section between "Declaration of competing interest"
    and "Acknowledgments", near the bottom of page 21 (the second-to-last
    page). This is the standard Elsevier data-availability statement slot.
  before: |-
    The authors declare that they have no known competing financial
    interests or personal relationships that could have appeared to
    influence the work reported in this paper.
  after: Acknowledgments
  bbox: null
comments: |-
  SUPPORTS the claim. "Made available on request" is explicitly NOT
  "publicly available." Publicly available would mean deposited in an
  open repository (GitHub, Zenodo, institutional archive) with a URL
  anyone can access. "On request" means you must contact the authors,
  who may or may not respond, may impose conditions, and provide no
  guarantee of timely access. This is a common practice in computational
  mechanics but falls short of modern open-science standards.

  Additionally, the paper provides NO code/software availability statement
  at all. The CRediT section credits Dongmei Zhao with "Software" (page
  21, line 13), confirming code exists, but no repository URL, no
  supplementary materials link, and no open-source license are mentioned
  anywhere in the 22-page paper. So neither the data NOR the code is
  publicly available.

  This is relevant to the PAT-Scan comparison: PAT-Scan's codebase lives
  in a git repository with version-controlled training scripts, while
  the Wei method's implementation is effectively inaccessible to other
  researchers without author cooperation.
---
index: 2
date: '2026-04-12'
status: matched
request: that the idea of "segmentation" is used to reduce problem dimensionality
  in the Wei paper
excerpt: In contrast to the prevalent implicit inverse approach, which focuses on
  optimizing the elastic properties of individual pixels, our proposed method optimizes
  the geometric parameters of deformable and movable components, as well as shear
  moduli of each component. As a result, the proposed approach requires far fewer
  optimization variables, streamlining the process.
location:
  page_pdf: 1
  page_printed: 126
  section: Abstract
  surrounding_context: |-
    Third and fourth sentences of the abstract, immediately after the
    introductory sentence about the "novel explicit inverse approach."
    This is the most concise statement of the pixel-to-region transition
    and its dimensionality consequence in the paper.
  before: |-
    In this paper, we present a novel explicit inverse approach designed to
    characterize the nonhomogeneous elastic property distribution of soft
    tissues using only surface displacement datasets.
  after: |-
    Numerical tests conducted in this study demonstrate the superiority of
    the explicit inverse method over the implicit inverse method, providing
    much-improved reconstructed results.
  bbox: null
comments: |-
  PARTIALLY SUPPORTS. The concept is clearly present but the word
  "segmentation" never appears in the paper (0 occurrences across all
  22 pages). What the paper describes IS functionally a segmentation:
  partition the domain into discrete regions (layers or components),
  assign one material property per region, and optimize the region
  boundaries + per-region shear moduli rather than per-pixel/per-node
  properties. This reduces optimization variables from 3691 (one per
  mesh node, page 8 line 43) to 56 (geometric + material, page 8
  line 38) — a 66x dimensionality reduction.

  The "partially" qualifier is important: the user's claim frames the
  paper's approach as "segmentation," which is a specific term from image
  processing / computer vision. The paper frames the same idea as
  "optimizing geometric parameters of deformable and movable components"
  — a continuum-mechanics / topology-optimization framing. The
  mathematical operation is the same (partition a domain into labeled
  regions); the conceptual framing is different. The paper uses B-spline
  interfaces and level-set TDF functions to define regions — these are
  topology-optimization tools, not segmentation tools, but the effect on
  dimensionality is identical to what segmentation achieves.

  Additional supporting passage on page 18 (Discussion, line 29):
  "it optimizes only the geometric parameters and shear moduli of each
  region, reducing the scale of the optimization problem."
---
index: 3
date: '2026-04-12'
status: matched
request: that the geometric parameters are not fixed but sample dependent in the Wei
  paper
excerpt: In the numerical cases presented in the paper, the total number of predefined
  interfaces (for layered structures) or components (for composites with embedded
  inclusions) in the initial guesses is intentionally kept not much larger than the
  actual target. This is because the topological information of the elastic property
  distribution can often be determined beforehand based on prior knowledge or observations.
  For example, in biomedical imaging, characteristics such as the number, location,
  and rough shape of tumors can be observed and identified. Similarly, for typical
  layered biological tissues like skins or corneas, the total number of layers has
  likely been well studied through in vitro experiments or other research.
location:
  page_pdf: 19
  page_printed: 144
  section: Discussion
  surrounding_context: |-
    First paragraph of the Discussion subsection. This is the passage where
    the authors explicitly acknowledge that the method's setup depends on
    prior knowledge about the specific sample being studied.
  before: (Discussion section heading)
  after: |-
    Due to this available prior knowledge, it is unnecessary to set a large
    number of deformable components or interfaces in the initial guess for
    the explicit inverse method.
  bbox: null
comments: |-
  SUPPORTS. The passage directly confirms that the geometric parameter
  configuration is sample-dependent, not fixed. The key phrase is
  "determined beforehand based on prior knowledge or observations" —
  meaning the user must decide, FOR EACH NEW SAMPLE, how many layers or
  inclusions to assume and how many control points to use.

  Concrete evidence across the three numerical cases shows the variability:
  - Case 1 (layered rectangle, page 8): 6 layers assumed, 10 control
    points per interface → 56 total variables
  - Case 2 (layered ring, page 11): 4 layers assumed, 14 control points
    per interface → 56 total variables (same total, different decomposition)
  - Case 3 (inclusions, page 17): 4 inclusions assumed, 16 geometric
    parameters per inclusion → 69 total variables

  Each case required a different initial guess topology chosen by the
  practitioner based on what they knew about the sample. The method cannot
  be run "out of the box" on a new, unknown sample without first making
  these topological choices. This is the fundamental architectural
  difference from PAT-Scan's U-Net approach, which requires NO topological
  prior — the U-Net learns the material distribution directly from the
  data without the user pre-specifying layers or inclusions.
---
index: 4
date: '2026-04-12'
status: matched
request: that the geometric parameters for some sample X were different than another
  sample Y in the Wei paper
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
    Second paragraph of Case 1 results, after describing the target
    shear modulus distributions for Samples 1-3 (Fig. 6). This passage
    defines the explicit method's variable count for the layered
    rectangular structure.
  before: |-
    The shear modulus values of each layer in the structure are of a
    similar order to that of skin tissue [36].
  after: |-
    From Fig. 7, it can be seen that both the shape of the interfaces
    between the neighboring layers and the shear modulus values of each
    layer varies and finally reach to the target shear modulus distribution.
  bbox: null
comments: |-
  SUPPORTS. The geometric parameters are demonstrably different across the
  three numerical cases. The primary excerpt (page 8) describes Case 1.
  Two contrasting passages prove the parameters change:

  CASE 1 — Layered rectangle (page 8, lines 36-38):
    6 layers assumed, 10 control points per interface
    → 50 geometric + 6 shear moduli = 56 total variables

  CASE 2 — Layered ring (page 11, lines 39-41):
    "the ring structure is considered to consist of 4 layers. For the
    explicit inverse method, each interface is represented by 14 control
    points, leading to a total of 56 optimization variables, which
    includes 4 shear moduli."
    → 52 geometric + 4 shear moduli = 56 total variables
    DIFFERENT from Case 1: fewer layers (4 vs 6), more control points
    per interface (14 vs 10), fewer shear moduli (4 vs 6)

  CASE 3 — Inclusion embedded (page 17, lines 10-11):
    "Each inclusion was represented by 16 geometric parameters and
    1 material property... the total number of optimization variables
    reached 69"
    → 64 geometric + 4 material + 1 background = 69 total variables
    DIFFERENT from both Case 1 and Case 2

  Three cases, three different configurations. The geometric parameter
  setup is NOT fixed — it changes based on the problem geometry (layered
  vs ring vs inclusion), the assumed topology (number of layers or
  inclusions), and the desired boundary resolution (control points per
  interface). The practitioner must re-choose these for each new sample.
---
index: 5
date: '2026-04-12'
status: matched
request: that some geometric parameters are hardcoded
excerpt: Note that we assume that each layer is homogeneous and did not consider spatial
  variations within a layer.
location:
  page_pdf: 4
  page_printed: 129
  section: Methods / Inverse problem
  surrounding_context: |-
    In the TDF (topology description function) formulation for layered
    structures, immediately after defining how the m-th layer is bounded
    by level-set functions phi_{m-1} and phi_m. This sentence establishes
    a fundamental hardcoded assumption about material uniformity within
    each region. Fig. 2 (bilayer example) is referenced just before.
  before: |-
    A bilayer structure can be expressed by the proposed method as shown
    in Fig. 2.
  after: |-
    For the domain with inclusion embedded, phi_i is the TDF of the i-th
    component and Omega_i is the region occupied by the i-th component.
  bbox: null
comments: |-
  SUPPORTS. Multiple geometric/topological parameters are hardcoded
  (fixed before optimization, not learned by the algorithm):

  1. HOMOGENEITY WITHIN REGIONS (page 4, line 22-23 — this excerpt):
     "we assume that each layer is homogeneous and did not consider spatial
     variations within a layer." This is the strongest form of hardcoding:
     the material model WITHIN each region is forced to be spatially
     uniform. The optimizer cannot discover intra-layer gradients.

  2. NUMBER OF LAYERS/COMPONENTS (page 8, line 36):
     "we initially assume a configuration with six layers" — hardcoded
     before optimization. Similarly, page 11: "four layers were predefined";
     page 15: "we employed four inclusions as the initial guess."

  3. NUMBER OF CONTROL POINTS PER INTERFACE (page 8, line 37):
     "we utilize 10 control points to interpolate each interface" — fixed
     before optimization. Page 11: "14 control points"; page 17: "16
     geometric parameters" per inclusion. The optimizer moves control
     points but cannot add or remove them.

  4. B-SPLINE ORDER (page 3): the order p of the B-spline basis functions
     is fixed (not discussed as a variable).

  5. The authors ACKNOWLEDGE this dependency (page 19, line 33):
     "the explicit inverse method relies on predefined interfaces or
     inclusions, which may not accurately capture the complexity of the
     underlying distribution."

  In total: the topology (how many regions), the resolution (how many
  control points), the material model (homogeneous within regions), and
  the curve representation (B-spline order) are ALL hardcoded. Only the
  control point POSITIONS and per-region shear moduli are optimized.
---
index: 6
date: '2026-04-12'
status: matched
request: of a CLEAR example of sample X where the practitioner HARD-CODED a geometric
  parameter with a choice P such that P was a unique choice NOT MADE FOR ANY OTHER
  SAMPLE Y, where Y is NOT X
excerpt: For the explicit inverse method, each interface is represented by 14 control
  points, leading to a total of 56 optimization variables, which includes 4 shear
  moduli.
location:
  page_pdf: 11
  page_printed: 136
  section: 'Results / Case 2: A layered ring structure'
  surrounding_context: |-
    Middle of the Case 2 setup description, after stating the ring has
    4 predefined layers and initial shear modulus guesses of 0.01-0.04 MPa.
  before: |-
    In the initial assumption, the ring structure is considered to consist
    of 4 layers.
  after: |-
    On the other hand, the implicit inverse method involves a much larger
    number of optimization variables, specifically 4030 variables in total.
  bbox: null
comments: |-
  SUPPORTS. Clear example identified:

  Sample X = Case 2 (layered ring structure, page 11)
  Choice P = 14 control points per interface
  This choice is UNIQUE to Case 2 — verified by searching all 22 raw.txt
  files for "14 control":

    - "14 control points" appears ONLY on page 11 (Case 2)
    - Case 1 (layered rectangle, page 8) uses 10 control points
    - Case 3 (inclusions, page 17) uses 16 geometric parameters per inclusion

  The number 14 is a practitioner choice hardcoded before optimization.
  The optimizer moves those 14 control points during the solve but cannot
  add a 15th or remove one. The choice of 14 (not 10, not 16, not 12)
  was made specifically for the ring geometry and was not reused for
  either the rectangular structure or the inclusion problem.

  WHY this matters: the practitioner had to decide "14 is the right number
  of control points for THIS geometry" — a problem-specific design decision
  that a fully automated method wouldn't require. Each new geometry gets
  its own hardcoded resolution choice, and the paper provides no guidance
  on HOW to choose these numbers (no sensitivity study on control-point
  count, no adaptive refinement strategy).
---
index: 7
date: '2026-04-12'
status: null
request: the Wei paper of a passage where the paper shows that for one specific sample,
  the practitioner made a geometric parameter choice that was demonstrably different
  from what they chose for another sample
comments: |-
  No single passage in the paper explicitly contrasts the geometric
  parameter choices between two samples. The paper describes each case's
  setup INDEPENDENTLY on separate pages:

    - Case 1 (page 8): "we utilize 10 control points" / "56 ... variables"
    - Case 2 (page 11): "14 control points" / "56 ... variables"
    - Case 3 (page 17): "16 geometric parameters" / "69 ... variables"

  Each passage states its OWN setup without referencing what was chosen
  for the other cases. The paper never writes something like "unlike
  Case 1 where we used 10 control points, here we use 14." The different
  choices are real and verifiable by cross-referencing three separate
  pages, but no single excerpt demonstrates the contrast.

  The closest passage to a comparative statement is on page 19
  (Discussion): "the total number of predefined interfaces (for layered
  structures) or components (for composites with embedded inclusions) in
  the initial guesses is intentionally kept not much larger than the
  actual target." This acknowledges that the setup varies but does not
  give specific numbers for comparison.

  This is a case where the evidence EXISTS in the paper but is
  DISTRIBUTED across non-adjacent passages rather than concentrated in
  one quotable excerpt.
---
index: 8
date: '2026-04-12'
status: matched
request: that the gradients need to be provided by the practitioner
excerpt: The optimization problem is solved by the limited Broyden-Fletcher-Goldfarb-Shanno(L-BFGS)
  method which requires the objective function and the gradients of the objective
  function with respect to the optimization variables. The gradients of the objective
  function with respect to the optimization variables are obtained by the adjoint
  method which has been thoroughly discussed in [27].
location:
  page_pdf: 6
  page_printed: 131
  section: Methods / Inverse problem (continued)
  surrounding_context: |-
    First paragraph on page 6, immediately before the "Gradient computation"
    subsection. This sentence establishes that L-BFGS REQUIRES gradients
    as input, and that those gradients come from the adjoint method —
    a hand-derived mathematical procedure, not automatic differentiation.
    The subsequent "Gradient computation" subsection (Eqs. 16-17, pages 6-7)
    then derives those gradients explicitly over ~2 pages of equations.
  before: (page break — continues from the Inverse problem subsection on page 5)
  after: |-
    The flow chart of the procedure to solve the inverse problem is shown
    in Fig. 5.
  bbox: null
comments: |-
  SUPPORTS. Two key words in this passage:

  (1) "REQUIRES" — L-BFGS does not compute its own gradients. It requires
      them as input. Someone must provide them.

  (2) "obtained by the ADJOINT METHOD which has been thoroughly discussed
      in [27]" — the gradients come from a hand-derived mathematical
      procedure (the adjoint method), not from automatic differentiation.
      The practitioner must:
      (a) Derive the adjoint equations for their specific forward model
      (b) Implement them in code
      (c) Feed the resulting gradients to L-BFGS

  The paper then spends ALL of pages 6-7 (Eqs. 16-26) deriving these
  gradients explicitly — linearizations of bilinear forms, chain-rule
  expansions through the TDF, adjoint state variable solves. This is
  ~2 pages of hand-derived calculus that the practitioner must get right
  for the optimizer to work. If the forward model changes (e.g., from
  neo-Hookean to a different constitutive law), ALL of these derivations
  must be REDONE from scratch.

  Contrast with autodiff (PyTorch loss.backward()): the gradient
  computation is handled automatically by the framework. The practitioner
  writes ONLY the forward model; gradients come for free. No adjoint
  derivation, no chain-rule expansions, no re-derivation when the model
  changes.
