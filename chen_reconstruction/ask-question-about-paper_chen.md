journal_metadata:
  paper: chen
  skill: ask-question-about-paper
  created: '2026-04-16'
  format: YAML documents separated by --- markers. First doc = metadata, subsequent
    docs = indexed entries. Machine-friendly format consumed by Cat-2 highlighting
    skills.
---
index: 1
date: '2026-04-16'
status: matched
request: What does the Chen paper do?
excerpt: In this study, we propose a novel tactile tomography system based on mechanical
  principles, featuring internal 3D imaging capabilities. The tactile tomography system
  is constructed from a scanning module, a probe module, and a closed-loop feedback
  control system, enabling it to sense force and control the movement of the probe
  tip. This tactile tomography system can further recognize the softness of soft matter.
  To simulate human tactile perception of softness, a novel parameter of "tactile
  softness of materials" is proposed. The tactile tomography system can achieve the
  function of internal 3D imaging with a maximum detection depth of 6 mm by the combined
  effects of the sample's size, thickness, and the applied force. The performance
  of the tactile tomography system is characterized by its scanning resolution, scanning
  accuracy, and the trade-off between scanning efficiency and imaging quality. Finally,
  the promising potential of the tactile tomography system is illustrated through
  its application in defect detection of encapsulated FPCBs.
location:
  page_pdf: 4
  page_printed: 4
  section: 1. Introduction (self-description paragraph)
  surrounding_context: |-
    Section 1 Introduction, the paragraph immediately following the
    MBT critique ("However, existing mechanical tomography methods
    face challenges..."). This is the standard "In this study, we
    propose..." self-summary paragraph that Elsevier-style papers
    use to announce their contribution before Section 2. Left
    column of printed page 4 (PDF page 4).
  before: However, existing mechanical tomography methods face challenges with low
    reconstruction accuracy and reliability, as displacement measurement noise and
    position deviations of the force can both lead to a significant deterioration
    in the quality of internal structure reconstruction.
  after: |-
    2. System design and instrumentation

    The configuration of the tactile tomography system is shown in Fig. 1, and the photograph of the developed tactile tomography system is presented in Fig. 2.
  bbox: null
comments: |-
  This is the authors' own one-paragraph self-description of the
  paper's contribution, placed where every Elsevier-style paper
  places it: in the Introduction, immediately after the prior-art
  critique pivot, and immediately before Section 2 begins.

  The paragraph covers four distinct things the paper does, in
  order:

  (1) BUILDS a novel tactile tomography system (the hardware
      contribution): scanning module + probe module + closed-loop
      feedback control system. This is an engineered instrument,
      not just an algorithm.

  (2) INTRODUCES a new measurable parameter: "tactile softness of
      materials," defined on page 6 as Softness = $\Delta d \cdot S / F$
      (equation 1). This simulates human haptic perception by
      combining compression depth, contact area, and applied force.

  (3) DEMONSTRATES internal 3D imaging via mechanical probing: up
      to 6 mm detection depth by sweeping the applied force across
      thresholds (layer-by-layer reconstruction).

  (4) APPLIES the system to a downstream use case: defect detection
      in encapsulated flexible printed circuit boards (FPCBs) — the
      hero application in Fig. 15.

  This page-4 passage is the prospective "we propose" framing.
  An alternative candidate is the Conclusion's retrospective "we
  developed" framing (page 13, Section 5, "In this work, a novel
  tactile tomography system based on mechanical principles with
  internal 3D imaging capabilities was developed..."), which adds
  the specific Shore-hardness range (3--60 HC) and the 0.5 mm
  xy-resolution / 0.1 $\mu$m z-resolution / 98.9\% accuracy
  numbers. The Conclusion is more specific; this Introduction
  passage is more self-contained and frames the contribution.

  For a "what does the paper do?" question, the Introduction
  passage is the better answer because it states the contribution
  without presupposing the experimental results.
---
index: 2
date: '2026-04-16'
status: matched
request: How does the core idea in the Chen paper differ from the MBT paper?
excerpt: |-
  MBT leverages solid mechanics equations and surface displacement measurements to reconstruct internal material property distributions through an optimization framework with regularization to handle ill-posed inverse problems. Experimental validations on composite silicone samples have demonstrated MBT's ability to visualize subsurface inclusions by resolving stiffness contrasts, highlighting its potential for non-destructive testing of soft materials [34]. [...] However, existing mechanical tomography methods face challenges with low reconstruction accuracy and reliability, as displacement measurement noise and position deviations of the force can both lead to a significant deterioration in the quality of internal structure reconstruction.
      In this study, we propose a novel tactile tomography system based on mechanical principles, featuring internal 3D imaging capabilities. The tactile tomography system is constructed from a scanning module, a probe module, and a closed-loop feedback control system, enabling it to sense force and control the movement of the probe tip.
location:
  page_pdf: 4
  page_printed: 4
  section: 1. Introduction (MBT positioning → self-proposal pivot)
  surrounding_context: |-
    Two adjacent paragraphs in the Introduction on printed page 4
    (PDF page 4). The first paragraph describes MBT's approach and
    levels a specific critique; the second paragraph ("In this
    study, we propose...") opens Chen's alternative. The excerpt
    bridges them to make the contrast explicit. The "[...]"
    represents the intervening text about ionizing-radiation
    avoidance and functional-imaging advantages, elided to keep
    focus on the methodological-contrast sentences.
  before: '...probe-sample interactions are too weak to penetrate without damaging
    the surface. Recent advancements in mechanical principle-based tomography have
    begun to address the gap in internal imaging, with Mechanics Based Tomography
    (MBT) emerging as a promising approach [32–34].'
  after: This tactile tomography system can further recognize the softness of soft
    matter. To simulate human tactile perception of softness, a novel parameter of
    "tactile softness of materials" is proposed.
  bbox: null
comments: |-
  The Chen paper does NOT contain a single sentence that says
  "our core idea differs from MBT because X." The contrast is
  conveyed IMPLICITLY across two adjacent paragraphs on page 4 —
  MBT's approach is described, MBT's reliability problem is
  stated, and Chen's alternative is announced. The excerpt
  stitches those three moves together so the reader can see the
  whole rhetorical pivot in one view.

  What the excerpt says directly (verbatim from Chen):

  (1) MBT's CORE IDEA: "leverages solid mechanics equations and
      surface displacement measurements to reconstruct internal
      material property distributions through an optimization
      framework with regularization to handle ill-posed inverse
      problems." — i.e., a GLOBAL inverse problem posed as
      constrained optimization over a PDE-governed continuum.

  (2) MBT's LIMITATION (per Chen): "displacement measurement noise
      and position deviations of the force can both lead to a
      significant deterioration in the quality of internal
      structure reconstruction." — i.e., the global-inversion
      approach is reliability-sensitive to noise in the surface
      displacement data.

  (3) CHEN's CORE IDEA: "a novel tactile tomography system based
      on mechanical principles... constructed from a scanning
      module, a probe module, and a closed-loop feedback control
      system, enabling it to sense force and control the movement
      of the probe tip." — i.e., a LOCAL probing instrument that
      senses force + displacement at each scanning point, closing
      the loop hardware-side rather than software-side.

  The methodological-class contrast the excerpt captures (but
  doesn't spell out): MBT solves a GLOBAL inverse problem over a
  PDE model of the whole sample; Chen does LOCAL point-wise
  probing at each (x,y) with force thresholds and stacks the
  results into 3D. Chen's page 6 Eq. 1 defines the local
  measurable (Softness = $\Delta d \cdot S / F$) that replaces
  MBT's optimization objective. Chen's page 5 workflow paragraph
  describes the sequential-grid-scan "stitching" of per-point
  (x, y, z) projection data — the mechanics of the local-to-global
  stitching that distinguishes their approach.

  So the single-sentence version would be: MBT infers the whole
  interior stiffness map by solving a regularized PDE inverse
  problem on the surface displacement field; Chen bypasses that
  inverse problem by physically probing each surface point at
  varying force thresholds and stacking per-point compression-
  depth measurements into a 3D map. Chen's framing trades
  computational mechanics complexity for hardware-integration
  complexity, on the premise that the trade is worth it because
  the global-inversion approach is noise-fragile.

  This is the "local-probing vs. global-system-modeling" class
  boundary that came out of the cluster analysis earlier in the
  session — page 4 is the passage where Chen explicitly stakes out
  which side of that boundary they're on.
---
index: 3
date: '2026-04-16'
status: matched
request: How does the Chen paper recognize the softness?
excerpt: |-
  The tactile tomography system can recognize softness by applying a constant force through a tip pressing against the soft material. This capability is attributed to the closed-loop feedback control system. As shown in Figs 5a, a series of two-component epoxy adhesives (AB glues) with Shore hardnesses of 3, 7.5, 12.5, 17.5, 25, 30, 35, 50, and 60 HC were prepared with a size of 15*15 mm2 and a thickness of 6 mm, and their hardness values were measured by a Shore hardness tester (LX-C) (Fig. 5b). The tactile tomography system pressed each AB glue sample with a constant pressure of 2.5 MPa. The scanning point for each AB glue is 12, within a scanning area of 12*12 mm2. The scanning results are shown in Fig. 5c, where the AB glues with different Shore hardnesses exhibit different compression depths. The average compression depth decreases as the Shore hardness of the AB glues increases (as shown in Fig. 5d), demonstrating a negative correlation. This indicates that the softness of soft materials can be characterized by the compression depth. Therefore, based on the tactile tomography system, the relationship between the softness and compression depth can be defined as
      Softness = Δd / P = Δd · S / F     (1)
  where Δd is the compression depth, F is the force when the tip pressing the sample, and S is the contact area between the tip and the sample.
location:
  page_pdf: 6
  page_printed: 6
  section: 3.1. Softness recognition of tactile tomography system (first paragraph
    + Eq. 1)
  surrounding_context: |-
    Section 3.1 "Softness recognition of tactile tomography system,"
    the full first paragraph, ending with the equation that formally
    defines the tactile-softness parameter. Printed page 6 (PDF page
    6), left column. The section heading itself is on the previous
    page (page 5) — the body of Section 3.1 spans the page-5/page-6
    boundary. This excerpt is the page-6 continuation plus Eq. 1.
    Immediately preceded by the AB-glue-sample preparation setup and
    immediately followed by Section 3.2 ("Tactile softness of
    materials").
  before: |-
    3.1. Softness recognition of tactile tomography system

    The ability of a material sample to resist the penetration of hard objects into its surface is known as hardness, which is an intrinsic property of the material. [36]. According to different measurement approach and standard, hardness can be subdivided into Rockwell hardness, Vickers hardness, Shore hardness, and Brinell hardness. Among these hardness measurements, Shore hardness has been widely applied to the hardness testing of soft materials, such as elastomers, rubber, soft plastics, and foam [37]. Therefore, the softness of soft matter can be characterized by Shore hardness, which exhibits a negative correlation with softness. The tactile tomography system can
  after: |-
    3.2. Tactile softness of materials

    The size and thickness of the sample must be taken into account when testing Shore hardness.
  bbox: null
comments: |-
  The excerpt gives the complete mechanism of softness recognition
  in one self-contained passage, moving from physical setup to
  formal definition in six steps:

  (1) PHYSICAL MECHANISM: "applying a constant force through a tip
      pressing against the soft material" — the measurement is
      force-in, deformation-out, with the constant-force regime
      enforced by the closed-loop feedback control system.

  (2) CONTROL ARCHITECTURE: "This capability is attributed to the
      closed-loop feedback control system" — the control system
      (described in Section 2.3 on page 5) is what makes the
      constant-force regime possible. Pressure sensor reads tip
      force; PLC compares to preset threshold; rodless cylinder
      stops Z-axis descent when the threshold is met.

  (3) CALIBRATION SUBSTRATES: "a series of two-component epoxy
      adhesives (AB glues) with Shore hardnesses of 3, 7.5, 12.5,
      17.5, 25, 30, 35, 50, and 60 HC" — 9 calibrated samples
      covering a 20× hardness range, measured independently by a
      commercial Shore tester (LX-C) to ground-truth their
      hardnesses.

  (4) OBSERVED RELATIONSHIP: "the AB glues with different Shore
      hardnesses exhibit different compression depths. The average
      compression depth decreases as the Shore hardness of the AB
      glues increases... demonstrating a negative correlation" —
      harder sample $\Rightarrow$ smaller compression depth. This
      monotonic relation is the empirical basis for using
      compression depth as a proxy for softness. The quantitative
      fit (Fig. 5d) is $Y = 6.23 - 1.62\,X^1 + 1.15\,X^2$ with
      $R^2 = 0.99596$.

  (5) DEFINITIONAL MOVE: "the softness of soft materials can be
      characterized by the compression depth" — the recognition
      primitive is $\Delta d$ (compression depth) itself.

  (6) FORMAL DEFINITION (Eq. 1): Softness $= \Delta d / P = \Delta
      d \cdot S / F$, where $\Delta d$ is compression depth, $F$
      is applied force, $S$ is contact area, and $P = F/S$ is
      pressure. This normalizes compression depth by pressure so
      softness values are comparable across different probe forces
      and contact geometries.

  To answer the question in one sentence: Chen recognizes softness
  by applying a known constant force (set by the closed-loop
  feedback controller) through a 0.5-mm-diameter probe tip onto
  the sample, measuring the resulting compression depth $\Delta d$
  at that point (via the grating sensor on the Z-axis), and
  computing Softness $= \Delta d \cdot S / F$. This is a local
  point-wise quantity — one softness reading per (x, y) scanning
  point — which is then stacked into the global softness map by
  the grid-scan pipeline on page 5.

  An alternative excerpt candidate was the Section 3.1 OPENING
  paragraph on page 5, which defines hardness and argues that
  softness is characterized by Shore hardness (negative
  correlation). I chose the page-6 passage instead because it is
  the operational answer (HOW recognition is actually performed by
  the instrument) rather than the conceptual setup (WHY hardness
  is related to softness).
---
index: 4
date: '2026-04-16'
status: matched
request: How is the image of a single layer reconstructed in the Chen paper?
excerpt: As shown in Fig. 8a, only the top two stages can be reconstructed and distinguished
  as the tactile tomography system pressed the sample with a pressure of 0.5 MPa.
  As the applied force increases, the deeper stage can be reconstructed and distinguished,
  such as the top 4 stages for the pressure of 1.5 MPa (Fig. 8b), the top 6 stages
  for the pressure of 2.5 MPa (Fig. 8c), and all stage for the pressure of 4 MPa (Fig.
  8d). Therefore, the tactile tomography system can detect and reconstruct the sample
  layer-by-layer by setting a series of threshold corresponding to increasing pressures.
  For example, nine thresholds of 0, 1, 2, 3, 4, 5, 6, and 7 that corresponding to
  the pressures of 0.5, 0.75, 1.10, 1.50, 1.85, 2.50, 4.00, and 5.00 MPa were set,
  and the reconstructed images can be displayed layer-by-layer in the host computer
  (as shown in Movie S1).
location:
  page_pdf: 7
  page_printed: 7
  section: Section 3.2 tail (layer-by-layer reconstruction paragraph, paragraph discussing
    Fig. 8)
  surrounding_context: |-
    Second-to-last paragraph of Section 3.2 on printed page 7 (PDF
    page 7), left column. This paragraph discusses Fig. 8 (the
    four-panel hierarchical-staircase scan at different applied
    pressures) and uses it to explain how each fixed pressure
    threshold produces one layer of the tomography. Immediately
    precedes the methodology-summary paragraph (the "Softness =
    $\Delta d \cdot S / F$" recap that sits just before Section 4).
  before: The object that is buried deep cannot be sensed by the fingers unless a
    large force is applied. Similarly, the tactile tomography system is unable to
    obtain the information of the sample with a thicker soft surface layer if the
    applied force is small.
  after: These results proved that this imaging system based on the detection of tactile
    softness of materials can image an object by tomography. Moreover, Fig. 8d also
    shows the maximum depth of 6 mm, indicating that the tactile tomography system
    can identify an object buried 6 mm below the surface.
  bbox: null
comments: |-
  A SINGLE LAYER in Chen's tomography corresponds to ONE pressure
  threshold. The image of that layer is reconstructed by the
  following sequence:

  (1) Fix a pressure threshold $T$ (e.g., 0.5 MPa for the
      shallowest layer, or 4 MPa for the deepest).
  (2) At each $(x, y)$ scanning point, the probe descends under
      closed-loop feedback control until the pressure sensor
      reading equals $T$; the Z-axis grating sensor records the
      compression depth $z$ at that moment. This is the per-point
      mechanism described in Section 2.3 (page 5).
  (3) The XY stage advances to the next $(x, y)$ and step 2
      repeats until all preset scanning points have been covered.
  (4) The set of $(x, y, z)$ tuples collected at threshold $T$ —
      one tuple per scanning point — IS the image of that layer.
      Fig. 8a shows this for $T = 0.5$ MPa; Fig. 8d shows it for
      $T = 4$ MPa.

  The page-7 excerpt above is the clearest statement of "one
  threshold $\to$ one layer." It uses Fig. 8a-d as a direct
  visual demonstration: four applied pressures, four single-layer
  images, showing how deeper stages of the hierarchical model
  become visible as the threshold increases. The nine-thresholds
  example (pressures 0.5, 0.75, 1.10, 1.50, 1.85, 2.50, 4.00, 5.00
  MPa) is the operational form of the full layer-by-layer 3D
  tomography: nine per-threshold scans, each producing one layer's
  image, which Movie S1 shows being displayed sequentially.

  An alternative excerpt would be the Section 2.3 workflow
  paragraph (page 5), which describes the per-point descent-to-
  threshold mechanism in more mechanical detail. I picked the
  page-7 passage because it explicitly connects the pressure
  threshold to a layer — the conceptual bridge the user's
  question ("how is a SINGLE LAYER reconstructed") hinges on —
  and because Fig. 8a-d concretely show what each single-layer
  image looks like. Page 5 describes HOW the mechanism works at
  each scanning point; page 7 is where Chen explains that doing
  step 2 across all $(x, y)$ at one threshold gives you one
  layer's image.

  Note: in Chen's framing, the "image of a single layer" is NOT
  a 2D top-down slice at fixed $z$. It is a 3D surface of $z$
  values (the interface at which pressure $= T$), covering all
  $(x, y)$ in the scanning grid. Fig. 8a-d render these surfaces
  as rainbow-colored 3D plots. Stacking these per-threshold
  surfaces across increasing $T$ values produces the full
  layered-tomography reconstruction.
---
index: 5
date: '2026-04-16'
status: null
request: How is the Force sensor in the Chen paper calibrated?
comments: |-
  null — Chen's paper does NOT describe any calibration procedure
  for the force (pressure) sensor. An exhaustive search across all
  14 raw.txt files confirms:

  KEYWORD COVERAGE:
    - "calibration" / "calibrat*" — zero hits in the paper body.
      The ONLY hit is in the references list (page 14, ref [31]:
      "Robert G. Cain, S. Biggs, Neil W. Page, Force calibration
      in lateral force microscopy, J. Colloid Interface Sci. 227
      (2000) 55--65"). That is a citation in the AFM-instrument
      background of the Introduction (ref [27-31] block on page
      2), not a description of Chen's own calibration procedure.
    - "pressure sensor" / "force sensor" — hits only describe the
      sensor's ROLE in the closed-loop (detects force changes,
      reaches thresholds, resets probe), not its calibration.
    - No mention of reference weights, known masses, load cells
      used for ground-truthing, linearity tests, hysteresis
      characterization, zero-point procedures, or any other
      standard force-sensor calibration practice.

  WHAT THE PAPER DOES SAY (closest relevant passages):

  (1) Page 5, Section 2.2 (Probe module): "and a carbon fiber
      beams based pressure sensor [35]. To achieve high scanning
      resolution without penetrating the soft layer of the sample,
      the tip was designed with a diameter of 0.5 mm. The springs
      are positioned behind the tip to transmit the force from
      the tip to the pressure sensor and to restore the tip's
      position. The pressure sensor, which is based on carbon
      fiber beams, is capable of quantitatively measuring force
      and responds to the force exerted by the springs."
      $\to$ describes the sensor's MECHANICAL PRINCIPLE
      (force $\to$ spring compression $\to$ carbon-fiber-beam
      deflection $\to$ resistance change) and cites reference [35]
      for the sensor itself. Reference [35] is Hu et al., "Cascade
      amplification effect for mechanical stimuli sensors by
      designing the current path through carbon fiber beams,"
      IEEE Sens. J. 21 (2021) 17410--17418 (page 14). Calibration
      details presumably live in that paper, not in this one.

  (2) Page 5, Section 2.3 (Closed-loop feedback control): "The
      resistance value of the pressure sensor changes when the
      tip of the probe module touches the surface of the sample.
      This resistance value is collected by an ADS1247 acquisition
      chip, which converts the analog signal into a digital
      signal. The digital signal is then transmitted to an
      STM32F030C8T6 microprocessor, where it is identified as the
      first threshold."
      $\to$ describes the signal-chain (sensor $\to$ ADS1247
      ADC $\to$ STM32 MCU $\to$ PLC) but treats "threshold" as a
      pre-set resistance-value target, not as a force value that
      has been calibrated to SI units. In particular, the
      "pressure" values reported throughout Results (0.5, 1.5,
      2.5, 4, 6, 8, 13 MPa) are stated without any description
      of how raw resistance was mapped to megapascals.

  (3) Section 4.3 "Scanning accuracy of the tactile tomography
      system" (page 8-9) — measures the system's ability to
      reproduce a known-geometry gauge block to within 98.9%,
      but this is a GEOMETRIC accuracy test (does the scanner
      reconstruct a 3-mm-high block as 3 mm?), not a force-sensor
      calibration (does a 4 MPa applied force actually read 4 MPa?).

  CONCLUSION:
  Force-sensor calibration is delegated to the cited sensor paper
  [35] (Hu et al., IEEE Sens. J. 2021). Within Chen et al.
  (this paper), the force sensor is treated as a black box that
  produces a resistance value; the mapping from resistance to
  force-in-MPa is never discussed. The paper is genuinely SILENT
  on this question, not just partially informative — there is no
  partial answer to stretch. To get the calibration procedure you
  would need to pull reference [35] and read that paper.

  $\Rightarrow$ status: null (paper does not answer this question)
---
index: 6
date: '2026-04-16'
status: matched
request: How does their Force sensor know the value of the force?
excerpt: |-
  The pressure sensor, which is based on carbon fiber beams, is capable of quantitatively measuring force and responds to the force exerted by the springs. Consequently, the probe module can accurately capture force information when the tip makes contact with the sample.
      [...]
      The resistance value of the pressure sensor changes when the tip of the probe module touches the surface of the sample. This resistance value is collected by an ADS1247 acquisition chip, which converts the analog signal into a digital signal. The digital signal is then transmitted to an STM32F030C8T6 microprocessor, where it is identified as the first threshold.
location:
  page_pdf: 5
  page_printed: 5
  section: Section 2.2 (Probe module) + Section 2.3 (Closed-loop feedback control
    system)
  surrounding_context: |-
    Two adjacent subsections in the system-design section on printed
    page 5 (PDF page 5). Section 2.2 describes the probe-module
    mechanics (tip, springs, pressure sensor); Section 2.3 describes
    the signal chain that routes the pressure-sensor output into the
    closed-loop controller. The excerpt bridges them with "[...]" to
    show the full transduction path: force on tip $\to$ resistance
    change in sensor $\to$ ADC $\to$ MCU threshold.
  before: |-
    2.2. Probe module

    As shown in Figs. 2 and 3, the probe module of the tactile tomography system was assembled from a shell structure, a tip, two springs, and a carbon fiber beams based pressure sensor [35]. To achieve high scanning resolution without penetrating the soft layer of the sample, the tip was designed with a diameter of 0.5 mm. The springs are positioned behind the tip to transmit the force from the tip to the pressure sensor and to restore the tip's position.
  after: Meanwhile, the position information of Z0 is read by the grating sensor,
    and the microprocessor sends this position information to the PLC. The rodless
    cylinder then continues to drive the probe downward, while the grating sensor
    reads a series of positional data as the resistance value of the pressure sensor
    reaches specific characteristic thresholds.
  bbox: null
comments: |-
  PARTIAL ANSWER.

  The paper answers HOW the sensor transduces force into a
  machine-readable value, but does NOT answer HOW that raw value
  is mapped to force in SI units (i.e., the calibration from
  resistance-in-ohms to force-in-newtons or pressure-in-MPa).
  This is the companion question to ask-question entry 5 (null
  on calibration procedure).

  WHAT THE EXCERPT TELLS YOU (the transduction chain, verbatim):

  (1) PHYSICAL TRANSDUCTION: the pressure sensor is "based on
      carbon fiber beams" and "responds to the force exerted by
      the springs." The carbon-fiber-beam design [ref 35, Hu et
      al. 2021, IEEE Sens. J.] is piezoresistive — applied force
      bends the carbon-fiber beams, which changes their electrical
      resistance. The springs between tip and sensor transmit the
      normal force from the tip onto the beam array.

  (2) SIGNAL ACQUISITION: "The resistance value of the pressure
      sensor changes when the tip of the probe module touches the
      surface of the sample. This resistance value is collected by
      an ADS1247 acquisition chip, which converts the analog signal
      into a digital signal." The ADS1247 is a 24-bit
      delta-sigma ADC commonly used with resistive sensors (TI
      part, typically bridge-excited). The raw resistance becomes
      a digital count.

  (3) THRESHOLD COMPARISON: "The digital signal is then transmitted
      to an STM32F030C8T6 microprocessor, where it is identified
      as the first threshold." The MCU compares the digitized
      reading against a pre-set threshold (set in software) to
      decide whether the probe has hit the target force; no
      conversion to SI force units is described.

  WHAT THE EXCERPT DOES NOT TELL YOU:

    - The piezoresistive gain coefficient (how many ohms per
      newton, or per MPa).
    - Whether the raw resistance is zero-corrected, temperature-
      compensated, or hysteresis-corrected.
    - The procedure that maps the set-threshold integer (used by
      the STM32 firmware) to the pressure values reported in
      Results (0.5, 1.5, 2.5, 4, 6, 8, 13 MPa).
    - Whether the "pressure" values quoted throughout Results are
      MEASURED by this sensor or COMMANDED to the rodless
      cylinder's pneumatic regulator (these are two very different
      things — commanded set-point vs. sensor-read actual force).

  So the honest answer to "how does their force sensor KNOW the
  value of the force?" is:

    - It SENSES force via carbon-fiber-beam piezoresistance
      (documented in this excerpt).
    - It ACQUIRES the sensed value via an ADS1247 ADC and
      STM32 MCU (documented in this excerpt).
    - It compares the acquired value to pre-set thresholds
      (documented in this excerpt).
    - But it KNOWS the absolute pressure value (in MPa) only
      if the raw resistance has been calibrated to SI units —
      and that calibration step is NOT documented in this paper
      (see ask-question entry 5 for the silence finding).

  The excerpt is the best available answer to the transduction
  half of the question. For the calibration half, the paper is
  silent and ref [35] would need to be consulted.
---
index: 7
date: '2026-04-16'
status: matched
request: Does the Chen paper allude to the CORE source instrumentation paper in the
  prose?
excerpt: As shown in Figs. 2 and 3, the probe module of the tactile tomography system
  was assembled from a shell structure, a tip, two springs, and a carbon fiber beams
  based pressure sensor [35].
location:
  page_pdf: 5
  page_printed: 5
  section: Section 2.2 (Probe module), opening sentence
  surrounding_context: |-
    Opening sentence of Section 2.2 "Probe module" on printed page
    5 (PDF page 5), left column. The citation [35] is parenthetical
    at the end of the sentence, appended to the noun phrase "carbon
    fiber beams based pressure sensor." Reference [35] itself is
    listed on page 14 as:
      J. Hu, Z. Chen, K. Lao, H. Liang, X. Huang, Z. Li, J. Huang,
      X. Hu, B. Liang, D. Ye, J. Wen, J. Luo, "Cascade amplification
      effect for mechanical stimuli sensors by designing the current
      path through carbon fiber beams," IEEE Sens. J. 21 (2021)
      17410--17418.
  before: 2.2. Probe module
  after: To achieve high scanning resolution without penetrating the soft layer of
    the sample, the tip was designed with a diameter of 0.5 mm.
  bbox: null
comments: |-
  YES — exactly once, as a single parenthetical citation in the
  opening sentence of Section 2.2 (page 5). That is the only
  allusion to the core source instrumentation paper in the entire
  14-page body text. A grep of the raw.txt files confirms "[35]"
  appears in only two places: once here in the body, and once on
  page 14 in the reference list.

  WHAT THIS MEANS OPERATIONALLY:

  (1) THE SENSOR IS CHEN'S OWN GROUP'S PRIOR WORK, not a
      third-party part. Reference [35]'s author list — J. Hu, Z.
      Chen, K. Lao, H. Liang, X. Huang, Z. Li, J. Huang, X. Hu,
      B. Liang, D. Ye, J. Wen, J. Luo — has heavy overlap with
      the Chen paper's own author list (Z. Chen, J. Wen, J. Luo
      appear in both; same Wuyi University research center, same
      Guangdong province funding). So the "core source
      instrumentation paper" is a self-citation to the same
      lab's 2021 IEEE Sensors paper that originally developed
      the carbon-fiber-beam pressure sensor.

  (2) THE ALLUSION IS MINIMAL — a single "[35]" token, no
      accompanying descriptive phrase (not "as described in
      [35]", not "see [35] for details", just "[35]"). The
      reader is expected to recognize that the pressure sensor's
      design, characterization, and calibration are all deferred
      to that one cited paper. There is no discussion in Chen
      (this paper) of what makes the sensor design a "cascade
      amplification effect" or why carbon-fiber-beam current-path
      geometry matters — those details live only in [35].

  (3) THIS EXPLAINS THE SILENCE ON CALIBRATION (ask-question
      entry 5): the Chen paper is NOT an instrumentation paper
      for the pressure sensor — it's an instrumentation paper
      for the whole scanning system that HAPPENS to use the
      group's earlier pressure sensor. Calibration of the sensor
      itself is an already-solved problem per [35], so Chen
      correctly doesn't re-derive or re-validate it. The same
      pattern appears elsewhere in the paper: Shore hardness
      measurement procedure [37], softness-perception physiology
      [38], encapsulation-layer-stiffness characterization [40]
      are all delegated via citation.

  IMPORTANT CLARIFICATION OF "ALLUDE":
    - If "allude" means "cites at least once in the body" $\to$
      YES (this sentence on page 5).
    - If "allude" means "discusses substantively in the prose"
      (e.g., summarizes the sensor's working principle, reports
      its calibration curve, compares it to alternatives) $\to$
      NO — the paper never describes reference [35] beyond the
      bare citation. It names the sensor category ("carbon fiber
      beams based pressure sensor") and cites [35] once; that is
      the full allusion.

  So in one sentence: the Chen paper alludes to the core source
  instrumentation paper with a single bracketed citation [35]
  at the top of Section 2.2, nothing more — and that citation
  points to the same research group's own 2021 IEEE Sensors
  paper, where all the sensor-design and calibration details
  live.
