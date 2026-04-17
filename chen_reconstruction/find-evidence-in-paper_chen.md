journal_metadata:
  paper: chen
  skill: find-evidence-in-paper
  created: '2026-04-16'
  format: YAML documents separated by --- markers. First doc = metadata, subsequent
    docs = indexed entries. Machine-friendly format consumed by Cat-2 highlighting
    skills.
---
index: 1
date: '2026-04-16'
status: matched
request: The authors explicitly name MBT (Mechanics Based Tomography) as the global-inversion
  alternative and position their own work against it — they deliberately chose the
  local-probing route because they view MBT's displacement-measurement noise as a
  reliability problem.
excerpt: Recent advancements in mechanical principle-based tomography have begun to
  address the gap in internal imaging, with Mechanics Based Tomography (MBT) emerging
  as a promising approach [32–34]. MBT leverages solid mechanics equations and surface
  displacement measurements to reconstruct internal material property distributions
  through an optimization framework with regularization to handle ill-posed inverse
  problems. Experimental validations on composite silicone samples have demonstrated
  MBT's ability to visualize subsurface inclusions by resolving stiffness contrasts,
  highlighting its potential for non-destructive testing of soft materials [34]. Unlike
  X-CT or ultrasonic methods, mechanical principle-based tomography avoids ionizing
  radiation or high-energy waves, making it suitable for delicate samples. In addition,
  mechanical principle-based tomography can map mechanical properties (such as stiffness
  or elastic modulus), providing functional information beyond structural imaging.
  However, existing mechanical tomography methods face challenges with low reconstruction
  accuracy and reliability, as displacement measurement noise and position deviations
  of the force can both lead to a significant deterioration in the quality of internal
  structure reconstruction.
location:
  page_pdf: 4
  page_printed: 2
  section: 1. Introduction
  surrounding_context: |-
    Contiguous paragraph block in the Introduction immediately preceding the
    "In this study, we propose..." transition to the authors' own contribution.
    Left column of a two-column layout, roughly the bottom half of the printed
    page 2 (PDF page 4), directly above the start of Section 2.
  before: '...they rely on surface interactions with a penetration depth of less than
    100 nm, limiting analysis to the outermost surface or near-surface layers. Moreover,
    they cannot handle thick samples (>1 μm) or soft-encapsulated materials (such
    as flexible electronics with soft coatings), as probe-sample interactions are
    too weak to penetrate without damaging the surface.'
  after: In this study, we propose a novel tactile tomography system based on mechanical
    principles, featuring internal 3D imaging capabilities. The tactile tomography
    system is constructed from a scanning module, a probe module, and a closed-loop
    feedback control system...
  bbox: null
comments: |-
  SUPPORTS — all three sub-claims are directly confirmed by a single contiguous
  passage in the Introduction:

  (1) "MBT as the global-inversion alternative": explicit naming with full
      acronym expansion on first reference, plus a methodological description
      that identifies it as global-inversion — "leverages solid mechanics
      equations and surface displacement measurements to reconstruct internal
      material property distributions through an optimization framework with
      regularization to handle ill-posed inverse problems." Citations [32-34]
      point to the Goenezen/Texas A&M MBT papers (ref [34] is explicitly
      "Mechanics based tomography (MBT): validation using experimental data"
      per page 14 of the PDF).

  (2) "They position their own work against it": the passage pivots with
      "However, existing mechanical tomography methods face challenges..." —
      a direct contrast setup — and then the "In this study, we propose..."
      sentence on the next paragraph launches their alternative. The rhetorical
      structure is literally "MBT does X, but MBT has problem Y, therefore we
      propose Z."

  (3) "Displacement-measurement noise as the reliability problem": the
      contrast sentence names two specific causes — "displacement measurement
      noise and position deviations of the force" — and attributes them to
      "low reconstruction accuracy and reliability" and "significant
      deterioration in the quality of internal structure reconstruction."
      This is verbatim the reliability framing you attributed to them.

  No contradicting passage found. The paper is consistent with the claim
  throughout — MBT is the only named global-inversion method, and Chen's
  scanning-instrument approach is explicitly their local-probing alternative.
---
index: 2
date: '2026-04-16'
status: matched
request: Chen cites the MBT paper seriously.
excerpt: Recent advancements in mechanical principle-based tomography have begun to
  address the gap in internal imaging, with Mechanics Based Tomography (MBT) emerging
  as a promising approach [32–34]. MBT leverages solid mechanics equations and surface
  displacement measurements to reconstruct internal material property distributions
  through an optimization framework with regularization to handle ill-posed inverse
  problems. Experimental validations on composite silicone samples have demonstrated
  MBT's ability to visualize subsurface inclusions by resolving stiffness contrasts,
  highlighting its potential for non-destructive testing of soft materials [34]. Unlike
  X-CT or ultrasonic methods, mechanical principle-based tomography avoids ionizing
  radiation or high-energy waves, making it suitable for delicate samples. In addition,
  mechanical principle-based tomography can map mechanical properties (such as stiffness
  or elastic modulus), providing functional information beyond structural imaging.
location:
  page_pdf: 4
  page_printed: 2
  section: 1. Introduction
  surrounding_context: |-
    The "serious engagement" portion of the MBT paragraph in the Introduction —
    ends at "providing functional information beyond structural imaging" and
    is immediately followed by the "However, existing mechanical tomography
    methods face challenges..." critique pivot that was recorded in entry 1.
    Left column of printed page 2 (PDF page 4).
  before: '...as probe-sample interactions are too weak to penetrate without damaging
    the surface.'
  after: However, existing mechanical tomography methods face challenges with low
    reconstruction accuracy and reliability, as displacement measurement noise and
    position deviations of the force can both lead to a significant deterioration
    in the quality of internal structure reconstruction.
  bbox: null
comments: |-
  SUPPORTS — Chen engages with MBT as serious prior art along four measurable
  dimensions:

  (1) BREADTH OF CITATION: "[32-34]" is a triple citation, not a throwaway
      single reference. The three references (page 14 of the PDF) are:
        [32] Mei, Wang, Shen, Rabke, Goenezen — "Mechanics based tomography:
             a preliminary feasibility study" (Sensors, 2017) — the foundational
             MBT paper from the Texas A&M group.
        [33] Olson, Throne, Rusnak, Gannon — "Force-based stiffness mapping for
             early detection of breast cancer" (Inverse Prob. Sci. Eng., 2021)
             — independent MBT-adjacent work from the Rose-Hulman group.
        [34] Goenezen, Kim, Kotecha, Luo, Hematiyan — "Mechanics based
             tomography (MBT): validation using experimental data"
             (J. Mech. Phys. Solids, 2021) — the MBT experimental validation
             paper, cited AGAIN in the body text ("Experimental validations on
             composite silicone samples... [34]") to anchor a specific claim.
      This is multi-group, multi-paper engagement — not a token name-drop.

  (2) METHODOLOGICAL ACCURACY: Chen's one-sentence summary of MBT — "leverages
      solid mechanics equations and surface displacement measurements to
      reconstruct internal material property distributions through an
      optimization framework with regularization to handle ill-posed inverse
      problems" — is a correct and compact description of the MBT approach.
      Authors who cite without reading usually get this wrong.

  (3) POSITIVE FRAMING LANGUAGE: "emerging as a promising approach",
      "demonstrated MBT's ability to visualize subsurface inclusions by
      resolving stiffness contrasts", "highlighting its potential for
      non-destructive testing of soft materials". These are sincere
      endorsements, not faint praise.

  (4) ACKNOWLEDGMENT OF MBT'S COMPARATIVE ADVANTAGES: "Unlike X-CT or
      ultrasonic methods, mechanical principle-based tomography avoids
      ionizing radiation or high-energy waves, making it suitable for delicate
      samples. In addition, mechanical principle-based tomography can map
      mechanical properties (such as stiffness or elastic modulus), providing
      functional information beyond structural imaging." Chen makes MBT's
      case FOR it, explaining why it's a meaningful contribution to the field,
      before pivoting to the critique.

  The critique that follows (entry 1: "displacement measurement noise and
  position deviations") is therefore a substantive methodological
  disagreement, not a dismissal — it's the kind of critique you only earn the
  right to make by first taking the prior work seriously. The rhetorical
  pattern is "praise → specific technical limitation → alternative", which
  is engaged prior-art positioning.

  No contradicting passage. Chen cites MBT only in this paragraph plus the
  references list; the tone is consistent.
---
index: 3
date: '2026-04-16'
status: matched
request: This work came after Fall 2025.
excerpt: |-
  Received 25 October 2024; Received in revised form 12 December 2025; Accepted 2 January 2026
  Available online 3 January 2026
location:
  page_pdf: 1
  page_printed: 1
  section: Publication metadata (bottom-of-page journal footer)
  surrounding_context: |-
    Bottom-of-page Elsevier journal metadata block, below the
    corresponding-author addresses and the DOI line, above the
    copyright line. Printed page 1 (PDF page 1).
  before: https://doi.org/10.1016/j.measurement.2026.120300
  after: 0263-2241/\u00a9 2026 Elsevier Ltd. All rights are reserved, including those
    for text and data mining, AI training, and similar technologies.
  bbox: null
comments: |-
  PARTIALLY_SUPPORTS.

  The claim "this work came after Fall 2025" admits two readings,
  and the excerpt supports one reading while contradicting the
  other:

  (A) If "came" means "was published / accepted / made available":
      SUPPORTS the claim. The revised manuscript was submitted on
      12 December 2025 (late Fall 2025, technically still within
      Vivek's stated Fall-2025 window but near its end), accepted
      2 January 2026, and appeared online 3 January 2026. Both the
      journal volume header ("Measurement 264 (2026) 120300") and
      the DOI stub (".2026.120300") confirm the paper is a 2026
      publication. The accepted / published-online dates sit
      firmly AFTER Fall 2025 by any standard academic-calendar
      reading.

  (B) If "came" means "the research itself was carried out /
      originally submitted": CONTRADICTS the claim. The original
      submission date "Received 25 October 2024" is more than a
      year BEFORE Fall 2025 — meaning the underlying research and
      first manuscript predate Vivek's Fall-2025 literature-review
      window substantially. The authors were working on this
      system throughout 2024 and early-to-mid 2025.

  OPERATIONAL IMPLICATION for the PAT-Scan lit review: per reading
  (A), the Chen paper is genuinely a post-Fall-2025 addition that
  a Fall-2025-closed review could not have caught — same structural
  reason the Ergodic Exploration paper (IEEE RAL, accepted Feb 2026)
  was absent. Per reading (B), the UNDERLYING WORK existed during
  the review window, but only in a form (preprint / under-review
  manuscript) that Vivek's broad review wouldn't have had access
  to via standard search indices. Either way, the legitimacy of the
  paper being "missed" by the Fall-2025 review is preserved — the
  lit review wasn't negligent, the paper just wasn't findable by
  publication date.

  No contradicting passage elsewhere in the paper. The metadata
  block is the canonical record of submission/acceptance dates.
---
index: 4
date: '2026-04-16'
status: matched
request: Chen's method of computing softness is a local-estimate.
excerpt: As the probe interacts with the sample, the pressure sensor detects force
  changes, and the grating sensor records the corresponding compression depth (Δd)
  at each X-Y position. This compression depth, combined with the applied force (F)
  and the contact area (S) between the tip and the sample, defines the tactile softness
  (Softness = Δd*S/F), which characterizes the material's softness at that location.
  By collecting the projection data of positional information (x, y, z) corresponding
  to each pressure threshold, the system reconstructs layer-by-layer 3D images of
  the internal structure.
location:
  page_pdf: 7
  page_printed: 7
  section: Tail of Section 3.2 Tactile softness of materials
  surrounding_context: |-
    Immediately before Section 4 "Results and discussion" on
    printed page 7 (PDF page 7). Right column. This paragraph is
    the summary-of-methodology block that recapitulates the full
    softness-recognition mechanism and explicitly names its scope
    as per-point ("at each X-Y position," "at that location"),
    then describes the aggregation step ("By collecting the
    projection data...") that stitches the per-point readings
    into a 3D map. Directly adjacent to the Section 4 header.
  before: The tomography methodology employed in this tactile tomography system is
    based on mechanical principles and the novel parameter of "tactile softness of
    materials", which simulates human haptic perception by integrating the effects
    of sample size, thickness, and applied force. The system utilizes a closed-loop
    feedback control system to drive the probe module to scan the sample in the X-Y
    plane with a preset scanning step and speed. During scanning, the Z-axis rodless
    cylinder applies a series of increasing pressures to the sample through the probe.
  after: |-
    4. Results and discussion

    4.1. Theoretical validity of the tactile tomography
  bbox: null
comments: |-
  SUPPORTS — unambiguous verbatim support, in the authors' own
  words, on three concurrent dimensions:

  (1) SPATIAL LOCALITY: "the grating sensor records the
      corresponding compression depth ($\Delta d$) at each X-Y
      position" — one measurement per (x, y) sampling point, not
      a global field. The paper uses "at each X-Y position" to
      describe the measurement locale.

  (2) POINT-WISE DEFINITION OF SOFTNESS: "defines the tactile
      softness (Softness = $\Delta d \cdot S / F$), which
      characterizes the material's softness AT THAT LOCATION" —
      verbatim "at that location" (lowercase-exact in the PDF at
      line 94 of page 7 raw). This is Chen explicitly calling
      softness a per-location quantity. The locality is not
      inferred — the authors state it.

  (3) EXPLICIT STITCHING CLAUSE: "By collecting the projection
      data of positional information (x, y, z) corresponding to
      each pressure threshold, the system reconstructs
      layer-by-layer 3D images of the internal structure." — the
      3D "global" map is built by collecting many per-point
      readings, not solved from a global inverse problem. This is
      the local-to-global stitching step that distinguishes
      Chen's approach from MBT's PDE-inverse approach (already
      recorded in entries 1 and 2 of this journal, and in
      ask-question entry 2).

  The word "local" does not appear in Chen's paper. But the
  concepts — per-point measurement, per-location softness value,
  aggregation-into-global-via-collection — are all stated
  verbatim. The user's claim "local-estimate" is the natural
  technical label for exactly what Chen describes.

  Alternative excerpt: page 6 Eq. (1) defines Softness $= \Delta d
  / P = \Delta d \cdot S / F$, which is also a per-point formula.
  I picked the page-7 passage instead because it is the only
  place in the paper where locality is named explicitly ("at each
  X-Y position", "at that location") AND the stitching step is
  stated adjacently — this single paragraph captures the entire
  local-to-global pipeline in one place.

  No contradicting passage. Nowhere in the paper does softness
  appear as anything other than a per-point scalar.
---
index: 5
date: '2026-04-16'
status: matched
request: CONCLUSIVE EVIDENCE that they START AT POINTWISE ESTIMATES and move out from
  there in the Chen paper -> POINT2LAYER -> LAYER2VOLUME
excerpt: As the probe interacts with the sample, the pressure sensor detects force
  changes, and the grating sensor records the corresponding compression depth (Δd)
  at each X-Y position. This compression depth, combined with the applied force (F)
  and the contact area (S) between the tip and the sample, defines the tactile softness
  (Softness = Δd*S/F), which characterizes the material's softness at that location.
  By collecting the projection data of positional information (x, y, z) corresponding
  to each pressure threshold, the system reconstructs layer-by-layer 3D images of
  the internal structure.
location:
  page_pdf: 7
  page_printed: 7
  section: End of Section 3.2 Tactile softness of materials — methodology summary
    paragraph, immediately before Section 4
  surrounding_context: |-
    Final three sentences of the methodology-summary paragraph at
    the end of Section 3.2 on printed page 7 (PDF page 7), right
    column. Directly adjacent to the Section 4 "Results and
    discussion" header. This is the paper's one-paragraph
    recapitulation of the full tactile-tomography methodology —
    the single place where all three levels of the local-to-global
    ladder are stated back-to-back.
  before: The tomography methodology employed in this tactile tomography system is
    based on mechanical principles and the novel parameter of "tactile softness of
    materials", which simulates human haptic perception by integrating the effects
    of sample size, thickness, and applied force. The system utilizes a closed-loop
    feedback control system to drive the probe module to scan the sample in the X-Y
    plane with a preset scanning step and speed. During scanning, the Z-axis rodless
    cylinder applies a series of increasing pressures to the sample through the probe.
  after: |-
    4. Results and discussion

    4.1. Theoretical validity of the tactile tomography
  bbox: null
comments: |-
  SUPPORTS — conclusive, in the authors' own words, in a single
  contiguous paragraph. The excerpt contains all three levels of
  the local-to-global ladder stated explicitly and in sequence:

  LEVEL 1 — POINTWISE ESTIMATE ("START AT POINTWISE ESTIMATES"):

    "the grating sensor records the corresponding compression
     depth ($\Delta d$) at each X-Y position. This compression
     depth, combined with the applied force ($F$) and the contact
     area ($S$) between the tip and the sample, defines the
     tactile softness (Softness $= \Delta d \cdot S / F$), which
     characterizes the material's softness at that location."

    $\to$ per-(x,y) softness value, named "at that location."
    This IS the pointwise estimate — one scalar per scanning
    point, computed from a local contact measurement.

  LEVEL 2 — POINT $\to$ LAYER ("POINT2LAYER"):

    "By collecting the projection data of positional information
     $(x, y, z)$ corresponding to each pressure threshold..."

    $\to$ for a FIXED pressure threshold, collect the $(x, y, z)$
    tuples across all scanning points. The set of tuples at one
    threshold IS one layer. (Fig. 8a--d on the same page visualize
    this directly: Fig. 8a = layer at 0.5 MPa, Fig. 8d = layer at
    4 MPa.) Note the verbatim "each pressure threshold" — one
    threshold, one layer.

  LEVEL 3 — LAYER $\to$ VOLUME ("LAYER2VOLUME"):

    "...the system reconstructs layer-by-layer 3D images of the
     internal structure."

    $\to$ stacking layers across multiple pressure thresholds
    produces the full 3D volumetric image. "Layer-by-layer 3D
    images" is the verbatim phrase for the layer-to-volume
    assembly. The earlier page-7 paragraph (Fig. 8 discussion)
    explicitly demonstrates this with nine thresholds (0.5, 0.75,
    1.10, 1.50, 1.85, 2.50, 4.00, 5.00 MPa) producing nine layers
    that together reconstruct the full 3D volume.

  NO CONTRADICTING PASSAGE exists. The paper never describes a
  top-down, volume-first, or layer-first procedure. The volume is
  always assembled from layers, and each layer is always assembled
  from per-point readings. The ladder is strict and one-directional:
  point $\to$ layer $\to$ volume.

  COMPLEMENTARY EVIDENCE in neighboring passages:

    - Page 5 (Section 2.3, find-evidence entry 1 + text-to-
      highlight entry 1): the per-scanning-point MECHANISM ("drive
      the sample to the next scanning point until all preset areas
      have been scanned. The positional information (x, y, z)
      corresponding to each threshold is saved as projection data
      to reconstruct a 3D image"). Shows HOW the points are
      collected at each threshold.

    - Page 7 Fig. 8 paragraph (ask-question entry 4): the LAYER-
      TO-VOLUME demonstration ("Therefore, the tactile tomography
      system can detect and reconstruct the sample layer-by-layer
      by setting a series of threshold corresponding to increasing
      pressures"). Nine thresholds $\to$ nine layers $\to$ full
      volume.

    - Page 7 methodology summary (this entry): the one-paragraph
      version that names all three levels in sequence. This is the
      excerpt I chose as the single strongest passage because the
      three levels appear in three consecutive sentences.

  The passage "$(x, y, z)$ at each X-Y position" also makes clear
  that the "pointwise estimate" Chen starts with is a per-scan-
  point $(\Delta d, F)$ pair, not a field or a model parameter.
  This is the strongest possible evidence for the user's claim
  that Chen's method is structured as a local-to-global ladder
  starting from pointwise estimates.
