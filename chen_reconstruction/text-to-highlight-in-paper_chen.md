journal_metadata:
  paper: chen
  skill: text-to-highlight-in-paper
  created: '2026-04-16'
  format: YAML documents separated by --- markers. First doc is metadata, subsequent
    docs are indexed entries. Machine-friendly format for Cat-2 highlighting skills.
---
index: 1
date: '2026-04-16'
status: matched
request: 'where they talk about stitching together local estimates to create a global
  map

  '
excerpt: 'The X and Y axes then drive the sample to the next scanning point until
  all preset areas have been scanned. The positional information (x, y, z) corresponding
  to each threshold is saved as projection data to reconstruct a 3D image of the internal
  structure of the sample. This projection data is stored in a CSV file and displayed
  in real time through a visual interface.

  '
location:
  page_pdf: 5
  page_printed: 3
  section: 2.3. Closed-loop feedback control system
  surrounding_context: 'Right column of printed page 3 (PDF page 5), second half of
    Section 2.3.

    This is the final three sentences of the workflow description, concluding

    Section 2 "System design and instrumentation". Immediately followed by

    Section 3 "Imaging principle of tactile tomography" header.

    '
  before: 'The rodless cylinder then continues to drive the probe downward, while
    the grating sensor reads a series of positional data as the resistance value of
    the pressure sensor reaches specific characteristic thresholds. The PLC stops
    and resets the probe once the resistance value of the pressure sensor attains
    the target threshold.

    '
  after: '3. Imaging principle of tactile tomography


    3.1. Softness recognition of tactile tomography system

    '
  bbox: null
comments: "This is the canonical workflow statement of Chen's local-to-global stitching:\n\
  scan one point → record (x, y, z) threshold data → advance to next point →\nrepeat\
  \ until the whole preset area is covered → aggregate all per-point\n(x, y, z) tuples\
  \ into projection data → reconstruct a 3D image. That is\nliterally \"stitch local\
  \ estimates into a global map.\"\n\nThe three ingredients in the user's cue map\
  \ 1:1 to the excerpt:\n  - \"local estimates\" = \"positional information (x, y,\
  \ z) corresponding to\n                        each threshold\" (one tuple per scanning\
  \ point)\n  - \"stitching\" = \"saved as projection data\" + \"drive the sample\
  \ to the\n                  next scanning point until all preset areas have been\n\
  \                  scanned\" (sequential accumulation across the grid)\n  - \"global\
  \ map\" = \"reconstruct a 3D image of the internal structure of\n              \
  \     the sample\"\n\nALTERNATIVE PASSAGE CONSIDERED: Page 6, Section 3.2, has a\
  \ more literal\n\"combining\" sentence: \"Significantly, by combining the compression\
  \ depth\nand its corresponding location information (x and y), Fig. 7c also shows\n\
  the internal hierarchical structure of the sample, indicating that the\ntactile\
  \ tomography system can obtain the internal structure information\nof the sample\
  \ with a soft surface layer based on tactile softness of\nmaterials.\" This is a\
  \ demonstration-scoped statement (tied to Fig. 7c)\nrather than the general workflow.\
  \ I picked the page 5 passage because it\nis the methodological definition, not\
  \ a per-experiment instance — so it\ncarries the general claim the user was pointing\
  \ at.\n\nAdditional relevant context (not excerpted): page 10 discusses the TRADE-OFF\n\
  in this stitching workflow — \"The low number of scanning points make the\nprojection\
  \ data insufficient, which causes poor quality of the reconstructed\n3D images.\
  \ Therefore, increasing the scanning speed is one effective strategy\nto achieve\
  \ both high scanning efficiency and imaging quality.\" This confirms\nthat the \"\
  stitch local to global\" architecture is exactly what bottlenecks\nthe method (sparse\
  \ point sampling → poor 3D reconstruction), which is\ncoherent with the methodological-class\
  \ critique you recorded in the\ncluster-analysis verdict.\n"
---
index: 2
date: '2026-04-16'
status: matched
request: 'that shows a global map is stitched together from local estimates

  '
excerpt: 'By collecting the projection data of positional information (x, y, z) corresponding
  to each pressure threshold, the system reconstructs layer-by-layer 3D images of
  the internal structure.

  '
location:
  page_pdf: 7
  page_printed: 7
  section: Tail of Section 3.2 Tactile softness of materials (methodology-summary
    paragraph, immediately before Section 4)
  surrounding_context: 'Last sentence of the methodology-summary paragraph on printed

    page 7 (PDF page 7), right column. Directly adjacent to the

    Section 4 "Results and discussion" header. This is the paper''s

    one-sentence summary of the local-to-global stitching pipeline

    — the stitching action ("By collecting"), the local inputs

    ("positional information (x, y, z)"), and the global output

    ("layer-by-layer 3D images of the internal structure") all sit

    in a single sentence.

    '
  before: 'This compression depth, combined with the applied force (F) and the contact
    area (S) between the tip and the sample, defines the tactile softness (Softness
    = Δd*S/F), which characterizes the material''s softness at that location.

    '
  after: '4. Results and discussion


    4.1. Theoretical validity of the tactile tomography

    '
  bbox: null
comments: "The user's cue has three ingredients — \"global map,\"\n\"stitched together,\"\
  \ and \"local estimates\" — and this single\nsentence maps each to explicit text\
  \ in the paper:\n\n  \"local estimates\"   $\\to$ \"positional information (x, y,\
  \ z)\n                             corresponding to each pressure threshold\"\n\
  \  \"stitched together\" $\\to$ \"By collecting the projection data\"\n  \"global\
  \ map\"        $\\to$ \"layer-by-layer 3D images of the\n                      \
  \       internal structure\"\n\nThis is the one-sentence version of the stitching\
  \ pipeline.\nPage 5 (text-to-highlight entry 1 of this journal) has the\nstep-by-step\
  \ workflow version — \"drive the sample to the next\nscanning point until all preset\
  \ areas have been scanned... saved\nas projection data to reconstruct a 3D image\
  \ of the internal\nstructure of the sample.\" The page 7 sentence compresses the\n\
  same idea into a single declarative statement suitable for a\nsingle highlight.\
  \ I picked page 7 because the user's request\nemphasized the GLOBAL MAP half of\
  \ the stitching (\"shows a\nglobal map is stitched together\") and the page 7 sentence\
  \ names\n\"layer-by-layer 3D images of the internal structure\" as the\noutput explicitly,\
  \ whereas page 5 names only \"a 3D image\" more\ngenerically.\n\nThe page 5 and\
  \ page 7 highlights complement each other: page 5\nis the workflow, page 7 is the\
  \ summary. Both live in the\ntext-to-highlight journal; they're not duplicates.\n"
---
index: 3
date: '2026-04-16'
status: matched
request: 'the complementary evidence: the page-7 Fig. 8 paragraph showing the layer-to-volume
  demonstration (nine thresholds example)

  '
excerpt: 'Therefore, the tactile tomography system can detect and reconstruct the
  sample layer-by-layer by setting a series of threshold corresponding to increasing
  pressures. For example, nine thresholds of 0, 1, 2, 3, 4, 5, 6, and 7 that corresponding
  to the pressures of 0.5, 0.75, 1.10, 1.50, 1.85, 2.50, 4.00, and 5.00 MPa were set,
  and the reconstructed images can be displayed layer-by-layer in the host computer
  (as shown in Movie S1).

  '
location:
  page_pdf: 7
  page_printed: 7
  section: Section 3.2 Tactile softness of materials (paragraph discussing Fig. 8)
  surrounding_context: 'Left column of printed page 7 (PDF page 7), the paragraph
    that

    immediately discusses Fig. 8 (the four-panel hierarchical-

    staircase scan at 0.5 / 1.5 / 2.5 / 4 MPa). This is the

    complementary passage to the page-7 methodology summary already

    highlighted in entry 2: entry 2 compresses the local-to-global

    ladder into one sentence, this entry expands the LAYER $\to$

    VOLUME step into a demonstration with an explicit nine-threshold

    example and a reference to Movie S1.

    '
  before: 'As shown in Fig. 8a, only the top two stages can be reconstructed and distinguished
    as the tactile tomography system pressed the sample with a pressure of 0.5 MPa.
    As the applied force increases, the deeper stage can be reconstructed and distinguished,
    such as the top 4 stages for the pressure of 1.5 MPa (Fig. 8b), the top 6 stages
    for the pressure of 2.5 MPa (Fig. 8c), and all stage for the pressure of 4 MPa
    (Fig. 8d).

    '
  after: 'These results proved that this imaging system based on the detection of
    tactile softness of materials can image an object by tomography. Moreover, Fig.
    8d also shows the maximum depth of 6 mm, indicating that the tactile tomography
    system can identify an object buried 6 mm below the surface.

    '
  bbox: null
comments: "This is the LAYER $\\to$ VOLUME demonstration — the second rung\nof the\
  \ POINT $\\to$ LAYER $\\to$ VOLUME ladder, made operational\nwith a concrete nine-threshold\
  \ recipe. Three reasons to highlight\nit alongside the two earlier text-to-highlight\
  \ entries:\n\n(1) ENTRY 1 (page 5 workflow, Section 2.3) highlights the\n    per-scanning-point\
  \ MECHANISM (\"drive the sample to the next\n    scanning point until all preset\
  \ areas have been scanned...\n    saved as projection data to reconstruct a 3D image\"\
  ).\n    $\\to$ POINT-level and POINT$\\to$LAYER step in operational\n    terms.\n\
  \n(2) ENTRY 2 (page 7 methodology summary) highlights the FULL\n    LADDER IN ONE\
  \ SENTENCE (\"By collecting the projection data\n    of positional information (x,\
  \ y, z) corresponding to each\n    pressure threshold, the system reconstructs layer-by-layer\n\
  \    3D images of the internal structure\").\n    $\\to$ all three levels stated\
  \ once, compact.\n\n(3) THIS ENTRY (page 7 Fig. 8 discussion) highlights the\n \
  \   LAYER$\\to$VOLUME DEMONSTRATION with a concrete numerical\n    example: nine\
  \ pressure thresholds producing nine layers\n    that together reconstruct the 3D\
  \ volume, with Movie S1 as\n    the visual proof.\n    $\\to$ the layer-to-volume\
  \ step made tangible and countable.\n\nTogether the three highlights form a complete\
  \ trace: mechanism\n(page 5) $\\to$ ladder summary (page 7 methodology summary)\
  \ $\\to$\nladder demonstration (page 7 Fig. 8 paragraph). A reader hovering\non\
  \ the three highlighted passages in sequence sees Chen's\nlocal-to-global assembly\
  \ at three zoom levels.\n\nAlternative that I did NOT add: a text-to-highlight entry\
  \ for\nthe page-7 methodology-summary passage focused specifically on\nthe POINT\
  \ part (\"compression depth at each X-Y position... at\nthat location\"). That passage\
  \ is already doing double duty via\nfind-evidence entry 4 (local-estimate) and entry\
  \ 5 (ladder\nevidence); a third citation in text-to-highlight would be\nredundant.\n"
