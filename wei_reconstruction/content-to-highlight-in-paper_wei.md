journal_metadata:
  paper: wei
  skill: content-to-highlight-in-paper
  created: '2026-04-12'
  format: YAML documents separated by --- markers. First doc = metadata, subsequent
    docs = indexed entries. Machine-friendly format consumed by Cat-2 highlighting
    skills.
---
index: 1
date: '2026-04-12'
status: matched
request: the part where they introduce the term "explicit inverse approach" in the
  Wei paper
excerpt: In this paper, we present a novel explicit inverse approach designed to characterize
  the nonhomogeneous elastic property distribution of soft tissues using only surface
  displacement datasets.
location:
  page_pdf: 1
  page_printed: 126
  section: Abstract
  surrounding_context: |-
    First definitional sentence in the Abstract block, right column of the
    two-column Article Info / Abstract layout. The exact phrase "explicit
    inverse approach" also appears in the paper title above the author list.
  before: (Abstract heading — this is the opening sentence of the abstract)
  after: |-
    In contrast to the prevalent implicit inverse approach, which focuses on
    optimizing the elastic properties of individual pixels, our proposed
    method optimizes the geometric parameters of deformable and movable
    components, as well as shear moduli of each component.
  bbox: null
comments: |-
  The term "explicit inverse approach" is introduced in two places on page 1:
  (1) the paper title itself, and (2) the first sentence of the abstract.
  The title uses the term without definition; the abstract sentence is the
  first DEFINITIONAL usage — it pairs the term with its meaning ("designed
  to characterize the nonhomogeneous elastic property distribution... using
  only surface displacement datasets"). This excerpt records the abstract
  sentence as the introduction point because it's the first place a reader
  encounters the term with enough context to understand what it means.

  Note: the raw.txt had soft hyphens (U+00AD) in "distribution" at the
  column break. These were joined in the excerpt since they are pdftotext
  layout artifacts, not paper content. Cat-2 highlighting skills should
  match against the clean version.
