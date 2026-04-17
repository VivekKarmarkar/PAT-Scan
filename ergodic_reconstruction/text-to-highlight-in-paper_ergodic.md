journal_metadata:
  paper: ergodic
  skill: text-to-highlight-in-paper
  created: '2026-04-16'
  format: YAML documents separated by --- markers. First doc is metadata, subsequent
    docs are indexed entries. Machine-friendly format for Cat-2 highlighting skills.
---
index: 1
date: '2026-04-16'
status: matched
request: 'that the ergodic exploration paper is an example of adaptive sampling

  '
excerpt: 'A closed-loop combination between online force-based viscoelastic parameter
  estimation and ergodic trajectory planning, where stiffness estimates continuously
  reshape the information objective driving exploration;

  '
location:
  page_pdf: 2
  page_printed: 2
  section: I. INTRODUCTION — bulleted contributions list (second bullet)
  surrounding_context: 'Second item in the four-bullet contributions list that closes
    Section I (Introduction), right before Section II (Related Work). The list enumerates
    (1) EID design, (2) closed-loop estimation-planning coupling — THIS bullet, (3)
    real-time HEDAC implementation, (4) convergence-based stopping criterion. Left
    column of page 2, roughly mid-page.

    '
  before: 'The design of an EID tailored to autonomous stiffness mapping, which explicitly
    combines uncertainty, stiffness magnitude, and spatial gradients to guide ergodic
    exploration toward diagnostically relevant regions;

    '
  after: 'A real-time implementation of ergodic palpation using HEDAC, enabling continuous
    exploration–exploitation trade-offs at 100 Hz with continuous motion;

    '
  bbox: null
comments: 'This bullet is the single cleanest statement in the paper that their method
  is adaptive sampling. The key phrase is "stiffness estimates continuously reshape
  the information objective driving exploration" — that IS the definition of adaptive
  sampling: the sampling policy is continuously updated based on what has been learned
  so far, rather than being fixed in advance.


  Alternative passages considered:

  - Page 1 abstract: "a Heat Equation Driven Area Coverage controller enables adaptive,
  continuous trajectory planning." Uses the word "adaptive" but the phrasing focuses
  on trajectory-planning adaptivity, not sampling-policy adaptivity.

  - Page 8 conclusion: "Exploration is guided by an EID tailored to stiffness mapping,
  allowing adaptive focus on diagnostically relevant regions." Also uses "adaptive"
  explicitly and is self-contained, but the mechanism (what adapts based on what)
  is less explicit.

  - Page 3, Section III-A: the α(t) time-varying formula (Eq. 3) where the exploration–exploitation
  weight changes as the ergodic metric evolves — this is the mathematical embodiment
  of the adaptive mechanism described in the bullet.

  - Page 2, Section II intro: "Typically, search algorithms rely on three key components:
  (i) a model of the underlying distribution, (ii) a strategy to select the next sampling
  location, and (iii) a mechanism to store and update the collected data." Frames
  the paper''s work in the standard adaptive-sampling three-component architecture,
  but is generic/definitional rather than a self-claim.


  The page-2 contribution bullet wins because it is BOTH self-contained AND mechanism-explicit:
  it names the feedback loop (estimates → information objective → exploration → new
  estimates) that is the adaptive-sampling signature.

  '
