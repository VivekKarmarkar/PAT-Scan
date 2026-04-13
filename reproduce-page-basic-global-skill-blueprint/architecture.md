# `reproduce-page` — Global Skill Blueprint

> Status: **planning document, not yet built.**
> Date drafted: 2026-04-09
> Author: Vivek + the orange robot, jointly
> Working test case: `../wei_reconstruction/` (Wei et al., page 1 + page 2)

---

## Purpose

`reproduce-page` is a globally-scoped Claude Code skill that takes a research paper PDF + a page number and produces a faithful LaTeX reconstruction of that page (`.tex` source + compiled `.pdf` output). Faithful means:

- **Prose**: byte-identical to the PDF's text layer (extracted via `pdftotext`, no LLM interpretation in the loop)
- **Equations**: hand-transcribed to LaTeX from the rendered page image (visual fidelity, since `pdftotext` mangles math)
- **Layout**: hand-styled to mirror the publisher's structural blocks (title, authors, abstract, sections, etc.) without requiring the publisher's `.cls` file
- **Citation numbers**: preserved as literal `[N]` text markers (no BibTeX bibliography needed)
- **Figures**: not reproduced (placeholders only); equations and structure are the focus

The skill's purpose is **personal study notes**, not republishing. The user already has the original PDF; the LaTeX reconstruction is a transformation that makes the content searchable, manipulable, and re-readable in a clean form.

---

## The Architecture

```
~/.claude/skills/reproduce-page/
├── SKILL.md                          # invocation instructions for Claude
│
├── helpers/                          # small atomic Python scripts
│   ├── extract_pdf_text.py           # wraps `pdftotext -f N -l N <pdf>`
│   ├── clean_soft_hyphens.py         # joins `distri­bution` → `distribution`
│   ├── reflow_paragraphs.py          # collapses PDF line breaks into paragraphs
│   └── escape_latex.py               # %, &, $, _, en-dashes → LaTeX-safe
│
└── examples/                         # concrete worked reconstructions
    ├── wei-explicit-inverse/
    │   ├── Wei-explicit-inverse.pdf  # original (immutable reference)
    │   ├── wei_page1.tex             # reconstruction source
    │   ├── wei_page1.pdf             # compiled output (what good looks like)
    │   ├── wei_page2.tex
    │   ├── wei_page2.pdf
    │   └── NOTES.md                  # what was tricky in this paper
    └── (more papers added over time as worked examples)
```

### What lives where

| Location | What it contains | Why |
|---|---|---|
| `SKILL.md` | Short prose instructions for Claude on how to invoke the helpers and consult the examples | The skill's contract |
| `helpers/` | Standalone runnable Python scripts, one per mechanical text-munging step | The deterministic part of the pipeline |
| `examples/` | Real reconstructions of real paper pages (PDF + tex + compiled PDF + notes) | Few-shot examples; what "good" looks like |

### Key design constraint: helpers are standalone CLI tools

Each helper is invokable from a shell, takes input on stdin or as an argument, writes to stdout. They compose with pipes:

```bash
python3 helpers/extract_pdf_text.py paper.pdf --page 2 \
  | python3 helpers/clean_soft_hyphens.py \
  | python3 helpers/reflow_paragraphs.py \
  > clean_text.txt
```

This is non-negotiable. The helpers are useful **outside** the skill (Vivek can call them directly from a shell) and they document exactly what the skill is doing.

### Examples carry the judgment

The helpers handle the mechanical steps. The examples carry the *judgment* parts that can't be captured in code:

- "The Wei paper has a two-column abstract block — used `minipage` with widths 0.30 and 0.66"
- "The Wei paper's function spaces are 𝓜 × 𝓟 and ℓ × 𝓟 (rendered via `\mathscr{M}`, `\ell`, `\mathscr{P}`)"
- "Page 2's stabilization term subscript is `Ωᵢ`, not `Ωᵉ` — pdftotext shows it correctly even though it mangles the rest of the equation"

These live in each example's `NOTES.md`. They become a growing knowledge base of edge cases.

### Examples double as a regression test corpus

Whenever the helpers or `SKILL.md` change, re-run the examples and visually diff the output PDFs. If `wei_page2.pdf` still compiles to the same byte stream (or visually identical), the helpers are still working. Free regression testing.

---

## The Granularity Decision: `reproduce-page`, not `reproduce-paper`

This is settled. The skill is per-page. Whole-paper workflows are user-level shell composition, not part of the skill.

### What McIlroy would do (settled by appeal to authority)

Doug McIlroy's Unix philosophy: *"do one thing and do it well."* Reproducing one page is one thing. Reproducing a whole paper is "do that one thing N times and concatenate the results" — which is **composition**, not a separate primitive. The Unix instinct is: **never bake the loop into the tool.**

The proof is in how Unix actually evolved:

- There is no `cat-all-files-in-directory` because `cat` takes a file and you use shell to iterate.
- There is no `grep-everything` because `grep` does one search and `find` (or shell globbing) does the iteration.
- There is no `pdftotext-whole-folder` because `pdftotext` does one document and `xargs` does the loop.

**The loop is the user's job, not the tool's job. Always.**

### Three reasons to keep the skill atomic

1. **Premature feature commitment.** A `reproduce-paper` skill bakes in the assumption "the user wants all pages." But maybe Vivek only wants the Methods section (pages 5–8). Or just the page he's currently studying. The atomic version supports all of those workflows for free; the bundled version only supports "do everything." McIlroy was deeply suspicious of features that constrain how the tool gets used.

2. **Failure isolation.** If `reproduce-page` works on pages 1–5 but fails on page 6 because of some weird layout, you can see exactly where the failure is, iterate on that one page, and ship the rest. If `reproduce-paper` is monolithic, the failure is "the whole paper didn't reproduce" and you have to debug from scratch. McIlroy's instinct was always toward graceful partial-failure.

3. **Composition is a universal interface.** A `reproduce-page` skill that takes `(paper, page) → (tex, pdf)` is composable with anything: shell loops, Makefiles, Python scripts, other skills, Jupyter notebooks. A `reproduce-paper` skill is composable with… being run. Once. On a whole paper. Much less power.

### The "but ergonomics" counter, and the McIlroy answer

You could argue: "but it's annoying to have to write the for-loop every time I want a whole paper." McIlroy's response would have been a polite shrug: write the for-loop *once* into a tiny shell script, name it `reproduce-paper.sh`, and put it in your `~/bin`. Five lines:

```bash
#!/bin/bash
# reproduce-paper.sh — wrapper around reproduce-page
PAPER="$1"
N=$(pdfinfo "$PAPER" | awk '/Pages:/ {print $2}')
for p in $(seq 1 $N); do reproduce-page "$PAPER" $p; done
pdfunite $(basename "$PAPER" .pdf)_page*.pdf $(basename "$PAPER" .pdf)_full.pdf
```

That's not a skill, it's a *user-level shortcut*. The skill stays atomic; the convenience lives where convenience belongs — at the user level, in the user's own scripts. That's exactly how Unix was meant to be extended: the system ships with atomic primitives, the users ship workflow scripts that compose them.

### Asymmetric composition costs

The deeper McIlroy principle: **building atomic-first and composing later is cheap; building bundled-first and decomposing later is expensive.** Adding a 5-line wrapper to `reproduce-page` is trivial. Removing the loop from a bundled `reproduce-paper` later (because you need single-page granularity for some new use case) is a refactor + version + migration. The atomic-first design preserves optionality in every direction. McIlroy thought about software the way a cabinet-maker thinks about wood: prefer the move that preserves optionality.

---

## Design Principles (Unix philosophy applied here)

1. **Each helper does one thing.** `extract_pdf_text.py` extracts text. `clean_soft_hyphens.py` cleans soft hyphens. They do not combine concerns.

2. **Helpers compose via stdin/stdout.** Text streams are the universal interface. Helpers can be piped together in any order the user finds useful.

3. **The fewer LLM-flavored steps, the higher the precision.** This is the key insight from the page-2 breakthrough. Every model-based step is a potential hallucination site. Every traditional Unix-tool step is exact by construction. The architecture wins by *minimizing* LLM involvement, not maximizing it.

4. **Each tool stays in its strength zone.**
   - `pdftotext` → prose only (mangles math)
   - Visual Read tool → equations only (hallucinates prose)
   - LLM (me) → recognize document structure + write LaTeX glue
   - `pdflatex` → compile deterministically

5. **Failure is isolated and informative.** When something goes wrong, you can tell which step failed because each step is its own tool with its own input/output.

6. **Examples teach what prose can't.** Few-shot worked examples carry structural patterns (e.g., "Elsevier two-column abstract block as a `minipage`") that are easier to *show* than to *explain*.

7. **No premature features.** The skill does pages. The skill does not do papers. The skill does not download PDFs. The skill does not summarize. The skill does not auto-generate bibliographies. Each of those is its own potential skill, decided independently when actually needed.

---

## Composition Recipe — how to use `reproduce-page` for whole papers

Once `reproduce-page` exists, the whole-paper workflow is a one-liner:

```bash
# all pages
for p in $(seq 1 22); do reproduce-page wei.pdf $p; done

# combine into a single PDF
pdfunite wei_page*.pdf wei_full_reconstruction.pdf
```

Or for a subset (just the Methods section):

```bash
for p in $(seq 5 12); do reproduce-page wei.pdf $p; done
pdfunite wei_page0{5,6,7,8,9}.pdf wei_page1{0,1,2}.pdf wei_methods.pdf
```

Or for a single page Vivek is studying right now:

```bash
reproduce-page wei.pdf 7
xdg-open wei_page7.pdf
```

All three workflows fall out of the same atomic skill. None of them required `reproduce-paper` to exist.

---

## Open Questions / TODOs (decide when actually building)

1. **Output filename convention.** `wei_page2.tex`/`wei_page2.pdf`? `<paper>_<NN>.pdf` (zero-padded)? Where does the output land — current directory? A configurable output dir? Same dir as the source PDF?

2. **The `escape_latex.py` helper's exact responsibilities.** Should it handle all special chars (`% & $ # _ { } ~ ^ \`)? En-dashes? Em-dashes? Curly quotes? It needs a clear contract.

3. **What does the `extract_pdf_text.py` helper output exactly?** Just the raw `pdftotext` dump? Or a "cleaned" version with the soft hyphens already joined? Lean toward raw — composition wants atoms.

4. **How should the `SKILL.md` reference the examples?** "Look at `examples/wei-explicit-inverse/wei_page2.tex` for a reference"? Or "look at the most recent example in `examples/`"? The first is more deterministic; the second auto-uses new examples.

5. **Does the skill produce its own `NOTES.md`** for the new paper it's reproducing? Probably yes — every reproduced paper should accumulate notes about its quirks. This becomes the next example for next time.

6. **Should `reproduce-page` create a new directory per paper** in some `reproductions/` folder, or just dump files in the current working directory? Per-paper directory feels cleaner.

7. **What about the original-PDF immutability rule?** The skill should never modify the input PDF. Worth saying explicitly in `SKILL.md`.

8. **Test coverage strategy.** Re-running all examples after every helper edit is the regression test. But how do we automate "compare output PDFs"? Visual diff? Byte-identical? Skip the check and trust manual review?

9. **Equation transcription is still an LLM step (the orange robot has to do it).** How do we make the contract between the mechanical helpers and the LLM-driven equation step explicit? Probably: the helpers produce a "skeleton" `.tex` with placeholders for equations, and the LLM fills in the equation regions by visually reading the PDF page. The handoff is the seam between mechanical and judgment.

---

## Provenance — where this came from

This blueprint emerged from a session on 2026-04-09 in which Vivek and the orange robot:

1. Tried to build infrastructure around Mathpix MCP, then concluded Mathpix wasn't necessary
2. Discovered (after Vivek pushed back on the robot's gatekeeping) that the Read tool + LaTeX-writing capability could reproduce paper pages directly
3. Built a one-off proof in `wei_reconstruction/wei_page2.tex` (the "IT IS FUCKING AMAZING" moment)
4. Replicated it on page 1 (the "IT IS NEXT LEVEL PERFECTION" moment)
5. Realized the pipeline was actually a hybrid of `pdftotext` (deterministic prose) + visual reads (equations) + LaTeX glue + pdflatex
6. Recognized this is structurally identical to the Unix philosophy of small tools composed via text streams
7. Decided to formalize it as a globally-scoped skill, with helpers as standalone CLI tools and examples as few-shot reference material

The blueprint is the planning artifact for that formalization. The actual skill build will happen when Vivek gives the green light. Until then, this document is the contract.

---

*"Make each program do one thing well. To do a new job, build afresh rather than complicate old programs by adding new 'features'."* — Doug McIlroy, paraphrased
