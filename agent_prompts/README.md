# Agent Prompts for PAT Scan Comps Document Generation

This folder contains 7 specialized agent prompts designed to analyze your PAT Scan project and generate comprehensive exam documents.

## Agent Overview

### Phase 1: Analysis Agents (Read & Understand)
1. **phase1_quickie.txt** - Fast analysis (~12-15 min)
2. **phase1_good.txt** - Thorough analysis (~25-30 min)
3. **phase1_obsessed.txt** - Exhaustive analysis (~50-70 min)

### Phase 2: Writing Agents (Create Documents)
4. **phase2_quickie.txt** - Rapid draft (~8-12 min)
5. **phase2_good.txt** - Solid draft with LaTeX (~20-28 min)
6. **phase2_obsessed.txt** - Publication-quality draft (~35-50 min)

### Utility Agents
7. **ai_hound_humanizer.txt** - Detects and removes AI writing signatures (~15-25 min)

## How to Use These Prompts

### Method 1: Via Claude Code CLI

```bash
# Create an agent using the /agent command
# Copy the content from one of the .txt files and paste it as the prompt
```

### Method 2: Programmatically (if supported)

Each file contains:
- Agent name
- Subagent type (general-purpose)
- Model (sonnet)
- Description
- Full prompt

## Workflow

### Recommended Sequence:

**Step 1: Run Phase 1 Agents**
- Run all 3 Phase 1 agents in parallel (quickie, good, obsessed)
- They will create analysis files in `/agents/phase1/`
- Review their outputs

**Step 2: Run Phase 2 Agents**
- Run all 3 Phase 2 agents in parallel
- They read Phase 1 analysis and generate comps documents
- Documents saved to `/agents/phase2/`

**Step 3: Run AI Hound**
- Run AI Hound on each Phase 2 output
- Removes AI writing signatures
- Creates humanized versions

**Final Outputs:**
- 3 skeleton documents (quickie, good, obsessed)
- 3 final markdown documents
- 1-2 LaTeX versions
- 1-2 Word-compatible versions
- 3 humanized versions
- Detection and humanization reports

## Key Features

### Phase 1 Agents Include:
- Codebase reality check (what's implemented vs. speculative)
- Literature landscape analysis
- Citation strategy
- Style & audience analysis (Katie Bouman's writing for Suresh Raghavan's perspective)
- Gap identification

### Phase 2 Agents Produce:
- Skeleton outlines (bullet points)
- Polished prose documents
- LaTeX versions (good & obsessed)
- Word-compatible markdown

### AI Hound Provides:
- AI signature detection (1-9 scale)
- Strategic humanization
- Multi-audience calibration (skeptics, pragmatists, forward-thinkers)

## Customization

Each prompt can be edited to:
- Adjust time targets
- Add/remove sections
- Change output formats
- Modify style guidelines
- Target different audiences

## Output Locations

```
/agents/
├── phase1/
│   ├── quickie_analysis.md
│   ├── good_analysis.md
│   └── obsessed_analysis.md
├── phase2/
│   ├── comps_skeleton_refined_[style].md
│   ├── comps_final_[style].md
│   ├── comps_final_[style].tex
│   ├── comps_final_[style]_word.md
│   └── comps_final_[style]_humanized.md
└── utilities/
    ├── ai_detection_report_[style].md
    └── humanization_report_[style].md
```

## Notes

- All agents use `sonnet` model by default
- Phase 1 agents are independent (can run in parallel)
- Phase 2 agents depend on Phase 1 outputs (run after Phase 1 completes)
- AI Hound depends on Phase 2 outputs (run last)
- Estimated total time: 60-90 minutes for full workflow

## Created

2026-01-06

## Project

Palpation-Assisted Tomography (PAT Scan) - Comprehensive Exam Document Generation
