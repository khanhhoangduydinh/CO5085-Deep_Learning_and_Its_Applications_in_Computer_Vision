# CO5085 Report Theory Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (native implementation in this session). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the GitHub repository to the full course name and expand the CO5085 report with accurate beginner-friendly theory, formulas, and code workflows.

**Architecture:** The existing report remains the source of measured results. New LaTeX sections are inserted beside the E1, E2, and E3 discussions, so each formula directly explains code in the matching Python file. The static site keeps its current layout but points to the renamed repository.

**Tech Stack:** GitHub REST API, Git, static HTML/CSS, LaTeX/PDF, PyTorch source code.

**Spec:** `docs/superpowers/specs/2026-10-28-co5085-report-theory-design.md`

## Global Constraints

- The new repository name is `CO5085-Deep_Learning_and_Its_Applications_in_Computer_Vision`.
- Preserve every executed result in `reports/e1_results.json`, `reports/e2_results.json`, and `reports/e3_results.json`.
- Every formula must define its symbols in nearby plain language.
- Documentation must describe only operations that exist in the submitted code.

## Review Focus

- A report reader can connect each algorithm step to a named code file.
- Formulas render without clipped symbols or unresolved references.
- The public GitHub and Pages hyperlinks use the renamed repository path.
- The existing PDF report link remains reachable after Pages deploys.

### Task 1: Rename GitHub and update links

**Files:**
- Modify: `index.html`
- Modify: `README.md`

- [x] Rename the GitHub repository with the REST API and update the `origin` URL.
- [x] Replace all old repository URLs in local Pages and README files.
- [x] Push the link update and verify the new Pages deployment serves it.

### Task 2: Add shared and E1 theory

**Files:**
- Modify: `CO5085 Report Latex/report.tex`

- [x] Add a shared split, softmax, cross-entropy, gradient, and accuracy explanation.
- [x] Add E1 formulas and an algorithm that follows `common.py` and `E1/run_e1.py`.

### Task 3: Add E2 and E3 theory

**Files:**
- Modify: `CO5085 Report Latex/report.tex`

- [x] Add tokenization and multi-head self-attention formulas with manual-code mapping.
- [x] Add LSTM and GRU equations, row-sequence steps, and a CNN comparison explanation.

### Task 4: Verify and publish

- [x] Compile LaTeX twice and inspect formula/workflow pages as PNGs.
- [x] Run focused static and source tests.
- [x] Commit, push, compare local and remote SHA, and verify public Pages plus PDF HTTP responses.
