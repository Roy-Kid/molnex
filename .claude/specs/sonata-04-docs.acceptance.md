---
slug: sonata-04-docs
criteria:
  - id: ac-001
    summary: SONATA.md exists at the canonical location and is non-trivial
    type: docs
    pass_when: |
      File `src/molpot/composition/SONATA.md` exists and is at least
      120 lines of markdown.
  - id: ac-002
    summary: SONATA.md contains the verbatim one-line Sonata definition
    type: docs
    pass_when: |
      The file contains the literal string "Sonata = MolNex's static
      permanent electrostatics model: learned permanent multipoles
      from local representations, evaluated through a differentiable
      periodic long-range electrostatic term, composed with the
      existing short-range potential."
  - id: ac-003
    summary: SONATA.md documents the σ vs α convention
    type: docs
    pass_when: |
      The file contains the conversion "α = 1/(σ√2)" (or the
      ASCII-equivalent "alpha = 1/(sigma * sqrt(2))") and explicitly
      names σ as the public knob matching
      `EwaldMultipoleEnergy(sigma=...)`.
  - id: ac-004
    summary: SONATA.md states the composition contract
    type: docs
    pass_when: |
      The file contains the equation "E_total = E_short + E_perm_elec"
      (or an ASCII rendering thereof) and lists the per-graph
      decomposition keys `short_range`, `electrostatic`, `total`.
  - id: ac-005
    summary: SONATA.md carries all three hard scope warnings
    type: docs
    pass_when: |
      The file contains all three warning blocks: (i) non-portability
      of learned charges to a different short-range half, (ii)
      forbidden `Polarization` co-composition (induction
      double-count), (iii) `kappa_head` / `alpha_head` belong to the
      future `LesPolarizable` composer.
  - id: ac-006
    summary: SONATA.md contains the (a)/(b)/(c) decision tree
    type: docs
    pass_when: |
      The file presents three labelled branches: (a) pure permanent
      `Sonata(kappa_head=None, alpha_head=None)`, (b) `Polarization`
      (CG Thole) self-consistent path, (c) future `LesPolarizable`
      composer.
  - id: ac-007
    summary: SONATA.md cites the corrected DOIs and excludes Kim 2025
    type: docs
    pass_when: |
      The file cites `10.1038/s41524-025-01577-7` (Cheng 2025),
      `10.1038/s41467-025-63852-x` (King 2025), Aguado & Madden 2003
      (J. Chem. Phys. 119:7471), Stone *Theory of Intermolecular
      Forces* 2nd ed., and the URL `github.com/ChengUCB/les`. The
      file does NOT cite `10.1021/acs.jctc.5c01400`.
  - id: ac-008
    summary: SONATA.md points at the benchmark and integration tests
    type: docs
    pass_when: |
      The file references both `benchmarks/bm_molpot/bm_sonata.py`
      and a path matching
      `tests/test_molpot/test_composition/test_sonata_*.py`.
  - id: ac-009
    summary: SONATA.md states migration-notes posture vs. existing specs
    type: docs
    pass_when: |
      The file states (i) `multipole-layer.md` is unaffected (Sonata
      composes, does not replace), (ii) `EwaldMultipoleEnergy.forward`'s
      `kappa=` / `alpha=` kwargs remain user-facing, (iii) the
      `multipole-layer.md` draft-status mismatch is out of scope.
  - id: ac-010
    summary: molpot module docstring surfaces Sonata
    type: code
    pass_when: |
      `src/molpot/__init__.py` lines 1-10 contain a "Model lines"
      mention of Sonata and a `from molpot.composition import
      Sonata, build_sonata` import example. The module's `__all__`
      list and import statements are byte-identical to the
      sonata-01 post-state.
  - id: ac-011
    summary: molpot README has a Model lines subsection
    type: docs
    pass_when: |
      `src/molpot/README.md` contains a `## Model lines` heading
      with a row referencing `composition/SONATA.md` and naming
      Sonata.
  - id: ac-012
    summary: CLAUDE.md flags Sonata as a model line
    type: docs
    pass_when: |
      `CLAUDE.md` contains a one-line mention of Sonata under the
      "Adding New Components" block (or equivalent location chosen
      by the implementer per the parent prompt) pointing at
      `src/molpot/composition/SONATA.md`.
  - id: ac-013
    summary: Lint and format clean after docstring edits
    type: runtime
    pass_when: |
      `ruff check src/` and `ruff format --check src/` both succeed
      with exit code 0 after the changes to
      `src/molpot/__init__.py`.
  - id: ac-014
    summary: Doc-described error paths agree with sonata-01 behaviour
    type: docs
    pass_when: |
      Every error condition described in SONATA.md's "Hard scope
      warnings" maps to a `raise` in
      `src/molpot/composition/sonata.py` (or wherever sonata-01
      lands the refusal logic). No warning describes an error path
      that does not exist in code.
  - id: ac-015
    summary: Full test suite still green
    type: runtime
    pass_when: |
      `python -m pytest tests/ -v` exits 0; no test is added,
      removed, or modified by this sub-spec.
---

# Acceptance criteria

The criteria above bind the doc's content (ac-001 to ac-009 and
ac-014), the three discoverability touch-points (ac-010 to ac-012),
and the no-regression bar (ac-013 and ac-015). ac-002 is verbatim
to lock the one-liner so subsequent reviewers don't paraphrase it
into drift. ac-007 is split out from the general references bar
because the explicit *non*-citation of Kim 2025 is load-bearing
(induced-response is out of Sonata scope; an accidental citation
would mislead users into expecting α-mode under Sonata). ac-014
is the cross-check between the prose claims and the implemented
behaviour from sonata-01 — if they diverge, the doc is wrong, not
the code.
