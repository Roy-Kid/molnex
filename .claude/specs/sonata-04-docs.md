---
title: Sonata 04 — Boundary documentation and design doc
status: approved
created: 2026-05-10
---

# Sonata 04 — Boundary documentation and design doc

## Summary

Close the Sonata model-line work-stream by writing the user-facing
boundary contract in prose. Add `src/molpot/composition/SONATA.md` —
the single design doc a third party reads before instantiating
`Sonata(...)` — covering the one-line definition, scientific
motivation, scope (permanent-only), σ vs α convention, composition
contract, hard scope warnings, decision tree (pure permanent vs.
`Polarization` vs. future `LesPolarizable`), corrected references,
and pointers to the benchmark and integration tests delivered by
sonata-02 and sonata-03. Also surface Sonata in the discoverability
funnel: a one-line mention in `src/molpot/__init__.py` module
docstring (with import example), a "Model lines" subsection in
`src/molpot/README.md`, and a single-line entry in `CLAUDE.md` near
the existing module-tier signposting. No production code, no test,
no behavioural change.

## Domain basis

The doc inherits the equations and references already cited in
sub-spec 01 — it does not re-derive them. Citation block in
`SONATA.md` must list (verbatim DOIs from the parent prompt's
scientist output):

- Cheng B., *Latent Ewald summation for machine-learning potentials*,
  npj Comput. Mater. **11**:80 (2025), doi:10.1038/s41524-025-01577-7.
- King D. S. et al., *Latent equivariant ML force fields with
  long-range electrostatics*, Nat. Commun. **16**:8763 (2025),
  doi:10.1038/s41467-025-63852-x.
- Aguado A., Madden P. A., *J. Chem. Phys.* **119**:7471 (2003) —
  canonical multipole-Ewald-with-self-correction reference.
- Stone A. J., *The Theory of Intermolecular Forces*, 2nd ed.
  (Oxford, 2013) — §3, multipole electrostatics.
- LES upstream code: https://github.com/ChengUCB/les.

The σ vs α convention block records the conversion `α = 1/(σ √2)`
used by the inner periodic kernel, fixing σ as the canonical public
knob (matches the existing `EwaldMultipoleEnergy(sigma=...)` API).

The doc explicitly does **not** cite Kim et al. *J. Chem. Theory
Comput.*, doi:10.1021/acs.jctc.5c01400 — induced-response, out of
Sonata scope. (Existing `ewald.py` flag-block citations are not
edited here; sub-spec 01 owns the inline `ewald.py` reference fixes.)

## Design

Single new design doc owned by `molpot.composition` (the layer that
provides `Sonata` / `build_sonata`). The doc is a free-form markdown
file, not bound to the molzoo 10-section template — Sonata is a
composer model line, not an encoder, so the molzoo apparatus does
not apply. Structure mirrors the in-tree design-prose patterns in
`src/molpot/heads/multipole.py` and
`src/molpot/potentials/electrostatics/ewald.py` module docstrings:
narrative summary → scope → "What this is NOT" / scope warnings →
references.

Sections of `SONATA.md` (in order):

1. **One-line definition** — verbatim from the parent prompt's user
   spec §13.
2. **Scientific motivation** — paraphrased from user §2 and the
   parent scientist output §1; cites Cheng 2025 and King 2025.
3. **What Sonata is / is not** — the strict scope boundary from
   user §3. Permanent multipoles only; induced response excluded.
   Names the future composer `LesPolarizable` so readers don't
   open a "missing feature" issue.
4. **σ vs α convention** — `α = 1/(σ √2)`. σ is the canonical
   public knob; matches `EwaldMultipoleEnergy(sigma=...)`.
5. **Composition contract** — `E_total = E_short + E_perm_elec`,
   per-graph decomposition `{short_range, electrostatic, total}`,
   pointer to the Sonata forward-output dict shape.
6. **Hard scope warnings** — three:
   - Charge/dipole/quadrupole **non-portability**: do not export
     into a fixed-charge solver paired with a different
     short-range model; the trained force is end-to-end
     definition.
   - **No `Polarization` (CG Thole) double-compose**: induction
     would be double-counted; the composer raises at construction
     time (sonata-01 enforces this; this doc surfaces the rule).
   - **No `kappa_head` / `alpha_head`**: those belong to the
     future `LesPolarizable` composer; passing them to `Sonata`
     / `build_sonata` raises at construction time (also
     enforced by sonata-01; this doc surfaces the rule).
7. **Decision tree** — three branches answering "which model line
   should I instantiate?":
   - (a) Pure permanent, periodic or finite — `Sonata(...)` with
     `kappa_head=None`, `alpha_head=None`.
   - (b) Self-consistent CG Thole induction (gas-phase /
     molecular-mechanics-like) — compose `Polarization` directly,
     not via Sonata.
   - (c) Non-self-consistent linear response (LES α-mode) — wait
     for the `LesPolarizable` composer (planned, not in this
     work-stream).
8. **References** — the corrected list above.
9. **Pointers**:
   - benchmark: `benchmarks/bm_molpot/bm_sonata.py` (delivered by
     sonata-03);
   - integration tests:
     `tests/test_molpot/test_composition/test_sonata_*.py`
     (delivered by sonata-02);
   - upstream API surface: `molpot.composition.{Sonata, build_sonata}`
     (delivered by sonata-01).
10. **Migration notes** — three:
    - `multipole-layer.md` (the existing `PermMultipoleHead` spec)
      is unaffected — Sonata composes the head, does not replace
      it.
    - `EwaldMultipoleEnergy.forward(..., kappa=, alpha=)` keyword
      arguments remain available for users who construct an
      induced-response model line manually; `Sonata` simply does
      not pass them.
    - The `multipole-layer.md` `status: draft` mismatch is **out
      of scope** of Sonata. A separate spec (e.g.
      `multipole-layer-promote`) can transition it later.

Three secondary touch-points, kept minimal:

- `src/molpot/__init__.py` module docstring (lines 1-4): add a
  one-sentence "Model lines (composers): Sonata" paragraph with a
  3-line `from molpot.composition import Sonata, build_sonata`
  example. Touch the docstring only — `__all__` and imports stay
  exactly as sonata-01 left them.
- `src/molpot/README.md`: append a "## Model lines" subsection
  after the existing "## Modules" section, with a one-row table
  pointing at `composition/SONATA.md`. Keep tone consistent with
  the existing terse README.
- `CLAUDE.md`: insert one line under the "Adding New Components"
  bullet block (around line 250-260), under a new bullet
  **"Model lines"**, naming Sonata and pointing at
  `src/molpot/composition/SONATA.md`. The "Module Dependency
  Graph" diagram already shows `molpot.composition` — no need to
  redraw it; Sonata is a citizen of that already-listed node.

Lifecycle: the doc is markdown, never imported. No symbol is added,
no Python module changes shape, so no registry / discoverability
update is needed beyond the three docstring/README touch-points.

## Files to create or modify

- (new) `src/molpot/composition/SONATA.md` — Sonata design /
  boundary doc, ten sections per § Design above.
- `src/molpot/__init__.py` — module docstring only (lines 1-4):
  add the "Model lines" sentence and `Sonata` / `build_sonata`
  import example.
- `src/molpot/README.md` — append "## Model lines" subsection
  with a one-row table linking to `composition/SONATA.md`.
- `CLAUDE.md` — insert a one-line "Model lines" bullet under
  "Adding New Components" pointing at
  `src/molpot/composition/SONATA.md`.

## Tasks

- [ ] Draft `src/molpot/composition/SONATA.md` covering all ten
  sections from § Design (one-liner, motivation, scope, σ vs α,
  composition contract, three hard warnings, decision tree,
  references, pointers, migration notes).
- [ ] Verify every reference in `SONATA.md` resolves: open each
  DOI URL, confirm Cheng 2025 → `s41524-025-01577-7`,
  King 2025 → `s41467-025-63852-x`, no Kim 2025 cited, LES
  upstream URL renders.
- [ ] Update the `src/molpot/__init__.py` module docstring with a
  "Model lines" sentence and a `from molpot.composition import
  Sonata, build_sonata` example; leave imports and `__all__`
  unchanged.
- [ ] Append a "## Model lines" subsection to
  `src/molpot/README.md` with a one-row table linking to
  `composition/SONATA.md`.
- [ ] Add a "Model lines" bullet to `CLAUDE.md` under "Adding New
  Components" pointing at `src/molpot/composition/SONATA.md`.
- [ ] Cross-check `SONATA.md` warnings against the sonata-01
  composer behaviour: confirm the doc only describes errors that
  are actually raised (`Polarization` reject, `kappa_head` /
  `alpha_head` reject); if any divergence, fix the doc to match
  the implemented behaviour, never the other way around.
- [ ] Run full check + test suite (`ruff check src/ && ruff
  format --check src/ && python -m pytest tests/ -v`) — prose
  only, but the markdown link rot and any docstring formatting
  must lint clean.

## Testing strategy

This sub-spec is prose-only; there are no Python tests added. The
verification ladder is:

- **Lint & format pass** — `ruff check src/` and `ruff format
  --check src/` succeed unchanged after the docstring edits to
  `src/molpot/__init__.py`. Markdown files are not lint targets in
  this repo (`build.check` only covers `src/`), so `SONATA.md`
  and `README.md` need no machine validation beyond rendering.
- **Reference resolution** — every DOI in `SONATA.md` resolves to
  the publication named in the doc (manual check, recorded in the
  PR description).
- **Behavioural agreement with sonata-01** — `SONATA.md` describes
  only error paths that exist in the composer. Specifically the
  doc claims that (i) constructing `Sonata` with `Polarization`
  in `short_range_terms` raises, and (ii) passing `kappa_head` or
  `alpha_head` raises. Both must already be enforced by
  sonata-01's `Sonata.__init__`; this sub-spec verifies, it does
  not introduce.
- **No drift in test pointers** — the pointer to
  `tests/test_molpot/test_composition/test_sonata_*.py` matches
  the file glob produced by sonata-02; the pointer to
  `benchmarks/bm_molpot/bm_sonata.py` matches the file produced
  by sonata-03. If the upstream sub-specs landed under different
  paths, fix the pointers, do not fork the doc.
- **Domain validation (`$META.science.required`)** — the σ vs α
  block (`α = 1/(σ √2)`) matches the convention in
  `src/molpot/potentials/electrostatics/ewald.py` line 247
  (`a = 1/(σ √2)`). The composition contract `E_total = E_short
  + E_perm_elec` matches the Sonata forward output schema
  delivered in sonata-01. Both are checked by reading the source
  files, not by a runtime test.

## Out of scope

- Promoting `multipole-layer.md` from `status: draft` to a higher
  bar. That is a separate transition spec (e.g.
  `multipole-layer-promote`) and is explicitly punted by the
  parent prompt's migration-notes block.
- Editing `src/molpot/heads/multipole.py` or
  `src/molpot/potentials/electrostatics/ewald.py` module
  docstrings. Those already have rich domain-prose blocks; this
  sub-spec adds new prose, not edits existing.
- Adding `Sonata` / `build_sonata` to
  `src/molpot/__init__.py:__all__`. Sonata-01 owns the export
  surface; this sub-spec only edits the module docstring.
- A `mol:litrev` for the citation list. The DOIs are already
  vetted upstream (`/mol:spec` parent + scientist output); this
  sub-spec re-uses them, it does not re-validate them.
- Any change to `benchmarks/bm_molpot/bm_sonata.py` itself; that
  belongs to sonata-03.
- Any change to `tests/test_molpot/test_composition/test_sonata_*.py`;
  that belongs to sonata-02.
