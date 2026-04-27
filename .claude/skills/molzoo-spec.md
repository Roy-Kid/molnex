---
name: molzoo-spec
description: One skill for molzoo encoder specs — scaffold, fill from reference, lookup, or log a run. Spec-first: §2/§3.1/§5 must be filled from the paper + reference impl BEFORE any encoder code is written. Operates on `src/molzoo/specs/<encoder>.md` following the 10-section traceability template embedded below. Mode is auto-selected from args + filesystem.
---

# molzoo-spec

A spec is a **traceable engineering document**, not a tutorial. The reader must be able to reproduce and debug the encoder ONLY from `<encoder>.md`. If they can't, the spec is wrong — fix the spec, not this skill.

## Spec-first principle (load-bearing)

The spec is written from the **paper + reference implementation**, not from `<encoder>.py`. Implementation is a mechanical translation of the spec — if the implementer must invent something, that's a spec gap and `<encoder>.md` must be updated first.

Concretely:

- `src/molzoo/<encoder>.py` MUST NOT be created or modified while §2 (Paper↔Code Mapping), §3.1 (Identical), or §5 (Mathematical Contract) of `<encoder>.md` contain `<...>` placeholders. Reading the paper and the reference impl into the spec comes first.
- The flow is **scaffold → fill → implement → audit**, in that order. Skipping `fill` and going straight to implementation is the failure mode this skill exists to prevent — every paper-port deviation we have ever shipped has been traced back to "wrote code first, retrofitted the spec from what the code happened to do".
- `molzoo-auditor` treats the spec as the source of truth for what the code SHOULD do, and the paper + reference impl as the source of truth for what the spec SHOULD say. A `code ↔ spec` disagreement, when the spec was correctly transcribed from the reference, is a **code defect** to be fixed in code, not a spec patch.
- Reverse workflows ("write code first, then back-fill the spec from the implementation") are forbidden. They guarantee improvisation will be encoded as fact.

## Modes

- **scaffold** — `<encoder>.md` absent → write a fresh spec from the template below; only header + §9 + Changelog populated. Status starts `draft`.
- **fill**     — spec exists with placeholder §2/§3.1/§5 → populate from the paper + reference impl at the pinned sha. Required before `<encoder>.py` is written. Status moves `draft` → `partial`.
- **lookup**   — spec exists, topic given → return matching sections verbatim; refuse to summarise; refuse outright if §2/§3/§5 are still placeholders.
- **log**      — spec exists, `--log k=v ...` given → append a row to §7.4 (Run log).

## Args

`/molzoo-spec <encoder> [--fill | <topic> | --log <k=v ...>] [--paper <arxiv_url>] [--ref <org>/<repo>@<sha>]`

- `<encoder>` — must match `^[a-z][a-z0-9_]*$`.
- `--paper` — required for **scaffold**, ignored otherwise.
- `--fill` — switch to **fill** mode; the reference repo must already be pinned in the spec header (or supplied via `--ref` if missing).
- `--ref` — pin a reference repo at `<org>/<repo>@<sha>` for **scaffold** (write into header) or **fill** (write into header before reading) when not yet set.

## Mode: scaffold

- Precondition: `<encoder>.md` does not exist; `--paper` provided. Refuse otherwise.
- Resolve via `WebFetch` on the arXiv abstract page; never invent citations. On failure, ask the user.
- Write `src/molzoo/specs/<encoder>.md` from the **Spec template** below, verbatim.
- At scaffold time fill ONLY:
  - Header table — Module, Entry point (`<Class>` + `<Spec>`; ask if not derivable), Paper / arXiv / DOI, Reference impl (`<org>/<repo>` @ `<sha>` from `--ref`, `git ls-remote`, or user; never invent), This repo commit (`git -C <root> rev-parse --short HEAD`), Spec status: `draft`.
  - §9 Version Pinning rows you actually have data for. Other rows: `<…>`.
  - Changelog: today + sha + "scaffolded".
- Everything else stays as `<...>` placeholders. Fabricated §2/§4/§5 rows are worse than empty.
- Update `src/molzoo/README.md`'s 4-column table (`Model | Spec | Walkthrough | Paper`); preserve order.
- Do NOT write `<encoder>.py`. Do NOT touch other encoders.
- Print created paths + the **mandatory next step**:
  > `/molzoo-spec <encoder> --fill` — populate §2 / §3.1 / §5 from the pinned reference impl + paper. Code authoring is forbidden until that completes (see Spec-first principle).

## Mode: fill

- Precondition: `<encoder>.md` exists; §2 / §3.1 / §5 contain `<...>` placeholders; reference repo + sha pinned in the header (Reference impl row) or supplied via `--ref` (and written into the header before reading).
- Read the reference impl directly. If the reference repo is not yet on disk, fetch it (shallow clone into `/tmp/upstream_refs/<repo>@<sha>/`) at the pinned commit. Never read a different sha — the pin is what makes the spec reproducible.
- Re-fetch the paper (arXiv abstract + PDF or HTML) for equation references. `WebFetch` only; no paraphrasing from memory.
- Populate, in this order, citing the source for every row:
  - **§2 Paper ↔ Code.** One row per paper concept. **Reference impl (file:line)** comes from the pinned reference. **This repo (file:line)** stays `<…>` until the encoder code lands. **Status** starts `unknown` (will move to `matched` / `adapted` etc. once code exists and `molzoo-auditor` confirms).
  - **§3 Reference Implementation Alignment.** §3.1 lists components to be ported verbatim, with a one-line reason. §3.2 stays empty — no deviations exist before code is written.
  - **§5 Mathematical Contract.** Equations transcribed from the paper (display math + symbol → code table). Code-side `Implemented at` cells stay `<...>` until code lands.
- Anything not directly evidenced by the fetched paper passage or the reference source MUST stay `<...>`. Inventing rows from memory is the failure mode this mode exists to prevent — exactly as in scaffold.
- Bump Spec status: `draft` → `partial`. Append Changelog: today + this-repo sha + `"filled §2/§3.1/§5 from <ref-org>/<ref-repo>@<ref-sha>"`.
- Do NOT write `<encoder>.py`. Do NOT add §3.2 or §4 rows (no deviations exist yet).
- Print: "Spec is ready to drive implementation. Implement `src/molzoo/<encoder>.py` mechanically against §3.1 / §5; resolve `<…>` cells in §2 'This repo' as code lands; run `molzoo-auditor` to flip §2 statuses from `unknown`."

## Mode: lookup

- Read `<encoder>.md` only — never `WebFetch` the paper, never paraphrase from memory.
- If §2 / §3.1 / §5 are still placeholders, refuse with: `"spec incomplete; run /molzoo-spec <encoder> --fill before asking implementation questions"`. A topic question against an unfilled spec is premature — the spec is by definition silent.
- Match: split topic on whitespace, lowercase, drop stopwords (`the`, `a`, `and`, `of`, `is`, `what`, `does`, `how`, `why`); ≥1 keyword in heading or body wins.
- Print matched sections **verbatim** in priority order: §2 (Paper↔Code) → §4 (Adaptation Ledger) → §5 equations → others. §2 / §4 are load-bearing; surface them first.
- If a match is > 40 lines, head 40 + `(see <encoder>.md:<line>-<end>)` pointer.
- On miss (filled spec): refuse to answer. Print `spec gap on "<topic>"; invoke molzoo-auditor`. Do NOT summarise from memory — a topic missing from §2/§4 is a concrete spec gap, not an LLM-summary opportunity.

## Mode: log

- Precondition: spec exists, `--log` given.
- Accepted keys: `run_id`, `date`, `commit`, `dirty`, `dataset`, `config`, `steps`, `train_mae`, `val_mae`, `fwd_ms`, `bwd_ms`, `compiled`, `note`.
- Auto-fill defaults: `run_id` = previous + 1; `date` = today (`date +%Y-%m-%d`); `commit` = `git rev-parse --short HEAD`; `dirty` = 0/1 from `git status --porcelain`. Other keys: ask once if missing.
- Append the row at the **end** of §7.4 (Run log) — never reformat earlier rows.
- After appending: read the previous row; compute `val_mae_ratio`. If `> 1.10` or `dirty=1`, print one-line warning and `consider /molzoo-auditor for run_id=<N>`. Do NOT auto-invoke; the user decides.

## Hard rules (all modes)

- **Spec-first.** `src/molzoo/<encoder>.py` MUST NOT be created or modified while §2 / §3.1 / §5 contain `<...>` placeholders. Implementation is a translation of the spec; if the spec is empty, there is nothing to translate. The natural flow is `scaffold → fill → implement → audit`; skipping `fill` is a process violation.
- One file per encoder: `src/molzoo/specs/<encoder>.md`. No CSV, no JSON, no walkthrough file. The `molzoo-auditor` agent prints its verdict report to the developer (chat-time artifact); on 📝/🆚 it patches `<encoder>.md` directly — the diff is the persistent trace. ⚠️ (code-drift) is print-only — recommendations don't get applied to the spec.
- The 10-section structure of the template is **immutable**. Adding / removing / renaming a section is a §10.2 breaking change.
- Never invent §2 / §4 / §5 rows from model memory. Empty placeholders are correct; fabricated rows are not. Both `scaffold` and `fill` MUST cite a reference `file:line` (for code rows) or a paper `§ + eq.` (for equation rows) for every populated cell.
- Reference repo is pinned at a specific sha in the header. Reading any other sha — including the default branch tip — is a violation; the pin is what makes the spec reproducible across re-audits.
- `fill` is a one-way ratchet: never downgrade a populated §2/§3.1/§5 cell back to `<...>`. Corrections go through `molzoo-auditor` (📝 verdict patches the spec cell in place; ⚠️ verdict means the code drifted and needs a code change instead).

---

## Spec template (copy verbatim on scaffold)

````markdown
# <Encoder> — Specification

| Field | Value |
|-------|-------|
| Module | `molzoo.<encoder>` |
| Entry point | `<Class>` (config `<Spec>`) |
| Paper | <Authors>, *"<Title>"*, <Venue Year> |
| arXiv | https://arxiv.org/abs/<id> |
| DOI | <link or —> |
| Reference impl | `<org>/<repo>` @ `<sha>` (<date>) |
| This repo commit | `<sha>` (<date>) |
| Spec status | draft \| partial \| stable | <!-- draft = scaffold only; partial = §2/§3.1/§5 filled from reference, ready to drive implementation; stable = code landed + audited green -->

## 1. Scope & Boundary
- **Does:** <one precise sentence>
- **Does NOT:** <explicit non-goals — e.g. "no readout MLP, no force gradient">
- **Sits at:** <position in the system>

### 1.1 I/O contract
| Direction | TensorDict path | Shape | Dtype | Source/sink |
|-----------|------------------|-------|-------|-------------|
| In | `("atoms","Z")` | `(N,)` | `int64` | `DataModule` |
| In | `("edges","edge_index")` | `(E,2)` | `int64` | `NeighborList` |
| Out | `("edges","edge_features")` | `(E,L,F)` | float | downstream readout |

Anything not listed MUST NOT be read or written by this module.

## 2. Paper ↔ Code Mapping
- One row per paper concept; never silently absent.
- Status ∈ `matched` / `adapted` / `missing` / `unknown`. Non-`matched` rows MUST link to a §4 ID.

| Concept | Paper §/eq. | Reference impl (file:line) | This repo (file:line) | Status |
|---------|-------------|----------------------------|------------------------|--------|
| <…> | <…> | <…> | <…> | <…> |

## 3. Reference Implementation Alignment
- **Repo / commit:** `<org>/<repo>` @ `<sha>` (<date>). **Diff link:** <…>.
- **§3.1 Identical:** bullet list of components ported verbatim, with reason.
- **§3.2 Differs:** every row also appears in §4.

### 3.2 Differs from reference
| Concern | Reference | This repo | Why | §4 ID |
|---------|-----------|-----------|-----|-------|
| Kernel | <…> | <…> | <…> | A1 |
| Tensor layout | <…> | <…> | <…> | A2 |

## 4. Adaptation Ledger
- One row per deviation — bundling forbids attribution.
- Risk: `low` (<1 % metric shift) / `medium` (1–5 %) / `high` (>5 % or symmetry break).
- Validation MUST resolve to a test path or a §7.4 `run_id`. `pending` only with a linked issue.

| ID | Change | Reason | Impact (metric, dataset) | Risk | Validation |
|----|--------|--------|---------------------------|------|------------|
| A1 | <…> | <…> | <…> | low | <…> |

## 5. Mathematical Contract
- ONLY equations directly implemented and necessary for behavior.
- Each subsection: equation in display math + symbol→code table.
- Anything not implemented goes to §4 as `missing`, not here.

### 5.1 <Equation name>
$$<\LaTeX>$$
| Symbol | Domain/shape | Implemented at |
|--------|--------------|----------------|
| `<x>` | `(E,F)` | `src/molzoo/<encoder>.py:<line>` |

## 6. Config Mapping
- Flag naming differences, default deltas, unit conventions. Missing entries use `—` (never blank).

| `<Spec>` field | Reference name | Meaning | Default this/ref | Notes |
|----------------|----------------|---------|-------------------|-------|
| <…> | <…> | <…> | <…> | <…> |

## 7. Benchmark Contract

### 7.1 Reproduction targets
| Dataset | Metric | Paper | This-repo target | Tolerance |
|---------|--------|-------|-------------------|-----------|
| <…> | <…> | <…> | <…> | <…> |

### 7.2 Engineering benchmarks
| Quantity | Configuration | Target | Tested at |
|----------|---------------|--------|-----------|
| Forward time / batch | bs=32 N̄=18 L=2 A100 | <ms> | `tests/bench/test_<enc>_perf.py::test_fwd` |
| Backward time / batch | same | <ms> | `…::test_bwd` |
| Peak memory | same | <GiB> | `…::test_mem` |
| Scaling: time vs E | E ∈ {1k,10k,100k} | linear | `…::test_scaling` |

### 7.3 Invariance / equivariance tests
| Symmetry | Test path | Tolerance |
|----------|-----------|-----------|
| Translation | `tests/test_symmetry.py::test_translation_invariance[<enc>]` | rel ≤ 1e-5 |
| Rotation | `…::test_rotation_invariance[<enc>]` | rel ≤ 1e-5 |
| Permutation | `…::test_permutation_equivariance[<enc>]` | rel ≤ 1e-5 |
| Parity (if claimed) | `…::test_parity[<enc>]` | rel ≤ 1e-5 |

Tests in §7.3 MUST exist before any §2 row dependent on that symmetry can be marked `matched`.

### 7.4 Run log (append-only)
- One row per benchmark / training run, written by `/molzoo-spec <enc> --log ...`.
- Append at the bottom; never reformat earlier rows. `note` may carry a short audit memo (e.g. `📝 §5 eq.(3) corrected`, `⚠️ code change recommended`, `✅ confirmed`) backfilled by `molzoo-auditor`.

| run_id | date | commit | dirty | dataset | config | steps | train_mae | val_mae | fwd_ms | bwd_ms | compiled | note |
|--------|------|--------|-------|---------|--------|-------|-----------|---------|--------|--------|----------|------|

## 8. System Boundary
| Concern | Owner | Contract |
|---------|-------|----------|
| Edge construction | `molix.data.NeighborList` | full bidirectional; `edge_index[:,0]=src`, `[:,1]=tgt`; `bond_diff = pos[tgt] − pos[src]` |
| Encoder forward | `molzoo.<encoder>` | reads §1.1 In; writes §1.1 Out; mutates `GraphBatch` in place |
| Energy aggregation | `molpot.PotentialComposer` | reads `("edges","edge_features")`; owns `("graphs","E")` |
| Force gradients | `molpot.BasePotential.calc_forces` | owns `("atoms","F")`; uses `torch.autograd.grad` |
| Training loop | `molix.core.Trainer` | owns `TrainState` namespaces |

**Hard rules.** Encoder MUST NOT read/write `("graphs",*)`, MUST NOT call `torch.autograd.grad`, MUST NOT mutate input keys. Any change here is a §10.2 breaking change.

## 9. Version Pinning
| Item | Value |
|------|-------|
| Paper | <author> <year> arXiv:<id>v<N> (<date>) |
| Reference repo | `<org>/<repo>` @ `<sha>` (<date>) |
| Reference revision diff link | <https://github.com/<org>/<repo>/compare/...> |
| PyTorch | `>=2.6` (verified `<v>`) |
| cuequivariance / cuet | `>=<x>` (verified `<v>`) |
| This repo commit | `<sha>` (<date>) |

## 10. Spec Drift Policy
- **10.1 Triggers — must update spec when:** §5 equation changes → patch §2 row → if behaviour shifts, add §4 row → bump §9; new/renamed config field → §6; §7.1 range moves → §7.4 row + §4 if outside tolerance; reference pin moves → re-audit §3 + refresh §9; touched a file in §2 → verify line resolves, otherwise patch §2.
- **10.2 Breaking — must bump major:** output dict-key rename or shape change; §2 row flips `matched` ⇄ non-`matched`; §4 row removed without replacement; §8 contract change.
- **10.3 Enforcement:** code change without matching `<encoder>.md` diff is reviewable as drift. Code authored while §2 / §3.1 / §5 are still placeholders (spec status `draft`) is a process violation, not just drift — revert and run `/molzoo-spec <enc> --fill` first. Run `/molzoo-spec <enc> --log` after every benchmark. Run `molzoo-auditor` for any §2 row whose code line numbers no longer resolve, or whose status is still `unknown` after code has landed.

## Changelog (append-only)
- <YYYY-MM-DD> · `<sha>` · scaffolded.
````
