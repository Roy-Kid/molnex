---
mol_project:
  name: molnex
  language: python
  build:
    install: "pip install -e '.[dev]'"
    check: "ruff check src/ && ruff format --check src/"
    test: "python -m pytest tests/ -v"
    test_single: "python -m pytest {path} -v"
    coverage: "python -m pytest tests/ --cov=src --cov-report=term-missing"
  arch:
    style: package-tree
    rules_section: "## Architecture"
  doc:
    style: google
  perf:
    focus: pytorch
  science:
    required: true
  notes_path: .claude/NOTES.md
  specs_path: .claude/specs/
---

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**MolNex** (v2.0.0) is a dict-first molecular ML framework for unified modeling of molecular potentials and properties with physics-aware ML. It is composed of four packages:

| Package | Role | Key Patterns |
|---------|------|-------------|
| **molix** | Training infrastructure | Trainer, TrainState (dict), Step protocol, Hook lifecycle |
| **molrep** | Representation learning | Embedding → Interaction → Readout pipeline, equivariance via cuEquivariance |
| **molpot** | Potential functions | BasePotential (nn.Module + ABC), autograd forces, PotentialComposer |
| **molzoo** | Pre-built encoders | Encoder-only (MACE, Allegro), no readout — downstream uses molpot |

## Build & Development

```bash
# Install (editable, with C++ extensions via scikit-build-core + CMake >=4.0)
pip install -e ".[dev]"

# Run all tests
python -m pytest tests/ -v

# Run single test file
python -m pytest tests/test_molzoo/test_mace.py -v

# Run single test
python -m pytest tests/test_molzoo/test_mace.py::test_mace_forward -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=term-missing
```

Python >=3.10 required. Requires `torch>=2.6` (always use latest stable PyTorch).

## Architecture

### Two-tier data contract (raw sample vs collated batch)

The pipeline speaks **two** shapes. They are intentionally different — get
this wrong and `KeyError(('edges','edge_index'))` is the first thing you see.

| Stage | Container | Example access |
|---|---|---|
| `DataSource[i]`, pipeline `SampleTask` / `DatasetTask` I/O, `MmapDataset[i]` | **flat `dict`** | `sample["Z"]`, `sample["edge_index"]`, `sample["targets"]["U0"]` |
| Post-`collate_molecules` `GraphBatch` (what encoders / losses receive) | **nested `TensorDict`** | `batch["atoms", "Z"]`, `batch["edges", "edge_index"]`, `batch["graphs", "energy"]` |

The single conversion point is `molix.data.collate.collate_molecules`
(invoked by `DataModule._CollateFn`). Stats / diagnostics operating on
individual cached samples use **flat** keys; model / loss / metric code
operating on batches uses **nested tuple-keys**.

#### Post-collate batch schema

```
GraphBatch (batch_size=[])
├── "atoms": AtomData (batch_size=[N])
│   ├── Z: atomic numbers (N,)
│   ├── pos: positions (N, 3)
│   └── batch: graph membership (N,)
├── "edges": EdgeData (batch_size=[E])
│   ├── edge_index: source-target pairs (E, 2)   # [:,0]=source, [:,1]=target
│   ├── bond_diff: edge vectors (E, 3)            # pos[target] - pos[source]
│   └── bond_dist: edge distances (E,)
└── "graphs": GraphData (batch_size=[B])  [optional]
    ├── num_atoms: (B,)
    └── <targets>
```

Access: `batch["atoms", "Z"]`, `batch["edges", "bond_dist"]`. Encoders
mutate the batch in place, writing `node_features` under `atoms` and
`edge_features` under `edges` — no subclass swap.

### Cache format

The on-disk cache is `molix.data.cache.PackedCache` — **one** `.pt` file
per cache with concatenated per-atom / per-edge / per-graph tensors and
`atom_ptr` / `edge_ptr` cumsum pointers. Loaded via `torch.load(mmap=True)`
for zero-copy reads; each DataLoader worker shares the OS page cache.

**Do not migrate to `TensorDict.memmap_()`.**
`LazyStackedTensorDict.memmap_(prefix)` writes one subdirectory per
sample (~1M files + ~260k dirs for QM9 130k molecules) — blows past HPC
shared-fs inode budgets and destroys `rsync` / `tar` / `rm` performance.
`PackedCache`'s single-file packed-bucket layout is strictly better for
fixed-schema molecular datasets. See `src/molix/data/cache.py` module
docstring for full rationale.

### Edge Convention (MUST follow everywhere)

```
edge_index[:, 0]  — source atom  (the "centre" in Allegro; the "sender" in MACE ConvTP)
edge_index[:, 1]  — target atom  (the "neighbour" in Allegro; the "receiver" in MACE ConvTP)
bond_diff         — pos[target] - pos[source]   (displacement vector, source → target)
bond_dist         — ‖bond_diff‖
```

`NeighborList` defaults to **full bidirectional** edges (`symmetry=True`, `E = 2 × n_pairs`).
Pass `symmetry=False` to get only the upper-triangle half-pairs (`E = n_pairs`) when you explicitly
want to exploit Newton's-3rd-law symmetry.  The two modes produce different `task_id`s so pipeline
caches are kept separate.

**Why bond_diff = pos[target] − pos[source]?**  This makes the displacement vector point in the
same direction as the edge (source → target), which is the convention expected by `SphericalHarmonics`
and all `cuEquivariance`-based tensor products in this repo.  The C++ `getNeighborPairs` kernel
returns `pos[rows] − pos[cols]` (opposite sign); `NeighborList.execute` negates it.

### Module Dependency Graph

```
molix.config (global dtype singleton)
    ↓
molrep.embedding → molrep.interaction → molrep.readout
    ↓                                       ↓
molzoo (MACE, Allegro encoders)         molpot.heads
    ↓                                       ↓
molpot.composition (PotentialComposer)  molpot.potentials
    ↓
molix.core (Trainer, TrainState, Step, Hook)
    ↓
molix.data (Dataset, collate, preprocess)
molix.datasets (QM9, MD17)
```

### State namespace contract

`TrainState` (`src/molix/core/state.py`) is a dict with a **fixed
top-level layout**. Every metric / scalar produced during training lives
inside one of four **namespace sub-dicts**, never at the top level with
a slash-prefix key.

| top-level key         | owner                    | kind        |
|-----------------------|--------------------------|-------------|
| `epoch`               | `Trainer`                | counter     |
| `global_step`         | `Trainer`                | counter     |
| `stage`               | `Trainer` (`Stage` enum) | counter     |
| `steps_since_last_eval` | `Trainer`              | counter     |
| `best_metric`         | `CheckpointHook`         | scalar      |
| `train`               | train-phase hooks        | **dict**    |
| `eval`                | eval-phase hooks         | **dict**    |
| `performance`         | throughput/timing hooks  | **dict**    |
| `gpu`                 | GPU telemetry hooks      | **dict**    |

**Writes — always nest.** Hooks must write into the namespace
sub-dict, never into the flat root with a slash:

```python
# ✅ correct
state["train"]["loss"] = loss.item()
state["eval"]["MAE"] = mae
state["performance"]["step_per_second"] = rate
state["gpu"]["peak_gib"] = peak

# ❌ rejected at __setitem__ — raises KeyError
state["train/loss"] = loss.item()
```

This prevents two hooks from silently sharing a flat namespace and
colliding on keys (which is exactly how the train/MAE-exploded-to-203
bug happened: eval path wrote into an accumulator shared with the
train path).

**Reads — tuple paths internally, dotted strings only for display.**
Hooks access scalars through nested dict lookups (``state["train"]["loss"]``)
or via the :data:`~molix.core.state.Path` type — a bare string for
top-level keys, a tuple for nested ones. Helpers at
:mod:`molix.core.state`:

```python
from molix.core.state import resolve, display

resolve(state, ("train", "loss"))     # → state["train"]["loss"] or None
resolve(state, "epoch")               # → top-level key
display(("train", "loss"))            # → "train/loss"  (for log column headers)
```

The dotted string form is **only** used as a display label (log column
headers, TensorBoard tags); never as an addressing scheme inside the
codebase.

**Phase-ownership rules for hooks.**

1. `on_train_batch_end`, `on_after_backward` — may write to
   `state["train"]`, `state["performance"]`, `state["gpu"]`.
2. `on_eval_batch_end`, `on_eval_step_complete` — may write to
   `state["eval"]`.
3. `on_epoch_start` / `on_epoch_end` — may write anywhere they own
   by phase, but should **prefer the per-phase callbacks** so readers
   have deterministic ordering.
4. **Hooks must not share mutable buffers between train and val.**
   If a hook needs the same metric on both sides (e.g. `MetricsHook`
   with `MAE`), it must hold two distinct instances — typically via
   `copy.deepcopy` in `__init__` — so a `.reset()` / `.update()` on
   the val side is invisible to the train side.

**Trainer guarantees for eval-phase hooks.**
`_run_eval_phase` fires `on_eval_step_complete` at the end of **every**
eval phase (both step-based `eval_every_n_steps` and epoch-end),
before the LR scheduler reads `best_metric_name`. Eval-publishing
hooks (`MetricsHook`, `TensorBoardHook`) should write their
`state["eval"]` scalars from `on_eval_step_complete`, not
`on_epoch_end`, to be available for the scheduler's read.

### Key Design Patterns

- **Encoder-only molzoo**: Encoders return raw features `(N, layers, features)`, readout/potentials handled by molpot
- **Pydantic configs**: All block configs use `BaseModel` with `ConfigDict(arbitrary_types_allowed=True)`
- **cuEquivariance**: Tensor products use `cuequivariance` / `cuequivariance_torch` for GPU-accelerated equivariant operations
- **Autograd forces**: `BasePotential.calc_forces()` computes `F = -dE/dx` via `torch.autograd.grad`
- **Functional composition**: `PotentialComposer` chains pooling → parameter heads → potential terms → aggregation
- **Hook protocol**: Lifecycle callbacks (`on_train_start`, `on_epoch_end`, etc.) via `Hook` protocol
- **Step protocol**: `DefaultTrainStep` / `DefaultEvalStep` wrap forward → loss → backward → optimizer

### Adding New Components

**New encoder** (molzoo): Accept `(Z, bond_dist, bond_diff, edge_index)`, return `(N, layers, features)`. Use `molrep` building blocks. Add paper reference.

**New potential** (molpot): Inherit `BasePotential`, implement `forward() -> scalar energy Tensor`. Forces come from autograd automatically.

**New embedding/interaction** (molrep): Pure `nn.Module`, use `cuequivariance` for equivariant layers.

## Scientific Correctness Requirements

Every implementation of a physical model, potential, or operator MUST:
1. Reference the original publication (arXiv/DOI) in the module docstring
2. Match the equations in the paper — document any deviations with rationale
3. Include numerical validation tests against reference implementations or published values
4. Preserve physical symmetries (energy conservation, rotational/translational invariance, permutation equivariance)

## PyTorch Version Policy

This project tracks **latest stable PyTorch** (currently >=2.6). Use modern PyTorch APIs:
- `torch.compile` for performance-critical paths
- `torch.export` for model serialization
- `torch.nn.functional` over deprecated module alternatives
- Native `torch.nested` for variable-length sequences where appropriate

## Docstring Convention

Google-style docstrings with tensor shape annotations:

```python
def forward(self, node_feats: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Message passing step.

    Args:
        node_feats: Node features ``(n_nodes, hidden_dim)``.
        edge_index: Edge indices ``(n_edges, 2)``.

    Returns:
        Updated node features ``(n_nodes, hidden_dim)``.

    Reference:
        Author et al. "Paper Title" Venue Year
        https://arxiv.org/abs/XXXX.XXXXX
    """
```

## Generic workflow — use the `mol` plugin

The workflow verbs (plan → build → verify → document → ship) live in the
[`mol` plugin](../claude-plugin/). Install once from any cwd:

```
/plugin marketplace add /Users/roykid/work/molcrafts/claude-plugin
/plugin install mol@molcrafts
```

From the MolNex root, the `mol_project:` frontmatter at the top of this
file tells every skill which commands to run. Use:

| Was (deprecated)   | Use instead        | Notes                                   |
|--------------------|--------------------|-----------------------------------------|
| `/molnex-impl`     | `/mol:impl`        | Full scope → litrev → spec → TDD → impl |
| `/molnex-spec`     | `/mol:spec`        | Natural-language → `.claude/specs/`     |
| `/molnex-litrev`   | `/mol:litrev`      | Paper + reference-impl credibility      |
| `/molnex-arch`     | `/mol:arch`        | Reads `arch.rules_section` (this file)  |
| `/molnex-review`   | `/mol:review`      | Fans out architect ∥ optimizer ∥ documenter ∥ undergrad ∥ scientist |
| `/molnex-test`     | `/mol:test`        | Runs `build.test`                       |
| `/molnex-perf`     | `/mol:perf`        | Uses `perf.focus: pytorch`              |
| `/molnex-docs`     | `/mol:docs`        | Uses `doc.style: google`                |
| —                  | `/mol:note`        | Capture into `.claude/NOTES.md`         |
| —                  | `/mol:fix`         | Minimal-diff bug fix                    |
| —                  | `/mol:debug`       | Diagnose-only — never writes code       |
| —                  | `/mol:refactor`    | Restructure preserving invariants       |
| —                  | `/mol:ship commit` | Pre-commit gate (format + lint)         |
| —                  | `/mol:ship push`   | Pre-push gate (+ full test suite)       |
| —                  | `/mol:ship merge`  | Pre-merge CI-parity gate                |

MolNex-specific agents remain: `ml-expert` (training dynamics — ML
axis not covered by the generic plugin) and `molzoo-auditor` (the
molzoo spec-walkthrough writer — see below). The former
`molnex-compute-scientist` and `molnex-pm` have been promoted into
the plugin as generic `compute-scientist` and `pm` agents and are now
invoked via `/mol:review`.

## Molzoo spec workflow — skills and agent

Each encoder in `src/molzoo/` owns three sibling artifacts in `src/molzoo/specs/`:

- `<encoder>.md` — paper-aligned spec (architecture, module math, I/O contract).
- `<encoder>_walkthrough.md` — code↔spec↔paper audit with ✅ ℹ️ ⚠️ 🆚 verdict rows.
- `<encoder>_experiments.csv` — append-only run log. Schema: `run_id,date,commit,dirty,dataset,config_label,steps,train_mae,val_mae,fwd_ms,bwd_ms,compiled,note_ref`.

The following skills and agent keep those three artifacts in sync. **All are repo-local** under `.claude/skills/` and `.claude/agents/` — do not promote to `~/.claude/` without a second repo adopting the same pattern.

| Command | Type | Purpose |
|---------|------|---------|
| `/molzoo-spec-new <encoder> <arxiv_url>` | skill | Scaffold `<encoder>.md`, `<encoder>_walkthrough.md`, `<encoder>_experiments.csv` with the Reference section populated from the arXiv page. Refuses to overwrite existing artifacts. |
| `/molzoo-spec-log <encoder>` | skill | Append one row to `<encoder>_experiments.csv` after a bench/train run. On MAE regression (> 10 %) or dirty tree, prompts to open a `molzoo-auditor` investigation and stubs a `run-<N>-<slug>` heading. |
| `/molzoo-spec-lookup <encoder> <topic>` | skill | Read-only retrieval of spec + walkthrough sections for a topic. **Refuses to fabricate**: if the topic is uncovered, it suggests `molzoo-auditor` rather than summarising the paper from memory. |
| `molzoo-auditor` | agent | Fetches the paper, compares to code, and appends a verdict row to `<encoder>_walkthrough.md`. Edits `<encoder>.md` only when the verdict is ⚠️/🆚. Required to write ≥ 1 walkthrough entry per invocation. |

**Closed-loop contract.** Every operation either updates an artifact or explicitly delegates the update — no operation closes silently:

1. `/molzoo-spec-log` is append-only on the CSV; it **must** check the previous row and prompt auditor hand-off on anomalies, backfilling `note_ref` on the just-written row if the user accepts.
2. `/molzoo-spec-lookup` **must** refuse on a topic miss and point at `molzoo-auditor`.
3. `molzoo-auditor` **must** produce ≥ 1 walkthrough row per invocation (even just `✅ confirmed, no drift`) citing the triggering `run_id` or question + paper section + code file:line.
4. `/molzoo-spec-new` is the only operation that writes to all three artifacts at once and only for a previously-absent encoder.

Invariants are enforced **inside each skill**, not via `settings.json` hooks, for now.
