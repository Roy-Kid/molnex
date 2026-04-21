# molzoo

Molecular encoder zoo. Provides encoder-only architectures (MACE, Allegro) without built-in energy/force readout. Downstream potential composition is handled by `molpot.composition`.

## Model Specifications

Each model in this package ships with **three sibling artifacts** in `specs/`:

- `<encoder>.md` — paper-aligned spec (architecture, module math, I/O contract, symmetries, deviations).
- `<encoder>_walkthrough.md` — code↔spec↔paper audit (✅ ℹ️ ⚠️ 🆚 verdicts + run-linked investigations).
- `<encoder>_experiments.csv` — append-only run log (date, commit, dataset, config tag, MAE, fwd/bwd ms).

**Read the spec before modifying the model** — any change to a module's math MUST be reflected in the corresponding spec.

| Model   | Spec | Walkthrough | Experiments | Paper |
|---------|------|-------------|-------------|-------|
| Allegro | [`specs/allegro.md`](specs/allegro.md) | [`specs/allegro_walkthrough.md`](specs/allegro_walkthrough.md) | [`specs/allegro_experiments.csv`](specs/allegro_experiments.csv) | Musaelian et al., Nat. Commun. 2023 ([arXiv](https://arxiv.org/abs/2204.05249)) |
| MACE    | *(todo)* | *(todo)* | *(todo)* | Batatia et al., NeurIPS 2022 ([arXiv](https://arxiv.org/abs/2206.07697)) |

### Spec workflow

Three skills + one agent maintain these artifacts as a closed loop:

| Trigger | Command | Writes |
|---------|---------|--------|
| Introducing a new encoder | `/molzoo-spec-new <encoder> <arxiv_url>` | seeds `<encoder>.md` + `_walkthrough.md` + `_experiments.csv` |
| After a benchmark or training run | `/molzoo-spec-log <encoder>` | appends one row to `_experiments.csv`; prompts `molnex-scientist` on MAE regression or dirty tree |
| Debugging a question | `/molzoo-spec-lookup <encoder> <topic>` | read-only; refuses to fabricate, suggests `molnex-scientist` on a miss |
| Verify code vs paper | `molnex-scientist` agent | appends ≥ 1 verdict row to `_walkthrough.md`; edits `<encoder>.md` only on ⚠️/🆚 |

**The loop never closes silently.** Every operation either updates an artifact or explicitly delegates the update. See `CLAUDE.md` for the full contract.

## Input Conventions

Both encoders accept keyword tensors:

- `Z`: Atomic numbers `(N,)`
- `bond_dist`: Edge distances `(E,)`
- `bond_diff`: Edge vectors `(E, 3)`
- `edge_index`: Edge indices `(E, 2)`

Output: `(N, num_layers, feature_dim)` — per-atom, per-layer features.

## Usage

```python
import torch
from molzoo import MACE, MACESpec
from molrep.embedding.node import DiscreteEmbeddingSpec
from molpot import LayerPooling, PotentialComposer, LJParameterHead, LJ126

encoder = MACE(MACESpec(
    node_attr_specs=[DiscreteEmbeddingSpec(input_key="Z", num_classes=119, emb_dim=64)],
    num_elements=119,
    num_features=64,
    r_max=5.0,
))

Z = torch.randint(0, 10, (20,))
features = encoder(
    Z=Z,
    bond_dist=torch.rand(80),
    bond_diff=torch.randn(80, 3),
    edge_index=torch.randint(0, 20, (80, 2)),
)

pool = LayerPooling("mean")
node_features = pool(features)  # (20, 64)
```
