# molzoo

Molecular encoder zoo. Provides encoder-only architectures (MACE, Allegro) without built-in energy/force readout. Downstream potential composition is handled by `molpot.composition`.

## Model Specifications

Each model in this package ships with **one** spec artifact in `specs/`:

- `<encoder>.md` — paper↔code↔reference traceable spec (10-section template; includes I/O, paper↔code mapping, adaptation ledger, benchmark contract with embedded run log).

**Read the spec before modifying the model** — any change to a module's math MUST be reflected in the corresponding spec.

| Model   | Spec | Paper |
|---------|------|-------|
| Allegro | [`specs/allegro.md`](specs/allegro.md) | Musaelian et al., Nat. Commun. 2023 ([arXiv](https://arxiv.org/abs/2204.05249)) |
| MACE    | *(todo)* | Batatia et al., NeurIPS 2022 ([arXiv](https://arxiv.org/abs/2206.07697)) |

### Spec workflow

One skill + one agent keep `<encoder>.md` aligned with code and paper:

| Trigger | Command | Effect |
|---------|---------|--------|
| Introducing a new encoder | `/molzoo-spec <encoder> --paper <arxiv_url>` | seeds `<encoder>.md` from the 10-section template |
| Debugging a question | `/molzoo-spec <encoder> <topic>` | read-only lookup; refuses to fabricate, suggests `molzoo-auditor` on a miss |
| After a benchmark or training run | `/molzoo-spec <encoder> --log <k=v ...>` | appends one row to §7.4 (Run log) of `<encoder>.md`; warns + suggests `molzoo-auditor` on MAE regression or dirty tree |
| Verify code vs paper | `molzoo-auditor` agent | **prints** ≥ 1 verdict report to the developer; on ⚠️/🆚 patches `<encoder>.md` (§2/§3.2/§4/§5) — the spec diff is the persistent trace |

**The loop never closes silently.** Every operation either updates the spec, prints a verdict, or explicitly delegates. See `CLAUDE.md` for the full contract.

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
