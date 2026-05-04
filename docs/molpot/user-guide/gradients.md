# Gradients and Forces

MolPot derives forces from energy with PyTorch autograd:

```text
forces = -dE / dpos
```

## Usage

```python
import torch
from molpot.derivation import ForceDerivation

pos = torch.randn(10, 3, requires_grad=True)
energy = (pos**2).sum().reshape(1)

force_deriv = ForceDerivation()
forces = force_deriv(energy, pos)

print(forces.shape)  # torch.Size([10, 3])
```

`pos.requires_grad` must be `True`.

## With PotentialComposer

`PotentialComposer` can derive forces when positions are present in `data`:

```python
data = {
    "edge_index": edge_index,
    "batch": batch,
    "pos": pos.requires_grad_(True),
}

outputs = composer(
    node_features=node_features,
    data=data,
    compute_forces=True,
)

forces = outputs["forces"]
```

During training, force derivation keeps the graph when the composer is in train
mode so force losses can backpropagate.
