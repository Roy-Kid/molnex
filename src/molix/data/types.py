"""Molecular data types built on nested TensorDict.

Defines a composable type hierarchy for molecular graph data as it appears
*after* ``collate_molecules`` — i.e. the batch-side shape. Raw samples from
``DataSource.__getitem__`` / ``MmapDataset[i]`` remain plain flat dicts; the
nested form here is reached only via ``collate.collate_molecules``.

- **AtomData** (batch_size=[N]): per-atom tensors (Z, pos, batch)
- **EdgeData** (batch_size=[E]): per-edge tensors (edge_index, bond_diff, bond_dist)
- **GraphData** (batch_size=[B]): per-graph tensors (num_atoms, targets)
- **GraphBatch** (batch_size=[]): top-level container nesting atoms + edges [+ graphs]

Encoder outputs are written into the existing sub-dicts in place — e.g.
``batch["atoms", "node_features"] = ...`` and ``batch["edges", "edge_features"] = ...``.
There is no subclass swap; the shape evolves by key addition, not by type.

Example:
    >>> atoms = AtomData(
    ...     Z=torch.tensor([6, 1, 1]),
    ...     pos=torch.randn(3, 3),
    ...     batch=torch.tensor([0, 0, 0]),
    ...     batch_size=[3],
    ... )
    >>> edges = EdgeData(
    ...     edge_index=torch.tensor([[0, 1], [1, 0]]),
    ...     bond_diff=torch.randn(2, 3),
    ...     bond_dist=torch.rand(2),
    ...     batch_size=[2],
    ... )
    >>> batch = GraphBatch(atoms=atoms, edges=edges, batch_size=[])
    >>> batch["atoms", "Z"]  # tensor([6, 1, 1])
"""

from __future__ import annotations

from tensordict import TensorDict


# ---------------------------------------------------------------------------
# Atom-level (batch_size=[N_total])
# ---------------------------------------------------------------------------


class AtomData(TensorDict):
    """Per-atom tensors. ``batch_size=[N]``.

    Schema produced by ``collate_molecules``:
        - ``Z``: Atomic numbers ``(N,)``
        - ``pos``: Atomic positions ``(N, 3)``
        - ``batch``: Graph membership ``(N,)``

    Encoders may add ``node_features`` of shape ``(N, [L,] D)`` in place.
    Atom-level targets (e.g. ``forces``) are also added here by the
    collate function according to ``TargetSchema``.
    """


# ---------------------------------------------------------------------------
# Edge-level (batch_size=[E_total])
# ---------------------------------------------------------------------------


class EdgeData(TensorDict):
    """Per-edge tensors. ``batch_size=[E]``.

    Schema produced by ``collate_molecules``:
        - ``edge_index``: Source-target pairs ``(E, 2)``
        - ``bond_diff``: Edge displacement vectors ``(E, 3)``
        - ``bond_dist``: Edge distances ``(E,)``

    Encoders may add ``edge_features`` of shape ``(E, [L,] D)`` in place.
    """


# ---------------------------------------------------------------------------
# Graph-level (batch_size=[B])
# ---------------------------------------------------------------------------


class GraphData(TensorDict):
    """Per-graph tensors. ``batch_size=[B]``.

    Schema produced by ``collate_molecules``:
        - ``num_atoms``: Atom counts per graph ``(B,)``
        - Graph-level targets (e.g. ``energy``, ``U0``) per ``TargetSchema``.
    """


# ---------------------------------------------------------------------------
# Top-level batch (batch_size=[])
# ---------------------------------------------------------------------------


class GraphBatch(TensorDict):
    """Top-level molecular graph batch. ``batch_size=[]``.

    Nests three levels with independent batch dimensions:
        - ``atoms``: :class:`AtomData` ``(batch_size=[N])``
        - ``edges``: :class:`EdgeData` ``(batch_size=[E])``
        - ``graphs``: :class:`GraphData` ``(batch_size=[B])`` (optional)

    Access with tuple-keys: ``batch["atoms", "Z"]``, ``batch["edges", "edge_index"]``.
    This nested form exists only post-collate; raw samples are flat dicts.
    """
