"""Benchmarks for the Allegro encoder + ``EdgeEnergyHead`` energy pipeline.

Driver: ``pytest benchmarks/bm_molzoo/bm_allegro.py --benchmark-only``
(requires ``pytest-benchmark``). For a one-shot CLI report without that
dependency, see ``benchmarks/run_allegro_bench.py``.
"""

from __future__ import annotations

import pytest
import torch
import torch._dynamo

from molix.data.types import GraphData
from molpot.heads import EdgeEnergyHead
from molzoo.allegro import Allegro


@pytest.fixture
def module():
    return Allegro(
        num_elements=5,
        num_scalar_features=16,
        num_tensor_features=8,
        r_max=5.0,
        num_bessel=8,
        l_max=1,
        num_layers=2,
        type_embed_dim=16,
        latent_mlp_depth=1,
        latent_mlp_width=16,
        avg_num_neighbors=4.0,
    )


@pytest.fixture
def head(module):
    return EdgeEnergyHead(
        input_dim=module.output_dim,
        hidden_dim=128,
        avg_num_neighbors=4.0,
    )


@pytest.fixture
def batch(graph_batch_td):
    """Add a ``graphs`` subdict so ``EdgeEnergyHead`` can read ``batch_size``."""
    n_graphs = graph_batch_td["atoms"].batch.max().item() + 1
    graph_batch_td["graphs"] = GraphData(
        num_atoms=torch.tensor(
            [(graph_batch_td["atoms", "batch"] == g).sum() for g in range(n_graphs)],
            dtype=torch.long,
        ),
        batch_size=[n_graphs],
    )
    return graph_batch_td


class BMAllegro:
    def test_forward(self, benchmark, module, batch):
        with torch.no_grad():
            benchmark(module, batch.clone())

    def test_forward_energy(self, benchmark, module, head, batch):
        def _full():
            td = module(batch.clone())
            return head(td)

        with torch.no_grad():
            benchmark(_full)

    def test_backward_energy(self, benchmark, module, head, batch):
        def _full_with_grad():
            b = batch.clone()
            b["atoms", "pos"].requires_grad_(True)
            td = module(b)
            e = head(td)["energy"].sum()
            e.backward()

        benchmark(_full_with_grad)

    def test_graph_breaks(self, module, batch):
        explanation = torch._dynamo.explain(module)(batch.clone())
        print(f"Graph break count: {explanation.graph_break_count}")
        print(f"Break reasons: {explanation.break_reasons}")
