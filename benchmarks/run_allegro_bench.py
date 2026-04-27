"""Standalone Allegro performance evaluation.

Builds a representative QM9-scale Allegro encoder + ``EdgeEnergyHead`` energy
pipeline, runs forward / forward+backward (with force gradients) / scaling
sweeps on whichever device is available (CUDA preferred), and prints a single
markdown table to stdout. No pytest-benchmark dependency.

Usage::

    python benchmarks/run_allegro_bench.py
    python benchmarks/run_allegro_bench.py --device cuda --num-layers 3 --l-max 2

Reference target shape (QM9):
    bs=32, N̄=18 atoms / graph, r_cut=5 Å, l_max=2, num_layers=3.
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
from contextlib import contextmanager

import torch

from molix.data.types import AtomData, EdgeData, GraphBatch, GraphData
from molpot.heads import EdgeEnergyHead
from molzoo.allegro import Allegro


def _build_random_qm9_batch(
    *,
    n_graphs: int,
    n_atoms_per_graph: int,
    r_cut: float,
    num_elements: int,
    seed: int,
    device: torch.device,
) -> GraphBatch:
    torch.manual_seed(seed)
    n_atoms_total = n_graphs * n_atoms_per_graph

    pos = torch.randn(n_atoms_total, 3, device=device) * 1.5
    Z = torch.randint(1, num_elements, (n_atoms_total,), device=device, dtype=torch.long)
    batch_idx = torch.arange(n_graphs, device=device).repeat_interleave(n_atoms_per_graph)

    pairs = []
    diffs = []
    dists = []
    for g in range(n_graphs):
        base = g * n_atoms_per_graph
        for i in range(n_atoms_per_graph):
            for j in range(n_atoms_per_graph):
                if i == j:
                    continue
                d = pos[base + j] - pos[base + i]
                r = d.norm()
                if r.item() < r_cut:
                    pairs.append((base + i, base + j))
                    diffs.append(d)
                    dists.append(r)
    edge_index = torch.tensor(pairs, dtype=torch.long, device=device)
    bond_diff = torch.stack(diffs).contiguous()
    bond_dist = torch.stack(dists).contiguous()

    return GraphBatch(
        atoms=AtomData(
            Z=Z, pos=pos, batch=batch_idx, batch_size=[n_atoms_total]
        ),
        edges=EdgeData(
            edge_index=edge_index,
            bond_diff=bond_diff.float(),
            bond_dist=bond_dist.float(),
            batch_size=[edge_index.shape[0]],
        ),
        graphs=GraphData(
            num_atoms=torch.full((n_graphs,), n_atoms_per_graph, dtype=torch.long, device=device),
            batch_size=[n_graphs],
        ),
        batch_size=[],
    )


@contextmanager
def _sync(device: torch.device):
    yield
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_callable(
    fn,
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> tuple[float, float]:
    """Returns (median_ms, stdev_ms) over ``repeats`` calls, after ``warmup``."""
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    samples: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples), statistics.stdev(samples) if len(samples) > 1 else 0.0


def _peak_memory_mib(device: torch.device, fn) -> float:
    if device.type != "cuda":
        return float("nan")
    torch.cuda.reset_peak_memory_stats(device)
    fn()
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device) / (1024 * 1024)


def main() -> None:
    parser = argparse.ArgumentParser(description="Allegro encoder benchmark")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-layers", type=int, default=3, help="Allegro layers L")
    parser.add_argument("--l-max", type=int, default=2)
    parser.add_argument("--num-scalar-features", type=int, default=128)
    parser.add_argument("--num-tensor-features", type=int, default=32)
    parser.add_argument("--n-graphs", type=int, default=32)
    parser.add_argument("--n-atoms-per-graph", type=int, default=18, help="QM9 mean")
    parser.add_argument("--r-cut", type=float, default=5.0)
    parser.add_argument("--num-elements", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--scaling", action="store_true", help="Run E-scaling sweep")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"\n## Allegro benchmark — device={device}, torch={torch.__version__}")
    if device.type == "cuda":
        print(f"## GPU: {torch.cuda.get_device_name(device)}")

    encoder = Allegro(
        num_elements=args.num_elements,
        num_scalar_features=args.num_scalar_features,
        num_tensor_features=args.num_tensor_features,
        r_max=args.r_cut,
        num_bessel=8,
        l_max=args.l_max,
        num_layers=args.num_layers,
        type_embed_dim=args.num_scalar_features,
        latent_mlp_depth=2,
        latent_mlp_width=args.num_scalar_features,
        avg_num_neighbors=4.0,
    ).to(device)
    head = EdgeEnergyHead(
        input_dim=encoder.output_dim,
        hidden_dim=128,
        avg_num_neighbors=4.0,
    ).to(device)

    n_params = sum(p.numel() for p in encoder.parameters()) + sum(
        p.numel() for p in head.parameters()
    )
    print(
        f"## Model: L={args.num_layers}, l_max={args.l_max}, "
        f"F={args.num_scalar_features}, u={args.num_tensor_features}, "
        f"params={n_params/1e6:.3f} M\n"
    )

    base_batch = _build_random_qm9_batch(
        n_graphs=args.n_graphs,
        n_atoms_per_graph=args.n_atoms_per_graph,
        r_cut=args.r_cut,
        num_elements=args.num_elements,
        seed=0,
        device=device,
    )
    n_atoms = base_batch["atoms", "Z"].shape[0]
    n_edges = base_batch["edges", "edge_index"].shape[0]
    print(
        f"## Batch: graphs={args.n_graphs}, atoms={n_atoms}, edges={n_edges}, "
        f"⟨|N|⟩={n_edges/n_atoms:.1f}\n"
    )

    # --- forward only (no_grad) -------------------------------------------
    def fwd_only():
        with torch.no_grad():
            td = encoder(base_batch.clone())
            head(td)

    fwd_med, fwd_std = _time_callable(
        fwd_only, device=device, warmup=args.warmup, repeats=args.repeats
    )

    # --- forward + backward (force gradients to positions) ----------------
    def fwd_bwd():
        b = base_batch.clone()
        b["atoms", "pos"].requires_grad_(True)
        td = encoder(b)
        e = head(td)["energy"].sum()
        e.backward()

    bwd_med, bwd_std = _time_callable(
        fwd_bwd, device=device, warmup=args.warmup, repeats=args.repeats
    )

    # --- peak memory ------------------------------------------------------
    fwd_peak = _peak_memory_mib(device, fwd_only)
    bwd_peak = _peak_memory_mib(device, fwd_bwd)

    print("| Quantity | Value |")
    print("|----------|-------|")
    print(f"| Forward time / batch | {fwd_med:.2f} ± {fwd_std:.2f} ms |")
    print(f"| Forward+backward time / batch | {bwd_med:.2f} ± {bwd_std:.2f} ms |")
    print(f"| Forward / edge | {fwd_med/n_edges*1e3:.2f} µs |")
    print(f"| Forward+backward / edge | {bwd_med/n_edges*1e3:.2f} µs |")
    if device.type == "cuda":
        print(f"| Forward peak memory | {fwd_peak:.1f} MiB |")
        print(f"| Forward+backward peak memory | {bwd_peak:.1f} MiB |")

    # --- scaling sweep ----------------------------------------------------
    if args.scaling:
        print("\n### Scaling sweep (forward, no_grad)")
        print("| n_graphs | atoms | edges | fwd_ms | µs/edge |")
        print("|----------|-------|-------|--------|---------|")
        for n_graphs in (4, 16, 32, 64):
            b = _build_random_qm9_batch(
                n_graphs=n_graphs,
                n_atoms_per_graph=args.n_atoms_per_graph,
                r_cut=args.r_cut,
                num_elements=args.num_elements,
                seed=n_graphs,
                device=device,
            )
            n_e = b["edges", "edge_index"].shape[0]
            n_a = b["atoms", "Z"].shape[0]

            def f():
                with torch.no_grad():
                    head(encoder(b.clone()))

            med, _ = _time_callable(
                f, device=device, warmup=2, repeats=10
            )
            print(
                f"| {n_graphs} | {n_a} | {n_e} | "
                f"{med:.2f} | {med/n_e*1e3:.2f} |"
            )


if __name__ == "__main__":
    main()
