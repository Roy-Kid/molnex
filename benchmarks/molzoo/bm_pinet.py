"""Benchmarks for the PiNet encoder + PiNetPotential pipeline.

Two modes:

1. **pytest-benchmark** (micro)::

       pytest benchmarks/molzoo/bm_pinet.py --benchmark-only

2. **Standalone training launcher**::

       # QM9 — energy only, fp32
       python benchmarks/molzoo/bm_pinet.py --dataset qm9 --max-steps 10000

       # revMD17 — energy + forces, fp64, inductor compile
       python benchmarks/molzoo/bm_pinet.py --dataset revmd17 --molecule aspirin \\
           --precision fp64 --compile inductor --max-steps 50000

       # Smoke test
       python benchmarks/molzoo/bm_pinet.py --smoke
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from molix import config, Trainer
from molix.config import set_precision
from molix.core.losses.molecular import energy_force_mse, energy_mse
from molix.core.metrics import MAE, RMSE
from molix.data import DataModule, MmapDataset, NeighborList, Pipeline
from molix.data.types import AtomData, EdgeData, GraphBatch
from molix.datasets import QM9Source, RevMD17Source
from molix.hooks import (
    CheckpointHook,
    GPUMemoryHook,
    Log,
    MetricsHook,
    StepSpeedHook,
)
from molpot.pinet_potential import PiNetPotential
from molzoo.pinet import PiNet

# ==============================================================================
# Shared defaults
# ==============================================================================

DEFAULTS = {
    "batch_size": 32,
    "lr": 1e-3,
    "hidden_dim": 64,
    "depth": 3,
    "pp_nodes": [32, 32],
    "pi_nodes": [32, 32],
    "ii_nodes": [32, 32],
    "n_basis": 8,
    "r_max": 5.0,
    "rank": 3,
    "layer_reduction": "mean",
    "num_workers": 4,
    "eval_every": 500,
    "log_every": 50,
    "seed": 42,
    "qm9_target": "U0",
}

MOLECULES = [
    "aspirin", "azobenzene", "benzene", "ethanol", "malonaldehyde",
    "naphthalene", "paracetamol", "salicylic", "toluene", "uracil",
]

# ==============================================================================
# pytest-benchmark fixtures & classes
# ==============================================================================


@pytest.fixture
def module():
    return PiNet(
        atom_types=[1, 6, 7, 8],
        r_max=5.0,
        n_basis=8,
        pp_nodes=[32, 32],
        pi_nodes=[32, 32],
        ii_nodes=[32, 32],
        depth=3,
        rank=3,
    )


@pytest.fixture
def potential(module):
    return PiNetPotential(encoder=module, hidden_dim=64)


class BMPiNet:
    def test_forward(self, benchmark, module, graph_batch_td):
        with torch.no_grad():
            benchmark(module, graph_batch_td)

    def test_forward_energy(self, benchmark, potential, graph_batch_td):
        def _full():
            return potential(graph_batch_td, compute_forces=False)

        with torch.no_grad():
            benchmark(_full)

    def test_backward_energy_force(self, benchmark, potential, graph_batch_td):
        def _full_with_grad():
            b = graph_batch_td.clone()
            out = potential(b, compute_forces=True)
            (out["energy"].sum() + out["forces"].square().sum()).backward()

        benchmark(_full_with_grad)

    def test_forward_compiled(self, benchmark, potential, graph_batch_td):
        compiled = torch.compile(potential, backend="inductor")
        with torch.no_grad():
            compiled(graph_batch_td)
        with torch.no_grad():
            benchmark(compiled, graph_batch_td)

    def test_graph_breaks(self, potential, graph_batch_td):
        explanation = torch._dynamo.explain(potential)(graph_batch_td)
        print(f"Graph break count: {explanation.graph_break_count}")
        print(f"Break reasons: {explanation.break_reasons}")


# ==============================================================================
# Standalone CLI — training launcher
# ==============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PiNet training on QM9 / revMD17")

    # dataset
    p.add_argument("--dataset", choices=["qm9", "revmd17"], default="qm9")
    p.add_argument("--molecule", type=str, default="aspirin",
                   help="revMD17 molecule (default: aspirin)")
    p.add_argument("--data-dir", type=str, default="data",
                   help="Root data directory")

    # precision & compile
    p.add_argument("--precision", choices=["fp32", "fp64", "fp16-mixed", "bf16-mixed"],
                   default="fp32")
    p.add_argument("--compile", dest="compile_backend",
                   choices=["none", "eager", "inductor"], default="none")
    p.add_argument("--compile-mode", choices=["default", "reduce-overhead", "max-autotune"],
                   default="default")

    # training
    p.add_argument("--max-steps", type=int, default=10_000)
    p.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    p.add_argument("--eval-every", type=int, default=DEFAULTS["eval_every"])

    # model
    p.add_argument("--hidden-dim", type=int, default=DEFAULTS["hidden_dim"])
    p.add_argument("--depth", type=int, default=DEFAULTS["depth"])
    p.add_argument("--r-max", type=float, default=DEFAULTS["r_max"])
    p.add_argument("--n-basis", type=int, default=DEFAULTS["n_basis"])
    p.add_argument("--rank", type=int, choices=[1, 3, 5], default=DEFAULTS["rank"])
    p.add_argument("--layer-reduction", choices=["mean", "sum", "last"],
                   default=DEFAULTS["layer_reduction"])
    p.add_argument("--qm9-target", type=str, default=DEFAULTS["qm9_target"])

    # misc
    p.add_argument("--work-dir", type=str, default="runs/pinet",
                   help="Output directory for checkpoints and logs")
    p.add_argument("--device", type=str, default=None,
                   help="Device override (default: cuda if available else cpu)")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke test: tiny model, 10 steps")

    return p.parse_args()


def build_encoder(args: argparse.Namespace) -> PiNet:
    return PiNet(
        atom_types=[1, 6, 7, 8],
        r_max=args.r_max,
        cutoff_type="f1",
        basis_type="gaussian",
        n_basis=args.n_basis,
        pp_nodes=DEFAULTS["pp_nodes"],
        pi_nodes=DEFAULTS["pi_nodes"],
        ii_nodes=DEFAULTS["ii_nodes"],
        depth=args.depth,
        activation="tanh",
        rank=args.rank,
    )


def build_model(args: argparse.Namespace) -> PiNetPotential:
    enc = build_encoder(args)
    return PiNetPotential(
        encoder=enc,
        hidden_dim=args.hidden_dim,
        layer_reduction=args.layer_reduction,
    )


def build_pipeline(args: argparse.Namespace) -> Pipeline:
    pipe = Pipeline(f"pinet-{args.dataset}-rcut{args.r_max:.1f}")
    pipe.add(NeighborList(cutoff=args.r_max, max_num_pairs=512, pbc=False),
             name="neighbor_list")
    return pipe.build()


def build_datamodule(
    args: argparse.Namespace,
    sink: Path,
    batch_nodes: set[str],
    n_val: int,
) -> DataModule:
    dataset = MmapDataset(sink)
    n_total = len(dataset)
    n_train = n_total - n_val
    train_ds, val_ds = dataset.split(sizes=(n_train, n_val), seed=args.seed)

    if args.dataset == "qm9":
        target_schema = QM9Source.TARGET_SCHEMA
    else:
        target_schema = RevMD17Source.TARGET_SCHEMA

    return DataModule(
        train_ds, val_ds,
        target_schema=target_schema,
        batch_nodes=list(batch_nodes),
        batch_size=args.batch_size,
        num_workers=DEFAULTS["num_workers"],
        pin_memory=torch.cuda.is_available(),
        seed=args.seed,
    )


def build_loss_fn(args: argparse.Namespace):
    if args.dataset == "qm9":
        return energy_mse(target_key=args.qm9_target, pred_key="energy")
    return energy_force_mse(
        energy_target_key="energy", force_target_key="forces",
        energy_pred_key="energy", force_pred_key="forces",
        lambda_F=1000.0,
    )


def _warmup_forward(model: nn.Module, device: torch.device) -> None:
    """Run a dummy batch through the model to initialize LazyLinear parameters."""
    _ft = config.ftype
    _dummy = GraphBatch(
        atoms=AtomData(
            Z=torch.tensor([1, 6, 7, 8], dtype=torch.long, device=device),
            pos=torch.randn(4, 3, dtype=_ft, device=device),
            batch=torch.zeros(4, dtype=torch.long, device=device),
            batch_size=[4],
        ),
        edges=EdgeData(
            edge_index=torch.tensor(
                [[0, 1], [1, 0], [0, 2], [2, 0], [1, 3], [3, 1], [2, 3], [3, 2]],
                dtype=torch.long, device=device,
            ),
            bond_diff=torch.randn(8, 3, dtype=_ft, device=device),
            bond_dist=torch.randn(8, dtype=_ft, device=device).abs(),
            batch_size=[8],
        ),
        batch_size=[],
    )
    model(_dummy)


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.hidden_dim = 8
        args.depth = 1
        args.n_basis = 3
        args.batch_size = 4
        args.max_steps = 10
        args.eval_every = 5
        args.log_every = 1

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.manual_seed(args.seed)

    # ---- precision (MUST be set before model construction) ----------------
    set_precision(args.precision)

    # ---- data pipeline & caching ------------------------------------------
    data_dir = Path(args.data_dir)
    if args.dataset == "qm9":
        source = QM9Source(data_dir)
        n_val = 10_000
    else:
        source = RevMD17Source(data_dir, molecule=args.molecule)
        n_val = 50

    pipe = build_pipeline(args)
    cache_dir = Path(args.work_dir) / "cache"
    print(f"Caching dataset under {cache_dir} ...")
    dag = pipe.cache(source, base_dir=cache_dir)
    sink = dag.final.sink

    dm = build_datamodule(args, sink, pipe.batch_nodes, n_val)

    # ---- model ------------------------------------------------------------
    model = build_model(args).to(device)

    # ---- loss -------------------------------------------------------------
    loss_fn = build_loss_fn(args)

    # ---- hooks ------------------------------------------------------------
    artifacts = Path(args.work_dir) / "artifacts"
    shutil.rmtree(artifacts, ignore_errors=True)
    artifacts.mkdir(parents=True, exist_ok=True)

    pred_key = ("predictions", "energy")
    target_key = ("graphs", args.qm9_target if args.dataset == "qm9" else "energy")
    hooks: list = [
        MetricsHook(metrics=[MAE(), RMSE()], pred_key=pred_key, target_key=target_key),
        StepSpeedHook(window_size=50),
        Log(DEFAULTS["log_every"], keys=["train/loss", "train/MAE"]),
    ]
    hooks.append(
        CheckpointHook(checkpoint_dir=str(artifacts / "checkpoints"), save_every_n_epochs=50)
    )
    if device.type == "cuda":
        hooks.append(GPUMemoryHook())

    # ---- warmup forward to initialize LazyLinear parameters ----------------
    _warmup_forward(model, device)

    # ---- trainer ----------------------------------------------------------
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer_factory=lambda p: torch.optim.Adam(p, lr=args.lr),
        lr_scheduler_factory=lambda opt: torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.8, patience=100,
        ),
        eval_every_n_steps=args.eval_every,
        hooks=hooks,
        device=device,
    )

    # ---- compile ----------------------------------------------------------
    if args.compile_backend != "none":
        mode = args.compile_mode if args.compile_mode != "default" else None
        trainer.compile(backend=args.compile_backend, mode=mode)

    # ---- train ------------------------------------------------------------
    state = trainer.train(datamodule=dm, max_steps=args.max_steps)

    print(f"\nTraining complete. Final eval MAE: {state.get('eval/MAE', 'N/A')}")


if __name__ == "__main__":
    main()
