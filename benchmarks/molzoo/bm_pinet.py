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
from tensordict import TensorDict

from molix import Trainer, config
from molix.config import set_precision
from molix.core.losses.molecular import energy_force_mse, energy_mse
from molix.core.metrics import MAE, RMSE
from molix.data import AtomicDress, DataModule, MmapDataset, NeighborList, Pipeline
from molix.datasets import QM9Source, RevMD17Source
from molix.hooks import (
    CheckpointHook,
    GPUMemoryHook,
    GradClipHook,
    Log,
    MetricsHook,
    StepSpeedHook,
    TensorBoardHook,
)
from molzoo.pinet import PiNet, PiNetPotential

# ==============================================================================
# Shared defaults
# ==============================================================================

# Aligned with Teoroo-CMC/PiNN reference YAMLs:
#   inputs/qm9-pinet.yml + inputs/md17-pinet2.yml (commit b592996c).
# depth/pp_nodes/pi_nodes/ii_nodes/out_nodes/n_basis/rc come verbatim from
# the YAML; optimizer (Adam, lr=1e-4, ExponentialDecay 0.994/10k step,
# global_clipnorm=0.01) and loss (e=1.0, f=10.0) wired in build_*.
DEFAULTS = {
    "batch_size": 32,
    "lr": 1e-4,
    "hidden_dim": 64,
    "depth": 5,
    "pp_nodes": [64, 64, 64, 64],
    "pi_nodes": [64],
    "ii_nodes": [64, 64, 64, 64],
    "n_basis": 10,
    "r_max": 4.5,
    "rank": 3,
    "layer_reduction": "mean",
    "num_workers": 4,
    "eval_every": 500,
    "log_every": 5000,
    "tb_every": 2000,
    "log_start_step": 10_000,
    "seed": 42,
    "qm9_target": "U0",
    # QM9 covers H, C, N, O, F. Atomic dressing baselines are fit on all
    # five even though the encoder embedding may only model a subset —
    # unknown atoms fall back to a 0 baseline (see AtomicDress.execute).
    "qm9_dress_elements": (1, 6, 7, 8, 9),
}

MOLECULES = [
    "aspirin",
    "azobenzene",
    "benzene",
    "ethanol",
    "malonaldehyde",
    "naphthalene",
    "paracetamol",
    "salicylic",
    "toluene",
    "uracil",
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
    p.add_argument(
        "--molecule", type=str, default="aspirin", help="revMD17 molecule (default: aspirin)"
    )
    p.add_argument("--data-dir", type=str, default="data", help="Root data directory")

    # precision & compile
    p.add_argument(
        "--precision",
        choices=["fp32", "fp64", "amp-fp16", "amp-bf16", "fp16-mixed", "bf16-mixed"],
        default="fp32",
        help="amp-fp16/amp-bf16 are aliases for fp16-mixed/bf16-mixed",
    )
    p.add_argument(
        "--compile",
        dest="compile_backend",
        choices=["none", "dynamo-eager", "aot-eager", "inductor", "eager", "aot_eager"],
        default="none",
        help="eager/aot_eager are accepted as legacy aliases",
    )
    p.add_argument(
        "--compile-mode", choices=["default", "reduce-overhead", "max-autotune"], default="default"
    )

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
    p.add_argument(
        "--layer-reduction", choices=["mean", "sum", "last"], default=DEFAULTS["layer_reduction"]
    )
    p.add_argument("--qm9-target", type=str, default=DEFAULTS["qm9_target"])

    # misc
    p.add_argument(
        "--work-dir",
        type=str,
        default="runs/pinet",
        help="Output directory for checkpoints and logs",
    )
    p.add_argument(
        "--tb-every",
        type=int,
        default=DEFAULTS["tb_every"],
        help="TensorBoard scalar logging cadence (steps)",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=DEFAULTS["log_every"],
        help="Console-table logging cadence (steps)",
    )
    p.add_argument(
        "--log-start-step",
        type=int,
        default=DEFAULTS["log_start_step"],
        help="Suppress train/perf scalar logging before this step "
        "(eval points still logged). Keeps the initial loss "
        "explosion from compressing plots / log tables.",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (default: cuda if available else cpu)",
    )
    p.add_argument("--smoke", action="store_true", help="Smoke test: tiny model, 10 steps")

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
        compute_forces=(args.dataset == "revmd17"),
    )


def build_pipeline(args: argparse.Namespace) -> Pipeline:
    pipe = Pipeline(f"pinet-{args.dataset}-rcut{args.r_max:.1f}")

    # Per-element baseline subtraction. For QM9 this is the standard
    # PiNet/SchNet protocol — without it the model has to learn the
    # ~-4×10⁴ eV mean of U0 from a near-zero initialization, which
    # wrecks the optimization landscape and makes AMP-BF16 numerically
    # dead (7-bit mantissa can't resolve sub-eV residuals on top of an
    # O(10⁴ eV) mean). For revMD17 the molecule composition is constant
    # within a sweep, so the lstsq fit collapses to one constant per
    # element that subtracts the molecule's mean energy — equivalent to
    # a global mean-shift, which is what every MD17 reference impl
    # applies. The cache encodes the fitted baselines so train / val /
    # test see the same subtraction.
    if args.dataset == "qm9":
        dress_target = args.qm9_target
    else:
        dress_target = "energy"
    pipe.add(
        AtomicDress(
            elements=list(DEFAULTS["qm9_dress_elements"]),
            target_key=dress_target,
            output_key=dress_target,
        ),
        name="atomic_dress",
    )
    pipe.add(NeighborList(cutoff=args.r_max, max_num_pairs=512, pbc=False), name="neighbor_list")
    return pipe.build()


def build_datamodule(
    args: argparse.Namespace,
    sink: Path,
    batch_nodes: set[str],
    n_train: int,
    n_val: int,
) -> DataModule:
    dataset = MmapDataset(sink)
    n_total = len(dataset)
    n_test = n_total - n_train - n_val
    if n_test < 0:
        raise ValueError(f"Dataset has {n_total} samples but n_train+n_val={n_train + n_val}.")
    # Held-out test partition is kept off-loader (PiNN paper protocol):
    # 110k/10k/10k for QM9, 950/50/~99k for revMD17.
    if n_test > 0:
        train_ds, val_ds, _test_ds = dataset.split(
            sizes=(n_train, n_val, n_test),
            seed=args.seed,
        )
    else:
        train_ds, val_ds = dataset.split(sizes=(n_train, n_val), seed=args.seed)

    if args.dataset == "qm9":
        target_schema = QM9Source.TARGET_SCHEMA
    else:
        target_schema = RevMD17Source.TARGET_SCHEMA

    return DataModule(
        train_ds,
        val_ds,
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
        energy_target_key="energy",
        force_target_key="forces",
        energy_pred_key="energy",
        force_pred_key="forces",
        lambda_F=10.0,
    )


def _warmup_forward(model: nn.Module, device: torch.device) -> None:
    """Run a dummy batch through the model to initialize LazyLinear parameters."""
    _ft = config.ftype
    _dummy = TensorDict(
        atoms=TensorDict(
            Z=torch.tensor([1, 6, 7, 8], dtype=torch.long, device=device),
            pos=torch.randn(4, 3, dtype=_ft, device=device),
            batch=torch.zeros(4, dtype=torch.long, device=device),
            batch_size=[4],
        ),
        edges=TensorDict(
            edge_index=torch.tensor(
                [[0, 1], [1, 0], [0, 2], [2, 0], [1, 3], [3, 1], [2, 3], [3, 2]],
                dtype=torch.long,
                device=device,
            ),
            batch_size=[8],
        ),
        graphs=TensorDict(
            num_atoms=torch.tensor([4], dtype=torch.long, device=device),
            batch_size=[1],
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
        args.tb_every = 1
        args.log_start_step = 0

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)

    # ---- precision (MUST be set before model construction) ----------------
    precision_alias = {"amp-fp16": "fp16-mixed", "amp-bf16": "bf16-mixed"}
    set_precision(precision_alias.get(args.precision, args.precision))

    # ---- data pipeline & caching ------------------------------------------
    data_dir = Path(args.data_dir)
    if args.dataset == "qm9":
        # Smoke loads a tiny subset (200 molecules, 20 val) to skip the ~30s
        # parse of all 130k xyz files; the same SOURCE_VERSION+total identity
        # also keeps the smoke cache separate from the production cache.
        # Production split mirrors PiNet2 paper: 110k train / 10k val / 10k test.
        smoke_total = 200 if args.smoke else None
        source = QM9Source(data_dir, total=smoke_total)
        n_val = 20 if args.smoke else 10_000
        n_train = 160 if args.smoke else 110_000
    else:
        # revMD17: paper protocol is 950 train / 50 val / remainder held out
        # as test set (~99k unused during training).
        smoke_total = 200 if args.smoke else None
        source = RevMD17Source(data_dir, molecule=args.molecule, total=smoke_total)
        n_val = 20 if args.smoke else 50
        n_train = 160 if args.smoke else 950

    pipe = build_pipeline(args)
    cache_dir = Path(args.work_dir) / "cache"
    print(f"Caching dataset under {cache_dir} ...")
    dag = pipe.cache(source, base_dir=cache_dir)
    sink = dag.final.sink

    dm = build_datamodule(args, sink, pipe.batch_nodes, n_train, n_val)

    # ---- model ------------------------------------------------------------
    model = build_model(args).to(device)

    # ---- loss -------------------------------------------------------------
    loss_fn = build_loss_fn(args)

    # ---- hooks ------------------------------------------------------------
    artifacts = Path(args.work_dir) / "artifacts"
    shutil.rmtree(artifacts, ignore_errors=True)
    artifacts.mkdir(parents=True, exist_ok=True)

    tb_dir = Path(args.work_dir) / "tensorboard"
    tb_dir.mkdir(parents=True, exist_ok=True)

    # Metric wiring. Energy metrics get an ``E_`` prefix and force metrics
    # an ``F_`` prefix on both console rows and TB tags, so the meaning is
    # unambiguous even on energy-only datasets. revMD17 publishes both
    # channels; QM9 publishes ``E_*`` only.
    energy_metrics = MetricsHook(
        metrics=[MAE(), RMSE()],
        pred_key=("predictions", "energy"),
        target_key=("graphs", args.qm9_target if args.dataset == "qm9" else "energy"),
        name_prefix="E_",
    )
    if args.dataset == "qm9":
        metric_hooks = [energy_metrics]
        log_keys = ["train/loss", "train/E_MAE", "train/E_RMSE", "eval/E_MAE", "eval/E_RMSE"]
    else:  # revmd17
        metric_hooks = [
            energy_metrics,
            MetricsHook(
                metrics=[MAE(), RMSE()],
                pred_key=("predictions", "forces"),
                target_key=("atoms", "forces"),
                name_prefix="F_",
            ),
        ]
        log_keys = [
            "train/loss",
            "train/E_MAE",
            "train/F_MAE",
            "eval/E_MAE",
            "eval/E_RMSE",
            "eval/F_MAE",
            "eval/F_RMSE",
        ]

    hooks: list = [
        *metric_hooks,
        StepSpeedHook(window_size=50),
        Log(
            args.log_every,
            keys=log_keys,
            start_step=args.log_start_step,
        ),
        TensorBoardHook(
            every_n_steps=args.tb_every,
            log_dir=str(tb_dir),
            start_step=args.log_start_step,
        ),
    ]
    # PiNN reference uses Adam(global_clipnorm=0.01) on all precisions — keep
    # always-on instead of toggling per AMP mode.
    hooks.append(GradClipHook(max_norm=0.01))
    hooks.append(
        # Step-based cadence. With --eval-every 5000 and 3M total steps:
        #   * last.pt fires at every eval (5000 / 10000 / ... / 3M)
        #   * step_N.pt archives every 50k steps (60 snapshots total)
        # No epoch-tied saves — matches every other hook's step-based clock.
        CheckpointHook(
            checkpoint_dir=str(artifacts / "checkpoints"),
            save_every_n_steps=50_000,
            save_last=True,
        )
    )
    if device.type == "cuda":
        hooks.append(GPUMemoryHook())

    # ---- warmup forward to initialize LazyLinear parameters ----------------
    _warmup_forward(model, device)

    # ---- trainer ----------------------------------------------------------
    # PiNN reference: tf.keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate=1e-4, decay_rate=0.994, decay_steps=10_000)
    # which is `lr_t = lr_0 * 0.994 ** (t / 10000)` per step (continuous).
    # Torch equivalent: ExponentialLR stepped per train step with
    # gamma = 0.994 ** (1 / 10000).
    decay_gamma_per_step = 0.994 ** (1.0 / 10_000)
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer_factory=lambda p: torch.optim.Adam(p, lr=args.lr),
        lr_scheduler_factory=lambda opt: torch.optim.lr_scheduler.ExponentialLR(
            opt,
            gamma=decay_gamma_per_step,
        ),
        eval_every_n_steps=args.eval_every,
        hooks=hooks,
        device=device,
    )

    # ---- compile ----------------------------------------------------------
    if args.compile_backend != "none":
        backend_alias = {"dynamo-eager": "eager", "aot-eager": "aot_eager"}
        backend = backend_alias.get(args.compile_backend, args.compile_backend)
        mode = args.compile_mode if args.compile_mode != "default" else None
        trainer.compile(backend=backend, mode=mode)

    # ---- train ------------------------------------------------------------
    state = trainer.train(datamodule=dm, max_steps=args.max_steps)

    print(f"\nTraining complete. Final eval E_MAE: {state.get('eval/E_MAE', 'N/A')}")


if __name__ == "__main__":
    main()
