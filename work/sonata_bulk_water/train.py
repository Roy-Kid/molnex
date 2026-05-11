"""Sonata training driver — bulk-water RPBE-D3 on a single A100.

End-to-end script: build the WaterLES dataset → wire the Allegro encoder +
``build_sonata`` head + Ewald multipole term → train with energy/force MSE →
write ``metrics.json`` / TensorBoard / Zarr-journal / checkpoint artifacts
under ``--out-dir``. No DDP, no rank-0 guards — one process on one A100 is
enough to reproduce the paper-scale numbers (Cheng B. 2025).

References:
    Cheng B., *Latent Ewald summation for machine-learning potentials*,
    **npj Comput. Mater.** 11:80 (2025).
    https://doi.org/10.1038/s41524-025-01577-7

    Allegro-class MLIP on RPBE-D3 liquid water force-RMSE baseline:
    *J. Chem. Phys.* **163**:104102 (2025).
    https://pubs.aip.org/aip/jcp/article/163/10/104102/

    Encoder hyperparameter provenance: ``benchmarks/bm_molpot/bm_sonata.py``
    lines 51-95 (see ``_build_sonata_and_baseline``). The Sonata branch in
    this driver matches those defaults verbatim.

Usage::

    python -u work/sonata_bulk_water/train.py \\
        --data-root /path/to/water_les \\
        --out-dir runs/water_rpbe_d3/sonata \\
        --max-epochs 100 --batch-size 4 --lr 1e-3 --seed 0

Exit codes:
    0 — training completed normally; ``metrics.json`` written.
    2 — :class:`NaNStopHook` aborted on a non-finite loss or parameter;
        ``<out_dir>/nan_checkpoint.pt`` is the dump of the model just
        before the abort and ``train.log`` ends with ``"NaN detected"``.
    1 — any other Python exception (propagated unchanged).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from molix.core.hook import BaseHook
from molix.core.losses import energy_force_mse
from molix.core.metrics import MAE, RMSE
from molix.core.steps import batch_to_device
from molix.core.trainer import Trainer
from molix.data.collate import TargetSchema
from molix.data.datamodule import DataModule
from molix.data.dataset import MmapDataset
from molix.data.pipeline import Pipeline
from molix.data.tasks import NeighborList
from molix.datasets import WaterLESSource
from molix.hooks import (
    CheckpointHook,
    JournalHook,
    MetricsHook,
    ProgressBarHook,
    TensorBoardHook,
)
from molix.io import JournalWriter
from molpot.composition import build_sonata
from molpot.heads import EdgeEnergyHead
from molzoo import Allegro

logger = logging.getLogger("sonata_bulk_water.train")

# eV → meV. metrics.json reports energy MAE in meV/atom and force RMSE in
# meV·Å⁻¹ to match the units the LES paper uses in its result tables.
EV_TO_MEV = 1000.0

# Paper hyperparameters — grep-locked here so a future edit can't silently
# drift away from Cheng 2025 §IV / bm_sonata.py:51-95. ac-005 enforces this
# via grep on the literal numeric values; keep the assignments verbatim.
R_MAX = 5.0  # Å — Allegro cutoff (Cheng 2025 §IV; bm_sonata.py:86)
L_MAX = 2  # Sonata requires l_max ≥ 2 for dipole + quadrupole (bm_sonata.py:87)
NUM_FEATURES = 64  # Allegro num_scalar_features (bm_sonata.py:88)
NUM_LAYERS = 2  # Allegro num_layers (bm_sonata.py:89)
TYPE_EMBED_DIM = 32  # bm_sonata.py:90
LATENT_MLP_WIDTH = 64  # bm_sonata.py:91
AVG_NUM_NEIGHBORS = 12.0  # bulk-water default (bm_sonata.py:92)
NUM_ELEMENTS = 20  # comfortable upper bound; bulk water uses {1, 8}
SIGMA = 1.0  # Å — σ-Gaussian charge-smearing length (Cheng 2025; bm_sonata.py:93)
DL = 2.0  # Å — Ewald reciprocal-space grid resolution (bm_sonata.py:94)

# 192-atom periodic box at water density (~0.033 atoms/Å³) holds roughly
# 17 neighbours per atom inside a 5 Å sphere, i.e. ~1700 half-pairs.
# 4096 leaves a 2× margin without blowing the C++ buffer.
NLIST_MAX_NUM_PAIRS = 4096

# Bulk-water boxes carry no net charge. Sonata's PermMultipoleHead with
# ``constrain_total_charge=True`` projects per-graph charge sums onto
# ``batch["graphs", "total_charge"]``; we inject this key per sample
# (value 0.0) since WaterLESSource doesn't carry it. The extended
# :class:`TargetSchema` routes the flat ``total_charge`` field through
# the collator into the nested ``graphs`` sub-dict.
WATER_LES_SCHEMA: TargetSchema = TargetSchema(
    graph_level=WaterLESSource.TARGET_SCHEMA.graph_level | {"total_charge"},
    atom_level=WaterLESSource.TARGET_SCHEMA.atom_level,
)


def _to_targets_dict(sample: dict) -> dict:
    """Pipeline task: nest ``energy``/``forces``/``total_charge`` under ``targets``.

    :class:`WaterLESSource` returns a flat sample dict with ``energy`` and
    ``forces`` at the top level, but :func:`molix.data.collate.collate_molecules`
    only inspects ``sample["targets"]`` for graph- and atom-level fields.
    We also inject ``total_charge=0.0`` here — bulk water carries no net
    charge, and ``build_sonata(..., constrain_total_charge=True)`` needs
    that key under ``("graphs", "total_charge")`` post-collate.
    """
    out = {k: v for k, v in sample.items() if k not in ("energy", "forces")}
    out["targets"] = {
        "energy": sample["energy"],
        "forces": sample["forces"],
        "total_charge": 0.0,
    }
    return out


# ---------------------------------------------------------------------------
# Distinct metric classes
# ---------------------------------------------------------------------------
#
# :class:`molix.hooks.MetricsHook` writes scalars under
# ``state["train"][type(metric).__name__]``. Energy and force MAE/RMSE
# must end up at distinct keys, so we subclass the bare :class:`MAE` /
# :class:`RMSE` once each — the only purpose is the ``__name__`` rename.
# Same accumulator semantics as the parents.


class EnergyMAE(MAE):
    """MAE on per-graph energies (eV), published as ``train/EnergyMAE``."""


class ForceRMSE(RMSE):
    """RMSE on per-atom forces (eV·Å⁻¹), published as ``train/ForceRMSE``."""


# ---------------------------------------------------------------------------
# Custom training/eval steps — pass compute_forces=True, keep autograd live
# ---------------------------------------------------------------------------
#
# :class:`Sonata.forward` derives forces via :func:`torch.autograd.grad`,
# which is incompatible with the :class:`DefaultEvalStep`'s
# ``torch.no_grad()`` wrapper. These two thin step classes
#
#   * call ``model(batch, compute_forces=True)`` so the head returns both
#     energy and forces;
#   * keep the eval pass under :func:`torch.enable_grad` so the autograd
#     graph survives long enough for the force derivative.


class _SonataTrainStep:
    """Training step: forward → loss → backward → optimiser step."""

    def on_train_batch(self, trainer, state, batch):
        assert trainer.model is not None
        assert trainer.loss_fn is not None
        assert trainer.optimizer is not None
        preds = trainer.model(batch, compute_forces=True)
        loss = trainer.loss_fn(preds, batch)
        trainer.optimizer.zero_grad()
        loss.backward()
        trainer._call_hooks("on_after_backward", trainer, state)
        trainer.optimizer.step()
        state["train"]["loss"] = loss.item()
        return {"loss": loss, "predictions": preds}

    def on_eval_batch(self, trainer, state, batch):
        raise NotImplementedError


class _SonataEvalStep:
    """Eval step: forward + loss only, autograd kept live for force grad."""

    def on_train_batch(self, trainer, state, batch):
        raise NotImplementedError

    def on_eval_batch(self, trainer, state, batch):
        assert trainer.model is not None
        assert trainer.loss_fn is not None
        with torch.enable_grad():
            preds = trainer.model(batch, compute_forces=True)
            loss = trainer.loss_fn(preds, batch)
        state["eval"]["loss"] = loss.item()
        return {"loss": loss, "predictions": preds}


# ---------------------------------------------------------------------------
# NaN early-stop hook — aborts training, exit code 2 via main()
# ---------------------------------------------------------------------------


class NaNStopHook(BaseHook):
    """Abort training when ``state["train"]["loss"]`` or a parameter is non-finite.

    Catches the two failure modes that silently kill a long run:

    1. Loss explodes to ``NaN`` or ``+inf`` — usually a runaway gradient
       (the Ewald long-range term is the standard culprit at small
       :data:`SIGMA`).
    2. A parameter tensor turns non-finite — happens when the optimiser
       follows a NaN gradient.

    On detection the hook dumps the model state to
    ``<out_dir>/nan_checkpoint.pt`` and raises :class:`RuntimeError` so
    :class:`Trainer._train` unwinds cleanly. The caller in :func:`main`
    catches it and translates to exit code ``2``, which lets SLURM
    ``--mail-type=FAIL`` and any outer wrapper distinguish "NaN" from
    "other crash" (exit code 1).
    """

    def __init__(self, model: nn.Module, out_dir: Path) -> None:
        self.model = model
        self.out_dir = out_dir

    def on_train_batch_end(self, trainer, state, batch, outputs) -> None:
        loss = state["train"].get("loss")
        if loss is not None and not math.isfinite(float(loss)):
            self._abort(state, reason=f"non-finite loss={loss}")
            return
        for name, p in self.model.named_parameters():
            if not torch.isfinite(p).all():
                self._abort(state, reason=f"non-finite parameter {name}")
                return

    def _abort(self, state, *, reason: str) -> None:
        step = int(state.get("global_step", 0))
        logger.error("NaN detected at step=%d — %s", step, reason)
        torch.save(self.model.state_dict(), self.out_dir / "nan_checkpoint.pt")
        raise RuntimeError(f"NaN detected — early stopping ({reason})")


# ---------------------------------------------------------------------------
# Debug-only loss-injection hook — used by the ac-002 NaN-stop smoke test
# ---------------------------------------------------------------------------


class _DebugNaNInjectorHook(BaseHook):
    """Overwrite ``state["train"]["loss"]`` with NaN at a fixed step.

    Sole purpose is to exercise :class:`NaNStopHook` from a synthetic
    smoke run without having to wait for the optimiser to blow up. Wired
    only when ``--debug-inject-nan`` is passed.
    """

    def __init__(self, *, at_step: int = 2) -> None:
        self.at_step = at_step

    def on_train_batch_end(self, trainer, state, batch, outputs) -> None:
        if int(state.get("global_step", 0)) == self.at_step:
            state["train"]["loss"] = float("nan")


# ---------------------------------------------------------------------------
# Model builder — Allegro encoder + EdgeEnergyHead short range + Sonata Ewald
# ---------------------------------------------------------------------------


def _build_sonata(*, seed: int) -> nn.Module:
    """Wire Allegro + EdgeEnergyHead + ``build_sonata`` per Cheng 2025.

    Equivalent to ``benchmarks/bm_molpot/bm_sonata.py``'s Sonata branch
    (``_build_sonata_and_baseline`` with ``expose_tensor_track=True``);
    we re-seed twice so the encoder and head start from a deterministic
    init independent of any global RNG state pre-call.
    """
    torch.manual_seed(seed)
    encoder = Allegro(
        num_elements=NUM_ELEMENTS,
        num_scalar_features=NUM_FEATURES,
        num_tensor_features=NUM_FEATURES // 4,
        r_max=R_MAX,
        num_bessel=8,
        l_max=L_MAX,
        num_layers=NUM_LAYERS,
        type_embed_dim=TYPE_EMBED_DIM,
        latent_mlp_depth=1,
        latent_mlp_width=LATENT_MLP_WIDTH,
        avg_num_neighbors=AVG_NUM_NEIGHBORS,
        expose_tensor_track=True,
    )
    torch.manual_seed(seed + 1)
    short_head = EdgeEnergyHead(
        input_dim=encoder.output_dim,
        hidden_dim=128,
        avg_num_neighbors=AVG_NUM_NEIGHBORS,
        out_key="energy_short",
    )
    return build_sonata(
        encoder,
        sigma=SIGMA,
        dl=DL,
        charge=True,
        dipole=True,
        quadrupole=True,
        constrain_total_charge=True,
        avg_num_neighbors=AVG_NUM_NEIGHBORS,
        short_range_head=short_head,
    )


# ---------------------------------------------------------------------------
# Dataset builder — one cache per split, MmapDataset wrap (zero-copy mmap)
# ---------------------------------------------------------------------------


def _build_split_dataset(
    *,
    data_root: Path,
    split: str,
    out_dir: Path,
) -> MmapDataset:
    """Materialise ``<split>`` through a ``Pipeline().add(NeighborList).cache``.

    Each split gets its own cache file under ``out_dir/cache/<split>``;
    the source's ``source_id`` already embeds ``split=`` so collisions
    are impossible. We wrap the resulting :class:`PackedCache` with
    :class:`MmapDataset` so the DataLoader workers can fan out across
    A100 PCIe bandwidth via zero-copy mmap without each holding the
    full split in resident memory.
    """
    source = WaterLESSource(root=data_root, split=split)
    pipe = (
        Pipeline(f"water-les-{split}")
        .add(_to_targets_dict, name="targets")
        .add(
            NeighborList(
                cutoff=R_MAX,
                max_num_pairs=NLIST_MAX_NUM_PAIRS,
                pbc=True,
                symmetry=True,
            )
        )
        .build()
    )
    packed = pipe.cache(source, base_dir=out_dir / "cache" / split)
    return MmapDataset(packed.sink)


# ---------------------------------------------------------------------------
# Manual test-eval — gives full unit control over the metrics.json payload
# ---------------------------------------------------------------------------


def _evaluate_test_split(
    *,
    model: nn.Module,
    test_dm: DataModule,
    device: torch.device,
) -> dict[str, float]:
    """Run the test dataloader once and return final-units metrics.

    Trainer hooks publish raw MAE/RMSE in eV / eV·Å⁻¹; the LES paper
    reports meV/atom and meV·Å⁻¹ instead. Rather than threading unit
    conversion through :class:`MetricsHook` we compute it ourselves
    here — one pass over the test loader, no batch-level statistics
    leak into ``state["eval"]`` (which would pollute the
    ``best_metric`` history used by :class:`CheckpointHook`).

    Returns:
        ``{"energy_mae_meV_per_atom": …, "force_rmse_meV_per_A": …}``.
    """
    model.eval()
    abs_err_sum = 0.0  # Σ |E_pred − E_ref|  (eV)
    n_atoms_sum = 0  # Σ N_atoms over all graphs
    sq_err_sum = 0.0  # Σ (F_pred − F_ref)²   (eV·Å⁻¹)²
    n_force_components = 0  # Σ 3·N_atoms over all graphs
    for batch in test_dm.val_dataloader():
        batch = batch_to_device(batch, device)
        with torch.enable_grad():
            preds = model(batch, compute_forces=True)
        e_pred = preds["energy"].detach()
        e_ref = batch["graphs", "energy"].to(dtype=e_pred.dtype).view_as(e_pred)
        f_pred = preds["forces"].detach()
        f_ref = batch["atoms", "forces"].to(dtype=f_pred.dtype).view_as(f_pred)
        num_atoms = batch["graphs", "num_atoms"].to(dtype=torch.long)
        abs_err_sum += (e_pred - e_ref).abs().sum().item()
        n_atoms_sum += int(num_atoms.sum().item())
        sq_err_sum += ((f_pred - f_ref) ** 2).sum().item()
        n_force_components += int(f_pred.numel())
    if n_atoms_sum == 0 or n_force_components == 0:
        raise RuntimeError("test split is empty; cannot write metrics.json")
    energy_mae_eV_per_atom = abs_err_sum / n_atoms_sum
    force_rmse_eV_per_A = math.sqrt(sq_err_sum / n_force_components)
    return {
        "energy_mae_meV_per_atom": EV_TO_MEV * energy_mae_eV_per_atom,
        "force_rmse_meV_per_A": EV_TO_MEV * force_rmse_eV_per_A,
    }


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------


def _parse_argv(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sonata bulk-water RPBE-D3 training — single A100, no DDP.",
    )
    p.add_argument("--data-root", type=Path, required=True, help="WaterLES dataset root.")
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory.")
    p.add_argument("--max-epochs", type=int, default=100, help="Maximum number of epochs.")
    p.add_argument("--batch-size", type=int, default=4, help="Batch size (per A100).")
    p.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    p.add_argument("--seed", type=int, default=0, help="Master seed.")
    p.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker count; pass 0 for synchronous loading.",
    )
    p.add_argument(
        "--debug-inject-nan",
        action="store_true",
        help="Force loss=NaN at step 2 (ac-002 smoke check only).",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _configure_logging(out_dir: Path) -> None:
    """Root logger at INFO; mirror to stderr and ``<out_dir>/train.log``."""
    handlers: list[logging.Handler] = [
        logging.StreamHandler(stream=sys.stderr),
        logging.FileHandler(out_dir / "train.log"),
    ]
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    for h in handlers:
        h.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # Reset prior handlers so re-running in the same process is idempotent.
    root.handlers = handlers


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = _parse_argv(argv)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    _configure_logging(out_dir)
    torch.manual_seed(args.seed)
    logger.info(
        "seed=%d max_epochs=%d batch_size=%d lr=%g",
        args.seed,
        args.max_epochs,
        args.batch_size,
        args.lr,
    )

    # --- data ---
    logger.info("building datasets under %s", args.data_root)
    train_ds = _build_split_dataset(data_root=args.data_root, split="train", out_dir=out_dir)
    val_ds = _build_split_dataset(data_root=args.data_root, split="val", out_dir=out_dir)
    test_ds = _build_split_dataset(data_root=args.data_root, split="test", out_dir=out_dir)

    dm = DataModule(
        train_ds,
        val_ds,
        target_schema=WATER_LES_SCHEMA,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    test_dm = DataModule(
        test_ds,
        test_ds,
        target_schema=WATER_LES_SCHEMA,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # --- model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s", device)
    model = _build_sonata(seed=args.seed)

    # --- hooks ---
    journal_writer = JournalWriter(out_dir / "journal", run_id="train")
    metrics_energy = MetricsHook(
        metrics=[EnergyMAE()],
        pred_key=("predictions", "energy"),
        target_key=("graphs", "energy"),
    )
    metrics_force = MetricsHook(
        metrics=[ForceRMSE()],
        pred_key=("predictions", "forces"),
        target_key=("atoms", "forces"),
    )
    hooks: list[Any] = [
        metrics_energy,
        metrics_force,
        NaNStopHook(model=model, out_dir=out_dir),
        CheckpointHook(
            checkpoint_dir=str(out_dir / "checkpoints"),
            save_last=True,
            save_best=True,
            best_metric_name=("eval", "ForceRMSE"),
            best_metric_mode="min",
        ),
        TensorBoardHook(every_n_steps=10, log_dir=str(out_dir / "tb")),
        JournalHook(every_n_steps=10, store=journal_writer),
        ProgressBarHook(desc="Sonata"),
    ]
    if args.debug_inject_nan:
        hooks.insert(2, _DebugNaNInjectorHook(at_step=2))
        logger.warning("--debug-inject-nan active; loss will be forced to NaN at step 2")

    # --- trainer ---
    trainer = Trainer(
        model=model,
        loss_fn=energy_force_mse(),
        optimizer_factory=lambda p: torch.optim.Adam(p, lr=args.lr),
        train_step=_SonataTrainStep(),
        eval_step=_SonataEvalStep(),
        hooks=hooks,
        device=device,
    )

    try:
        trainer.train(dm, max_epochs=args.max_epochs)
    except RuntimeError as e:
        if "NaN detected" in str(e):
            logger.error("training aborted by NaNStopHook: %s", e)
            return 2
        raise

    # --- test eval + metrics.json ---
    logger.info("training complete; running test eval")
    metrics = _evaluate_test_split(model=model, test_dm=test_dm, device=device)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info(
        "metrics.json written: energy_mae=%.4f meV/atom, force_rmse=%.4f meV/Å",
        metrics["energy_mae_meV_per_atom"],
        metrics["force_rmse_meV_per_A"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
