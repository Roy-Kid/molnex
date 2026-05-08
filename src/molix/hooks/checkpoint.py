"""Checkpoint persistence hook."""

from __future__ import annotations

from molix import logger as _logger_mod
from molix import logging as _logging_mod
from molix.core.hook import BaseHook
from molix.core.state import Path, resolve

logger = _logger_mod.getLogger(__name__)


class CheckpointHook(BaseHook):
    """Saves model checkpoints during training.

    Delegates to :meth:`trainer._checkpoint.state_dict` so that the saved
    payload always carries the *complete* set of resume-critical state
    (model, optimizer, lr_scheduler, AMP scaler, RNG states, counters,
    best-metric tracking).

    Args:
        checkpoint_dir: Directory to save checkpoints into.
        save_every_n_epochs: Save an ``epoch_{N}.pt`` snapshot every N epochs.
        save_last: Write ``last.pt`` at end of training and on every
            epoch-end save (so a kill -9'd run still has a resumable
            snapshot).
        save_best: Track a scalar metric in :class:`TrainState` and write
            ``best.pt`` whenever it improves.
        best_metric_name: Path into ``state`` to track for ``save_best``.
            Default ``("eval", "loss")``.
        best_metric_mode: ``"min"`` (smaller = better) or ``"max"``.
        register_artifacts: Register each written checkpoint via
            ``trainer.ctx.save_artifact`` (if present).
    """

    _MINUS_INF = float("-inf")
    _PLUS_INF = float("inf")

    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        save_every_n_epochs: int = 1,
        save_last: bool = True,
        save_best: bool = False,
        best_metric_name: Path = ("eval", "loss"),
        best_metric_mode: str = "min",
        register_artifacts: bool = False,
    ):
        import os

        import torch

        if best_metric_mode not in ("min", "max"):
            raise ValueError(f"best_metric_mode must be 'min' or 'max', got {best_metric_mode!r}")

        self.os = os
        self.torch = torch

        self.checkpoint_dir = checkpoint_dir
        self.save_every_n_epochs = save_every_n_epochs
        self.save_last = save_last
        self.save_best = save_best
        self.best_metric_name = best_metric_name
        self.best_metric_mode = best_metric_mode
        self.register_artifacts = register_artifacts

        self._best_value: float | None = None
        self._last_announced: tuple[str, int] | None = None

    def on_train_start(self, trainer, state):
        """Create checkpoint directory and sync best-metric metadata."""
        self.os.makedirs(self.checkpoint_dir, exist_ok=True)
        ckpt = getattr(trainer, "_checkpoint", None)
        if ckpt is not None:
            ckpt.best_metric_name = self.best_metric_name

    def on_epoch_end(self, trainer, state):
        """Save periodic / best / last checkpoints at epoch end."""
        if self.save_every_n_epochs > 0 and (state.epoch + 1) % self.save_every_n_epochs == 0:
            self._save_checkpoint(trainer, state, f"epoch_{state.epoch}.pt")
        if self.save_best:
            self._maybe_save_best(trainer, state)
        if self.save_last:
            self._save_checkpoint(trainer, state, "last.pt")

    def on_train_end(self, trainer, state):
        """Save final ``last.pt`` unless ``on_epoch_end`` just wrote it."""
        if self.save_last:
            self._save_checkpoint(trainer, state, "last.pt")

    def _is_improvement(self, candidate: float) -> bool:
        if self._best_value is None:
            return True
        if self.best_metric_mode == "min":
            return candidate < self._best_value
        return candidate > self._best_value

    def _maybe_save_best(self, trainer, state):
        raw = resolve(state, self.best_metric_name)
        if raw is None:
            return
        value = float(raw)
        if not self._is_improvement(value):
            return
        self._best_value = value
        ckpt = getattr(trainer, "_checkpoint", None)
        if ckpt is not None:
            ckpt.best_metric = value
        self._save_checkpoint(trainer, state, "best.pt")

    def _build_state_dict(self, trainer, state) -> dict:
        """Produce the complete resume-ready state dict."""
        ckpt = getattr(trainer, "_checkpoint", None)
        if ckpt is not None:
            ckpt.epoch = int(state.epoch)
            ckpt.global_step = int(state.global_step)
            if self.save_best and self._best_value is not None:
                ckpt.best_metric = self._best_value
            return ckpt.state_dict()
        sd: dict = {
            "epoch": int(state.epoch),
            "global_step": int(state.global_step),
        }
        if trainer.model is not None:
            sd["model_state_dict"] = trainer.model.state_dict()
        if trainer.optimizer is not None:
            sd["optimizer_state_dict"] = trainer.optimizer.state_dict()
        return sd

    def _save_checkpoint(self, trainer, state, filename):
        """Save checkpoint to disk; dedupe redundant console announces."""
        step = int(state.get("global_step", 0))

        filepath = self.os.path.join(self.checkpoint_dir, filename)
        checkpoint = self._build_state_dict(trainer, state)
        self.torch.save(checkpoint, filepath)

        logger.info(f"Saved checkpoint to {filepath}")

        marker = (filename, step)
        if self._last_announced == marker:
            return
        self._last_announced = marker
        message = f"ckpt: {filename} @ step={step}"
        evt = _logging_mod.events_logger()
        width = _logging_mod.get_table_width()
        if _logging_mod.has_effective_handlers(evt):
            evt.info(
                message,
                kind="announce",
                table_width=width,
                category="checkpoint",
                filename=filename,
                step=step,
                filepath=filepath,
            )
        else:
            prefix = f"─── {message} "
            pad = max(3, width - len(prefix))
            print(prefix + "─" * pad, flush=True)

        if self.register_artifacts and hasattr(trainer, "ctx"):
            ctx = trainer.ctx
            if ctx:
                from pathlib import Path as _PathLib

                checkpoint_path = _PathLib(filepath)
                if checkpoint_path.exists():
                    ctx.save_artifact(
                        name=f"checkpoint_{filename}",
                        src=checkpoint_path,
                    )
                    logger.info(f"Registered checkpoint as artifact: {checkpoint_path}")
