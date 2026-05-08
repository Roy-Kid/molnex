"""Append-only persistence sink for the training-event journal.

:class:`JournalHook` is the producer side of
:class:`molix.io.JournalWriter`: it mirrors the
:class:`~molix.core.state.TrainState` namespaces (``train/``,
``performance/``, ``gpu/`` on the train phase; ``eval/`` on the eval
phase) into the writer at a configured cadence, and emits optional
``hparams`` / weight + gradient histogram records at the
appropriate lifecycle moments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from molix.core.hook import BaseHook
from molix.hooks._utils import _as_scalar

if TYPE_CHECKING:
    from molix.core.state import TrainState
    from molix.core.trainer import Trainer
    from molix.io import JournalWriter


class JournalHook(BaseHook):
    """Append-only persistence sink for the training-event journal.

    Mirrors the same :class:`~molix.core.state.TrainState` namespaces as
    :class:`molix.hooks.tensorboard.TensorBoardHook` (``train/``,
    ``performance/``, ``gpu/`` on the train phase; ``eval/`` on the eval
    phase) but writes records to a :class:`~molix.io.JournalWriter`
    backend instead of TensorBoard event files.

    Args:
        every_n_steps: Mirror frequency for ``train``/``performance``/
            ``gpu`` namespace scalars during the train phase. Must be
            positive.
        store: A :class:`~molix.io.JournalWriter` (kept named
            ``store`` for source-compat with the legacy ``Journal``
            constructor; the writer is the only first-party concrete
            backend).
        log_hparams: When ``True`` and ``hparams`` is provided, emit
            one ``type="json", key="hparams", step=0`` record at
            ``on_train_start``.
        log_histograms: When ``True``, emit ``type="histogram"``
            records for ``Weights/<name>`` and ``Gradients/<name>`` on
            every ``histogram_freq``-th epoch end. Bins are computed
            via :func:`numpy.histogram` (per-record).
        hparams: Hyperparameter mapping; required when
            ``log_hparams=True``.
        histogram_freq: Histogram emission cadence in epochs. Default 1.
        histogram_bins: Number of histogram bins. Default 64.
    """

    TRAIN_NAMESPACES: tuple[str, ...] = ("train", "performance", "gpu")
    EVAL_NAMESPACES: tuple[str, ...] = ("eval",)

    def __init__(
        self,
        every_n_steps: int,
        store: "JournalWriter",
        *,
        log_hparams: bool = False,
        log_histograms: bool = False,
        hparams: dict[str, Any] | None = None,
        histogram_freq: int = 1,
        histogram_bins: int = 64,
    ) -> None:
        if every_n_steps <= 0:
            raise ValueError(f"every_n_steps must be positive, got {every_n_steps}")
        if histogram_freq <= 0:
            raise ValueError(f"histogram_freq must be positive, got {histogram_freq}")

        self._store = store
        self._every_n_steps = every_n_steps
        self._log_hparams = log_hparams
        self._log_histograms = log_histograms
        self._hparams = hparams or {}
        self._histogram_freq = histogram_freq
        self._histogram_bins = histogram_bins

    @staticmethod
    def _now_ns() -> int:
        import time

        return time.time_ns()

    def on_train_start(self, trainer: "Trainer | None", state: "TrainState") -> None:
        """Emit the ``hparams`` JSON record if enabled."""
        if self._log_hparams and self._hparams:
            self._store.append(
                type="json",
                key="hparams",
                step=0,
                wall_time_ns=self._now_ns(),
                value=dict(self._hparams),
            )

    def on_train_batch_end(
        self,
        trainer: "Trainer | None",
        state: "TrainState",
        batch: Any,
        outputs: Any,
    ) -> None:
        """Mirror ``train/`` ``performance/`` ``gpu/`` scalars at the configured cadence."""
        global_step = int(state.get("global_step", 0))
        if global_step % self._every_n_steps != 0:
            return
        self._mirror_namespaces(state, self.TRAIN_NAMESPACES, global_step)

    def on_eval_step_complete(self, trainer: "Trainer | None", state: "TrainState") -> None:
        """Mirror ``eval/`` scalars whenever an eval phase completes."""
        global_step = int(state.get("global_step", 0))
        self._mirror_namespaces(state, self.EVAL_NAMESPACES, global_step)

    def on_epoch_end(self, trainer: "Trainer | None", state: "TrainState") -> None:
        """Emit weight / gradient histogram records when enabled."""
        if not self._log_histograms or trainer is None:
            return
        epoch = int(state.get("epoch", 0))
        if (epoch + 1) % self._histogram_freq != 0:
            return
        self._emit_histograms(trainer, state)

    def on_train_end(self, trainer: "Trainer | None", state: "TrainState") -> None:
        """Close the underlying writer."""
        self._store.close()

    def _mirror_namespaces(
        self, state: "TrainState", namespaces: tuple[str, ...], global_step: int
    ) -> None:
        wall = self._now_ns()
        for ns in namespaces:
            sub = state[ns]
            if not isinstance(sub, dict):
                continue
            for k, value in sub.items():
                scalar = _as_scalar(value)
                if scalar is None:
                    continue
                self._store.append(
                    type="scalar",
                    key=f"{ns}/{k}",
                    step=global_step,
                    wall_time_ns=wall,
                    value=float(scalar),
                )

    def _emit_histograms(self, trainer: "Trainer", state: "TrainState") -> None:
        import numpy as np

        epoch = int(state.get("epoch", 0))
        wall = self._now_ns()
        model = getattr(trainer, "model", None)
        if model is None:
            return
        for name, param in model.named_parameters():
            data = param.detach().cpu().numpy().ravel()
            counts, bins = np.histogram(data, bins=self._histogram_bins)
            self._store.append(
                type="histogram",
                key=f"Weights/{name}",
                step=epoch,
                wall_time_ns=wall,
                value={"bins": bins.tolist(), "counts": counts.tolist()},
            )
            if param.grad is not None:
                gdata = param.grad.detach().cpu().numpy().ravel()
                gcounts, gbins = np.histogram(gdata, bins=self._histogram_bins)
                self._store.append(
                    type="histogram",
                    key=f"Gradients/{name}",
                    step=epoch,
                    wall_time_ns=wall,
                    value={"bins": gbins.tolist(), "counts": gcounts.tolist()},
                )
