"""Early-stop hook — abort training on non-finite loss or parameters."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import torch
import torch.nn as nn

from molix.core.hook import BaseHook

logger = logging.getLogger("molix.hooks.early_stop")


class EarlyStop(BaseHook):
    """Abort training when a stop condition is met.

    Args:
        if_nan: Abort when ``state["train"]["loss"]`` or any model
            parameter becomes non-finite (NaN or ±inf). On detection,
            saves ``<out_dir>/nan_checkpoint.pt`` and raises
            ``RuntimeError`` so the Trainer unwinds cleanly. The
            caller can catch it and translate to a distinct exit code.

    Example::

        EarlyStop(if_nan=True)
    """

    def __init__(
        self,
        *,
        if_nan: bool = False,
        model: nn.Module | None = None,
        out_dir: str | Path | None = None,
    ) -> None:
        if if_nan and model is None:
            raise ValueError("EarlyStop(if_nan=True) requires `model=` to check parameters.")
        self._if_nan = if_nan
        self._model = model
        self._out_dir = Path(out_dir) if out_dir is not None else None

    def on_train_batch_end(self, trainer, state, batch, outputs) -> None:
        if not self._if_nan:
            return
        loss = state["train"].get("loss")
        if loss is not None and not math.isfinite(float(loss)):
            self._abort(state, reason=f"non-finite loss={loss}")
            return
        assert self._model is not None
        for name, p in self._model.named_parameters():
            if not torch.isfinite(p).all():
                self._abort(state, reason=f"non-finite parameter {name}")
                return

    def _abort(self, state, *, reason: str) -> None:
        step = int(state.get("global_step", 0))
        logger.error("NaN detected at step=%d — %s", step, reason)
        if self._out_dir is not None and self._model is not None:
            torch.save(self._model.state_dict(), self._out_dir / "nan_checkpoint.pt")
        raise RuntimeError(f"NaN detected — early stopping ({reason})")
