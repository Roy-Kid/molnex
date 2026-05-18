"""Default training step implementation."""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

import torch

from molix.config import config

if TYPE_CHECKING:
    from molix.core.state import TrainState
    from molix.core.trainer import Trainer


class DefaultTrainStep:
    """Default training step with optional AMP support.

    Precision is controlled globally via :meth:`molix.config.MolnexConfig.set_precision`
    which writes ``use_amp`` and ``amp_dtype`` into the global
    :data:`molix.config.config`. When ``use_amp`` is true the forward pass
    runs under :func:`torch.amp.autocast` and backward uses the
    :class:`torch.amp.GradScaler` owned by the :class:`Trainer`.

    Flow:
        1. Forward pass (under autocast if AMP enabled)
        2. Loss computation
        3. Backward pass (with ``trainer.scaler`` if AMP enabled)
        4. ``on_after_backward`` hook point (gradients are unscaled)
        5. Optimizer step
    """

    def on_train_batch(self, trainer: "Trainer", state: "TrainState", batch: Any) -> dict[str, Any]:
        assert trainer.model is not None
        assert trainer.loss_fn is not None
        assert trainer.optimizer is not None

        device_type = next(trainer.model.parameters()).device.type
        amp_enabled = bool(config["use_amp"])
        amp_dtype = config["amp_dtype"]

        ctx = torch.amp.autocast(device_type, dtype=amp_dtype) if amp_enabled else nullcontext()
        with ctx:
            predictions = trainer.model(batch)
            loss = trainer.loss_fn(predictions, batch)

        trainer.optimizer.zero_grad()
        scaler = trainer.scaler if amp_enabled else None
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(trainer.optimizer)
        else:
            loss.backward()

        trainer._call_hooks("on_after_backward", trainer, state)

        if scaler is not None:
            scaler.step(trainer.optimizer)
            scaler.update()
        else:
            trainer.optimizer.step()

        state["train"]["loss"] = loss.item()
        return {"loss": loss, "predictions": predictions}

    def on_eval_batch(self, trainer: "Trainer", state: "TrainState", batch: Any) -> dict[str, Any]:
        raise NotImplementedError(
            "DefaultTrainStep.on_eval_batch() is not implemented. "
            "Use DefaultEvalStep for evaluation batches."
        )
