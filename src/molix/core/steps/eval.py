"""Default evaluation step implementation."""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

import torch

from molix.config import config

if TYPE_CHECKING:
    from molix.core.state import TrainState
    from molix.core.trainer import Trainer


class DefaultEvalStep:
    """Default evaluation step with optional AMP support.

    Precision is controlled globally via :func:`molix.config.set_precision`.
    When ``config["use_amp"]`` is true the forward pass runs under
    :func:`torch.amp.autocast` with ``config["amp_dtype"]``. No
    ``GradScaler`` is used during evaluation.

    Args:
        no_grad: Wrap forward in ``torch.no_grad()`` (default ``True``).
            Set ``False`` for models that derive forces via
            ``torch.autograd.grad`` (e.g. Sonata), which needs an active
            autograd graph.
    """

    def __init__(self, *, no_grad: bool = True) -> None:
        self._no_grad = no_grad

    def on_train_batch(self, trainer: "Trainer", state: "TrainState", batch: Any) -> dict[str, Any]:
        raise NotImplementedError(
            "DefaultEvalStep.on_train_batch() is not implemented. "
            "Use DefaultTrainStep for training batches."
        )

    def on_eval_batch(self, trainer: "Trainer", state: "TrainState", batch: Any) -> dict[str, Any]:
        assert trainer.model is not None
        assert trainer.loss_fn is not None

        device_type = next(trainer.model.parameters()).device.type
        amp_enabled = bool(config["use_amp"])
        amp_dtype = config["amp_dtype"]

        grad_ctx = torch.no_grad() if self._no_grad else torch.enable_grad()
        amp_ctx = torch.amp.autocast(device_type, dtype=amp_dtype) if amp_enabled else nullcontext()
        with grad_ctx, amp_ctx:
            predictions = trainer.model(batch)
            loss = trainer.loss_fn(predictions, batch)

        state["eval"]["loss"] = loss.item()
        return {"loss": loss, "predictions": predictions}
