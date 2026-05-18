"""Step implementations for Molix training.

This module provides the Step protocol and default implementations for
training and evaluation computation.
"""

from __future__ import annotations

from typing import Any

import torch

from molix.core.steps.base import Step
from molix.core.steps.eval import DefaultEvalStep
from molix.core.steps.train import DefaultTrainStep


def batch_to_device(
    batch: Any,
    device: torch.device | str,
) -> Any:
    """Move a batch to the target device.

    Handles TensorDict/GraphBatch (via ``.to()``), plain dicts (recursive),
    and bare tensors. Non-tensor leaves are returned unchanged.

    Args:
        batch: Batch data — TensorDict, dict, Tensor, or other.
        device: Target device.

    Returns:
        Batch with all tensors on ``device``.
    """
    if hasattr(batch, "to"):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: batch_to_device(v, device) for k, v in batch.items()}
    return batch


__all__ = [
    "Step",
    "DefaultTrainStep",
    "DefaultEvalStep",
    "batch_to_device",
]
