"""Molecular-ML loss presets built on top of the generic :mod:`losses`.

The generic :class:`MSELoss` / :class:`MAELoss` are schema-agnostic: they
extract values by plain-string key. Training scripts then wrap them in
a per-script closure that knows *where* to read targets from the
:class:`GraphBatch` TensorDict — leading to three copies of

.. code-block:: python

    def loss_fn(preds, batch):
        return mse(preds["energy"], batch["graphs", target_key])

These presets absorb that wrapping. They return ``Callable[[preds, batch], Tensor]``
that consumes ``preds`` (plain dict from the model forward) and ``batch``
(a :class:`GraphBatch`), following the convention encoded in
:class:`~molix.data.collate.TargetSchema`:

* graph-level targets (e.g. ``energy``) live at ``batch["graphs", key]``;
* atom-level targets (e.g. ``forces``) live at ``batch["atoms", key]``.

The factories do **not** auto-discover the schema — the caller passes the
exact target key name, so the resulting callable is a pure function with
static shape/key access (keeps ``torch.compile`` graphs stable).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch
import torch.nn as nn

__all__ = ["energy_mse", "energy_force_mse"]


def energy_mse(
    target_key: str = "energy",
    *,
    pred_key: str = "energy",
    reduction: str = "mean",
) -> Callable[[Mapping[str, Any], Any], torch.Tensor]:
    """MSE between predicted energy and a graph-level target.

    Args:
        target_key: Graph-level target name stored at
            ``batch["graphs", target_key]`` by the collator.
        pred_key: Key under which the model forward writes the energy
            prediction in its output dict.
        reduction: ``'mean'``, ``'sum'``, or ``'none'`` — forwarded to
            :class:`torch.nn.MSELoss`.

    Returns:
        ``loss_fn(preds, batch) -> Tensor`` suitable for the Trainer's
        ``step.loss_fn`` slot.
    """
    mse = nn.MSELoss(reduction=reduction)

    def _fn(preds: Mapping[str, Any], batch: Any) -> torch.Tensor:
        e_pred = preds[pred_key]
        e_true = batch["graphs", target_key].view_as(e_pred)
        return mse(e_pred, e_true)

    return _fn


def energy_force_mse(
    *,
    energy_target_key: str = "energy",
    force_target_key: str = "forces",
    energy_pred_key: str = "energy",
    force_pred_key: str = "forces",
    lambda_F: float = 1.0,
    reduction: str = "mean",
) -> Callable[[Mapping[str, Any], Any], torch.Tensor]:
    """Energy MSE + ``lambda_F`` × forces MSE.

    The energy target lives at ``batch["graphs", energy_target_key]``
    (graph-level scalar). The forces target lives at
    ``batch["atoms", force_target_key]`` (atom-level vector, shape
    ``(N_total, 3)``).

    Args:
        energy_target_key / force_target_key: GraphBatch keys.
        energy_pred_key / force_pred_key: keys on the model-forward output dict.
        lambda_F: Weight on the force-MSE term.
        reduction: Forwarded to both :class:`torch.nn.MSELoss` instances.

    Returns:
        ``loss_fn(preds, batch) -> Tensor``.
    """
    energy_fn = energy_mse(
        energy_target_key,
        pred_key=energy_pred_key,
        reduction=reduction,
    )
    mse_f = nn.MSELoss(reduction=reduction)

    def _fn(preds: Mapping[str, Any], batch: Any) -> torch.Tensor:
        f_pred = preds[force_pred_key]
        f_true = batch["atoms", force_target_key].view_as(f_pred)
        return energy_fn(preds, batch) + lambda_F * mse_f(f_pred, f_true)

    return _fn
