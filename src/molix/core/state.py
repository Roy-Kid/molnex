"""Training state and result types for MolNex.

Scalars produced during training live inside the ``train`` / ``eval`` /
``performance`` / ``gpu`` sub-dicts of :class:`TrainState`, never at the
top level with a slash-prefix key. Hooks read/write nested dicts
directly (``state["train"]["loss"] = x``); dotted-path access via a
resolver method is **not** provided — callers that need to address a
scalar from the outside use tuple paths (``("train", "loss")``), with
dotted strings reserved for display only.

See CLAUDE.md "State namespace contract" for the full rules.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class Stage(str, Enum):
    """Training stage enumeration."""

    TRAIN = "train"
    EVAL = "eval"
    TEST = "test"
    PREDICT = "predict"


#: Sub-dict namespaces that :class:`TrainState` pre-creates and enforces
#: ownership for.
NAMESPACES: tuple[str, ...] = ("train", "eval", "performance", "gpu")


class TrainState(dict):
    """Training state container with namespace sub-dicts.

    Top-level keys: ``epoch``, ``global_step``, ``stage``,
    ``steps_since_last_eval``, ``best_metric``, plus the four sub-dict
    namespaces ``train``, ``eval``, ``performance``, ``gpu``. Hooks
    write scalars into the appropriate sub-dict.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setdefault("epoch", 0)
        self.setdefault("global_step", 0)
        self.setdefault("stage", Stage.TRAIN)
        self.setdefault("steps_since_last_eval", 0)
        for ns in NAMESPACES:
            if not isinstance(super().get(ns), dict):
                super().__setitem__(ns, {})

    def __setitem__(self, key, value):
        """Reject slash-prefix keys and non-dict namespace replacements."""
        if isinstance(key, str) and "/" in key:
            ns, _, sub = key.partition("/")
            raise ValueError(
                f"TrainState does not accept slash-prefix keys at the top level. "
                f"Write state[{ns!r}][{sub!r}] = ... instead of state[{key!r}] = .... "
                f"See CLAUDE.md 'State namespace contract'."
            )
        if key in NAMESPACES and not isinstance(value, dict):
            raise ValueError(
                f"TrainState[{key!r}] must be a dict (sub-namespace), got {type(value).__name__}."
            )
        super().__setitem__(key, value)

    def increment_step(self) -> None:
        self["global_step"] = self["global_step"] + 1

    def increment_epoch(self) -> None:
        self["epoch"] = self["epoch"] + 1

    def set_stage(self, stage: Stage) -> None:
        self["stage"] = stage

    @property
    def epoch(self) -> int:
        return self["epoch"]

    @epoch.setter
    def epoch(self, value: int) -> None:
        self["epoch"] = value

    @property
    def global_step(self) -> int:
        return self["global_step"]

    @global_step.setter
    def global_step(self, value: int) -> None:
        self["global_step"] = value

    @property
    def stage(self) -> Stage:
        return self["stage"]

    @stage.setter
    def stage(self, value: Stage) -> None:
        self["stage"] = value

    @property
    def steps_since_last_eval(self) -> int:
        return self["steps_since_last_eval"]

    @steps_since_last_eval.setter
    def steps_since_last_eval(self, value: int) -> None:
        self["steps_since_last_eval"] = value

    @property
    def best_metric(self) -> float | None:
        return self.get("best_metric")

    @best_metric.setter
    def best_metric(self, value: float | None) -> None:
        self["best_metric"] = value


# ---------------------------------------------------------------------------
# Path helpers (tuple-path → nested lookup / display string)
# ---------------------------------------------------------------------------


#: A path into :class:`TrainState`. A bare ``str`` names a top-level key
#: (``"epoch"``); a ``tuple`` walks the namespace hierarchy
#: (``("train", "loss")``). Used by :class:`molix.core.hooks.Log`,
#: :class:`CheckpointHook`, and the LR scheduler metric lookup.
Path = str | tuple[str, ...]


def resolve(state: Mapping[str, Any], path: Path, default: Any = None) -> Any:
    """Look up ``path`` in ``state``; return ``default`` if any segment misses."""
    if isinstance(path, str):
        return state.get(path, default)
    node: Any = state
    for part in path:
        if not isinstance(node, Mapping) or part not in node:
            return default
        node = node[part]
    return node


def display(path: Path) -> str:
    """Render a path for user-facing display: ``("train","loss")`` → ``"train/loss"``."""
    if isinstance(path, str):
        return path
    return "/".join(path)


@dataclass
class StepResult:
    """Result from executing a training step."""

    loss: Any = None
    result: Any = None
    logs: Mapping[str, Any] = field(default_factory=dict)
