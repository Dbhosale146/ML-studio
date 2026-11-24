"""Shared types of the ML engine."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

ProblemType = Literal["binary", "multiclass", "regression", "forecasting"]
Arr = npt.NDArray[Any]

# Engine -> caller progress callback: (kind, payload). Kinds: "log", "fold", "trial", "run".
Progress = Callable[[str, dict[str, Any]], None]


def noop_progress(kind: str, payload: dict[str, Any]) -> None:
    return None


@dataclass
class Detection:
    problem_type: ProblemType
    target: str | None
    time_column: str | None = None
    series_column: str | None = None
    entity_column: str | None = None
    positive_class: str | None = None
    confidence: float = 0.5
    signals: list[str] = field(default_factory=list)
    alternatives: list[str] = field(default_factory=list)
    candidate_targets: list[str] = field(default_factory=list)
    leakage_suspects: list[dict[str, str]] = field(default_factory=list)
    clarification: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Strategy:
    metric: str  # roc_auc | pr_auc | macro_f1 | rmse | mae | mape | wape
    metric_label: str
    greater_is_better: bool
    secondary: list[str]
    validation: str
    validation_kind: Literal["stratified_kfold", "kfold", "time_series", "rolling_origin"]
    n_folds: int
    holdout_fraction: float
    class_weighting: bool
    time_budget_s: int
    reasons: list[str]
    extras: list[tuple[str, str]] = field(default_factory=list)
    horizon: int | None = None
    freq: str | None = None
    season: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CleaningStep:
    id: str
    title: str
    detail: str
    count: int
    optional: bool
    enabled: bool = True
    columns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainConfig:
    """Everything that defines how a pipeline is built. Changed by the cleaning approval and by critic fixes."""

    exclude: list[str] = field(default_factory=list)
    class_weight: bool = False
    log_target: bool = False
    regularise: bool = False
    calibrate: bool = False
    cap_outliers: bool = True
    missing_flags: bool = True
    normalise_labels: bool = True
    group_rare: bool = True
    derived: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class RunResult:
    run_id: str
    candidate: str
    family: str
    round: int
    params: dict[str, Any]
    config: dict[str, Any]
    scores: list[float]  # metric in natural units per fold / window
    mean: float
    std: float
    selection_score: float  # higher is better
    train_score: float | None
    fit_seconds: float
    tuned: bool = False
    oof: Arr | None = field(default=None, repr=False)

    def summary(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("oof", None)
        return d


@dataclass
class Finding:
    id: str
    check: str  # leakage | overfitting | stability | calibration | baseline | residuals | imbalance | data_size
    severity: Literal["high", "medium", "low", "info", "pass"]
    title: str
    detail: str
    evidence: dict[str, Any] = field(default_factory=dict)
    fix: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def as_float(x: Any) -> float:
    v = float(x)
    return v if np.isfinite(v) else float("nan")
