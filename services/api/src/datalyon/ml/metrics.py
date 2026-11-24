"""Metric computation and display formatting."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from .types import Arr

LABELS = {
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
    "macro_f1": "Macro F1",
    "rmse": "RMSE",
    "mae": "MAE",
    "mape": "MAPE",
    "wape": "WAPE",
    "r2": "R²",
    "f1": "F1",
    "precision": "Precision",
    "recall": "Recall",
    "accuracy": "Accuracy",
    "brier": "Brier score",
    "weighted_f1": "Weighted F1",
    "coverage_80": "80% interval coverage",
}
LOWER_IS_BETTER = {"rmse", "mae", "mape", "wape", "brier"}
PERCENT = {"mape", "wape", "f1", "precision", "recall", "accuracy", "macro_f1", "weighted_f1", "coverage_80"}


def fmt_num(v: float | None, digits: int = 3) -> str:
    if v is None or not math.isfinite(v):
        return "—"
    a = abs(v)
    if a >= 1e9:
        return f"{v / 1e9:.2f}B"
    if a >= 1e6:
        return f"{v / 1e6:.2f}M"
    if a >= 1e4:
        return f"{v / 1e3:.1f}K"
    if a >= 100:
        return f"{v:,.0f}"
    if a >= 1:
        return f"{v:.2f}".rstrip("0").rstrip(".")
    if a == 0:
        return "0"
    return f"{v:.{digits}g}"


def display(metric: str, v: float | None) -> str:
    if v is None or not math.isfinite(v):
        return "—"
    if metric in PERCENT:
        return f"{v * 100:.1f}%"
    if metric in ("rmse", "mae"):
        return fmt_num(v)
    return f"{v:.3f}"


def selection_score(metric: str, v: float) -> float:
    """Higher is always better."""
    return -v if metric in LOWER_IS_BETTER else v


def binary_metrics(y: Arr, p: Arr, threshold: float) -> dict[str, float]:
    yb = y.astype(int)
    pred = (p >= threshold).astype(int)
    two = len(np.unique(yb)) > 1
    return {
        "roc_auc": float(roc_auc_score(yb, p)) if two else float("nan"),
        "pr_auc": float(average_precision_score(yb, p)) if two else float("nan"),
        "f1": float(f1_score(yb, pred, zero_division=0)),
        "precision": float(precision_score(yb, pred, zero_division=0)),
        "recall": float(recall_score(yb, pred, zero_division=0)),
        "accuracy": float(accuracy_score(yb, pred)),
        "brier": float(brier_score_loss(yb, p)),
    }


def multiclass_metrics(y: Arr, proba: Arr) -> dict[str, float]:
    pred = proba.argmax(1)
    out = {
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y, proba, multi_class="ovr", labels=list(range(proba.shape[1]))))
    except ValueError:
        out["roc_auc"] = float("nan")
    return out


def regression_metrics(y: Arr, pred: Arr) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    out = {
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)) if len(y) > 1 else float("nan"),
    }
    if (y > 0).all():
        out["mape"] = float(np.mean(np.abs((y - pred) / y)))
    return out


def forecast_metrics(y: Arr, pred: Arr) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err = np.abs(y - pred)
    denom = float(np.abs(y).sum())
    nz = np.abs(y) > 1e-9
    return {
        "wape": float(err.sum() / denom) if denom > 0 else float("nan"),
        "mape": float(np.mean(err[nz] / np.abs(y[nz]))) if nz.any() else float("nan"),
        "mae": float(err.mean()) if len(err) else float("nan"),
        "rmse": float(np.sqrt(np.mean(err**2))) if len(err) else float("nan"),
    }


def score_task(metric: str, y: Arr, pred: Arr, pos_index: int = 1) -> float:
    """Primary metric (natural units) for CV folds. pred: probabilities for classification, values otherwise."""
    if metric == "roc_auc":
        if pred.ndim == 2 and pred.shape[1] > 2:
            return float(roc_auc_score(y, pred, multi_class="ovr", labels=list(range(pred.shape[1]))))
        yb = (y == pos_index).astype(int)
        return float(roc_auc_score(yb, pred[:, pos_index])) if len(np.unique(yb)) > 1 else float("nan")
    if metric == "pr_auc":
        return float(average_precision_score((y == pos_index).astype(int), pred[:, pos_index]))
    if metric == "macro_f1":
        return float(f1_score(y, pred.argmax(1), average="macro", zero_division=0))
    if metric in ("rmse", "mae", "r2", "mape"):
        return regression_metrics(y, pred).get(metric, float("nan"))
    if metric == "wape":
        return forecast_metrics(y, pred)["wape"]
    raise ValueError(f"unknown metric {metric}")


def metric_table(model: dict[str, float], baseline: dict[str, float], order: list[str]) -> list[dict[str, Any]]:
    rows = []
    for m in order:
        v = model.get(m)
        if v is None or not math.isfinite(v):
            continue
        b = baseline.get(m)
        good = None
        if b is not None and math.isfinite(b):
            good = v <= b if m in LOWER_IS_BETTER else v >= b
        rows.append(
            {
                "key": m,
                "name": LABELS.get(m, m),
                "value": v,
                "display": display(m, v),
                "baseline": display(m, b),
                "baseline_value": b,
                "good": good,
            }
        )
    return rows
