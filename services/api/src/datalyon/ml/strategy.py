"""Metric and validation selection from rules (the LLM only explains the choice)."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from .metrics import LABELS
from .types import ProblemType, Strategy

FREQ_INFO: dict[str, dict[str, Any]] = {
    "h": {"label": "Hourly", "unit": "hour", "days": 1 / 24, "season": 24, "default_h": 48, "options": [24, 48, 168]},
    "D": {"label": "Daily", "unit": "day", "days": 1, "season": 7, "default_h": 30, "options": [7, 14, 30]},
    "W": {"label": "Weekly", "unit": "week", "days": 7, "season": 52, "default_h": 13, "options": [4, 8, 13]},
    "M": {"label": "Monthly", "unit": "month", "days": 30.44, "season": 12, "default_h": 6, "options": [3, 6, 12]},
    "Q": {"label": "Quarterly", "unit": "quarter", "days": 91.3, "season": 4, "default_h": 4, "options": [2, 4, 8]},
    "Y": {"label": "Yearly", "unit": "year", "days": 365.25, "season": 1, "default_h": 2, "options": [1, 2, 3]},
}


def infer_frequency(dates: pd.Series) -> str | None:
    """Pandas period alias for a date column, or None when it cannot be inferred (irregular / too short)."""
    d = pd.to_datetime(pd.Series(dates), errors="coerce").dropna().drop_duplicates().sort_values()
    if len(d) < 3:
        return None
    diffs = d.diff().dropna().dt.total_seconds() / 86400
    med = float(np.median(diffs))
    if med <= 0:
        return None
    for code, lo, hi in (("h", 0.03, 0.08), ("D", 0.8, 1.6), ("W", 5.5, 8.5), ("M", 27, 32), ("Q", 85, 95), ("Y", 350, 380)):
        if lo <= med <= hi:
            return code
    return None


def horizon_from_goal(goal: str, freq: str) -> int | None:
    unit_days = {"hour": 1 / 24, "day": 1, "week": 7, "month": 30.44, "quarter": 91.3, "year": 365.25}
    m = re.search(r"next\s+(\d+)\s*(hour|day|week|month|quarter|year)s?", goal, re.I)
    if m:
        n, unit = int(m.group(1)), m.group(2).lower()
    else:
        m2 = re.search(r"next\s+(hour|day|week|month|quarter|year)\b", goal, re.I)
        if not m2:
            return None
        n, unit = 1, m2.group(1).lower()
    return max(1, round(n * unit_days[unit] / FREQ_INFO[freq]["days"]))


def folds_for(rows: int) -> int:
    return 5 if rows <= 50_000 else 3


def choose_strategy(
    problem_type: ProblemType,
    df: pd.DataFrame,
    target: str,
    time_column: str | None,
    budget_s: int,
    positive_class: str | None = None,
    horizon: int | None = None,
    freq: str | None = None,
) -> Strategy:
    rows = len(df)
    reasons: list[str] = []
    extras: list[tuple[str, str]] = []
    n_folds = folds_for(rows)

    if problem_type == "forecasting":
        assert freq is not None and horizon is not None
        y = pd.to_numeric(df[target], errors="coerce")
        metric = "wape" if (y <= 0).mean() > 0.001 else "mape"
        info = FREQ_INFO[freq]
        reasons.append(
            "Planners think in percentage error. "
            + (
                "WAPE is used because the series contains zeros, where MAPE is undefined."
                if metric == "wape"
                else "MAPE is the headline; MAE is reported in units."
            )
        )
        reasons.append(
            "A rolling-origin backtest trains on the past and forecasts the next horizon at several points in time, just like real use."
        )
        extras += [
            ("Frequency", info["label"]),
            ("Horizon", f"{horizon} {info['unit']}{'s' if horizon > 1 else ''}"),
            ("Baseline", "Seasonal naive"),
        ]
        return Strategy(
            metric=metric,
            metric_label=LABELS[metric],
            greater_is_better=False,
            secondary=["mae", "rmse", "wape" if metric == "mape" else "mape"],
            validation=f"Rolling-origin backtest: up to 4 windows of {horizon} {info['unit']}s",
            validation_kind="rolling_origin",
            n_folds=4,
            holdout_fraction=0.0,
            class_weighting=False,
            time_budget_s=budget_s,
            reasons=reasons,
            extras=extras,
            horizon=horizon,
            freq=freq,
            season=int(info["season"]),
        )

    time_split = time_column is not None
    if problem_type == "binary":
        vc = df[target].dropna().astype(str).value_counts()
        pos_share = float(vc.get(positive_class, vc.min()) / vc.sum()) if len(vc) else 0.5
        metric = "pr_auc" if pos_share < 0.10 else "roc_auc"
        if metric == "pr_auc":
            reasons.append(
                f"Only {pos_share * 100:.1f}% of rows are positive. ROC-AUC looks flattering at that rate, so PR-AUC is the primary metric."
            )
        else:
            reasons.append("ROC-AUC measures how well the model ranks positives above negatives, independent of any threshold.")
        reasons.append("F1, precision, recall and the Brier score are reported too.")
        weighting = pos_share < 0.2
        extras += [
            ("Positive class", f"{positive_class} ({pos_share * 100:.1f}%)"),
            ("Class weights", "yes" if weighting else "not needed"),
        ]
        secondary = ["roc_auc" if metric == "pr_auc" else "pr_auc", "f1", "precision", "recall", "brier"]
        greater = True
    elif problem_type == "multiclass":
        vc = df[target].dropna().astype(str).value_counts()
        metric = "macro_f1"
        reasons.append(
            f"With {len(vc)} classes, macro F1 weighs every class equally, so small classes are not ignored. Accuracy and a per-class report are shown too."
        )
        weighting = bool(vc.min() / vc.sum() < 0.05)
        extras += [("Classes", str(len(vc))), ("Smallest class", f"{vc.index[-1]} ({vc.iloc[-1] / vc.sum() * 100:.1f}%)")]
        secondary = ["accuracy", "weighted_f1"]
        greater = True
    else:
        y = pd.to_numeric(df[target], errors="coerce").dropna()
        skew = float(y.skew()) if len(y) > 2 else 0.0
        metric = "mae" if abs(skew) > 2 else "rmse"
        reasons.append(
            f"The target is heavily skewed (skew {skew:.1f}), so MAE is the primary metric: it is not dominated by a few huge values."
            if metric == "mae"
            else "RMSE penalises big misses, which are usually the costly ones."
        )
        positive = bool((y > 0).all())
        reasons.append("R² is reported too" + (", and MAPE because the target is always positive." if positive else "."))
        weighting = False
        extras += [("Target range", f"{y.min():,.4g} to {y.max():,.4g}"), ("Skew", f"{skew:.2f}")]
        secondary = [m for m in ("mae" if metric == "rmse" else "rmse", "r2", "mape" if positive else "") if m]
        greater = False

    if time_split:
        validation = f"Time-ordered CV on the oldest 80% (by `{time_column}`) + newest 20% as holdout"
        kind = "time_series"
        reasons.append(
            f"`{time_column}` orders the rows in time, so the model is always tested on rows newer than the ones it learned from."
        )
    else:
        validation = f"{'Stratified ' if problem_type != 'regression' else ''}{n_folds}-fold CV + 20% {'stratified ' if problem_type != 'regression' else ''}holdout"
        kind = "stratified_kfold" if problem_type != "regression" else "kfold"
        reasons.append("Cross-validation compares models; the holdout is kept aside for the final, honest score.")
    if rows < 300:
        extras.append(("Data size", f"Only {rows} rows: scores will be noisy"))
    return Strategy(
        metric=metric,
        metric_label=LABELS[metric],
        greater_is_better=greater,
        secondary=secondary,
        validation=validation,
        validation_kind=kind,  # type: ignore[arg-type]
        n_folds=n_folds,
        holdout_fraction=0.2,
        class_weighting=weighting,
        time_budget_s=budget_s,
        reasons=reasons,
        extras=extras,
    )
