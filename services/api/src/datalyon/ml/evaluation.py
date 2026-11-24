"""Final selection on the holdout, metrics vs baseline, and chart data computed from real predictions."""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve, precision_recall_fscore_support, roc_curve

from .candidates import Candidate
from .critic import calibration_curve
from .explain import permutation_shares
from .metrics import binary_metrics, metric_table, multiclass_metrics, regression_metrics, score_task, selection_score
from .training import TabularTask, TrainingError, build, fit, predict, sample_weights
from .types import Arr, RunResult, TrainConfig


@dataclass
class FinalResult:
    candidate_key: str
    candidate: str
    params: dict[str, Any]
    config: TrainConfig
    threshold: float
    model: Any  # fitted on the training split
    holdout: dict[str, float]
    baseline: dict[str, float]
    metrics: list[dict[str, Any]]
    charts: dict[str, Any]
    ranking: list[dict[str, Any]]
    importance: list[list[Any]]
    cv_mean: float
    cv_std: float
    fit_seconds: float
    runner_up: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)


def tune_threshold(p: Arr, yb: Arr) -> float:
    """Threshold maximising F1 on out-of-fold probabilities."""
    ok = np.isfinite(p)
    p, yb = p[ok], yb[ok]
    if len(np.unique(yb)) < 2:
        return 0.5
    grid = np.unique(np.quantile(p, np.linspace(0.5, 0.995, 120)))
    best_t, best_f = 0.5, -1.0
    for t in grid:
        f = f1_score(yb, (p >= t).astype(int), zero_division=0)
        if f > best_f:
            best_t, best_f = float(t), float(f)
    return round(best_t, 4)


def fit_final(task: TabularTask, cand: Candidate, params: dict[str, Any], cfg: TrainConfig, X: pd.DataFrame, y: Arr) -> Any:
    pipe = build(task, cand, params, cfg)
    if task.is_clf and cfg.calibrate and not cand.baseline:
        method = "isotonic" if len(y) >= 2000 else "sigmoid"
        model = CalibratedClassifierCV(pipe, method=method, cv=3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X, y)
        return model
    return fit(pipe, X, y, sample_weights(y, task, cfg))


def task_metrics(task: TabularTask, pred: Arr, y: Arr, threshold: float) -> dict[str, float]:
    if task.problem_type == "binary":
        return binary_metrics((y == task.pos_index).astype(int), pred[:, task.pos_index], threshold)
    if task.problem_type == "multiclass":
        return multiclass_metrics(y, pred)
    return regression_metrics(y, pred)


def _curve(xs: Arr, ys: Arr, n: int = 60) -> tuple[Arr, Arr]:
    if len(xs) <= n:
        return xs, ys
    idx = np.unique(np.linspace(0, len(xs) - 1, n).astype(int))
    return xs[idx], ys[idx]


def charts(task: TabularTask, pred: Arr, y: Arr, threshold: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if task.problem_type == "binary":
        p, yb = pred[:, task.pos_index], (y == task.pos_index).astype(int)
        fpr, tpr, _ = roc_curve(yb, p)
        fx, fy = _curve(fpr, tpr)
        out["roc"] = [{"fpr": round(float(a), 4), "tpr": round(float(b), 4)} for a, b in zip(fx, fy, strict=True)]
        prec, rec, _ = precision_recall_curve(yb, p)
        order = np.argsort(rec)
        rx, ry = _curve(rec[order], prec[order])
        out["pr"] = [{"recall": round(float(a), 4), "precision": round(float(b), 4)} for a, b in zip(rx, ry, strict=True)]
        out["prevalence"] = float(yb.mean())
        cm = confusion_matrix(yb, (p >= threshold).astype(int), labels=[0, 1])
        out["confusion"] = {"labels": task.classes, "matrix": cm.tolist(), "threshold": threshold}
        _, curve = calibration_curve(yb, p)
        out["calibration"] = curve
        bins = np.linspace(0, 1, 21)
        out["scores"] = [
            {
                "bin": f"{bins[i]:.2f}",
                "negative": int(((p >= bins[i]) & (p < bins[i + 1] + (1e-9 if i == 19 else 0)) & (yb == 0)).sum()),
                "positive": int(((p >= bins[i]) & (p < bins[i + 1] + (1e-9 if i == 19 else 0)) & (yb == 1)).sum()),
            }
            for i in range(20)
        ]
    elif task.problem_type == "multiclass":
        yp = pred.argmax(1)
        top = list(np.argsort(-np.bincount(y.astype(int), minlength=task.n_classes))[:12])
        cm = confusion_matrix(y, yp, labels=top)
        out["confusion"] = {"labels": [task.classes[i] for i in top], "matrix": cm.tolist(), "threshold": None}
        pr, rc, f1, sup = precision_recall_fscore_support(y, yp, labels=list(range(task.n_classes)), zero_division=0)
        out["per_class"] = [
            {"class": task.classes[i], "precision": float(pr[i]), "recall": float(rc[i]), "f1": float(f1[i]), "support": int(sup[i])}
            for i in range(task.n_classes)
        ]
    else:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(y), min(len(y), 400), replace=False)
        out["scatter"] = [{"actual": float(y[i]), "predicted": float(pred[i])} for i in idx]
        res = y - pred
        counts, edges = np.histogram(res, bins=25)
        out["residuals"] = [{"x0": float(edges[i]), "x1": float(edges[i + 1]), "count": int(counts[i])} for i in range(len(counts))]
        out["residual_quantiles"] = {"q10": float(np.quantile(res, 0.1)), "q90": float(np.quantile(res, 0.9))}
    return out


def choose_final(
    task: TabularTask, runs: list[RunResult], candidates: dict[str, Candidate], cfg: TrainConfig, top_k: int = 3
) -> FinalResult:
    metric = task.strategy.metric
    by_name = {c.name: c for c in candidates.values()}
    models = sorted((r for r in runs if r.family != "Baseline" and np.isfinite(r.mean)), key=lambda r: -r.selection_score)
    # best run per candidate
    seen: set[str] = set()
    shortlist = []
    for r in models:
        if r.candidate not in seen:
            seen.add(r.candidate)
            shortlist.append(r)
    shortlist = shortlist[:top_k]
    if not shortlist:
        raise TrainingError("No candidate model finished training, so there is nothing to evaluate.")
    baseline_cand = next(c for c in candidates.values() if c.baseline)

    base_cfg = TrainConfig.from_dict({**cfg.to_dict(), "class_weight": False, "calibrate": False})  # the baseline is never re-weighted
    base_model = fit_final(task, baseline_cand, {}, base_cfg, task.X_train, task.y_train)
    base_pred = predict(base_model, task.X_hold, task)
    base_metrics = task_metrics(task, base_pred, task.y_hold, 0.5)

    evaluated = []
    for r in shortlist:
        cand = by_name[r.candidate]
        run_cfg = TrainConfig.from_dict(r.config)
        threshold = 0.5
        if task.problem_type == "binary" and run_cfg.class_weight and r.oof is not None:
            threshold = tune_threshold(r.oof[:, task.pos_index], (task.y_search == task.pos_index).astype(int))
        t0 = time.perf_counter()
        model = fit_final(task, cand, r.params, run_cfg, task.X_train, task.y_train)
        secs = time.perf_counter() - t0
        if task.problem_type == "binary" and run_cfg.calibrate and run_cfg.class_weight:
            threshold = tune_threshold(predict(model, task.X_train, task)[:, task.pos_index], (task.y_train == task.pos_index).astype(int))
        pred = predict(model, task.X_hold, task)
        hold = task_metrics(task, pred, task.y_hold, threshold)
        evaluated.append(
            {
                "run": r,
                "cand": cand,
                "cfg": run_cfg,
                "model": model,
                "pred": pred,
                "metrics": hold,
                "threshold": threshold,
                "secs": secs,
                "score": selection_score(metric, hold.get(metric, float("nan"))),
            }
        )

    best_score = max(e["score"] for e in evaluated)
    tol = 0.005 * max(abs(best_score), 1e-9)
    tied = [e for e in evaluated if e["score"] >= best_score - tol]
    tied.sort(key=lambda e: (e["run"].std, e["secs"]))  # stability, then speed
    best = tied[0]
    notes = []
    if len(tied) > 1:
        notes.append(f"{len(tied)} models were within 0.5% on the holdout; {best['cand'].name} won on stability and speed.")
    runner = next((e for e in sorted(evaluated, key=lambda e: -e["score"]) if e is not best), None)

    def score_fn(Xp: pd.DataFrame, yp: Arr) -> float:
        return selection_score(metric, score_task(metric, yp, predict(best["model"], Xp, task), task.pos_index))

    importance = permutation_shares(
        score_fn, task.X_hold[[c for c in task.columns if c not in best["cfg"].exclude]], task.y_hold, n_repeats=4, max_rows=2000
    )
    order = [metric, *task.strategy.secondary]
    order += [m for m in best["metrics"] if m not in order]
    ranking = [
        {
            "candidate": e["cand"].name,
            "cv": e["run"].mean,
            "cv_std": e["run"].std,
            "holdout": e["metrics"].get(metric),
            "fit_seconds": round(e["secs"], 2),
            "selected": e is best,
        }
        for e in evaluated
    ]
    ranking.append(
        {
            "candidate": baseline_cand.name,
            "cv": None,
            "cv_std": None,
            "holdout": base_metrics.get(metric),
            "fit_seconds": None,
            "selected": False,
        }
    )
    return FinalResult(
        candidate_key=best["cand"].key,
        candidate=best["cand"].name,
        params=best["run"].params,
        config=best["cfg"],
        threshold=best["threshold"],
        model=best["model"],
        holdout=best["metrics"],
        baseline=base_metrics,
        metrics=metric_table(best["metrics"], base_metrics, order),
        charts=charts(task, best["pred"], task.y_hold, best["threshold"]),
        ranking=ranking,
        importance=importance,
        cv_mean=best["run"].mean,
        cv_std=best["run"].std,
        fit_seconds=round(best["secs"], 2),
        runner_up=(
            {"candidate": runner["cand"].name, "holdout": runner["metrics"].get(metric), "key": runner["cand"].key} if runner else None
        ),
        notes=notes,
    )
