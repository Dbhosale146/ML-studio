"""Critic checks on round-1 results. The code produces the findings; the LLM only narrates them."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from .candidates import Candidate
from .explain import permutation_shares
from .metrics import LABELS, LOWER_IS_BETTER, display, score_task, selection_score
from .training import TabularTask, build, fit, predict, sample_weights
from .types import Arr, Finding, RunResult, TrainConfig

AUC_LEAK = 0.97
CORR_LEAK = 0.95


def _auc(yb: Arr, s: Any) -> float:
    try:
        a = float(roc_auc_score(yb, s))
        return max(a, 1 - a)
    except ValueError:
        return 0.5


def _oof_target_mean(s: pd.Series, y: Arr, seed: int = 0) -> Arr:
    """Out-of-fold target-mean encoding so high-cardinality columns cannot fake a perfect fit."""
    fold = np.random.default_rng(seed).integers(0, 2, len(s))
    out = np.full(len(s), np.nan)
    keys = s.astype(str).fillna("__nan__").to_numpy()
    prior = float(np.mean(y))
    for f in (0, 1):
        tr = fold != f
        means = pd.Series(y[tr]).groupby(keys[tr]).mean()
        out[~tr] = pd.Series(keys[~tr]).map(means).fillna(prior).to_numpy()
    return out


def univariate_leakage(task: TabularTask, columns: list[str]) -> list[dict[str, Any]]:
    """Columns that predict the target almost perfectly on their own."""
    X, y = task.X_search, task.y_search
    rng = np.random.default_rng(1)
    idx = rng.choice(len(y), size=min(len(y), 20_000), replace=False)
    yy = y[idx]
    binary = task.problem_type == "binary"
    if task.problem_type == "multiclass":
        return []
    yb = (yy == task.pos_index).astype(int) if binary else None
    out: dict[str, dict[str, Any]] = {}
    for c in columns:
        s = X[c].iloc[idx]
        k = task.kinds.get(c)
        if k in ("number", "integer", "boolean"):
            s_num = pd.to_numeric(s, errors="coerce")
            miss = s_num.isna()
        else:
            s_num = None
            miss = s.isna() | (s.astype(str).str.strip() == "")
        if binary and 0.05 <= miss.mean() <= 0.95:
            a = _auc(yb, miss.astype(int).to_numpy())  # type: ignore[arg-type]
            if a >= 0.95:
                out[c] = {
                    "column": c,
                    "score": a,
                    "evidence": f"it is empty for {miss.mean() * 100:.0f}% of rows, and whether it is empty predicts `{task.target}` with ROC-AUC {a:.2f}",
                }
                continue
        if s_num is not None:
            if s_num.notna().sum() < 10:
                continue
            v = s_num.fillna(s_num.median()).to_numpy()
            if binary:
                a = _auc(yb, v)  # type: ignore[arg-type]
                if a > AUC_LEAK:
                    out[c] = {"column": c, "score": a, "evidence": f"on its own it separates `{task.target}` with ROC-AUC {a:.3f}"}
            else:
                rho = abs(float(spearmanr(v, yy).statistic or 0.0))
                if rho > CORR_LEAK:
                    out[c] = {
                        "column": c,
                        "score": rho,
                        "evidence": f"it moves in lockstep with `{task.target}` (rank correlation {rho:.3f})",
                    }
        elif k == "category":
            enc = _oof_target_mean(s, yb if binary else yy.astype(float))  # type: ignore[arg-type]
            if binary:
                a = _auc(yb, enc)  # type: ignore[arg-type]
                if a > AUC_LEAK:
                    out[c] = {"column": c, "score": a, "evidence": f"its categories alone separate `{task.target}` with ROC-AUC {a:.3f}"}
            elif np.std(enc) > 0:
                rho = abs(float(np.corrcoef(enc, yy.astype(float))[0, 1]))
                if rho > CORR_LEAK:
                    out[c] = {
                        "column": c,
                        "score": rho,
                        "evidence": f"its categories alone explain `{task.target}` almost exactly (correlation {rho:.3f})",
                    }
    return sorted(out.values(), key=lambda d: -d["score"])


def calibration_curve(yb: Arr, p: Arr, bins: int = 10, quantile: bool = False) -> tuple[float, list[dict[str, float]]]:
    """ECE + reliability points. Uniform bins for ECE; quantile bins give every chart point the same number of rows."""
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1))) if quantile else np.linspace(0, 1, bins + 1)
    ece, curve = 0.0, []
    last = edges[-1]
    for lo, hi in itertools.pairwise(edges):
        m = (p >= lo) & ((p < hi) if hi < last else (p <= hi))
        if m.any():
            ece += abs(float(p[m].mean()) - float(yb[m].mean())) * float(m.mean())
            curve.append({"predicted": float(p[m].mean()), "observed": float(yb[m].mean()), "count": int(m.sum())})
    return float(ece), curve


def run_critic(
    task: TabularTask,
    runs: list[RunResult],
    candidates: dict[str, Candidate],
    cfg: TrainConfig,
    name_suspects: list[dict[str, str]],
) -> list[Finding]:
    metric = task.strategy.metric
    label = LABELS[metric]
    findings: list[Finding] = []
    base = next((r for r in runs if r.family == "Baseline"), None)
    models = sorted((r for r in runs if r.family != "Baseline" and np.isfinite(r.mean)), key=lambda r: -r.selection_score)
    if not models:
        return [
            Finding(
                "no_model",
                "baseline",
                "high",
                "No model could be trained",
                "Every candidate failed during cross-validation.",
                fix={"stop": True},
            )
        ]
    leader = models[0]
    cand = next(c for c in candidates.values() if c.name == leader.candidate)
    columns = [c for c in task.columns if c not in cfg.exclude]

    # 1. leakage ------------------------------------------------------------------------
    leaks = univariate_leakage(task, columns)
    suspicious = (metric in ("roc_auc", "pr_auc", "macro_f1") and leader.mean >= 0.97) or (
        task.problem_type == "regression" and _r2(task, leader) >= 0.97
    )
    shares: list[tuple[str, float]] = []
    try:
        n = len(task.X_search)
        cut = int(n * 0.8)
        pipe = build(task, cand, leader.params, TrainConfig.from_dict(leader.config))
        fit(pipe, task.X_search.iloc[:cut], task.y_search[:cut], sample_weights(task.y_search[:cut], task, cfg))

        def score_fn(Xp: pd.DataFrame, yp: Arr) -> float:
            return selection_score(metric, score_task(metric, yp, predict(pipe, Xp, task), task.pos_index))

        shares = [
            (str(c), float(s))
            for c, s in permutation_shares(score_fn, task.X_search.iloc[cut:][columns], task.y_search[cut:], n_repeats=3, max_rows=1500)
        ]
    except Exception:
        shares = []
    if shares and shares[0][1] >= 0.5 and suspicious and shares[0][0] not in {lk["column"] for lk in leaks}:
        leaks.append(
            {
                "column": shares[0][0],
                "score": shares[0][1],
                "evidence": f"it alone drives {shares[0][1] * 100:.0f}% of the model's accuracy while the score is suspiciously high",
            }
        )
    leak_cols = [lk["column"] for lk in leaks]
    for lk in leaks:
        findings.append(
            Finding(
                f"leak_{lk['column']}",
                "leakage",
                "high",
                f"Target leakage: `{lk['column']}`",
                f"`{lk['column']}` looks like it is only known after the outcome: {lk['evidence']}. Removing it gives an honest score.",
                {"column": lk["column"], "score": round(lk["score"], 4)},
                {"exclude": [lk["column"]]},
            )
        )
    for s in name_suspects:
        if s["column"] in columns and s["column"] not in leak_cols:
            findings.append(
                Finding(
                    f"suspect_{s['column']}",
                    "leakage",
                    "medium",
                    f"Possible leakage by name: `{s['column']}`",
                    f"`{s['column']}`: {s['reason']}. The data does not prove it, so it stays unless you remove it.",
                    {"column": s["column"]},
                    None,
                )
            )
    if not leaks:
        findings.append(
            Finding(
                "leak_pass",
                "leakage",
                "pass",
                "No target leakage found",
                f"No single column predicts `{task.target}` too well on its own (ROC-AUC ≤ {AUC_LEAK} / correlation ≤ {CORR_LEAK}).",
            )
        )

    # 2. must beat the baseline ---------------------------------------------------------------
    if base is not None and np.isfinite(base.mean):
        lift = leader.selection_score - base.selection_score
        noise = 1.5 * leader.std  # a lift inside the fold-to-fold noise is not a real improvement
        if metric == "pr_auc":  # the PR-AUC baseline is the positive rate: require a clear multiple of it
            weak = leader.mean < 2 * base.mean or lift < max(0.01, noise)
        else:
            weak = lift <= max(0.02 if metric not in LOWER_IS_BETTER else 0.02 * abs(base.mean), noise)
        if weak:
            findings.append(
                Finding(
                    "baseline",
                    "baseline",
                    "high",
                    "No model clearly beats the baseline",
                    f"The best model ({leader.candidate}, {label} {display(metric, leader.mean)}) is not meaningfully better than the naive baseline ({display(metric, base.mean)}). "
                    "The columns in this sheet do not seem to carry enough signal for this target.",
                    {"model": leader.mean, "baseline": base.mean},
                    {"stop": True},
                )
            )
        else:
            findings.append(
                Finding(
                    "baseline_pass",
                    "baseline",
                    "pass",
                    "Beats the baseline",
                    f"{leader.candidate} {display(metric, leader.mean)} vs baseline {display(metric, base.mean)}.",
                )
            )

    # 3. overfitting ----------------------------------------------------------------------
    if leader.train_score is not None and np.isfinite(leader.train_score):
        if metric in LOWER_IS_BETTER:
            gap = leader.mean / max(leader.train_score, 1e-9)
            over = gap > 1.6
            gap_txt = f"validation error is {gap:.1f}× the training error"
        else:
            gap = leader.train_score - leader.mean
            over = gap > 0.08
            gap_txt = f"train {display(metric, leader.train_score)} vs validation {display(metric, leader.mean)}"
        if over:
            findings.append(
                Finding(
                    "overfit",
                    "overfitting",
                    "medium",
                    f"{leader.candidate} overfits",
                    f"There is a large gap between training and validation ({gap_txt}). Round 2 uses stronger regularisation.",
                    {"train": leader.train_score, "validation": leader.mean},
                    {"regularise": True},
                )
            )
        else:
            findings.append(
                Finding(
                    "overfit_pass",
                    "overfitting",
                    "pass",
                    "No worrying overfitting",
                    f"Training and validation scores are close ({gap_txt}).",
                )
            )

    # 4. stability -----------------------------------------------------------------------
    rel = leader.std / max(abs(leader.mean), 1e-9)
    if (metric in LOWER_IS_BETTER and rel > 0.15) or (metric not in LOWER_IS_BETTER and leader.std > 0.04):
        findings.append(
            Finding(
                "stability",
                "stability",
                "low",
                "Scores vary between folds",
                f"{label} ranges from {display(metric, min(leader.scores))} to {display(metric, max(leader.scores))} across folds, so expect some variation on new data.",
                {"scores": leader.scores},
            )
        )

    # 5. imbalance -------------------------------------------------------------------------
    if task.is_clf:
        counts = np.bincount(task.y_search.astype(int), minlength=task.n_classes)
        minority = counts.min() / counts.sum()
        limit = 0.2 if task.n_classes == 2 else 0.4 / task.n_classes
        if minority < limit and not cfg.class_weight:
            findings.append(
                Finding(
                    "imbalance",
                    "imbalance",
                    "medium",
                    "Imbalanced classes",
                    f"The smallest class is only {minority * 100:.1f}% of rows. Round 2 weights the classes so the model does not just predict the majority"
                    + (", re-calibrates the probabilities and tunes the decision threshold." if task.problem_type == "binary" else "."),
                    {"minority_share": float(minority)},
                    {"class_weight": True, "calibrate": task.problem_type == "binary"},
                )
            )

    # 6. calibration (binary) ------------------------------------------------------------------
    if task.problem_type == "binary" and leader.oof is not None:
        p = leader.oof[:, task.pos_index]
        ok = np.isfinite(p)
        if ok.sum() > 50:
            ece, curve = calibration_curve((task.y_search[ok] == task.pos_index).astype(int), p[ok])
            if ece > 0.05:
                findings.append(
                    Finding(
                        "calibration",
                        "calibration",
                        "low",
                        "Probabilities are not well calibrated",
                        f"Predicted probabilities are off by {ece * 100:.1f} points on average (expected calibration error). The final model gets isotonic or Platt calibration.",
                        {"ece": ece, "curve": curve},
                        {"calibrate": True},
                    )
                )
            else:
                findings.append(
                    Finding(
                        "calibration_pass",
                        "calibration",
                        "pass",
                        "Probabilities are well calibrated",
                        f"Expected calibration error {ece * 100:.1f} points.",
                        {"ece": ece, "curve": curve},
                    )
                )

    # 7. residual patterns (regression) ----------------------------------------------------------
    if task.problem_type == "regression" and leader.oof is not None:
        pred = leader.oof
        ok = np.isfinite(pred)
        res = task.y_search[ok] - pred[ok]
        if ok.sum() > 50:
            rho = float(spearmanr(np.abs(res), pred[ok]).statistic or 0.0)
            y_pos = bool((task.y_search > 0).all())
            skew = float(pd.Series(task.y_search).skew())
            if rho > 0.3 and y_pos and skew > 1 and not cfg.log_target:
                findings.append(
                    Finding(
                        "heteroscedastic",
                        "residuals",
                        "medium",
                        "Errors grow with the predicted value",
                        f"Bigger predictions come with bigger errors (rank correlation {rho:.2f}) and the target is skewed ({skew:.1f}). Round 2 trains on log({task.target}).",
                        {"correlation": rho, "skew": skew},
                        {"log_target": True},
                    )
                )
            seg = _worst_segment(task, ok, np.abs(res))
            if seg:
                findings.append(
                    Finding("segment_error", "residuals", "info", f"Weakest segment: {seg['column']} = {seg['value']}", seg["detail"], seg)
                )
    if len(task.y_train) + len(task.y_hold) < 300:
        findings.append(
            Finding(
                "data_size",
                "data_size",
                "info",
                "Small dataset",
                f"Only {len(task.y_train) + len(task.y_hold)} rows: treat the scores as rough estimates.",
            )
        )
    return findings


def _r2(task: TabularTask, run: RunResult) -> float:
    if run.oof is None:
        return 0.0
    ok = np.isfinite(run.oof)
    return score_task("r2", task.y_search[ok], run.oof[ok]) if ok.sum() > 2 else 0.0


def _worst_segment(task: TabularTask, ok: Arr, abs_err: Arr) -> dict[str, Any] | None:
    overall = float(abs_err.mean()) or 1e-9
    best: dict[str, Any] | None = None
    X = task.X_search[ok]
    for c in task.columns:
        if task.kinds.get(c) != "category":
            continue
        g = pd.Series(abs_err).groupby(X[c].astype(str).to_numpy()).agg(["mean", "size"])
        g = g[g["size"] >= max(20, 0.02 * len(X))]
        if g.empty:
            continue
        worst = g["mean"].idxmax()
        ratio = float(g.loc[worst, "mean"]) / overall
        if ratio > 1.4 and (best is None or ratio > best["ratio"]):
            best = {
                "column": c,
                "value": str(worst),
                "ratio": ratio,
                "detail": f"Errors are {ratio:.1f}× the average when {c} = {worst} ({int(g.loc[worst, 'size'])} rows). More data for this segment would help most.",
            }
    return best


def apply_fixes(cfg: TrainConfig, findings: list[Finding], skip: set[str] | None = None) -> tuple[TrainConfig, list[Finding]]:
    """Return the round-2 config and the findings whose fixes were applied."""
    skip = skip or set()
    new = TrainConfig.from_dict(cfg.to_dict())
    applied = []
    for f in findings:
        if not f.fix or f.id in skip or f.fix.get("stop"):
            continue
        if "exclude" in f.fix:
            new.exclude = sorted(set(new.exclude) | set(f.fix["exclude"]))
        for key in ("class_weight", "calibrate", "log_target", "regularise"):
            if f.fix.get(key):
                setattr(new, key, True)
        applied.append(f)
    return new, applied
