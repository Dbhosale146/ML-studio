"""Explanations: permutation importance (global) and SHAP drivers per prediction (local), on raw input columns."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from .types import Arr

log = logging.getLogger("datalyon.ml.explain")


def permutation_shares(
    score_fn: Callable[[pd.DataFrame, Arr], float], X: pd.DataFrame, y: Arr, n_repeats: int = 3, max_rows: int = 2000, seed: int = 0
) -> list[list[Any]]:
    """Raw-column permutation importance normalised to shares summing to 1. score_fn(X, y): higher is better."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    if len(X) > max_rows:
        idx = rng.choice(len(X), max_rows, replace=False)
        X, y = X.iloc[idx], y[idx]
    X = X.reset_index(drop=True)
    base = score_fn(X, y)
    imps: dict[str, float] = {}
    for c in X.columns:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[c] = rng.permutation(Xp[c].to_numpy())
            drops.append(base - score_fn(Xp, y))
        imps[str(c)] = max(0.0, float(np.nanmean(drops)))
    total = sum(imps.values())
    if total <= 0:
        return [[c, 0.0] for c in X.columns]
    return sorted(([c, v / total] for c, v in imps.items()), key=lambda t: -float(t[1]))


def _unwrap(model: Any) -> tuple[Any, Any]:
    """(sklearn Pipeline with features+prep+model, inner estimator) from a possibly calibrated pipeline."""
    pipe = model
    if hasattr(model, "calibrated_classifiers_"):
        pipe = model.calibrated_classifiers_[0].estimator
    est = pipe.named_steps["model"]
    if hasattr(est, "regressor_"):
        est = est.regressor_
    return pipe, est


def transformed_sources(pipe: Any) -> list[list[str]]:
    """For every column the model sees, the raw input columns it came from."""
    fb = pipe.named_steps["features"]
    prep = pipe.named_steps["prep"]
    names = [str(n) for n in prep.get_feature_names_out()]
    fb_names = sorted(fb.source_map_.keys(), key=len, reverse=True)
    out = []
    for n in names:
        if n in fb.source_map_:
            out.append(fb.source_map_[n])
            continue
        hit = next((f for f in fb_names if n.startswith(f + "_")), None)
        out.append(fb.source_map_[hit] if hit else [n])
    return out


def shap_drivers(
    model: Any, X: pd.DataFrame, problem_type: str, top: int = 5, background: pd.DataFrame | None = None
) -> tuple[list[list[dict[str, Any]]], str] | None:
    """Top drivers per row as [{feature, effect}], aggregated to raw columns. None when SHAP cannot explain the model."""
    try:
        import shap
    except ImportError:  # pragma: no cover
        return None
    try:
        pipe, est = _unwrap(model)
        pre = pipe[:-1]
        Xt = np.asarray(pre.transform(X), dtype=float)
        kind = type(est).__name__
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if kind in ("LogisticRegression", "Ridge"):
                bg = np.asarray(pre.transform(background), dtype=float) if background is not None else np.zeros((1, Xt.shape[1]))
                explainer = shap.LinearExplainer(est, bg)
                values = explainer.shap_values(Xt)
                name = "SHAP (linear)"
            elif kind.startswith(("LGBM", "XGB", "RandomForest", "HistGradientBoosting")):
                explainer = shap.TreeExplainer(est)
                values = explainer.shap_values(Xt, check_additivity=False)
                name = "SHAP (tree)"
            else:
                return None
        vals = np.asarray(values)
        if isinstance(values, list):
            vals = np.stack(values, axis=-1)
        if vals.ndim == 3:  # (rows, features, classes)
            if problem_type == "binary":
                vals = vals[:, :, -1]
            else:
                proba = np.asarray(model.predict_proba(X))
                vals = np.stack([vals[i, :, int(proba[i].argmax())] for i in range(len(X))])
        sources = transformed_sources(pipe)
        result = []
        for i in range(len(X)):
            agg: dict[str, float] = {}
            for j, srcs in enumerate(sources):
                for s in srcs:
                    agg[s] = agg.get(s, 0.0) + float(vals[i, j]) / len(srcs)
            ranked = sorted(agg.items(), key=lambda t: -abs(t[1]))[:top]
            result.append([{"feature": f, "effect": e} for f, e in ranked if abs(e) > 1e-12])
        return result, name
    except Exception as e:  # SHAP is best-effort; inference falls back to occlusion
        log.warning("SHAP failed: %s", e.__class__.__name__)
        return None


def occlusion_drivers(
    predict_one: Callable[[pd.DataFrame], Arr], X: pd.DataFrame, reference: dict[str, Any], columns: list[str], top: int = 5
) -> list[list[dict[str, Any]]]:
    """Effect of replacing each input with its typical value (fallback when SHAP is unavailable)."""
    out = []
    for i in range(len(X)):
        row = X.iloc[[i]]
        variants = pd.concat([row] * len(columns), ignore_index=True)
        for k, c in enumerate(columns):
            variants.loc[k, c] = reference.get(c, np.nan)
        base = float(predict_one(row)[0])
        preds = predict_one(variants)
        effects = [(c, base - float(p)) for c, p in zip(columns, preds, strict=True)]
        ranked = sorted(effects, key=lambda t: -abs(t[1]))[:top]
        out.append([{"feature": c, "effect": e} for c, e in ranked if abs(e) > 1e-12])
    return out
