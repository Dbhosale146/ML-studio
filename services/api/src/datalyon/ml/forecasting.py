"""Time-series forecasting for one or many series.

A single global gradient-boosting model is trained on lag features where every lag is >= the
horizon, so one direct model forecasts all `horizon` steps from data that is already known
(nothing from the future can leak). Validation is a rolling-origin backtest; the baseline is
seasonal naive; prediction intervals come from backtest residuals.
"""

from __future__ import annotations

import math
import time
import uuid
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import optuna
import pandas as pd

from .metrics import LABELS, display, forecast_metrics, selection_score
from .strategy import FREQ_INFO
from .types import Arr, Finding, Progress, RunResult, noop_progress

try:
    from lightgbm import LGBMRegressor

    HAS_LGBM = True
except ImportError:  # pragma: no cover
    HAS_LGBM = False
try:
    from xgboost import XGBRegressor

    HAS_XGB = True
except ImportError:  # pragma: no cover
    HAS_XGB = False

TOTAL = "Total"


class ForecastError(Exception):
    pass


@dataclass
class FcConfig:
    lags: list[int]
    windows: list[int]
    difference: bool = False  # model y - y[t-h] (removes level / trend bias)
    calendar_dummies: bool = False  # raw period-of-year integers in addition to cyclical encodings

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ForecastTask:
    target: str
    time_column: str
    series_column: str | None
    freq: str
    horizon: int
    season: int
    agg: str
    metric: str
    panel: pd.DataFrame  # columns: series, ds, y
    series: list[str]
    notes: list[dict[str, Any]] = field(default_factory=list)
    seed: int = 42

    @property
    def unit(self) -> str:
        return str(FREQ_INFO[self.freq]["unit"])


def prepare_forecast(
    df: pd.DataFrame,
    target: str,
    time_column: str,
    series_column: str | None,
    freq: str,
    horizon: int,
    metric: str,
    seed: int,
    agg: str = "sum",
) -> ForecastTask:
    notes: list[dict[str, Any]] = []
    d = pd.DataFrame({"ds": pd.to_datetime(df[time_column], errors="coerce"), "y": pd.to_numeric(df[target], errors="coerce")})
    d["series"] = df[series_column].astype(str).str.strip() if series_column else TOTAL
    bad = int(d["ds"].isna().sum())
    if bad:
        notes.append({"issue": f"{bad:,} rows without a valid date", "action": "Dropped: they cannot be placed in time", "count": bad})
        d = d.dropna(subset=["ds"])
    d["period"] = d["ds"].dt.to_period(freq)
    dup = int(d.duplicated(["series", "period"]).sum())
    g = d.groupby(["series", "period"])["y"]
    agg_df = (g.sum(min_count=1) if agg == "sum" else g.mean()).reset_index()
    if dup:
        notes.append(
            {
                "issue": f"{dup:,} rows share a {FREQ_INFO[freq]['unit']} with another row of the same series",
                "action": f"Aggregated them with {agg}",
                "count": dup,
            }
        )
    end = agg_df["period"].max()
    frames, gaps, dropped = [], 0, []
    min_len = 2 * horizon + 8
    for s, part in agg_df.groupby("series"):
        full = pd.period_range(part["period"].min(), end, freq=freq)
        ser = part.set_index("period")["y"].reindex(full)
        gaps += int(ser.isna().sum())
        ser = ser.interpolate(limit_direction="both")
        if ser.notna().sum() < min_len:
            dropped.append(str(s))
            continue
        frames.append(pd.DataFrame({"series": str(s), "ds": full.to_timestamp(), "y": ser.to_numpy(dtype=float)}))
    if gaps:
        notes.append(
            {
                "issue": f"{gaps:,} missing {FREQ_INFO[freq]['unit']}s in the timelines",
                "action": "Filled by linear interpolation between neighbours",
                "count": gaps,
            }
        )
    if dropped:
        notes.append(
            {
                "issue": f"{len(dropped)} series shorter than {min_len} periods",
                "action": f"Left out ({', '.join(dropped[:5])})",
                "count": len(dropped),
            }
        )
    if not frames:
        raise ForecastError(f"Each series needs at least {min_len} {FREQ_INFO[freq]['unit']}s of history to forecast {horizon} ahead.")
    panel = pd.concat(frames, ignore_index=True).sort_values(["series", "ds"]).reset_index(drop=True)
    season = int(FREQ_INFO[freq]["season"])
    return ForecastTask(
        target,
        time_column,
        series_column,
        freq,
        horizon,
        season,
        agg,
        metric,
        panel,
        sorted(panel["series"].unique().tolist()),
        notes,
        seed,
    )


def default_config(task: ForecastTask) -> FcConfig:
    h, m = task.horizon, task.season
    n_min = int(task.panel.groupby("series").size().min())
    lags = {h, h + 1, h + 2, h + 3}
    if m > 1 and m >= h and m + 2 < n_min - h:
        lags |= {m, m + 1}
    lags = {k for k in lags if k < n_min - h}
    windows = sorted({w for w in (4, max(4, m // 4), m) if 2 <= w <= max(2, n_min - h - 2)})
    return FcConfig(lags=sorted(lags) or [h], windows=windows or [2])


def build_features(panel: pd.DataFrame, task: ForecastTask, cfg: FcConfig, codes: dict[str, int]) -> pd.DataFrame:
    """Lag/rolling/calendar features for every row of the panel (future rows have y = NaN)."""
    h = task.horizon
    out = panel[["series", "ds", "y"]].copy()
    grp = out.groupby("series")["y"]
    for k in cfg.lags:
        out[f"lag_{k}"] = grp.shift(k)
    shifted = grp.shift(h)
    for w in cfg.windows:
        out[f"roll_mean_{w}"] = shifted.groupby(out["series"]).transform(lambda s, w=w: s.rolling(w, min_periods=max(1, w // 2)).mean())
    out["roll_std_4"] = shifted.groupby(out["series"]).transform(lambda s: s.rolling(4, min_periods=2).std())
    ds = out["ds"]
    if task.freq in ("D", "W", "h"):
        woy = ds.dt.isocalendar().week.astype(float)
        out["woy_sin"], out["woy_cos"] = np.sin(2 * np.pi * woy / 52.18), np.cos(2 * np.pi * woy / 52.18)
    if task.freq in ("D", "W", "M", "h"):
        mon = ds.dt.month.astype(float)
        out["month_sin"], out["month_cos"] = np.sin(2 * np.pi * (mon - 1) / 12), np.cos(2 * np.pi * (mon - 1) / 12)
    if task.freq in ("D", "h"):
        dow = ds.dt.dayofweek.astype(float)
        out["dow_sin"], out["dow_cos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    if task.freq == "h":
        hr = ds.dt.hour.astype(float)
        out["hour_sin"], out["hour_cos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
    if task.freq == "Q":
        out["quarter"] = ds.dt.quarter.astype(float)
    if cfg.calendar_dummies:
        out["month"] = ds.dt.month.astype(float)
        if task.freq in ("D", "W", "h"):
            out["week_of_year"] = ds.dt.isocalendar().week.astype(float)
    out["t"] = (ds - ds.min()).dt.days.astype(float)
    out["series_code"] = out["series"].map(codes).astype(float)
    return out


def feature_columns(feat: pd.DataFrame) -> list[str]:
    return [c for c in feat.columns if c not in ("series", "ds", "y")]


@dataclass
class FcCandidate:
    key: str
    name: str
    family: str
    baseline: bool = False


def forecast_candidates() -> list[FcCandidate]:
    c = [FcCandidate("snaive", "Seasonal naive", "Baseline", baseline=True)]
    if HAS_LGBM:
        c.append(FcCandidate("lgbm", "LightGBM (global)", "Boosting"))
    if HAS_XGB:
        c.append(FcCandidate("xgb", "XGBoost (global)", "Boosting"))
    return c


def make_estimator(key: str, params: dict[str, Any], seed: int, n_jobs: int) -> Any:
    if key == "lgbm":
        base: dict[str, Any] = {
            "n_estimators": 400,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 10,
            "subsample": 0.9,
            "subsample_freq": 1,
            "colsample_bytree": 0.9,
        }
        return LGBMRegressor(**{**base, **params, "random_state": seed, "n_jobs": n_jobs, "verbose": -1})
    base = {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.9, "colsample_bytree": 0.9}
    return XGBRegressor(**{**base, **params, "random_state": seed, "n_jobs": n_jobs, "tree_method": "hist", "verbosity": 0})


def suggest(key: str, t: optuna.Trial) -> dict[str, Any]:
    if key == "lgbm":
        return {
            "n_estimators": t.suggest_int("n_estimators", 150, 800, step=50),
            "learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": t.suggest_int("num_leaves", 8, 96, log=True),
            "min_child_samples": t.suggest_int("min_child_samples", 3, 50),
            "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0),
        }
    return {
        "n_estimators": t.suggest_int("n_estimators", 150, 800, step=50),
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": t.suggest_int("max_depth", 3, 9),
        "min_child_weight": t.suggest_float("min_child_weight", 0.5, 20, log=True),
        "subsample": t.suggest_float("subsample", 0.6, 1.0),
    }


def origins(task: ForecastTask, n_windows: int = 4) -> list[pd.Timestamp]:
    """Backtest cut-off dates, oldest first; each window covers the `horizon` periods after its origin."""
    dates = np.sort(task.panel["ds"].unique())
    n_min = int(task.panel.groupby("series").size().min())
    max_lag = max(default_config(task).lags)
    min_train = max_lag + max(2 * task.horizon, 10)
    possible = (n_min - min_train) // task.horizon
    w = int(min(n_windows, possible))
    if w < 1:
        raise ForecastError(f"The history is too short for a backtest: each series needs about {min_train + task.horizon} {task.unit}s.")
    return [pd.Timestamp(dates[-1 - k * task.horizon]) for k in range(w, 0, -1)]


def _seasonal_naive(task: ForecastTask, history: pd.DataFrame, target_rows: pd.DataFrame, origin: pd.Timestamp) -> Arr:
    m = task.season if task.season > 1 else 1
    out = np.empty(len(target_rows))
    hist = history.set_index(["series", "ds"])["y"]
    offsets = {ds: i + 1 for i, ds in enumerate(sorted(target_rows["ds"].unique()))}
    step = _offset(task.freq)
    for i, (s, ds) in enumerate(zip(target_rows["series"], target_rows["ds"], strict=True)):
        k = offsets[ds]
        back = m * math.ceil(k / m) if m > 1 else k
        ref = ds - back * step
        out[i] = hist.get((s, ref), np.nan)
        if not np.isfinite(out[i]):
            out[i] = hist.get((s, origin), np.nan)
    return out


def _offset(freq: str) -> Any:
    """Unanchored step matching Period.start_time timestamps (weeks start on Monday, months on the 1st...)."""
    return {
        "h": pd.offsets.Hour(),
        "D": pd.offsets.Day(),
        "W": pd.offsets.Week(),
        "M": pd.offsets.MonthBegin(),
        "Q": pd.offsets.QuarterBegin(startingMonth=1),
        "Y": pd.offsets.YearBegin(),
    }[freq]


def _fit_predict(
    task: ForecastTask,
    cand: FcCandidate,
    params: dict[str, Any],
    cfg: FcConfig,
    feat: pd.DataFrame,
    train_mask: Arr,
    test_mask: Arr,
    n_jobs: int,
) -> Arr:
    cols = feature_columns(feat)
    tr = feat[train_mask].dropna(subset=[f"lag_{k}" for k in cfg.lags] + ["y"])
    y = tr["y"].to_numpy()
    base_lag = f"lag_{task.horizon}"
    if cfg.difference:
        y = y - tr[base_lag].to_numpy()
    est = make_estimator(cand.key, params, task.seed, n_jobs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est.fit(tr[cols], y)
        pred = np.asarray(est.predict(feat.loc[test_mask, cols]), dtype=float)
    if cfg.difference:
        pred = pred + feat.loc[test_mask, base_lag].to_numpy()
    return np.maximum(pred, 0.0) if (task.panel["y"] >= 0).all() else pred


def backtest(
    task: ForecastTask,
    cand: FcCandidate,
    params: dict[str, Any],
    cfg: FcConfig,
    windows: list[pd.Timestamp],
    n_jobs: int,
    progress: Progress = noop_progress,
) -> tuple[list[float], pd.DataFrame]:
    codes = {s: i for i, s in enumerate(task.series)}
    feat = build_features(task.panel, task, cfg, codes)
    scores: list[float] = []
    preds = []
    step = _offset(task.freq)
    for wi, origin in enumerate(windows):
        end = origin + task.horizon * step
        train_mask = (feat["ds"] <= origin).to_numpy()
        test_mask = ((feat["ds"] > origin) & (feat["ds"] <= end)).to_numpy()
        rows = feat.loc[test_mask, ["series", "ds", "y"]].copy()
        if cand.baseline:
            yhat = _seasonal_naive(task, task.panel[task.panel["ds"] <= origin], rows, origin)
        else:
            yhat = _fit_predict(task, cand, params, cfg, feat, train_mask, test_mask, n_jobs)
        rows["yhat"] = yhat
        rows["window"] = wi + 1
        rows["step"] = rows.groupby("series").cumcount() + 1
        preds.append(rows)
        ok = np.isfinite(rows["yhat"].to_numpy())
        scores.append(forecast_metrics(rows["y"].to_numpy()[ok], rows["yhat"].to_numpy()[ok])[task.metric])
        progress("fold", {"fold": wi + 1, "folds": len(windows), "partial": float(np.nanmean(scores))})
    return scores, pd.concat(preds, ignore_index=True)


def evaluate(
    task: ForecastTask,
    cand: FcCandidate,
    params: dict[str, Any],
    cfg: FcConfig,
    windows: list[pd.Timestamp],
    round_: int,
    n_jobs: int,
    progress: Progress = noop_progress,
    tuned: bool = False,
) -> tuple[RunResult, pd.DataFrame]:
    def on_fold(kind: str, p: dict[str, Any]) -> None:
        progress("fold", {**p, "candidate": cand.name, "key": cand.key, "round": round_})

    t0 = time.perf_counter()
    scores, preds = backtest(task, cand, params, cfg, windows, n_jobs, on_fold)
    mean = float(np.nanmean(scores))
    run = RunResult(
        run_id=f"r{round_}-{cand.key}-{uuid.uuid4().hex[:4]}",
        candidate=cand.name,
        family=cand.family,
        round=round_,
        params=params,
        config=cfg.to_dict(),
        scores=scores,
        mean=mean,
        std=float(np.nanstd(scores)),
        selection_score=selection_score(task.metric, mean),
        train_score=None,
        fit_seconds=round(time.perf_counter() - t0, 2),
        tuned=tuned,
    )
    return run, preds


def tune(
    task: ForecastTask,
    cand: FcCandidate,
    cfg: FcConfig,
    windows: list[pd.Timestamp],
    budget_s: float,
    max_trials: int,
    n_jobs: int,
    progress: Progress = noop_progress,
) -> tuple[dict[str, Any], float, int]:
    if cand.baseline or budget_s <= 1:
        return {}, float("-inf"), 0
    wins = windows[-2:]
    t0 = time.perf_counter()

    def objective(trial: optuna.Trial) -> float:
        scores, _ = backtest(task, cand, suggest(cand.key, trial), cfg, wins, n_jobs)
        v = float(np.nanmean(scores))
        return selection_score(task.metric, v) if np.isfinite(v) else float("-inf")

    def cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        progress(
            "trial",
            {
                "candidate": cand.name,
                "key": cand.key,
                "trial": trial.number + 1,
                "value": trial.value,
                "best": study.best_value,
                "elapsed": time.perf_counter() - t0,
            },
        )

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=task.seed))
    study.optimize(objective, n_trials=max_trials, timeout=budget_s, callbacks=[cb], catch=(ValueError, RuntimeError))
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not done:
        return {}, float("-inf"), len(study.trials)
    return dict(study.best_trial.params), float(study.best_value), len(study.trials)


def interval_quantiles(task: ForecastTask, preds: pd.DataFrame) -> list[list[float]]:
    """Relative 10%/90% error quantiles per step ahead, from backtest residuals."""
    p = preds[np.isfinite(preds["yhat"]) & (preds["yhat"].abs() > 1e-9)]
    rel = (p["y"] - p["yhat"]) / p["yhat"].abs()
    out = []
    for s in range(1, task.horizon + 1):
        r = rel[p["step"] == s]
        if len(r) < 8:
            r = rel[(p["step"] >= max(1, s - 2)) & (p["step"] <= s + 2)]
        if len(r) < 4:
            r = rel
        out.append([float(np.quantile(r, 0.1)), float(np.quantile(r, 0.9))])
    return out


def coverage(preds: pd.DataFrame, q: list[list[float]]) -> float:
    lo = np.array([q[s - 1][0] for s in preds["step"]])
    hi = np.array([q[s - 1][1] for s in preds["step"]])
    base = preds["yhat"].abs().to_numpy()
    inside = (preds["y"] >= preds["yhat"] + lo * base) & (preds["y"] <= preds["yhat"] + hi * base)
    return float(inside.mean())


def critic(task: ForecastTask, runs: list[RunResult], preds_by_run: dict[str, pd.DataFrame], cfg: FcConfig) -> list[Finding]:
    metric, label = task.metric, LABELS[task.metric]
    findings = [
        Finding(
            "leak_pass",
            "leakage",
            "pass",
            "No future data can leak",
            f"Every lag feature is at least {task.horizon} {task.unit}s old, the forecast horizon, so the model only uses values known at forecast time.",
        )
    ]
    base = next((r for r in runs if r.family == "Baseline"), None)
    models = sorted((r for r in runs if r.family != "Baseline" and np.isfinite(r.mean)), key=lambda r: -r.selection_score)
    if not models:
        return [
            *findings,
            Finding(
                "no_model",
                "baseline",
                "high",
                "No model could be trained",
                "Every candidate failed during the backtest.",
                fix={"stop": True},
            ),
        ]
    leader = models[0]
    if base is not None:
        if leader.selection_score - base.selection_score <= 0.02 * abs(base.mean):
            findings.append(
                Finding(
                    "baseline",
                    "baseline",
                    "high",
                    "The model does not beat seasonal naive",
                    f"{leader.candidate} scores {label} {display(metric, leader.mean)} vs {display(metric, base.mean)} for simply repeating last season. "
                    "This series is dominated by its seasonal pattern; seasonal naive is the honest forecast.",
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
                    "Beats seasonal naive",
                    f"{display(metric, leader.mean)} vs {display(metric, base.mean)} for seasonal naive.",
                )
            )
    if leader.std / max(abs(leader.mean), 1e-9) > 0.25 and len(leader.scores) > 1:
        findings.append(
            Finding(
                "stability",
                "stability",
                "low",
                "Accuracy varies between backtest windows",
                f"{label} ranges from {display(metric, min(leader.scores))} to {display(metric, max(leader.scores))} across windows.",
                {"scores": leader.scores},
            )
        )
    preds = preds_by_run.get(leader.run_id)
    if preds is not None and len(preds):
        bias = float((preds["yhat"] - preds["y"]).sum() / max(preds["y"].abs().sum(), 1e-9))
        if abs(bias) > 0.04 and not cfg.difference:
            findings.append(
                Finding(
                    "bias",
                    "residuals",
                    "medium",
                    f"Forecasts are systematically {'too high' if bias > 0 else 'too low'}",
                    f"Across the backtest the forecast is {abs(bias) * 100:.1f}% {'above' if bias > 0 else 'below'} the actuals, a sign of trend or level shifts. "
                    f"Round 2 models the change from {task.horizon} {task.unit}s earlier instead of the raw level.",
                    {"bias": bias},
                    {"difference": True},
                )
            )
        err = (preds["yhat"] - preds["y"]).abs()
        by_month = err.groupby(preds["ds"].dt.month).sum() / preds["y"].abs().groupby(preds["ds"].dt.month).sum().clip(lower=1e-9)
        overall = float(err.sum() / max(preds["y"].abs().sum(), 1e-9))
        if len(by_month) >= 3 and overall > 0:
            worst_m = int(by_month.idxmax())
            ratio = float(by_month.max()) / overall
            if ratio > 1.5 and not cfg.calendar_dummies:
                findings.append(
                    Finding(
                        "seasonal_errors",
                        "residuals",
                        "medium",
                        f"Errors peak in {pd.Timestamp(2024, worst_m, 1).strftime('%B')}",
                        f"Errors in {pd.Timestamp(2024, worst_m, 1).strftime('%B')} are {ratio:.1f}× the average, typical for holiday peaks. Round 2 adds explicit month and week-of-year features.",
                        {"month": worst_m, "ratio": ratio},
                        {"calendar_dummies": True},
                    )
                )
        if task.series_column and len(task.series) > 1:
            per = preds.groupby("series").apply(
                lambda g: float((g["yhat"] - g["y"]).abs().sum() / max(g["y"].abs().sum(), 1e-9)), include_groups=False
            )
            worst = str(per.idxmax())
            if overall > 0 and per.max() / overall > 1.8:
                findings.append(
                    Finding(
                        "worst_series",
                        "residuals",
                        "info",
                        f"Hardest series: {worst}",
                        f"{worst} has {per.max() / overall:.1f}× the average error (WAPE {per.max() * 100:.1f}%).",
                        {"series": worst, "wape": float(per.max())},
                    )
                )
    return findings


def apply_fixes(cfg: FcConfig, findings: list[Finding]) -> tuple[FcConfig, list[Finding]]:
    new = FcConfig(**cfg.to_dict())
    applied = []
    for f in findings:
        if f.fix and not f.fix.get("stop"):
            if f.fix.get("difference"):
                new.difference = True
            if f.fix.get("calendar_dummies"):
                new.calendar_dummies = True
            applied.append(f)
    return new, applied


def fit_full(task: ForecastTask, cand: FcCandidate, params: dict[str, Any], cfg: FcConfig, n_jobs: int) -> Any:
    if cand.baseline:
        return None
    codes = {s: i for i, s in enumerate(task.series)}
    feat = build_features(task.panel, task, cfg, codes)
    tr = feat.dropna(subset=[f"lag_{k}" for k in cfg.lags] + ["y"])
    y = tr["y"].to_numpy()
    if cfg.difference:
        y = y - tr[f"lag_{task.horizon}"].to_numpy()
    est = make_estimator(cand.key, params, task.seed, n_jobs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est.fit(tr[feature_columns(feat)], y)
    return est


def forecast_future(task: ForecastTask, cand_key: str, model: Any, cfg: FcConfig, q: list[list[float]]) -> pd.DataFrame:
    """Forecast the next `horizon` periods for every series (columns: series, ds, yhat, lo, hi, step)."""
    step = _offset(task.freq)
    future = []
    for s, part in task.panel.groupby("series"):
        last = part["ds"].max()
        future.append(pd.DataFrame({"series": s, "ds": [last + k * step for k in range(1, task.horizon + 1)], "y": np.nan}))
    ext = pd.concat([task.panel, *future], ignore_index=True).sort_values(["series", "ds"]).reset_index(drop=True)
    codes = {s: i for i, s in enumerate(task.series)}
    feat = build_features(ext, task, cfg, codes)
    mask = feat["y"].isna().to_numpy()
    rows = feat.loc[mask, ["series", "ds"]].copy()
    if cand_key == "snaive" or model is None:
        origin = task.panel["ds"].max()
        yhat = _seasonal_naive(task, task.panel, rows.assign(y=np.nan), origin)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yhat = np.asarray(model.predict(feat.loc[mask, feature_columns(feat)]), dtype=float)
        if cfg.difference:
            yhat = yhat + feat.loc[mask, f"lag_{task.horizon}"].to_numpy()
        if (task.panel["y"] >= 0).all():
            yhat = np.maximum(yhat, 0.0)
    rows["yhat"] = yhat
    rows["step"] = rows.groupby("series").cumcount() + 1
    base = np.abs(rows["yhat"].to_numpy())
    rows["lo"] = rows["yhat"] + np.array([q[s - 1][0] for s in rows["step"]]) * base
    rows["hi"] = rows["yhat"] + np.array([q[s - 1][1] for s in rows["step"]]) * base
    if (task.panel["y"] >= 0).all():
        rows["lo"] = rows["lo"].clip(lower=0)
    return rows.reset_index(drop=True)


def chart_data(task: ForecastTask, future: pd.DataFrame, last_window: pd.DataFrame, tail: int | None = None) -> dict[str, Any]:
    """History tail + last backtest window + future forecast, per series and for the total."""
    tail = tail or max(3 * task.horizon, 2 * task.season if task.season > 1 else 0, 26)
    plots: dict[str, list[dict[str, Any]]] = {}

    def series_plot(hist: pd.DataFrame, back: pd.DataFrame, fut: pd.DataFrame) -> list[dict[str, Any]]:
        pts: dict[str, dict[str, Any]] = {}
        for ds, y in zip(hist["ds"].tail(tail), hist["y"].tail(tail), strict=True):
            pts[ds.strftime("%Y-%m-%d")] = {"date": ds.strftime("%Y-%m-%d"), "actual": round(float(y), 3)}
        for ds, yh in zip(back["ds"], back["yhat"], strict=True):
            pts.setdefault(ds.strftime("%Y-%m-%d"), {"date": ds.strftime("%Y-%m-%d")})["backtest"] = round(float(yh), 3)
        for ds, yh, lo, hi in zip(fut["ds"], fut["yhat"], fut["lo"], fut["hi"], strict=True):
            pts[ds.strftime("%Y-%m-%d")] = {
                "date": ds.strftime("%Y-%m-%d"),
                "forecast": round(float(yh), 3),
                "band": [round(float(lo), 3), round(float(hi), 3)],
            }
        return [pts[k] for k in sorted(pts)]

    for s in task.series[:50]:
        plots[s] = series_plot(
            task.panel[task.panel["series"] == s], last_window[last_window["series"] == s], future[future["series"] == s]
        )
    if len(task.series) > 1:
        hist = task.panel.groupby("ds", as_index=False)["y"].sum()
        back = last_window.groupby("ds", as_index=False)["yhat"].sum()
        fut = future.groupby("ds", as_index=False)[["yhat", "lo", "hi"]].sum()
        plots[TOTAL] = series_plot(hist, back, fut)
    return {"series": plots, "default": TOTAL if len(task.series) > 1 else task.series[0]}
