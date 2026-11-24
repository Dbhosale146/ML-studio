"""Supervised training: data split, cross-validation, Optuna HPO within a time budget, derived-feature acceptance."""

from __future__ import annotations

import logging
import time
import uuid
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit, train_test_split

from .candidates import Candidate
from .metrics import score_task, selection_score
from .preprocessing import column_kinds, input_columns, make_pipeline
from .types import Arr, Progress, RunResult, Strategy, TrainConfig, noop_progress

log = logging.getLogger("datalyon.ml.training")
optuna.logging.set_verbosity(optuna.logging.WARNING)


class TrainingError(Exception):
    pass


@dataclass
class TabularTask:
    """Prepared data, split and CV plan for binary / multiclass / regression problems."""

    problem_type: str
    target: str
    strategy: Strategy
    columns: list[str]
    kinds: dict[str, str]
    X_train: pd.DataFrame
    y_train: Arr
    X_hold: pd.DataFrame
    y_hold: Arr
    X_search: pd.DataFrame
    y_search: Arr
    splits: list[tuple[Arr, Arr]]
    classes: list[str] = field(default_factory=list)
    pos_index: int = 1
    dropped_classes: list[str] = field(default_factory=list)
    seed: int = 42
    time_column: str | None = None

    @property
    def is_clf(self) -> bool:
        return self.problem_type in ("binary", "multiclass")

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    def X_all(self) -> pd.DataFrame:
        return pd.concat([self.X_train, self.X_hold])

    def y_all(self) -> Arr:
        return np.concatenate([self.y_train, self.y_hold])


def prepare_task(
    df: pd.DataFrame,
    profile: dict[str, Any],
    problem_type: str,
    target: str,
    strategy: Strategy,
    exclude: list[str],
    time_column: str | None,
    positive_class: str | None,
    seed: int,
    search_rows: int = 30_000,
) -> TabularTask:
    columns = input_columns(profile, target, exclude=[], time_column=time_column)
    kinds = column_kinds(profile, columns)
    data = df[df[target].notna()].copy()
    classes: list[str] = []
    dropped: list[str] = []
    pos_index = 1
    if problem_type in ("binary", "multiclass"):
        ys = data[target].map(lambda v: str(int(v)) if isinstance(v, float) and v.is_integer() else str(v).strip())
        vc = ys.value_counts()
        min_rows = max(5, strategy.n_folds)
        rare = vc[vc < min_rows]
        if len(rare):
            dropped = [str(k) for k in rare.index]
            keep = ~ys.isin(rare.index)
            data, ys = data[keep], ys[keep]
            vc = ys.value_counts()
        if len(vc) < 2:
            raise TrainingError(f"`{target}` needs at least two classes with {min_rows}+ rows each to train a classifier.")
        if problem_type == "binary" or len(vc) == 2:
            pos = positive_class if positive_class is not None and positive_class in vc.index else str(vc.index[-1])
            neg = next(str(v) for v in vc.index if v != pos)
            classes = [neg, pos]
        else:
            classes = [str(v) for v in vc.index]
        index = {c: i for i, c in enumerate(classes)}
        y = ys.map(index).to_numpy(dtype=int)
    else:
        y = pd.to_numeric(data[target], errors="coerce").to_numpy(dtype=float)
        keep_mask = np.isfinite(y)
        data, y = data[keep_mask], y[keep_mask]
    if len(data) < 30:
        raise TrainingError(f"Only {len(data)} usable rows: at least 30 are needed to train and validate a model.")
    X = data[columns].reset_index(drop=True)
    times = pd.to_datetime(data[time_column], errors="coerce").reset_index(drop=True) if time_column else None
    is_clf = problem_type in ("binary", "multiclass")

    if strategy.validation_kind == "time_series" and times is not None:
        order = np.argsort(times.fillna(times.min()).to_numpy(), kind="stable")
        X, y = X.iloc[order].reset_index(drop=True), y[order]
        cut = int(len(X) * (1 - strategy.holdout_fraction))
        X_tr, X_ho, y_tr, y_ho = X.iloc[:cut], X.iloc[cut:], y[:cut], y[cut:]
    else:
        X_tr, X_ho, y_tr, y_ho = train_test_split(
            X, y, test_size=strategy.holdout_fraction, random_state=seed, stratify=y if is_clf else None
        )
    X_tr, X_ho = X_tr.reset_index(drop=True), X_ho.reset_index(drop=True)

    if len(X_tr) > search_rows:
        if strategy.validation_kind == "time_series":
            X_s, y_s = X_tr.iloc[-search_rows:].reset_index(drop=True), y_tr[-search_rows:]
        else:
            X_s, _, y_s, _ = train_test_split(X_tr, y_tr, train_size=search_rows, random_state=seed, stratify=y_tr if is_clf else None)
            X_s = X_s.reset_index(drop=True)
    else:
        X_s, y_s = X_tr, y_tr
    splits = make_splits(X_s, np.asarray(y_s), strategy, is_clf, seed, len(classes))
    return TabularTask(
        problem_type=problem_type,
        target=target,
        strategy=strategy,
        columns=columns,
        kinds=kinds,
        X_train=X_tr,
        y_train=np.asarray(y_tr),
        X_hold=X_ho,
        y_hold=np.asarray(y_ho),
        X_search=X_s,
        y_search=np.asarray(y_s),
        splits=splits,
        classes=classes,
        pos_index=pos_index,
        dropped_classes=dropped,
        seed=seed,
        time_column=time_column,
    )


def make_splits(
    X: pd.DataFrame, y: Arr, strategy: Strategy, is_clf: bool, seed: int, n_classes: int, n_folds: int | None = None
) -> list[tuple[Arr, Arr]]:
    k = n_folds or strategy.n_folds
    if strategy.validation_kind == "time_series":
        splits = list(TimeSeriesSplit(n_splits=k).split(X))
        if is_clf:
            splits = [(tr, va) for tr, va in splits if len(np.unique(y[tr])) == n_classes]
        if len(splits) >= 2:
            return splits
    if is_clf:
        k = max(2, min(k, int(np.bincount(y.astype(int)).min())))
        return list(StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(X, y))
    return list(KFold(n_splits=k, shuffle=True, random_state=seed).split(X))


# ---------------------------------------------------------------------------------------------
# Building, fitting and predicting
# ---------------------------------------------------------------------------------------------


def build(task: TabularTask, cand: Candidate, params: dict[str, Any], cfg: TrainConfig) -> Any:
    p = dict(params)
    if cfg.regularise and not cand.baseline:
        p = {**cand.regularised, **p} if not params else p
    est = cand.build(p)
    if task.problem_type == "regression" and cfg.log_target and not cand.baseline:
        est = TransformedTargetRegressor(regressor=est, func=np.log1p, inverse_func=np.expm1)
    return make_pipeline(task.columns, task.kinds, cfg, cand.prep, est)


def sample_weights(y: Arr, task: TabularTask, cfg: TrainConfig) -> Arr | None:
    if not (task.is_clf and cfg.class_weight):
        return None
    counts = np.bincount(y.astype(int), minlength=task.n_classes).astype(float)
    counts[counts == 0] = 1.0
    w = len(y) / (task.n_classes * counts)
    return np.asarray(w[y.astype(int)])


def fit(pipe: Any, X: pd.DataFrame, y: Arr, w: Arr | None) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if w is not None:
            pipe.fit(X, y, model__sample_weight=w)
        else:
            pipe.fit(X, y)
    return pipe


def predict(pipe: Any, X: pd.DataFrame, task: TabularTask) -> Arr:
    """Probabilities (n, k) for classification, values for regression."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if not task.is_clf:
            return np.asarray(pipe.predict(X), dtype=float)
        p = np.asarray(pipe.predict_proba(X), dtype=float)
    cls = list(getattr(pipe, "classes_", range(p.shape[1])))
    if p.shape[1] == task.n_classes and cls == list(range(task.n_classes)):
        return p
    full = np.zeros((len(X), task.n_classes))
    for j, c in enumerate(cls):
        full[:, int(c)] = p[:, j]
    return full


def cross_validate(
    task: TabularTask,
    cand: Candidate,
    params: dict[str, Any],
    cfg: TrainConfig,
    on_fold: Progress = noop_progress,
    splits: list[tuple[Arr, Arr]] | None = None,
    X: pd.DataFrame | None = None,
    y: Arr | None = None,
    check_train: bool = True,
) -> tuple[list[float], Arr, float | None, float]:
    X = task.X_search if X is None else X
    y = task.y_search if y is None else y
    splits = task.splits if splits is None else splits
    metric = task.strategy.metric
    oof: Arr = np.full((len(y), task.n_classes), np.nan) if task.is_clf else np.full(len(y), np.nan)
    scores: list[float] = []
    train_score = None
    t0 = time.perf_counter()
    for k, (tr, va) in enumerate(splits):
        pipe = build(task, cand, params, cfg)
        fit(pipe, X.iloc[tr], y[tr], sample_weights(y[tr], task, cfg))
        pred = predict(pipe, X.iloc[va], task)
        oof[va] = pred
        scores.append(score_task(metric, y[va], pred, task.pos_index))
        if k == 0 and check_train:
            idx = tr[: min(len(tr), 5000)]
            train_score = score_task(metric, y[idx], predict(pipe, X.iloc[idx], task), task.pos_index)
        on_fold("fold", {"fold": k + 1, "folds": len(splits), "partial": float(np.nanmean(scores))})
    return scores, oof, train_score, time.perf_counter() - t0


def make_run(
    task: TabularTask,
    cand: Candidate,
    round_: int,
    params: dict[str, Any],
    cfg: TrainConfig,
    scores: list[float],
    oof: Arr,
    train_score: float | None,
    secs: float,
    tuned: bool,
) -> RunResult:
    mean = float(np.nanmean(scores)) if scores else float("nan")
    return RunResult(
        run_id=f"r{round_}-{cand.key}-{uuid.uuid4().hex[:4]}",
        candidate=cand.name,
        family=cand.family,
        round=round_,
        params=params,
        config=cfg.to_dict(),
        scores=[float(s) for s in scores],
        mean=mean,
        std=float(np.nanstd(scores)) if scores else float("nan"),
        selection_score=selection_score(task.strategy.metric, mean) if np.isfinite(mean) else float("-inf"),
        train_score=train_score,
        fit_seconds=round(secs, 2),
        tuned=tuned,
        oof=oof,
    )


def evaluate_candidate(
    task: TabularTask,
    cand: Candidate,
    round_: int,
    params: dict[str, Any],
    cfg: TrainConfig,
    progress: Progress = noop_progress,
    tuned: bool = False,
) -> RunResult:
    def on_fold(kind: str, p: dict[str, Any]) -> None:
        progress("fold", {**p, "candidate": cand.name, "key": cand.key, "round": round_})

    scores, oof, train_score, secs = cross_validate(task, cand, params, cfg, on_fold)
    return make_run(task, cand, round_, params, cfg, scores, oof, train_score, secs, tuned)


def tune(
    task: TabularTask, cand: Candidate, cfg: TrainConfig, budget_s: float, max_trials: int, progress: Progress = noop_progress
) -> tuple[dict[str, Any], float, int]:
    """Optuna TPE search on a 3-fold CV of (at most) 10k rows. Returns (best params, best selection score, trials run)."""
    if cand.suggest is None or budget_s <= 1:
        return {}, float("-inf"), 0
    n = min(len(task.X_search), 10_000)
    if n < len(task.X_search):
        idx = np.random.default_rng(task.seed).choice(len(task.X_search), n, replace=False)
        idx.sort()
        X, y = task.X_search.iloc[idx].reset_index(drop=True), task.y_search[idx]
    else:
        X, y = task.X_search, task.y_search
    splits = make_splits(X, y, task.strategy, task.is_clf, task.seed, task.n_classes, n_folds=3)
    metric = task.strategy.metric

    def objective(trial: optuna.Trial) -> float:
        assert cand.suggest is not None
        params = cand.suggest(trial)
        scores, _, _, _ = cross_validate(task, cand, params, cfg, splits=splits, X=X, y=y, check_train=False)
        v = float(np.nanmean(scores))
        return selection_score(metric, v) if np.isfinite(v) else float("-inf")

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=task.seed))
    t0 = time.perf_counter()

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

    study.optimize(objective, n_trials=max_trials, timeout=budget_s, callbacks=[cb], catch=(ValueError, RuntimeError))
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        return {}, float("-inf"), len(study.trials)
    return dict(study.best_trial.params), float(study.best_value), len(study.trials)


def evaluate_derived(
    task: TabularTask, cand: Candidate, cfg: TrainConfig, specs: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], float]]:
    """Accept a derived feature only if it improves a quick 3-fold CV score. Returns [(spec, gain)] for accepted ones."""
    if not specs:
        return []
    n = min(len(task.X_search), 8000)
    X, y = task.X_search.iloc[:n], task.y_search[:n]
    splits = make_splits(X, y, task.strategy, task.is_clf, task.seed, task.n_classes, n_folds=3)
    metric = task.strategy.metric
    base_cfg = TrainConfig.from_dict({**cfg.to_dict(), "derived": []})

    def score(c: TrainConfig) -> float:
        s, _, _, _ = cross_validate(task, cand, {}, c, splits=splits, X=X, y=y, check_train=False)
        return selection_score(metric, float(np.nanmean(s)))

    base = score(base_cfg)
    accepted = []
    for spec in specs:
        trial_cfg = TrainConfig.from_dict({**base_cfg.to_dict(), "derived": [spec]})
        gain = score(trial_cfg) - base
        if gain > 1e-4:
            accepted.append((spec, float(gain)))
    return accepted


def time_left(deadline: float) -> float:
    return deadline - time.perf_counter()


def clone_pipeline(pipe: Any) -> Any:
    return clone(pipe)
