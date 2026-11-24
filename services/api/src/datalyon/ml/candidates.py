"""Model families per problem type, their Optuna search spaces and shortlisting by data size."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import optuna
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge

try:
    from lightgbm import LGBMClassifier, LGBMRegressor

    HAS_LGBM = True
except ImportError:  # pragma: no cover
    HAS_LGBM = False
try:
    from xgboost import XGBClassifier, XGBRegressor

    HAS_XGB = True
except ImportError:  # pragma: no cover
    HAS_XGB = False

Suggest = Callable[[optuna.Trial], dict[str, Any]]


@dataclass
class Candidate:
    key: str
    name: str
    family: str
    prep: str  # "linear" | "tree"
    build: Callable[[dict[str, Any]], Any]
    suggest: Suggest | None = None
    regularised: dict[str, Any] = field(default_factory=dict)
    baseline: bool = False


def _rf_space(t: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": t.suggest_int("n_estimators", 100, 400, step=50),
        "max_depth": t.suggest_categorical("max_depth", [None, 6, 10, 16]),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 20),
        "max_features": t.suggest_categorical("max_features", ["sqrt", 0.5, 0.8]),
    }


def _hgb_space(t: optuna.Trial) -> dict[str, Any]:
    return {
        "learning_rate": t.suggest_float("learning_rate", 0.02, 0.25, log=True),
        "max_leaf_nodes": t.suggest_int("max_leaf_nodes", 8, 64),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 5, 100),
        "l2_regularization": t.suggest_float("l2_regularization", 1e-4, 10, log=True),
        "max_iter": t.suggest_int("max_iter", 100, 500, step=50),
    }


def _lgbm_space(t: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": t.suggest_int("n_estimators", 100, 700, step=50),
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": t.suggest_int("num_leaves", 8, 128, log=True),
        "min_child_samples": t.suggest_int("min_child_samples", 5, 100),
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 10, log=True),
    }


def _xgb_space(t: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": t.suggest_int("n_estimators", 100, 700, step=50),
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": t.suggest_int("max_depth", 3, 10),
        "min_child_weight": t.suggest_float("min_child_weight", 0.5, 20, log=True),
        "subsample": t.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0),
    }


def classification_candidates(n_rows: int, n_classes: int, seed: int, n_jobs: int) -> tuple[list[Candidate], list[dict[str, str]]]:
    skipped: list[dict[str, str]] = []
    c = [
        Candidate("baseline", "Baseline (majority class)", "Baseline", "tree", lambda p: DummyClassifier(strategy="prior"), baseline=True),
        Candidate(
            "logreg",
            "Logistic Regression",
            "Linear",
            "linear",
            lambda p: LogisticRegression(max_iter=3000, **{"C": 1.0, **p}),
            lambda t: {"C": t.suggest_float("C", 1e-3, 1e2, log=True)},
            {"C": 0.1},
        ),
    ]
    if n_rows <= 100_000:
        c.append(
            Candidate(
                "rf",
                "Random Forest",
                "Bagging",
                "tree",
                lambda p: RandomForestClassifier(
                    **{"n_estimators": 200, "min_samples_leaf": 2, "n_jobs": n_jobs, "random_state": seed, **p}
                ),
                _rf_space,
                {"max_depth": 8, "min_samples_leaf": 10},
            )
        )
    else:
        skipped.append({"name": "Random Forest", "reason": f"too slow on {n_rows:,} rows within the time budget"})
    c.append(
        Candidate(
            "hgb",
            "Hist Gradient Boosting",
            "Boosting",
            "tree",
            lambda p: HistGradientBoostingClassifier(**{"max_iter": 250, "learning_rate": 0.08, "random_state": seed, **p}),
            _hgb_space,
            {"max_leaf_nodes": 12, "min_samples_leaf": 60, "l2_regularization": 1.0},
        )
    )
    if HAS_LGBM:
        c.append(
            Candidate(
                "lgbm",
                "LightGBM",
                "Boosting",
                "tree",
                lambda p: LGBMClassifier(
                    **{
                        "n_estimators": 300,
                        "learning_rate": 0.05,
                        "num_leaves": 31,
                        "subsample": 0.9,
                        "subsample_freq": 1,
                        "colsample_bytree": 0.9,
                        "n_jobs": n_jobs,
                        "random_state": seed,
                        "verbose": -1,
                        **p,
                    }
                ),
                _lgbm_space,
                {"num_leaves": 15, "min_child_samples": 60, "reg_lambda": 5.0},
            )
        )
    if HAS_XGB:
        c.append(
            Candidate(
                "xgb",
                "XGBoost",
                "Boosting",
                "tree",
                lambda p: XGBClassifier(
                    **{
                        "n_estimators": 300,
                        "learning_rate": 0.05,
                        "max_depth": 6,
                        "subsample": 0.9,
                        "colsample_bytree": 0.9,
                        "n_jobs": n_jobs,
                        "random_state": seed,
                        "tree_method": "hist",
                        "verbosity": 0,
                        **p,
                    }
                ),
                _xgb_space,
                {"max_depth": 4, "min_child_weight": 10},
            )
        )
    if n_rows < 150:
        c = [x for x in c if x.key in ("baseline", "logreg", "rf")]
        skipped.append({"name": "Gradient boosting", "reason": f"only {n_rows} rows: boosting overfits this little data"})
    return c, skipped


def regression_candidates(n_rows: int, seed: int, n_jobs: int) -> tuple[list[Candidate], list[dict[str, str]]]:
    skipped: list[dict[str, str]] = []
    c = [
        Candidate("baseline", "Baseline (mean)", "Baseline", "tree", lambda p: DummyRegressor(strategy="mean"), baseline=True),
        Candidate(
            "ridge",
            "Ridge Regression",
            "Linear",
            "linear",
            lambda p: Ridge(**{"alpha": 1.0, **p}),
            lambda t: {"alpha": t.suggest_float("alpha", 1e-3, 1e3, log=True)},
            {"alpha": 30.0},
        ),
    ]
    if n_rows <= 100_000:
        c.append(
            Candidate(
                "rf",
                "Random Forest",
                "Bagging",
                "tree",
                lambda p: RandomForestRegressor(
                    **{"n_estimators": 200, "min_samples_leaf": 2, "n_jobs": n_jobs, "random_state": seed, **p}
                ),
                _rf_space,
                {"max_depth": 8, "min_samples_leaf": 10},
            )
        )
    else:
        skipped.append({"name": "Random Forest", "reason": f"too slow on {n_rows:,} rows within the time budget"})
    c.append(
        Candidate(
            "hgb",
            "Hist Gradient Boosting",
            "Boosting",
            "tree",
            lambda p: HistGradientBoostingRegressor(**{"max_iter": 250, "learning_rate": 0.08, "random_state": seed, **p}),
            _hgb_space,
            {"max_leaf_nodes": 12, "min_samples_leaf": 60, "l2_regularization": 1.0},
        )
    )
    if HAS_LGBM:
        c.append(
            Candidate(
                "lgbm",
                "LightGBM",
                "Boosting",
                "tree",
                lambda p: LGBMRegressor(
                    **{
                        "n_estimators": 300,
                        "learning_rate": 0.05,
                        "num_leaves": 31,
                        "subsample": 0.9,
                        "subsample_freq": 1,
                        "colsample_bytree": 0.9,
                        "n_jobs": n_jobs,
                        "random_state": seed,
                        "verbose": -1,
                        **p,
                    }
                ),
                _lgbm_space,
                {"num_leaves": 15, "min_child_samples": 60, "reg_lambda": 5.0},
            )
        )
    if HAS_XGB:
        c.append(
            Candidate(
                "xgb",
                "XGBoost",
                "Boosting",
                "tree",
                lambda p: XGBRegressor(
                    **{
                        "n_estimators": 300,
                        "learning_rate": 0.05,
                        "max_depth": 6,
                        "subsample": 0.9,
                        "colsample_bytree": 0.9,
                        "n_jobs": n_jobs,
                        "random_state": seed,
                        "tree_method": "hist",
                        "verbosity": 0,
                        **p,
                    }
                ),
                _xgb_space,
                {"max_depth": 4, "min_child_weight": 10},
            )
        )
    if n_rows < 150:
        c = [x for x in c if x.key in ("baseline", "ridge", "rf")]
        skipped.append({"name": "Gradient boosting", "reason": f"only {n_rows} rows: boosting overfits this little data"})
    return c, skipped


def shortlist_reason(problem: str, n_rows: int, n_cols: int, n_cat: int, skipped: list[dict[str, str]]) -> str:
    size = "small" if n_rows < 2000 else "medium-sized" if n_rows < 100_000 else "large"
    lin = "Ridge Regression" if problem == "regression" else "Logistic Regression"
    parts = [
        f"This is a {size} table ({n_rows:,} rows, {n_cols} input columns, {n_cat} categorical), where tree ensembles and gradient boosting usually win.",
        f"{lin} stays in as an interpretable reference, and a dummy baseline sets the bar every model must beat.",
    ]
    out = ["SVMs and k-NN scale poorly with rows and mixed types", "a neural network rarely beats boosting on tabular data this size"]
    out += [f"{s['name']}: {s['reason']}" for s in skipped]
    parts.append("Skipped: " + "; ".join(out) + ".")
    return " ".join(parts)
