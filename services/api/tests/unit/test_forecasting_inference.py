import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from datalyon.ml import forecasting as fc
from datalyon.ml.evaluation import choose_final
from datalyon.ml.explain import permutation_shares, shap_drivers
from datalyon.ml.inference import batch_frame, forecast_view, predict_tabular, validate_rows
from datalyon.ml.pipeline import build_forecast_artifact, build_tabular_artifact, candidates_for, forecast_round, supervised_round
from datalyon.ml.profiling import profile_dataframe
from datalyon.ml.registry import Tracker, library_versions, load_artifact, save_version
from datalyon.ml.strategy import choose_strategy
from datalyon.ml.training import prepare_task
from datalyon.ml.types import TrainConfig
from tests.unit.data import churn_df, weekly_panel


def _fc_task(stores: int = 3) -> fc.ForecastTask:
    df = weekly_panel(stores)
    df = df.drop(index=[10, 11])  # a gap to interpolate
    df = pd.concat([df, df.head(3)])  # duplicate periods to aggregate
    return fc.prepare_forecast(df, "units", "week", "store", "W", 8, "mape", 0)


def test_prepare_forecast_regularises_panel() -> None:
    task = _fc_task()
    assert task.series == ["S0", "S1", "S2"]
    assert not task.panel["y"].isna().any()
    issues = " ".join(n["issue"] for n in task.notes)
    assert "share a week" in issues and "missing" in issues
    with pytest.raises(fc.ForecastError):
        fc.prepare_forecast(weekly_panel(1, 10), "units", "week", None, "W", 8, "mape", 0)


def test_lags_never_leak_the_future() -> None:
    task = _fc_task()
    cfg = fc.default_config(task)
    assert min(cfg.lags) >= task.horizon
    feat = fc.build_features(task.panel, task, cfg, {s: i for i, s in enumerate(task.series)})
    s0 = feat[feat["series"] == "S0"].reset_index(drop=True)
    k = cfg.lags[0]
    assert s0.loc[k + 5, f"lag_{k}"] == pytest.approx(s0.loc[5, "y"])


def test_backtest_beats_naive_and_artifact_forecasts() -> None:
    task = _fc_task()
    cfg = fc.default_config(task)
    windows = fc.origins(task)
    assert 1 <= len(windows) <= 4
    runs, preds = forecast_round(task, cfg, windows, 1, time.perf_counter() + 25, 1)
    base = next(r for r in runs if r.family == "Baseline")
    best = max((r for r in runs if r.family != "Baseline"), key=lambda r: r.selection_score)
    assert best.mean < base.mean * 1.2
    findings = fc.critic(task, runs, preds, cfg)
    assert findings[0].check == "leakage" and findings[0].severity == "pass"
    cfg2, _ = fc.apply_fixes(cfg, findings)
    art, metrics, algo = build_forecast_artifact(task, runs, preds, 1)
    assert metrics[0]["key"] == "mape" and algo
    view = forecast_view(art, "S1", 5)
    assert view["horizon"] == 5 and len(view["table"]) == 5
    assert all(r["lower_80"] <= r["forecast"] <= r["upper_80"] for r in view["table"])
    total = forecast_view(art, None, None)
    assert total["series"] == "Total" and len(total["table"]) == 8
    assert cfg2.lags == cfg.lags


def test_inference_schema_validation_and_batch(tmp_path: Path) -> None:
    df = churn_df(400, leak=False)
    prof = profile_dataframe(df)
    strat = choose_strategy("binary", df, "churned", None, 30, "Yes")
    task = prepare_task(df, prof, "binary", "churned", strat, [], None, "Yes", 0)
    cands, _ = candidates_for(task, 1)
    small = {k: v for k, v in cands.items() if k in ("baseline", "logreg", "lgbm")}
    runs = supervised_round(task, small, TrainConfig(), 1, time.perf_counter() + 20, tune_top=0)
    final = choose_final(task, runs, small, TrainConfig())
    art, schema, _ = build_tabular_artifact(task, final, small, df, {"days_since_login": {"label": "Days since login", "help": ""}})
    by = {f["name"]: f for f in schema}
    assert by["plan"]["kind"] == "category" and {o["value"] for o in by["plan"]["options"]} >= {"Basic", "Pro"}
    assert by["days_since_login"]["label"] == "Days since login" and "min" in by["days_since_login"]
    X, warns = validate_rows([{"days_since_login": "500", "plan": "Gold", "sessions": "abc", "bogus": 1}], schema)
    text = " ".join(warns[0])
    assert "outside the training range" in text and "never seen" in text and "not a number" in text and "unknown" in text
    results, explainer = predict_tabular(art, X, explain=True)
    r = results[0]
    assert r["prediction"] in ("Yes", "No") and set(r["probabilities"]) == {"Yes", "No"}
    assert r["risk_band"] in ("High", "Medium", "Low") and r["drivers"] and explainer
    out, report = batch_frame(art, df.drop(columns=["churned"]).head(50))
    assert {"predicted_churned", "probability_Yes", "risk_band"} <= set(out.columns) and report["rows"] == 50
    path = save_version(tmp_path, "m1", 1, art, {"name": "x"}, {"fields": schema})
    assert load_artifact(path)["problem_type"] == "binary"
    assert "scikit-learn" in library_versions()
    Tracker(False, "", "x").log_run("r", {}, {})


def test_explain_helpers() -> None:
    from lightgbm import LGBMRegressor

    from datalyon.ml.preprocessing import make_pipeline

    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300), "c": rng.choice(["x", "y"], 300)})
    y = 3 * X["a"].to_numpy() + rng.normal(0, 0.1, 300)
    pipe = make_pipeline(
        ["a", "b", "c"], {"a": "number", "b": "number", "c": "category"}, TrainConfig(), "tree", LGBMRegressor(n_estimators=50, verbose=-1)
    )
    pipe.fit(X, y)
    shares = permutation_shares(lambda Xp, yp: -float(np.mean((pipe.predict(Xp) - yp) ** 2)), X, y)
    assert shares[0][0] == "a"
    out = shap_drivers(pipe, X.head(3), "regression")
    assert out is not None and out[0][0][0]["feature"] == "a"
