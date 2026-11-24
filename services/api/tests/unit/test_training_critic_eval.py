import time

import numpy as np
import pytest

from datalyon.ml.critic import apply_fixes, calibration_curve, run_critic
from datalyon.ml.evaluation import choose_final, tune_threshold
from datalyon.ml.pipeline import build_tabular_artifact, candidates_for, round2_keys, supervised_round
from datalyon.ml.profiling import profile_dataframe
from datalyon.ml.strategy import choose_strategy
from datalyon.ml.training import TrainingError, evaluate_derived, prepare_task, tune
from datalyon.ml.types import TrainConfig
from tests.unit.data import churn_df, multiclass_df, regression_df


def _task(df, ptype, target, time_col=None, pos=None):  # type: ignore[no-untyped-def]
    prof = profile_dataframe(df)
    strat = choose_strategy(ptype, df, target, time_col, 30, pos)
    return prepare_task(df, prof, ptype, target, strat, [], time_col, pos, 42), prof


def test_prepare_task_splits_and_encodes() -> None:
    task, _ = _task(churn_df(), "binary", "churned", pos="Yes")
    assert task.classes == ["No", "Yes"] and task.pos_index == 1
    assert len(task.X_hold) == pytest.approx(0.2 * 600, abs=2)
    assert "customer_id" not in task.columns and "churned" not in task.columns
    timed, _ = _task(churn_df(), "binary", "churned", time_col="signup", pos="Yes")
    assert timed.X_train["signup"].max() <= timed.X_hold["signup"].min()
    with pytest.raises(TrainingError):
        _task(churn_df(20), "binary", "churned", pos="Yes")


def test_round_critic_catches_leakage_and_round2_recovers() -> None:
    task, _ = _task(churn_df(), "binary", "churned", pos="Yes")
    cands, _ = candidates_for(task, 1)
    small = {k: v for k, v in cands.items() if k in ("baseline", "logreg", "hgb")}
    events: list[str] = []
    runs = supervised_round(task, small, TrainConfig(), 1, time.perf_counter() + 20, lambda k, p: events.append(k), tune_top=1)
    assert {"start", "fold", "run"} <= set(events)
    leader = max((r for r in runs if r.family != "Baseline"), key=lambda r: r.selection_score)
    assert leader.mean > 0.95  # inflated by the leaky cancel_reason column
    findings = run_critic(task, runs, small, TrainConfig(), [])
    leak = [f for f in findings if f.check == "leakage" and f.severity == "high"]
    assert leak and leak[0].evidence["column"] == "cancel_reason"
    cfg2, applied = apply_fixes(TrainConfig(), findings)
    assert "cancel_reason" in cfg2.exclude and applied
    assert round2_keys(runs, small, TrainConfig(), cfg2) == list(small)
    runs2 = supervised_round(task, small, cfg2, 2, time.perf_counter() + 20, tune_top=0)
    honest = max((r for r in runs2 if r.family != "Baseline"), key=lambda r: r.selection_score)
    assert honest.mean < leader.mean
    final = choose_final(task, runs2, small, cfg2)
    assert final.holdout["roc_auc"] > final.baseline["roc_auc"]
    assert {"roc", "pr", "confusion", "calibration", "scores"} <= set(final.charts)
    assert final.importance[0][0] in ("days_since_login", "sessions")
    art, schema, _ = build_tabular_artifact(task, final, small, churn_df(), {})
    assert {f["name"] for f in schema} >= {"days_since_login", "plan"} and "cancel_reason" not in {f["name"] for f in schema}
    assert art["examples"] and art["reference"]


def test_regression_and_multiclass_final() -> None:
    task, _ = _task(regression_df(), "regression", "price")
    cands, _ = candidates_for(task, 1)
    small = {k: v for k, v in cands.items() if k in ("baseline", "ridge", "hgb")}
    runs = supervised_round(task, small, TrainConfig(), 1, time.perf_counter() + 20, tune_top=0)
    final = choose_final(task, runs, small, TrainConfig())
    assert final.holdout["r2"] > 0.7 and {"scatter", "residuals", "residual_quantiles"} <= set(final.charts)
    mtask, _ = _task(multiclass_df(), "multiclass", "kind")
    mc, _ = candidates_for(mtask, 1)
    msmall = {k: v for k, v in mc.items() if k in ("baseline", "logreg", "rf")}
    mruns = supervised_round(mtask, msmall, TrainConfig(), 1, time.perf_counter() + 20, tune_top=0)
    mfinal = choose_final(mtask, mruns, msmall, TrainConfig())
    assert mfinal.holdout["macro_f1"] > 0.8 and len(mfinal.charts["per_class"]) == 3


def test_baseline_check_stops_on_noise() -> None:
    rng = np.random.default_rng(12345)  # independent of the generator that built the features
    df = churn_df(400, leak=False)
    df["churned"] = rng.choice(["Yes", "No"], len(df))
    task, _ = _task(df, "binary", "churned", pos="Yes")
    cands, _ = candidates_for(task, 1)
    small = {k: v for k, v in cands.items() if k in ("baseline", "logreg")}
    runs = supervised_round(task, small, TrainConfig(), 1, time.perf_counter() + 20, tune_top=0)
    findings = run_critic(task, runs, small, TrainConfig(), [])
    assert any(f.check == "baseline" and f.fix and f.fix.get("stop") for f in findings)


def test_tune_and_derived_acceptance() -> None:
    task, _ = _task(regression_df(), "regression", "price")
    cands, _ = candidates_for(task, 1)
    params, score, trials = tune(task, cands["ridge"], TrainConfig(), 3, 4)
    assert trials >= 1 and "alpha" in params and np.isfinite(score)
    spec = {"name": "area_x_rooms", "op": "product", "a": "area", "b": "rooms"}
    accepted = evaluate_derived(task, cands["ridge"], TrainConfig(), [spec])
    assert all(gain > 0 for _, gain in accepted)


def test_threshold_and_calibration_helpers() -> None:
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.1).astype(int)
    p = np.clip(y * 0.5 + rng.random(500) * 0.5, 0, 1)
    t = tune_threshold(p, y)
    assert 0.3 < t < 0.9
    ece, curve = calibration_curve(y, p)
    assert ece > 0 and curve
