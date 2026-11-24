import pandas as pd

from datalyon.ml.detection import detect, name_leakage_suspects, positive_class, problem_for_target, target_candidates
from datalyon.ml.profiling import profile_dataframe
from datalyon.ml.strategy import choose_strategy, horizon_from_goal, infer_frequency
from tests.unit.data import churn_df, multiclass_df, regression_df, weekly_panel


def test_detects_binary_target_from_goal_wording() -> None:
    df = churn_df()
    det = detect("Predict which customers will churn", profile_dataframe(df), df)
    assert det.problem_type == "binary" and det.target == "churned"
    assert det.positive_class == "Yes"
    assert det.entity_column == "customer_id"
    assert any(s["column"] == "cancel_reason" for s in det.leakage_suspects)
    assert det.confidence >= 0.8


def test_detects_regression_and_multiclass() -> None:
    reg = regression_df()
    assert detect("Estimate the price of a home", profile_dataframe(reg), reg).problem_type == "regression"
    mc = multiclass_df()
    det = detect("Predict the kind of product", profile_dataframe(mc), mc)
    assert det.problem_type == "multiclass" and det.target == "kind"


def test_detects_multi_series_forecasting() -> None:
    df = weekly_panel()
    det = detect("Forecast weekly units for the next 8 weeks", profile_dataframe(df), df)
    assert det.problem_type == "forecasting"
    assert det.target == "units" and det.time_column == "week" and det.series_column == "store"


def test_ambiguous_goal_asks_for_target() -> None:
    df = regression_df().rename(columns={"price": "zz"})
    det = detect("Help me with this data", profile_dataframe(df), df)
    assert det.target is None and det.clarification
    assert det.candidate_targets


def test_helpers() -> None:
    df = churn_df()
    prof = profile_dataframe(df)
    assert "churned" in target_candidates(prof)[:3]
    assert problem_for_target(prof, "sessions") == "regression"
    assert positive_class(df, "churned") == "Yes"
    assert name_leakage_suspects(profile_dataframe(regression_df()), "price") == []


def test_strategy_rules() -> None:
    df = churn_df()
    s = choose_strategy("binary", df, "churned", None, 60, "Yes")
    assert s.metric in ("roc_auc", "pr_auc") and s.validation_kind == "stratified_kfold"
    rare = df.assign(churned=["Yes" if i % 20 == 0 else "No" for i in range(len(df))])
    assert choose_strategy("binary", rare, "churned", None, 60, "Yes").metric == "pr_auc"
    timed = choose_strategy("binary", df, "churned", "signup", 60, "Yes")
    assert timed.validation_kind == "time_series"
    assert choose_strategy("multiclass", multiclass_df(), "kind", None, 60).metric == "macro_f1"
    skewed = pd.DataFrame({"y": [1, 2, 3, 4, 5] * 20 + [10_000] * 3, "x": range(103)})
    assert choose_strategy("regression", skewed, "y", None, 60).metric == "mae"
    assert choose_strategy("regression", regression_df(), "price", None, 60).metric == "rmse"
    fc = choose_strategy("forecasting", weekly_panel(), "units", "week", 60, horizon=8, freq="W")
    assert fc.validation_kind == "rolling_origin" and fc.metric in ("mape", "wape") and fc.season == 52


def test_frequency_and_horizon() -> None:
    assert infer_frequency(pd.Series(pd.date_range("2024-01-01", periods=30, freq="D"))) == "D"
    assert infer_frequency(pd.Series(pd.date_range("2024-01-01", periods=30, freq="W-MON"))) == "W"
    assert infer_frequency(pd.Series(pd.date_range("2024-01-01", periods=30, freq="MS"))) == "M"
    assert infer_frequency(pd.Series(pd.to_datetime(["2024-01-01", "2024-01-01"]))) is None
    assert horizon_from_goal("forecast the next quarter", "W") == 13
    assert horizon_from_goal("next 8 weeks please", "W") == 8
    assert horizon_from_goal("sales by store", "W") is None
