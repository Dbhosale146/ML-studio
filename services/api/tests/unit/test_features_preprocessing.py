import numpy as np
import pandas as pd

from datalyon.ml.features import FeatureBuilder, coerce_frame, describe_derived, rule_based_derived, validate_derived
from datalyon.ml.preprocessing import build_cleaning_plan, clean_rows, config_from_steps, make_pipeline
from datalyon.ml.profiling import profile_dataframe
from datalyon.ml.types import TrainConfig
from tests.unit.data import churn_df


def test_coerce_frame_handles_raw_form_values() -> None:
    kinds = {"n": "number", "b": "boolean", "d": "date", "c": "category"}
    X = coerce_frame([{"n": "$1,200", "b": "Yes", "d": "2024-02-03", "c": 3.0}, {"n": "", "b": "no"}], list(kinds), kinds)
    assert X["n"].tolist()[0] == 1200.0 and np.isnan(X["n"].iloc[1])
    assert X["b"].tolist() == [1.0, 0.0]
    assert X["d"].iloc[0] == pd.Timestamp("2024-02-03") and pd.isna(X["d"].iloc[1])
    assert X["c"].iloc[0] == "3" and pd.isna(X["c"].iloc[1])


def test_feature_builder_learns_on_fit_and_applies_on_transform() -> None:
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame(
        {
            "amount": np.r_[rng.normal(100, 10, n - 3), [10_000] * 3],
            "plan": rng.choice(["Pro", "pro ", "PRO", "Basic"], n).tolist(),
            "city": [f"c{i % 10}" for i in range(n - 2)] + ["tiny1", "tiny2"],
            "when": pd.date_range("2024-01-01", periods=n, freq="D"),
            "maybe": np.where(rng.random(n) < 0.3, np.nan, 1.0),
            "other": rng.normal(5, 1, n),
        }
    )
    kinds = {"amount": "number", "plan": "category", "city": "category", "when": "date", "maybe": "number", "other": "number"}
    derived = [
        {"name": "amount_per_other", "op": "ratio", "a": "amount", "b": "other"},
        {"name": "bad", "op": "ratio", "a": "plan", "b": "other"},
    ]
    fb = FeatureBuilder(list(kinds), kinds, derived).fit(df)
    out = fb.transform(df)
    assert out["amount"].max() < 10_000  # capped at the 99th percentile
    assert len(set(out["plan"].dropna())) == 2 and "Basic" in set(out["plan"])  # case/space variants collapsed to one label
    assert "__other__" in set(out["city"])  # rare categories grouped
    assert {"when__month_sin", "when__dow_cos", "maybe__missing", "amount_per_other"} <= set(out.columns)
    assert "bad" not in out.columns  # invalid derived spec ignored
    assert fb.source_map_["amount_per_other"] == ["amount", "other"]
    unseen = fb.transform(pd.DataFrame([{"plan": "brand-new", "amount": "50"}]))
    assert unseen.loc[0, "plan"] == "__other__" or unseen.loc[0, "plan"] == "brand-new"


def test_validate_and_describe_derived() -> None:
    cols, kinds = ["a", "b", "c"], {"a": "number", "b": "integer", "c": "category"}
    specs = [
        {"name": "a/b", "op": "ratio", "a": "a", "b": "b"},
        {"op": "log1p", "a": "a"},
        {"op": "exec", "a": "a", "b": "b"},
        {"op": "ratio", "a": "a", "b": "c"},
        {"op": "diff", "a": "a", "b": "target"},
    ]
    out = validate_derived(specs, cols, kinds, "target")
    assert [d["op"] for d in out] == ["ratio", "log1p"]
    assert out[0]["name"] == "a_b"
    assert describe_derived(out[0]) == "a ÷ b" and describe_derived(out[1]) == "log(1 + a)"
    prof = [{"name": "a", "min": 0, "skew": 5.0}, {"name": "b", "min": 0, "skew": 0.1}]
    assert [d["a"] for d in rule_based_derived(cols, kinds, prof, None)] == ["a"]


def test_cleaning_plan_counts_and_config() -> None:
    df = pd.concat([churn_df(200)] * 1 + [churn_df(200).head(5)])
    prof = profile_dataframe(df)
    steps = build_cleaning_plan(df, prof, "churned", ["cancel_reason"])
    ids = {s.id: s for s in steps}
    assert ids["drop_columns"].columns == ["customer_id"]
    assert ids["dedupe_rows"].count == 5 and ids["dedupe_rows"].optional
    assert ids["normalise_labels"].count > 0
    assert ids["date_parts"].columns == ["signup"]
    ids["normalise_labels"].enabled = False
    cfg = config_from_steps(steps, TrainConfig())
    assert cfg.normalise_labels is False and cfg.missing_flags is True
    clean, stats = clean_rows(df, "churned", dedupe=True)
    assert stats["dropped_duplicates"] == 5 and len(clean) == 200


def test_pipeline_fits_and_predicts_raw_rows() -> None:
    from sklearn.linear_model import LogisticRegression

    df = churn_df(300, leak=False)
    cols = ["plan", "days_since_login", "sessions", "nps", "signup"]
    kinds = {"plan": "category", "days_since_login": "integer", "sessions": "integer", "nps": "integer", "signup": "date"}
    pipe = make_pipeline(cols, kinds, TrainConfig(), "linear", LogisticRegression(max_iter=500))
    pipe.fit(df[cols], (df["churned"] == "Yes").astype(int))
    p = pipe.predict_proba(pd.DataFrame([{"plan": "pro", "days_since_login": "80", "sessions": 2}]))
    assert 0 <= p[0, 1] <= 1
