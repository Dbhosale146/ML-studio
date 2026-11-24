"""Full engine runs on every sample file: each model must beat its baseline, and the critic must catch the leaky churn column."""

from pathlib import Path

import pytest

from datalyon.ml.ingest import read_file, sniff_file_type
from datalyon.ml.pipeline import run_all
from tests.conftest import SAMPLES

CASES = [
    ("customer_churn.xlsx", "Predict which customers will churn", "binary", "churned"),
    ("card_fraud.xlsx", "Detect fraudulent transactions", "binary", "is_fraud"),
    ("house_prices.xlsx", "Predict the sale price of a house", "regression", "sale_price"),
    ("product_category.xlsx", "Predict the product category", "multiclass", "category"),
    ("store_weekly_demand.xlsx", "Forecast weekly units sold for the next 8 weeks", "forecasting", "units_sold"),
]


def _load(name: str):  # type: ignore[no-untyped-def]
    path: Path = SAMPLES / name
    if not path.exists():
        pytest.skip("run `make samples` first")
    sheets = read_file(path, sniff_file_type(path.name, path.read_bytes()[:16]), 500_000, 300)
    return max(sheets, key=lambda s: len(s.df)).df


@pytest.mark.slow
@pytest.mark.parametrize(("file", "goal", "ptype", "target"), CASES)
def test_sample_beats_baseline(file: str, goal: str, ptype: str, target: str) -> None:
    res = run_all(_load(file), goal, budget_s=25, n_jobs=2)
    assert res.detection.problem_type == ptype
    assert res.detection.target == target
    primary = res.metrics[0]
    assert primary["good"] is True, f"{file}: {primary}"
    assert res.stopped is None
    if file == "customer_churn.xlsx":
        assert "cancellation_reason" in res.excluded
        assert any(
            f.check == "leakage" and f.severity == "high" and f.evidence.get("column") == "cancellation_reason" for f in res.findings
        )
