import numpy as np
import pandas as pd

from datalyon.ml.profiling import column_kind, looks_pii, preview, profile_dataframe


def test_column_kinds() -> None:
    n = 100
    assert column_kind(pd.Series(range(n)), "row_id") == "id"
    assert column_kind(pd.Series(np.random.default_rng(0).normal(size=n)), "amount") == "number"
    assert column_kind(pd.Series(["Yes", "No"] * 50), "churned") == "boolean"
    assert column_kind(pd.Series([0, 1] * 50), "flag") == "boolean"
    assert column_kind(pd.Series(["a", "b", "c"] * 34), "plan") == "category"
    assert column_kind(pd.Series(["x"] * n), "constant_col") == "constant"
    assert column_kind(pd.Series([None] * n, dtype=object), "empty_col") == "empty"
    assert column_kind(pd.Series(pd.date_range("2024-01-01", periods=n)), "when") == "date"
    long_text = pd.Series([f"This is a long free text comment number {i} about the product" for i in range(n)])
    assert column_kind(long_text, "comment") == "text"


def test_pii_detection() -> None:
    assert looks_pii(pd.Series(["a@b.com", "c@d.org"]), "contact")
    assert looks_pii(pd.Series(["x", "y"]), "customer_name")
    assert not looks_pii(pd.Series(["Basic", "Pro"]), "plan")


def test_profile_reports_issues_and_quality() -> None:
    df = pd.DataFrame(
        {
            "id": range(60),
            "plan": ["Pro", "pro ", "Basic"] * 20,
            "spend": [float(i) for i in range(30)] + [None] * 30,
            "const": ["x"] * 60,
        }
    )
    df = pd.concat([df, df.head(3)])
    p = profile_dataframe(df, "f.xlsx", "Sheet1", ["note"])
    types = {i["type"] for i in p["issues"]}
    assert {"duplicates", "missing", "constant", "inconsistent_labels"} <= types
    assert 30 <= p["quality"] < 100
    assert p["quality_reasons"]
    spend = next(c for c in p["columns"] if c["name"] == "spend")
    assert spend["missing_pct"] > 40 and spend["histogram"]
    prev = preview(df, 5)
    assert prev["total_rows"] == 63 and len(prev["rows"]) == 5
