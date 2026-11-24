"""Dataset profiling: column kinds and roles, statistics, distributions, quality issues and a 0-100 quality score."""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd

ID_NAME = re.compile(r"(^|[_\s-])(id|uuid|guid|key|code|number|no|num)$|^(id|uuid|guid)([_\s-]|$)|_id$|Id$", re.I)
PII_NAME = re.compile(
    r"e-?mail|phone|mobile|\bcell\b|first.?name|last.?name|full.?name|^name$|customer.?name|address|street|ssn|passport|\bdob\b|birth", re.I
)
EMAIL_RE = re.compile(r"^[\w.+-]+@[\w-]+\.[\w.-]+$")
PHONE_RE = re.compile(r"^\+?[\d\s().-]{7,}$")
TEXT_MIN_LEN = 40

ROLE_LABEL = {
    "id": "Identifier",
    "number": "Numeric",
    "integer": "Numeric (integer)",
    "category": "Categorical",
    "boolean": "Boolean",
    "date": "Date / time",
    "text": "Free text",
    "constant": "Constant",
    "empty": "Empty",
}


def cell_str(v: Any, max_len: int = 60) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    if isinstance(v, pd.Timestamp):
        if pd.isna(v):
            return ""
        return v.strftime("%Y-%m-%d") if (v.hour, v.minute, v.second) == (0, 0, 0) else v.strftime("%Y-%m-%d %H:%M")
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return str(int(f)) if f.is_integer() and abs(f) < 1e15 else f"{f:.6g}"
    s = str(v)
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def column_kind(s: pd.Series, name: str) -> str:
    nn = s.dropna()
    n = len(nn)
    if n == 0:
        return "empty"
    nunique = int(nn.astype(str).nunique()) if not pd.api.types.is_numeric_dtype(s) else int(nn.nunique())
    if nunique <= 1:
        return "constant"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "date"
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_numeric_dtype(s):
        vals = nn.astype(float).to_numpy()
        is_int = bool(np.all(np.mod(vals, 1) == 0))
        ratio = nunique / n
        if is_int and ratio > 0.95 and n > 20:
            if ID_NAME.search(str(name)):
                return "id"
            sorted_vals = np.sort(vals)
            if np.all(np.diff(sorted_vals) == 1):
                return "id"
        if is_int and nunique == 2 and set(np.unique(vals)) <= {0.0, 1.0}:
            return "boolean"
        return "integer" if is_int else "number"
    as_str = nn.astype(str)
    ratio = nunique / n
    avg_len = float(as_str.str.len().mean())
    if ID_NAME.search(str(name)) and ratio > 0.9:
        return "id"
    if avg_len >= TEXT_MIN_LEN and ratio > 0.5:
        return "text"
    if ratio > 0.98 and n > 50:
        return "id"
    if nunique == 2 and {v.lower() for v in as_str.unique()} <= {"yes", "no", "true", "false", "y", "n", "t", "f"}:
        return "boolean"
    return "category"


def looks_pii(s: pd.Series, name: str) -> bool:
    if PII_NAME.search(str(name)):
        return True
    sample = s.dropna().astype(str).head(200)
    if sample.empty:
        return False
    return bool(
        sample.str.match(EMAIL_RE).mean() > 0.5 or (not pd.api.types.is_numeric_dtype(s) and sample.str.match(PHONE_RE).mean() > 0.8)
    )


def profile_dataframe(df: pd.DataFrame, file_name: str = "", sheet: str = "", notes: list[str] | None = None) -> dict[str, Any]:
    rows, cols = df.shape
    columns: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    total_missing = int(df.isna().sum().sum())
    try:
        dups = int(df.duplicated().sum())
    except TypeError:
        dups = int(df.astype(str).duplicated().sum())
    outlier_cols = inconsistent_cols = 0

    for c in df.columns:
        s = df[c]
        kind = column_kind(s, str(c))
        nn = s.dropna()
        miss = float(s.isna().mean() * 100) if rows else 0.0
        info: dict[str, Any] = {
            "name": str(c),
            "kind": kind,
            "role": ROLE_LABEL[kind],
            "missing_pct": round(miss, 1),
            "missing": int(s.isna().sum()),
            "unique": int(nn.astype(str).nunique()) if len(nn) else 0,
            "pii": looks_pii(s, str(c)),
            "sample": [cell_str(v, 40) for v in nn.head(3).tolist()],
        }
        if kind in ("number", "integer") and len(nn):
            v = nn.astype(float)
            q1, q3 = np.percentile(v, [25, 75])
            iqr = q3 - q1
            n_out = int(((v < q1 - 3 * iqr) | (v > q3 + 3 * iqr)).sum()) if iqr > 0 else 0
            hist_counts, edges = np.histogram(v, bins=min(20, max(5, int(np.sqrt(len(v))))))
            info.update(
                min=float(v.min()),
                max=float(v.max()),
                mean=float(v.mean()),
                median=float(v.median()),
                std=float(v.std()) if len(v) > 1 else 0.0,
                skew=float(v.skew()) if len(v) > 2 and v.std() > 0 else 0.0,
                outliers=n_out,
                histogram=[
                    {"x0": float(edges[i]), "x1": float(edges[i + 1]), "count": int(hist_counts[i])} for i in range(len(hist_counts))
                ],
            )
            if n_out > max(3, 0.002 * len(v)):
                outlier_cols += 1
                issues.append(
                    {"column": str(c), "type": "outliers", "severity": "low", "detail": f"{n_out:,} extreme values (beyond 3×IQR) in {c}"}
                )
        elif kind == "date" and len(nn):
            info.update(min=pd.Timestamp(nn.min()).isoformat(), max=pd.Timestamp(nn.max()).isoformat())
        elif kind in ("category", "boolean") and len(nn):
            vc = nn.astype(str).value_counts()
            info["top_values"] = [{"value": str(k), "count": int(n), "pct": round(100 * n / len(nn), 1)} for k, n in vc.head(10).items()]
            if kind == "category":
                norm = nn.astype(str).str.strip().str.lower()
                if norm.nunique() < nn.astype(str).nunique():
                    inconsistent_cols += 1
                    issues.append(
                        {
                            "column": str(c),
                            "type": "inconsistent_labels",
                            "severity": "low",
                            "detail": f"{c} has labels that differ only by case or spaces ({nn.astype(str).nunique()} raw → {norm.nunique()} distinct)",
                        }
                    )
        if miss >= 5:
            issues.append(
                {
                    "column": str(c),
                    "type": "missing",
                    "severity": "high" if miss >= 40 else "medium",
                    "detail": f"{miss:.1f}% missing in {c}",
                }
            )
        if kind == "constant":
            issues.append(
                {"column": str(c), "type": "constant", "severity": "low", "detail": f"{c} has a single value, so it carries no signal"}
            )
        if kind == "id":
            issues.append(
                {
                    "column": str(c),
                    "type": "id",
                    "severity": "info",
                    "detail": f"{c} looks like an identifier and will not be used as a feature",
                }
            )
        if kind == "text":
            issues.append(
                {
                    "column": str(c),
                    "type": "text",
                    "severity": "info",
                    "detail": f"{c} is free text and is not used as a feature in this version",
                }
            )
        columns.append(info)

    if dups:
        issues.insert(0, {"column": None, "type": "duplicates", "severity": "medium", "detail": f"{dups:,} duplicate rows"})

    cells = max(1, rows * cols)
    miss_pct = 100 * total_missing / cells
    dup_pct = 100 * dups / max(1, rows)
    n_constant = sum(1 for c in columns if c["kind"] in ("constant", "empty"))
    penalties = [
        ("Missing values", min(30.0, miss_pct * 1.5), f"{miss_pct:.1f}% of cells are empty"),
        ("Duplicate rows", min(15.0, dup_pct * 2), f"{dups:,} duplicate rows ({dup_pct:.1f}%)"),
        ("Constant columns", min(9.0, 3.0 * n_constant), f"{n_constant} column(s) with a single value"),
        ("Outliers", min(10.0, 2.0 * outlier_cols), f"{outlier_cols} column(s) with extreme values"),
        ("Inconsistent labels", min(8.0, 2.0 * inconsistent_cols), f"{inconsistent_cols} column(s) with case/space variants"),
    ]
    score = round(max(30.0, min(100.0, 100 - sum(p for _, p, _ in penalties))))
    reasons = [{"factor": f, "penalty": round(p, 1), "detail": d} for f, p, d in penalties if p > 0]

    date_cols = [c for c in columns if c["kind"] == "date" and "min" in c]
    period = None
    if date_cols:
        main = max(date_cols, key=lambda c: c["unique"])
        period = f"{pd.Timestamp(main['min']).strftime('%b %Y')} – {pd.Timestamp(main['max']).strftime('%b %Y')}"

    return {
        "file_name": file_name,
        "sheet": sheet,
        "rows": int(rows),
        "cols": int(cols),
        "duplicates": dups,
        "missing_pct": round(miss_pct, 2),
        "quality": score,
        "quality_reasons": reasons,
        "period": period,
        "columns": columns,
        "issues": issues,
        "notes": list(notes or []),
    }


def preview(df: pd.DataFrame, limit: int = 50) -> dict[str, Any]:
    rows = [[cell_str(v, 80) for v in row] for row in df.head(limit).itertuples(index=False, name=None)]
    return {"columns": [str(c) for c in df.columns], "rows": rows, "total_rows": len(df)}
