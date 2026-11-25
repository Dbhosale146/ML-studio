"""The only view of the data the LLM ever gets: the goal, column names/roles, summary statistics and at most
5 sample rows, with personal-looking values (emails, phones, names, identifiers) masked."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE = re.compile(r"\+?\d[\d\s().-]{6,}\d")
MASK = "[masked]"


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def mask_value(v: Any, pii: bool) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return ""
    if pii:
        return MASK
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    s = _fmt(v)
    s = EMAIL.sub(MASK, s)
    s = PHONE.sub(MASK, s) if not re.fullmatch(r"-?\d+(\.\d+)?", s) else s
    return s[:40]


def safe_summary(df: pd.DataFrame, profile: dict[str, Any], sample_rows: int = 5, max_columns: int = 80) -> dict[str, Any]:
    cols = profile["columns"][:max_columns]
    masked = {c["name"] for c in cols if c.get("pii") or c["kind"] == "id"}
    columns = []
    for c in cols:
        entry: dict[str, Any] = {"name": c["name"], "role": c["kind"], "missing_pct": c["missing_pct"], "unique": c["unique"]}
        if c["kind"] in ("number", "integer") and "min" in c:
            entry.update(min=round(c["min"], 4), median=round(c["median"], 4), max=round(c["max"], 4))
        elif c["kind"] == "date" and "min" in c:
            entry.update(min=str(c["min"])[:10], max=str(c["max"])[:10])
        elif c.get("top_values") and c["name"] not in masked:
            entry["top_values"] = [f"{mask_value(t['value'], False)} ({t['pct']}%)" for t in c["top_values"][:6]]
        if c["name"] in masked:
            entry["masked"] = True
        columns.append(entry)
    n = min(sample_rows, 5, len(df))
    sample = df.sample(n, random_state=0) if n else df.head(0)
    rows = [{c["name"]: mask_value(r.get(c["name"]), c["name"] in masked) for c in cols} for _, r in sample.iterrows()]
    return {"rows": profile["rows"], "columns": columns, "sample_rows": rows}


def summary_text(summary: dict[str, Any]) -> str:
    lines = [f"{summary['rows']} rows, {len(summary['columns'])} columns"]
    for c in summary["columns"]:
        extra = ""
        if "min" in c:
            extra = f" range {c['min']}..{c['max']}" + (f" median {c['median']}" if "median" in c else "")
        elif "top_values" in c:
            extra = " values: " + ", ".join(c["top_values"])
        lines.append(
            f"- {c['name']} [{c['role']}] missing {c['missing_pct']}% unique {c['unique']}{extra}{' (masked)' if c.get('masked') else ''}"
        )
    if summary["sample_rows"]:
        lines.append("sample rows:")
        lines += [str(r) for r in summary["sample_rows"]]
    return "\n".join(lines)
