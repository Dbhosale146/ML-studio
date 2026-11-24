"""Feature building inside the saved sklearn Pipeline.

FeatureBuilder is fitted on training folds only and travels with the model, so raw rows at
prediction time get exactly the same coercion, label normalisation, rare-category grouping,
outlier capping, date parts (with cyclical encodings), missing flags and derived features.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

TRUE_WORDS = {"true", "yes", "y", "1", "1.0", "t", "on"}
FALSE_WORDS = {"false", "no", "n", "0", "0.0", "f", "off"}
DERIVED_OPS = {"ratio", "diff", "product", "log1p", "sum"}
OTHER = "__other__"


def _blank(v: Any) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v)) or (isinstance(v, str) and v.strip() == "") or v is pd.NaT


def coerce_frame(X: Any, columns: list[str], kinds: dict[str, str]) -> pd.DataFrame:
    """Raw rows (DataFrame or list of dicts) -> typed frame with exactly `columns`."""
    X = pd.DataFrame(X).copy()
    for c in columns:
        if c not in X.columns:
            X[c] = np.nan
    X = X[list(columns)]
    for c in columns:
        k = kinds.get(c, "category")
        s = X[c]
        if k in ("number", "integer"):
            if not pd.api.types.is_numeric_dtype(s):  # object or pandas 3 "str" dtype
                s = s.map(lambda v: np.nan if _blank(v) else re.sub(r"[,\s$€£₹%]", "", str(v)) if isinstance(v, str) else v)
            X[c] = pd.to_numeric(s, errors="coerce").astype(float)
        elif k == "boolean":

            def to_b(v: Any) -> float:
                if _blank(v):
                    return np.nan
                if isinstance(v, (bool, np.bool_)):
                    return float(v)
                t = str(v).strip().lower()
                return 1.0 if t in TRUE_WORDS else 0.0 if t in FALSE_WORDS else np.nan

            X[c] = s.map(to_b).astype(float)
        elif k == "date":
            X[c] = pd.to_datetime(s, errors="coerce", format="mixed")
        else:
            X[c] = s.map(
                lambda v: np.nan if _blank(v) else (str(int(v)) if isinstance(v, float) and v.is_integer() else str(v).strip())
            ).astype(object)
    return X


class FeatureBuilder(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    def __init__(
        self,
        columns: list[str] | None = None,
        kinds: dict[str, str] | None = None,
        derived: list[dict[str, Any]] | None = None,
        cap_outliers: bool = True,
        missing_flags: bool = True,
        normalise_labels: bool = True,
        group_rare: bool = True,
    ) -> None:
        self.columns = columns
        self.kinds = kinds
        self.derived = derived
        self.cap_outliers = cap_outliers
        self.missing_flags = missing_flags
        self.normalise_labels = normalise_labels
        self.group_rare = group_rare

    def fit(self, X: Any, y: Any = None) -> FeatureBuilder:
        cols = list(self.columns or [])
        kinds = dict(self.kinds or {})
        X = coerce_frame(X, cols, kinds)
        self.num_cols_ = [c for c in cols if kinds.get(c) in ("number", "integer")]
        self.bool_cols_ = [c for c in cols if kinds.get(c) == "boolean"]
        self.date_cols_ = [c for c in cols if kinds.get(c) == "date"]
        self.cat_cols_ = [c for c in cols if c not in self.num_cols_ + self.bool_cols_ + self.date_cols_]
        self.cat_maps_: dict[str, dict[str, str]] = {}
        self.keep_cats_: dict[str, set[str]] = {}
        for c in self.cat_cols_:
            vals = X[c].dropna().astype(str)
            if vals.empty:
                continue
            if self.normalise_labels:
                canon = vals.groupby(vals.str.lower()).agg(lambda s: s.value_counts().index[0])
                self.cat_maps_[c] = {str(k): str(v) for k, v in canon.to_dict().items()}
                vals = vals.str.lower().map(self.cat_maps_[c])
            if self.group_rare:  # same rule the cleaning plan shows the user: columns with 10+ values, labels under 1%
                share = vals.value_counts(normalize=True)
                keep = share[share >= 0.01] if len(share) >= 10 else share
                if len(keep) < len(share):
                    self.keep_cats_[c] = set(map(str, keep.index))
        self.clip_: dict[str, tuple[float, float]] = {}
        if self.cap_outliers:
            for c in self.num_cols_:
                v = X[c].dropna()
                if len(v) < 30:
                    continue
                q1, q3 = np.percentile(v, [25, 75])
                iqr = q3 - q1
                if iqr <= 0:
                    continue
                if ((v < q1 - 3 * iqr) | (v > q3 + 3 * iqr)).mean() > 0.002:
                    lo, hi = np.percentile(v, [1, 99])
                    self.clip_[c] = (float(lo), float(hi))
        self.miss_cols_ = [c for c in cols if self.missing_flags and X[c].isna().mean() >= 0.05]
        self.derived_ = [d for d in (self.derived or []) if self._valid(d)]
        out = self._build(X.head(5))
        self.feature_names_out_ = list(out.columns)
        self.source_map_ = self._source_map()
        return self

    def _valid(self, d: dict[str, Any]) -> bool:
        if d.get("op") not in DERIVED_OPS:
            return False
        refs = [d.get("a")] + ([d.get("b")] if d["op"] != "log1p" else [])
        return all(r in self.num_cols_ for r in refs)

    def _source_map(self) -> dict[str, list[str]]:
        m: dict[str, list[str]] = {}
        for c in self.num_cols_ + self.bool_cols_ + self.cat_cols_:
            m[c] = [c]
        for c in self.date_cols_:
            for part in ("year", "month", "dayofweek", "is_weekend", "month_sin", "month_cos", "dow_sin", "dow_cos"):
                m[f"{c}__{part}"] = [c]
        for c in self.miss_cols_:
            m[f"{c}__missing"] = [c]
        for d in self.derived_:
            m[d["name"]] = [d["a"]] + ([d["b"]] if d.get("b") else [])
        return m

    def _build(self, X: pd.DataFrame) -> pd.DataFrame:
        out: dict[str, Any] = {}
        for c in self.num_cols_:
            v = X[c].astype(float)
            if c in self.clip_:
                lo, hi = self.clip_[c]
                v = v.clip(lo, hi)
            out[c] = v
        for c in self.bool_cols_:
            out[c] = X[c].astype(float)
        for c in self.cat_cols_:
            m = self.cat_maps_.get(c)
            keep = self.keep_cats_.get(c)

            def norm(v: Any, m: dict[str, str] | None = m, keep: set[str] | None = keep) -> Any:
                if not isinstance(v, str):
                    return np.nan
                s = m.get(v.lower(), v) if m is not None else v
                if keep is not None and s not in keep:
                    return OTHER
                return s

            out[c] = X[c].map(norm).astype(object)
        for c in self.date_cols_:
            d = X[c]
            month, dow = d.dt.month.astype(float), d.dt.dayofweek.astype(float)
            out[f"{c}__year"] = d.dt.year.astype(float)
            out[f"{c}__month"] = month
            out[f"{c}__dayofweek"] = dow
            out[f"{c}__is_weekend"] = (dow >= 5).astype(float).where(d.notna())
            out[f"{c}__month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
            out[f"{c}__month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
            out[f"{c}__dow_sin"] = np.sin(2 * np.pi * dow / 7)
            out[f"{c}__dow_cos"] = np.cos(2 * np.pi * dow / 7)
        for c in self.miss_cols_:
            out[f"{c}__missing"] = X[c].isna().astype(float)
        for d in self.derived_:
            a = X[d["a"]].astype(float)
            if d["op"] == "log1p":
                out[d["name"]] = np.log1p(a.clip(lower=0))
                continue
            b = X[d["b"]].astype(float)
            if d["op"] == "ratio":
                out[d["name"]] = (a / b.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
            elif d["op"] == "diff":
                out[d["name"]] = a - b
            elif d["op"] == "sum":
                out[d["name"]] = a + b
            else:
                out[d["name"]] = a * b
        return pd.DataFrame(out, index=X.index)

    def transform(self, X: Any) -> pd.DataFrame:
        return self._build(coerce_frame(X, list(self.columns or []), dict(self.kinds or {})))

    def get_feature_names_out(self, input_features: Any = None) -> Any:
        return np.array(self.feature_names_out_, dtype=object)


def describe_derived(d: dict[str, Any]) -> str:
    sym = {"ratio": "÷", "diff": "−", "product": "×", "sum": "+"}
    if d["op"] == "log1p":
        return f"log(1 + {d['a']})"
    return f"{d['a']} {sym[d['op']]} {d['b']}"


def validate_derived(specs: list[dict[str, Any]], columns: list[str], kinds: dict[str, str], target: str | None) -> list[dict[str, Any]]:
    """Keep only well-formed specs over existing numeric input columns (LLM output is never executed)."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    nums = {c for c in columns if kinds.get(c) in ("number", "integer") and c != target}
    for d in specs or []:
        if not isinstance(d, dict):
            continue
        op = str(d.get("op", "")).lower()
        a, b = d.get("a"), d.get("b")
        if op not in DERIVED_OPS or a not in nums or (op != "log1p" and (b not in nums or b == a)):
            continue
        raw_name = str(d.get("name") or f"{a}_{op}_{b or ''}")
        name = re.sub(r"[^a-zA-Z0-9_]+", "_", raw_name).strip("_")[:48] or f"{a}_{op}"
        if name in seen or name in columns:
            continue
        seen.add(name)
        out.append(
            {
                "name": name,
                "op": op,
                "a": a,
                "b": b if op != "log1p" else None,
                "why": str(d.get("why", ""))[:200],
                "llm": bool(d.get("llm", False)),
            }
        )
    return out[:6]


def rule_based_derived(
    columns: list[str], kinds: dict[str, str], profile_cols: list[dict[str, Any]], target: str | None
) -> list[dict[str, Any]]:
    """Deterministic suggestions: log of heavily skewed positive amounts."""
    by_name = {c["name"]: c for c in profile_cols}
    out = []
    for c in columns:
        info = by_name.get(c, {})
        if c != target and kinds.get(c) in ("number", "integer") and info.get("min", -1) >= 0 and abs(info.get("skew", 0)) > 2:
            out.append(
                {
                    "name": f"log_{c}",
                    "op": "log1p",
                    "a": c,
                    "b": None,
                    "why": f"{c} is heavily skewed (skew {info['skew']:.1f})",
                    "llm": False,
                }
            )
    return out[:3]
