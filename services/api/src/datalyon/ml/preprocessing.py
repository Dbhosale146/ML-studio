"""Cleaning plan (proposed with real counts, approved by the user) and the sklearn preprocessing pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .features import FeatureBuilder
from .types import CleaningStep, TrainConfig

UNUSABLE = ("id", "text", "constant", "empty")


def input_columns(
    profile: dict[str, Any], target: str | None, exclude: list[str], time_column: str | None = None, include_time: bool = True
) -> list[str]:
    cols = []
    for c in profile["columns"]:
        n = c["name"]
        if n == target or n in exclude or c["kind"] in UNUSABLE:
            continue
        if n == time_column and not include_time:
            continue
        cols.append(n)
    return cols


def column_kinds(profile: dict[str, Any], columns: list[str]) -> dict[str, str]:
    by = {c["name"]: c["kind"] for c in profile["columns"]}
    return {c: by[c] for c in columns}


def build_cleaning_plan(df: pd.DataFrame, profile: dict[str, Any], target: str | None, exclude: list[str]) -> list[CleaningStep]:
    """Concrete fixes with counts. Required steps cannot be skipped; optional ones can."""
    steps: list[CleaningStep] = []
    cols = {c["name"]: c for c in profile["columns"]}

    unusable = [c["name"] for c in profile["columns"] if c["kind"] in UNUSABLE and c["name"] != target]
    if unusable:
        kinds = sorted({cols[c]["role"].lower() for c in unusable})
        steps.append(
            CleaningStep(
                "drop_columns",
                f"Drop {len(unusable)} column{'s' if len(unusable) > 1 else ''} with no usable signal",
                f"{', '.join(unusable[:6])} ({', '.join(kinds)})",
                len(unusable),
                optional=False,
                columns=unusable,
            )
        )
    if exclude:
        steps.append(
            CleaningStep(
                "drop_excluded",
                f"Leave out {len(exclude)} column{'s' if len(exclude) > 1 else ''} you excluded",
                ", ".join(exclude),
                len(exclude),
                optional=False,
                columns=list(exclude),
            )
        )
    if target and target in df:
        miss = int(df[target].isna().sum())
        if miss:
            steps.append(
                CleaningStep(
                    "drop_missing_target",
                    f"Drop {miss:,} rows without a value for {target}",
                    "A model can only learn from rows where the outcome is known.",
                    miss,
                    optional=False,
                )
            )
    try:
        dups = int(df.duplicated().sum())
    except TypeError:
        dups = int(df.astype(str).duplicated().sum())
    if dups:
        steps.append(
            CleaningStep(
                "dedupe_rows",
                f"Remove {dups:,} duplicate rows",
                "Exact duplicates over-weight some records and can leak between train and test splits.",
                dups,
                optional=True,
            )
        )
    usable = [c for c in profile["columns"] if c["name"] not in unusable and c["name"] != target and c["name"] not in exclude]
    missing_cols = [c for c in usable if c["missing"] > 0]
    if missing_cols:
        total = sum(int(c["missing"]) for c in missing_cols)
        steps.append(
            CleaningStep(
                "impute",
                f"Fill {total:,} missing values in {len(missing_cols)} column{'s' if len(missing_cols) > 1 else ''}",
                "Median for numbers, a 'missing' category for labels. Learned from training folds only.",
                total,
                optional=False,
                columns=[c["name"] for c in missing_cols],
            )
        )
        flagged = [c for c in missing_cols if c["missing_pct"] >= 5]
        if flagged:
            steps.append(
                CleaningStep(
                    "missing_flags",
                    f"Add missing-value flags for {len(flagged)} column{'s' if len(flagged) > 1 else ''}",
                    "Whether a value is missing is often a signal in itself (e.g. a skipped survey).",
                    len(flagged),
                    optional=True,
                    columns=[c["name"] for c in flagged],
                )
            )
    outlier_cols = [c for c in usable if c.get("outliers", 0) > max(3, 0.002 * profile["rows"])]
    if outlier_cols:
        n = sum(int(c["outliers"]) for c in outlier_cols)
        steps.append(
            CleaningStep(
                "cap_outliers",
                f"Cap {n:,} extreme values at the 1st/99th percentiles",
                ", ".join(c["name"] for c in outlier_cols[:6]),
                n,
                optional=True,
                columns=[c["name"] for c in outlier_cols],
            )
        )
    label_cols, label_rows = [], 0
    for c in usable:
        if c["kind"] != "category":
            continue
        s = df[c["name"]].dropna().astype(str)
        canon = s.groupby(s.str.lower().str.strip()).transform(lambda g: g.value_counts().index[0])
        changed = int((canon != s).sum())
        if changed:
            label_cols.append(c["name"])
            label_rows += changed
    if label_cols:
        steps.append(
            CleaningStep(
                "normalise_labels",
                f"Normalise {label_rows:,} category labels that differ only by case or spaces",
                ", ".join(label_cols),
                label_rows,
                optional=True,
                columns=label_cols,
            )
        )
    rare_cols, rare_rows = [], 0
    for c in usable:
        if c["kind"] != "category" or c["unique"] < 10:
            continue
        share = df[c["name"]].dropna().astype(str).str.lower().str.strip().value_counts(normalize=True)
        rare = share[share < 0.01]
        if len(rare):
            rare_cols.append(c["name"])
            rare_rows += int(df[c["name"]].astype(str).str.lower().str.strip().isin(rare.index).sum())
    if rare_cols:
        steps.append(
            CleaningStep(
                "group_rare",
                f"Group {rare_rows:,} rare category values (<1%) into 'Other'",
                ", ".join(rare_cols),
                rare_rows,
                optional=True,
                columns=rare_cols,
            )
        )
    dates = [c["name"] for c in usable if c["kind"] == "date"]
    if dates:
        steps.append(
            CleaningStep(
                "date_parts",
                f"Turn {len(dates)} date column{'s' if len(dates) > 1 else ''} into year, month, weekday and seasonal features",
                ", ".join(dates),
                len(dates),
                optional=False,
                columns=dates,
            )
        )
    return steps


def config_from_steps(steps: list[CleaningStep], base: TrainConfig) -> TrainConfig:
    on = {s.id for s in steps if s.enabled or not s.optional}
    present = {s.id for s in steps}
    cfg = TrainConfig.from_dict(base.to_dict())
    cfg.missing_flags = "missing_flags" in on or "missing_flags" not in present
    cfg.cap_outliers = "cap_outliers" in on or "cap_outliers" not in present
    cfg.normalise_labels = "normalise_labels" in on or "normalise_labels" not in present
    cfg.group_rare = "group_rare" in on or "group_rare" not in present
    return cfg


def clean_rows(df: pd.DataFrame, target: str | None, dedupe: bool) -> tuple[pd.DataFrame, dict[str, int]]:
    stats = {"rows_in": len(df), "dropped_duplicates": 0, "dropped_missing_target": 0}
    out = df
    if dedupe:
        before = len(out)
        try:
            out = out.drop_duplicates()
        except TypeError:
            out = out.loc[~out.astype(str).duplicated()]
        stats["dropped_duplicates"] = before - len(out)
    if target:
        before = len(out)
        out = out[out[target].notna()]
        stats["dropped_missing_target"] = before - len(out)
    stats["rows_out"] = len(out)
    return out.reset_index(drop=True), stats


def make_preprocessor(family: str) -> ColumnTransformer:
    if family == "linear":
        num: Any = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
        cat = Pipeline(
            [
                ("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
                ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=30, sparse_output=False)),
            ]
        )
    else:
        num = SimpleImputer(strategy="median", keep_empty_features=True)
        cat = Pipeline(
            [
                ("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
                ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1)),
            ]
        )
    return ColumnTransformer(
        [("num", num, make_column_selector(dtype_include=np.number)), ("cat", cat, make_column_selector(dtype_include=object))],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_pipeline(columns: list[str], kinds: dict[str, str], cfg: TrainConfig, family: str, estimator: Any) -> Pipeline:
    cols = [c for c in columns if c not in cfg.exclude]
    derived = [d for d in cfg.derived if d["a"] in cols and (not d.get("b") or d["b"] in cols)]
    fb = FeatureBuilder(
        columns=cols,
        kinds={c: kinds[c] for c in cols},
        derived=derived,
        cap_outliers=cfg.cap_outliers,
        missing_flags=cfg.missing_flags,
        normalise_labels=cfg.normalise_labels,
        group_rare=cfg.group_rare,
    )
    return Pipeline([("features", fb), ("prep", make_preprocessor(family)), ("model", estimator)])
