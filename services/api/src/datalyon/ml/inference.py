"""Inference: input schema, validation of raw rows, single/batch predictions with drivers, forecasts."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .explain import occlusion_drivers, shap_drivers
from .features import coerce_frame
from .metrics import fmt_num
from .profiling import cell_str


# ---------------------------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------------------------
def _nice(v: float, integer: bool) -> float | int:
    if integer:
        return round(v)
    if v == 0:
        return 0.0
    mag = 10 ** (math.floor(math.log10(abs(v))) - 2)
    return float(round(v / mag) * mag)


def build_input_schema(
    df: pd.DataFrame, columns: list[str], kinds: dict[str, str], importance: dict[str, float], labels: dict[str, dict[str, str]]
) -> list[dict[str, Any]]:
    fields = []
    for c in columns:
        s = df[c]
        k = kinds.get(c, "category")
        info = labels.get(c, {})
        f: dict[str, Any] = {
            "name": c,
            "label": info.get("label") or c,
            "help": info.get("help", ""),
            "kind": k if k in ("number", "integer", "category", "boolean", "date") else "category",
            "importance": round(float(importance.get(c, 0.0)), 4),
            "required": False,
        }
        if k in ("number", "integer"):
            v = pd.to_numeric(s, errors="coerce").dropna()
            if len(v):
                p5, p50, p95 = np.percentile(v, [5, 50, 95])
                f.update(
                    min=float(v.min()),
                    max=float(v.max()),
                    p05=float(p5),
                    p95=float(p95),
                    median=float(p50),
                    default=_nice(float(p50), k == "integer"),
                    step=1 if k == "integer" else "any",
                )
        elif k == "boolean":
            v = s.dropna().astype(str).str.lower()
            truthy = v.isin(["1", "1.0", "true", "yes", "y", "t"]).mean() >= 0.5 if len(v) else False
            f.update(options=[{"value": "true", "label": "Yes"}, {"value": "false", "label": "No"}], default="true" if truthy else "false")
        elif k == "date":
            v = pd.to_datetime(s, errors="coerce").dropna()
            if len(v):
                f.update(min=v.min().strftime("%Y-%m-%d"), max=v.max().strftime("%Y-%m-%d"), default=v.max().strftime("%Y-%m-%d"))
        else:
            v = s.dropna().astype(str).str.strip()
            counts = v.str.lower().value_counts()
            canon = v.groupby(v.str.lower()).agg(lambda x: x.value_counts().index[0])
            opts = [{"value": str(canon[key]), "count": int(n)} for key, n in counts.head(80).items()]
            f.update(options=opts, allow_other=len(counts) > 80, default=opts[0]["value"] if opts else "")
        fields.append(f)
    fields.sort(key=lambda f: -f["importance"])
    return fields


def reference_row(df: pd.DataFrame, columns: list[str], kinds: dict[str, str]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for c in columns:
        s = df[c]
        if kinds.get(c) in ("number", "integer"):
            v = pd.to_numeric(s, errors="coerce").dropna()
            row[c] = float(v.median()) if len(v) else np.nan
        elif kinds.get(c) == "date":
            v = pd.to_datetime(s, errors="coerce").dropna()
            row[c] = v.max().strftime("%Y-%m-%d") if len(v) else None
        else:
            v = s.dropna()
            row[c] = str(v.mode().iloc[0]) if len(v) else None
    return row


def example_rows(df: pd.DataFrame, columns: list[str], target: str | None, n: int = 30, seed: int = 3) -> list[dict[str, Any]]:
    sample = df.sample(min(n, len(df)), random_state=seed)
    out = []
    for _, r in sample.iterrows():
        inputs: dict[str, Any] = {}
        for c in columns:
            v = r[c]
            if isinstance(v, pd.Timestamp):
                inputs[c] = None if pd.isna(v) else v.strftime("%Y-%m-%d")
            elif v is None or (isinstance(v, float) and np.isnan(v)):
                inputs[c] = None
            elif isinstance(v, (bool, np.bool_)):
                inputs[c] = "true" if v else "false"
            elif isinstance(v, (np.integer,)):
                inputs[c] = int(v)
            elif isinstance(v, (np.floating, float)):
                inputs[c] = float(v)
            else:
                inputs[c] = str(v)
        ex: dict[str, Any] = {"inputs": inputs}
        if target is not None:
            ex["actual"] = cell_str(r[target])
        out.append(ex)
    return out


# ---------------------------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------------------------
def validate_rows(rows: list[dict[str, Any]], schema: list[dict[str, Any]]) -> tuple[pd.DataFrame, list[list[str]]]:
    """Raw form/batch rows -> frame with exactly the model's input columns + human-readable warnings per row."""
    fields = {f["name"]: f for f in schema}
    warnings_out: list[list[str]] = []
    clean = []
    for row in rows:
        w: list[str] = []
        out: dict[str, Any] = {}
        unknown = [k for k in row if k not in fields]
        if unknown:
            w.append(f"Ignored unknown field{'s' if len(unknown) > 1 else ''}: {', '.join(unknown[:5])}")
        for name, f in fields.items():
            v = row.get(name)
            if v is None or (isinstance(v, str) and v.strip() == ""):
                out[name] = None
                if f["importance"] >= 0.05:
                    w.append(f"{f['label']} is empty, so the model fills in a typical value")
                continue
            if f["kind"] in ("number", "integer"):
                try:
                    num = float(str(v).replace(",", "").replace("$", "").strip())
                except ValueError:
                    w.append(f"{f['label']}: '{v}' is not a number and was ignored")
                    out[name] = None
                    continue
                out[name] = num
                if "min" in f and (num < f["min"] or num > f["max"]):
                    w.append(
                        f"{f['label']} = {fmt_num(num)} is outside the training range ({fmt_num(f['min'])} to {fmt_num(f['max'])}); the prediction may be unreliable"
                    )
            elif f["kind"] == "category":
                out[name] = str(v).strip()
                known = {str(o["value"]).lower() for o in f.get("options") or []}
                if known and out[name].lower() not in known and not f.get("allow_other"):
                    w.append(f"{f['label']}: '{v}' was never seen in training")
            else:
                out[name] = v
        clean.append(out)
        warnings_out.append(w)
    return pd.DataFrame(clean, columns=list(fields)), warnings_out


# ---------------------------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------------------------
def _proba(model: Any, X: pd.DataFrame, n_classes: int) -> Any:
    p = np.asarray(model.predict_proba(X), dtype=float)
    cls = list(getattr(model, "classes_", range(p.shape[1])))
    if p.shape[1] == n_classes and cls == list(range(n_classes)):
        return p
    full = np.zeros((len(X), n_classes))
    for j, c in enumerate(cls):
        full[:, int(c)] = p[:, j]
    return full


def risk_band(p: float, threshold: float) -> str:
    if p >= max(threshold, 0.5):
        return "High"
    if p >= threshold / 2:
        return "Medium"
    return "Low"


def predict_tabular(art: dict[str, Any], X: pd.DataFrame, explain: bool = True, top: int = 5) -> tuple[list[dict[str, Any]], str | None]:
    model = art["model"]
    pt = art["problem_type"]
    labels = {f["name"]: f["label"] for f in art["input_schema"]}
    results: list[dict[str, Any]] = []
    if pt in ("binary", "multiclass"):
        P = _proba(model, X, len(art["classes"]))
        for i in range(len(X)):
            probs = {art["classes"][j]: round(float(P[i, j]), 4) for j in range(len(art["classes"]))}
            if pt == "binary":
                p = float(P[i, art["pos_index"]])
                idx = art["pos_index"] if p >= art["threshold"] else 1 - art["pos_index"]
                label = art["classes"][idx]
                results.append(
                    {
                        "prediction": label,
                        "display": f"{label} ({p * 100:.0f}% {art['classes'][art['pos_index']]})",
                        "probabilities": probs,
                        "risk_band": risk_band(p, art["threshold"]),
                    }
                )
            else:
                j = int(P[i].argmax())
                label = art["classes"][j]
                results.append({"prediction": label, "display": f"{label} ({P[i, j] * 100:.0f}%)", "probabilities": probs})
    else:
        pred = np.asarray(model.predict(X), dtype=float)
        q = art.get("residual_q") or {}
        for v in pred:
            interval = [float(v + q["q10"]), float(v + q["q90"])] if q else None
            results.append({"prediction": float(v), "display": fmt_num(float(v)), "interval": interval})

    explainer: str | None = None
    if explain and len(X):
        shap_out = shap_drivers(model, X, pt, top=top, background=art.get("background"))
        if shap_out is not None:
            drivers, explainer = shap_out
        else:
            ref = art["reference"]

            def f(Z: pd.DataFrame) -> Any:
                if pt == "binary":
                    return _proba(model, Z, 2)[:, art["pos_index"]]
                if pt == "multiclass":
                    return _proba(model, Z, len(art["classes"])).max(axis=1)
                return np.asarray(model.predict(Z), dtype=float)

            drivers = occlusion_drivers(f, X, ref, [c for c in art["columns"]], top)
            explainer = "occlusion"
        for r, d, (_, row) in zip(results, drivers, X.iterrows(), strict=True):
            r["drivers"] = [
                {
                    "feature": x["feature"],
                    "label": labels.get(x["feature"], x["feature"]),
                    "value": _jsonable(row.get(x["feature"])),
                    "effect": float(x["effect"]),
                    "direction": "up" if x["effect"] > 0 else "down",
                }
                for x in d
            ]
    return results, explainer


def _jsonable(v: Any) -> Any:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    return v


def batch_frame(art: dict[str, Any], df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Predict every row of an uploaded sheet; returns the sheet with prediction columns added and a column report."""
    cols = art["columns"]
    missing = [c for c in cols if c not in df.columns]
    extra = [c for c in df.columns if c not in cols and c != art.get("target")]
    report = {"rows": len(df), "missing_columns": missing, "extra_columns": extra}
    if len(missing) == len(cols):
        raise ValueError("None of the model's input columns were found in this sheet.")
    X = coerce_frame(df, cols, art["kinds"])
    results, _ = predict_tabular(art, X, explain=False)
    out = df.copy()
    name = f"predicted_{art.get('target') or 'value'}"
    out[name] = [r["prediction"] for r in results]
    if art["problem_type"] == "binary":
        pos = art["classes"][art["pos_index"]]
        out[f"probability_{pos}"] = [r["probabilities"][pos] for r in results]
        out["risk_band"] = [r["risk_band"] for r in results]
    elif art["problem_type"] == "multiclass":
        out["confidence"] = [max(r["probabilities"].values()) for r in results]
    elif results and results[0].get("interval"):
        out["lower_80"] = [r["interval"][0] for r in results]
        out["upper_80"] = [r["interval"][1] for r in results]
    vals = out[name]
    if art["problem_type"] in ("binary", "multiclass"):
        report["distribution"] = {str(k): int(v) for k, v in vals.value_counts().items()}
    else:
        report["distribution"] = {"min": float(vals.min()), "median": float(vals.median()), "max": float(vals.max())}
    return out, report


def forecast_view(art: dict[str, Any], series: str | None, horizon: int | None) -> dict[str, Any]:
    fc = art["forecast"]
    key = series if series in fc["chart"]["series"] else fc["chart"]["default"]
    h = min(int(horizon or fc["horizon"]), int(fc["horizon"]))
    pts = fc["chart"]["series"][key]
    future = [p for p in pts if "forecast" in p][:h]
    keep_dates = {p["date"] for p in future}
    plot = [p for p in pts if "forecast" not in p or p["date"] in keep_dates]
    history = sum(1 for p in plot if "forecast" not in p)
    plot = plot[max(0, history - max(52, 6 * h)) :]  # recent history keeps the forecast readable
    return {
        "series": key,
        "horizon": h,
        "max_horizon": fc["horizon"],
        "unit": fc["unit"],
        "available_series": list(fc["chart"]["series"].keys()),
        "points": plot,
        "table": [{"date": p["date"], "forecast": p["forecast"], "lower_80": p["band"][0], "upper_80": p["band"][1]} for p in future],
    }
