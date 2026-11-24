"""Small synthetic datasets with known signal for engine unit tests."""

import numpy as np
import pandas as pd


def churn_df(n: int = 600, seed: int = 0, leak: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = rng.integers(0, 90, n)
    sessions = rng.poisson(20, n)
    plan = rng.choice(["Basic", "Pro", "pro ", "PRO", "Enterprise"], n, p=[0.4, 0.3, 0.05, 0.05, 0.2])
    logit = -2.5 + 0.06 * days - 0.06 * sessions + rng.normal(0, 0.5, n)
    churn = rng.random(n) < 1 / (1 + np.exp(-logit))
    df = pd.DataFrame(
        {
            "customer_id": [f"C{i:05d}" for i in range(n)],
            "signup": pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 400, n), unit="D"),
            "plan": plan,
            "days_since_login": days.astype(float),
            "sessions": sessions.astype(float),
            "nps": np.where(rng.random(n) < 0.3, np.nan, rng.integers(0, 11, n)).astype(float),
            "churned": np.where(churn, "Yes", "No"),
        }
    )
    if leak:
        df["cancel_reason"] = np.where(churn, rng.choice(["price", "support"], n), None)
    return df


def regression_df(n: int = 500, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    area = rng.uniform(500, 3000, n)
    hood = rng.choice(["A", "B", "C"], n)
    price = 50_000 + area * 150 * pd.Series(hood).map({"A": 1.2, "B": 1.0, "C": 0.8}).to_numpy() + rng.normal(0, 15_000, n)
    return pd.DataFrame({"area": area, "rooms": (area / 500).round(), "hood": hood, "price": price})


def multiclass_df(n: int = 450, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cls = rng.choice(["x", "y", "z"], n)
    w = pd.Series(cls).map({"x": 1.0, "y": 5.0, "z": 20.0}).to_numpy() * rng.uniform(0.7, 1.3, n)
    return pd.DataFrame({"weight": w, "color": rng.choice(["red", "blue"], n), "kind": cls})


def weekly_panel(stores: int = 3, weeks: int = 120, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-03", periods=weeks, freq="W-MON")
    rows = []
    for s in range(stores):
        scale = 100 * (s + 1)
        for i, d in enumerate(dates):
            season = 1 + 0.3 * np.sin(2 * np.pi * i / 52)
            rows.append({"week": d, "store": f"S{s}", "units": max(0.0, scale * season + 0.5 * i + rng.normal(0, scale * 0.05))})
    return pd.DataFrame(rows)
