"""Generate realistic synthetic sample datasets in data/samples/.

Each file has a known signal plus deliberate real-world mess (title rows above the header,
inconsistent labels, numbers stored as text, Excel error values, missing values, duplicate rows,
a second sheet) so the whole pipeline can be demonstrated and tested.

    uv run python ../../scripts/generate_samples.py      (from services/api)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font

OUT = Path(__file__).resolve().parents[1] / "data" / "samples"
SEED = 7


def _write(path: Path, sheets: dict[str, pd.DataFrame], title: str | None = None, title_sheet: str | None = None) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, df in sheets.items():
            start = 2 if (title and name == title_sheet) else 0
            df.to_excel(xw, sheet_name=name, index=False, startrow=start)
    if title and title_sheet:
        wb = load_workbook(path)
        ws = wb[title_sheet]
        ws["A1"] = title
        ws["A1"].font = Font(bold=True, size=14)
        wb.save(path)


def _sprinkle(rng: np.random.Generator, s: pd.Series, share: float, value: object = np.nan) -> pd.Series:
    s = s.astype(object).copy()
    idx = rng.choice(len(s), int(len(s) * share), replace=False)
    s.iloc[idx] = value
    return s


def customer_churn(rng: np.random.Generator) -> None:
    n = 3000
    tier = rng.choice(["Basic", "Pro", "Enterprise"], n, p=[0.5, 0.35, 0.15])
    contract = rng.choice(["Monthly", "Annual"], n, p=[0.62, 0.38])
    seats = np.where(tier == "Enterprise", rng.integers(20, 400, n), np.where(tier == "Pro", rng.integers(3, 40, n), rng.integers(1, 6, n)))
    fee = np.round(seats * np.where(tier == "Enterprise", 32, np.where(tier == "Pro", 24, 15)) * rng.uniform(0.9, 1.1, n), 2)
    sessions = rng.poisson(np.where(tier == "Basic", 14, 30), n)
    days_login = np.clip(rng.exponential(9, n) + (sessions < 8) * rng.exponential(18, n), 0, 180).round().astype(int)
    tickets = rng.poisson(1.2, n)
    pay_fail = rng.poisson(0.25, n)
    nps = rng.integers(0, 11, n).astype(float)
    region = rng.choice(["North America", "Europe", "APAC", "LATAM"], n, p=[0.45, 0.3, 0.17, 0.08])
    signup = pd.Timestamp("2022-01-01") + pd.to_timedelta(rng.integers(0, 1000, n), unit="D")
    logit = -2.6 + 0.07 * days_login - 0.045 * sessions + 0.85 * pay_fail + 0.22 * tickets - 1.0 * (contract == "Annual") - 0.08 * (nps - 5) + rng.normal(0, 0.6, n)
    churn = rng.random(n) < 1 / (1 + np.exp(-logit))
    reasons = np.where(churn, rng.choice(["Too expensive", "Missing features", "Switched to competitor", "Poor support", "Company closed"], n), "")
    df = pd.DataFrame(
        {
            "customer_id": [f"C-{i:05d}" for i in range(1, n + 1)],
            "signup_date": signup,
            "plan_tier": tier,
            "contract_type": contract,
            "seats": seats,
            "monthly_fee": fee,
            "sessions_last_30d": sessions,
            "days_since_last_login": days_login,
            "support_tickets_90d": tickets,
            "payment_failures_90d": pay_fail,
            "nps_score": nps,
            "region": region,
            "churned": np.where(churn, "Yes", "No"),
            "cancellation_reason": reasons,  # leaky: only filled in after the customer has churned
        }
    )
    # mess: label variants, $ amounts as text, missing NPS, Excel errors, duplicates
    df["plan_tier"] = df["plan_tier"].astype(object)
    df.loc[rng.choice(n, 90, replace=False), "plan_tier"] = "pro "
    df.loc[rng.choice(n, 60, replace=False), "plan_tier"] = "PRO"
    fee_txt = df["monthly_fee"].astype(object)
    idx = rng.choice(n, 400, replace=False)
    fee_txt.iloc[idx] = [f"${v:,.2f}" for v in df["monthly_fee"].iloc[idx]]
    df["monthly_fee"] = fee_txt
    df["nps_score"] = _sprinkle(rng, df["nps_score"], 0.3)
    df["sessions_last_30d"] = _sprinkle(rng, df["sessions_last_30d"], 0.01, "#N/A")
    df = pd.concat([df, df.sample(25, random_state=SEED)], ignore_index=True)
    notes = pd.DataFrame({"Field": ["churned", "cancellation_reason", "nps_score"], "Meaning": ["Yes if the customer cancelled within 30 days of the export", "Reason given at cancellation", "Last survey score 0-10"]})
    _write(OUT / "customer_churn.xlsx", {"Customers": df, "Notes": notes}, title="Acme SaaS: customer export (Sep 2026)", title_sheet="Customers")


def card_fraud(rng: np.random.Generator) -> None:
    n = 20000
    t = pd.Timestamp("2026-06-01") + pd.to_timedelta(np.sort(rng.integers(0, 90 * 24 * 3600, n)), unit="s")
    cat = rng.choice(["grocery", "electronics", "travel", "fuel", "restaurants", "online_retail", "jewelry"], n, p=[0.27, 0.1, 0.06, 0.15, 0.2, 0.19, 0.03])
    amount = np.round(np.exp(rng.normal(3.6, 1.0, n)), 2)
    foreign = rng.random(n) < 0.06
    dist = np.round(np.abs(rng.normal(8, 15, n)) + foreign * rng.uniform(200, 3000, n), 1)
    cnt1h = rng.poisson(0.6, n)
    age = rng.integers(10, 4000, n)
    present = rng.random(n) < 0.55
    dev = rng.random(n) < 0.05
    logit = -7.2 + 0.0016 * amount + 1.4 * foreign + 0.0006 * dist + 0.75 * cnt1h - 0.0009 * age + 1.2 * dev - 0.9 * present + 1.1 * np.isin(cat, ["electronics", "jewelry", "online_retail"])
    fraud = rng.random(n) < 1 / (1 + np.exp(-logit))
    amount = np.where(fraud, amount * rng.uniform(1.5, 4, n), amount).round(2)
    df = pd.DataFrame(
        {
            "txn_id": [f"T{i:07d}" for i in range(n)],
            "txn_time": t,
            "amount": amount,
            "merchant_category": cat,
            "card_present": present,
            "is_foreign": np.where(foreign, "Y", "N"),
            "distance_from_home_km": dist,
            "txn_count_last_1h": cnt1h,
            "account_age_days": age,
            "new_device_7d": dev.astype(int),
            "is_fraud": fraud.astype(int),
        }
    )
    df["distance_from_home_km"] = _sprinkle(rng, df["distance_from_home_km"], 0.04)
    _write(OUT / "card_fraud.xlsx", {"Transactions": df})


def house_prices(rng: np.random.Generator) -> None:
    n = 3000
    hood = rng.choice(["Maple Grove", "Riverside", "Old Town", "Hillcrest", "Lakeview", "Industrial Park"], n, p=[0.2, 0.2, 0.15, 0.15, 0.2, 0.1])
    hood_mult = pd.Series(hood).map({"Maple Grove": 1.15, "Riverside": 1.05, "Old Town": 0.95, "Hillcrest": 1.3, "Lakeview": 1.2, "Industrial Park": 0.75}).to_numpy()
    area = np.clip(rng.normal(1800, 600, n), 500, 5500).round()
    beds = np.clip((area / 600 + rng.normal(0, 0.8, n)).round(), 1, 7).astype(int)
    baths = np.clip((beds * 0.7 + rng.normal(0, 0.5, n)) * 2, 2, 12).round() / 2
    built = rng.integers(1950, 2025, n)
    cond = rng.choice(["Poor", "Fair", "Good", "Excellent"], n, p=[0.08, 0.27, 0.45, 0.2])
    cond_mult = pd.Series(cond).map({"Poor": 0.82, "Fair": 0.93, "Good": 1.0, "Excellent": 1.1}).to_numpy()
    school = np.clip(rng.normal(6.5, 1.7, n), 1, 10).round(1)
    dist = np.round(np.abs(rng.normal(9, 5, n)), 1)
    lot = np.round(area * rng.uniform(1.5, 4, n))
    garage = rng.integers(0, 4, n)
    sale = pd.Timestamp("2022-01-01") + pd.to_timedelta(rng.integers(0, 1700, n), unit="D")
    price = (60_000 + area * 165 * hood_mult * cond_mult + school * 9_000 - dist * 2_800 + garage * 8_000 + (built - 1950) * 450) * np.exp(rng.normal(0, 0.08, n))
    df = pd.DataFrame(
        {
            "property_id": [f"P-{i:05d}" for i in range(n)],
            "sale_date": sale.strftime("%d/%m/%Y"),  # dates stored as text
            "neighborhood": hood,
            "living_area_sqft": area.astype(int),
            "lot_size_sqft": lot,
            "bedrooms": beds,
            "bathrooms": baths,
            "year_built": built,
            "garage_spaces": garage,
            "condition": cond,
            "school_rating": school,
            "distance_to_center_km": dist,
            "sale_price": price.round(-2),
        }
    )
    df["lot_size_sqft"] = _sprinkle(rng, df["lot_size_sqft"], 0.06)
    sp = df["sale_price"].astype(object)
    idx = rng.choice(n, 300, replace=False)
    sp.iloc[idx] = [f"${v:,.0f}" for v in df["sale_price"].iloc[idx]]
    df["sale_price"] = sp
    df.loc[rng.choice(n, 40, replace=False), "condition"] = "good"
    df = pd.concat([df, df.sample(15, random_state=SEED)], ignore_index=True)
    _write(OUT / "house_prices.xlsx", {"Sales": df}, title="County housing sales 2022-2026", title_sheet="Sales")


def store_weekly_demand(rng: np.random.Generator) -> None:
    weeks = pd.date_range("2023-10-02", periods=156, freq="W-MON")
    stores = [f"S{i:02d}" for i in range(1, 11)]
    region = dict(zip(stores, rng.choice(["North", "South", "East", "West"], len(stores)), strict=True))
    rows = []
    for s in stores:
        scale = rng.uniform(600, 2000)
        trend = rng.uniform(-0.5, 2.0)
        for i, w in enumerate(weeks):
            woy = w.isocalendar().week
            season = 1 + 0.18 * np.sin(2 * np.pi * (woy - 10) / 52)
            holiday = 1.35 if woy in (47, 48, 51, 52) else 1.0
            promo = rng.random() < 0.15
            units = scale * season * holiday * (1.25 if promo else 1.0) + trend * i + rng.normal(0, scale * 0.06)
            rows.append({"week_start": w, "store_id": s, "region": region[s], "promo_flag": int(promo), "units_sold": max(0, round(units))})
    df = pd.DataFrame(rows)
    df = df.drop(rng.choice(len(df), 12, replace=False)).reset_index(drop=True)  # a few missing weeks
    stores_df = pd.DataFrame({"store_id": stores, "region": [region[s] for s in stores], "opened": rng.integers(2005, 2020, len(stores))})
    _write(OUT / "store_weekly_demand.xlsx", {"Weekly sales": df, "Stores": stores_df})


def product_category(rng: np.random.Generator) -> None:
    n = 2500
    cats = rng.choice(["Electronics", "Furniture", "Clothing", "Sports", "Kitchen"], n, p=[0.25, 0.18, 0.25, 0.17, 0.15])
    base = {
        "Electronics": (250, 1.5, 30, "plastic"),
        "Furniture": (400, 25, 120, "wood"),
        "Clothing": (45, 0.4, 40, "cotton"),
        "Sports": (90, 3, 70, "polyester"),
        "Kitchen": (60, 2, 35, "steel"),
    }
    price = np.array([base[c][0] for c in cats]) * np.exp(rng.normal(0, 0.45, n))
    weight = np.array([base[c][1] for c in cats]) * np.exp(rng.normal(0, 0.5, n))
    length = np.array([base[c][2] for c in cats]) * np.exp(rng.normal(0, 0.35, n))
    materials = ["plastic", "wood", "cotton", "polyester", "steel", "aluminium", "glass"]
    material = np.array([base[c][3] if rng.random() < 0.65 else rng.choice(materials) for c in cats])
    warranty = np.array([rng.integers(1, 4) if c == "Electronics" else rng.integers(0, 2) for c in cats])
    df = pd.DataFrame(
        {
            "sku": [f"SKU{i:06d}" for i in range(n)],
            "price": price.round(2),
            "weight_kg": weight.round(2),
            "length_cm": length.round(1),
            "material": material,
            "color": rng.choice(["black", "white", "grey", "blue", "red", "green", "brown"], n),
            "warranty_years": warranty,
            "avg_rating": np.clip(rng.normal(4.1, 0.5, n), 1, 5).round(1),
            "returns_rate": np.clip(rng.beta(2, 30, n) + (cats == "Clothing") * 0.05, 0, 1).round(3),
            "category": cats,
        }
    )
    df["material"] = _sprinkle(rng, df["material"], 0.03)
    _write(OUT / "product_category.xlsx", {"Products": df})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    for fn in (customer_churn, card_fraud, house_prices, store_weekly_demand, product_category):
        fn(rng)
        print(f"wrote {fn.__name__}")
    print(f"samples in {OUT}")


if __name__ == "__main__":
    sys.exit(main())
