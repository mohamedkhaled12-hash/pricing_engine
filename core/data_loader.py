"""
core/data_loader.py
===================
تحميل وتجهيز الداتا من الـ CSV files الجاهزة.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def parse_midpoint(bracket_str: str) -> float:
    """
    تحويل نص الـ bracket لـ midpoint رقمي.
    '25,000 - 49,000' -> 37000.0
    '< 25,000'        -> 12500.0
    '> 247,000'       -> 321100.0
    """
    s = str(bracket_str).replace(",", "").strip()
    if s.startswith("<"):
        return float(s[1:].strip()) * 0.5
    elif s.startswith(">"):
        return float(s[1:].strip()) * 1.3
    else:
        parts = s.split("-")
        return (float(parts[0].strip()) + float(parts[1].strip())) / 2


def load_macro_metrics() -> dict:
    """
    يرجع dict من ml_macro_metrics.csv.
    {
      'Urban Cumulative Inflation': 2.469,
      'Urban Purchasing Power %':  40.5,
      'Urban Food Budget Share %': 28.63,
      ...
    }
    """
    df = pd.read_csv(DATA_DIR / "ml_macro_metrics.csv")
    return dict(zip(df["Metric"], df["Value"]))


def load_distribution() -> pd.DataFrame:
    """
    يحمل ml_ready_income_distribution.csv ويضيف عمود midpoint.
    """
    df = pd.read_csv(DATA_DIR / "ml_ready_income_distribution.csv")
    df["midpoint"] = df["Annual household income"].apply(parse_midpoint)
    return df


def get_area_params(metrics: dict, area: str) -> dict:
    """
    يرجع الـ parameters الخاصة بكل منطقة.
    area: 'Urban' أو 'Rural' أو 'Total' (يعامل زي Urban)
    """
    # Total بيتعامل زي Urban (المتوسط المرجح أقرب للحضر)
    key = area if area in ("Urban", "Rural") else "Urban"
    return {
        "inflation":        metrics[f"{key} Cumulative Inflation"],
        "purchasing_power": metrics[f"{key} Purchasing Power %"] / 100,
        "food_budget":      metrics[f"{key} Food Budget Share %"] / 100,
    }


def compute_disposable(gross_annual: float, area_params: dict) -> dict:
    """
    حساب الـ Disposable Income الحقيقي من الدخل السنوي الاسمي.

    المعادلة:
        real_income    = gross × purchasing_power%
        food_cost      = real_income × food_budget%
        disposable_yr  = real_income - food_cost
        disposable_mo  = disposable_yr / 12

    المنطق:
        - purchasing_power يعكس إن الـ 40.5% بس من الدخل الاسمي
          بقي له قيمة حقيقية بعد التضخم
        - food_budget هو النسبة اللي بتتصرف إجبارياً على الأكل
          فالباقي هو اللي المستهلك حر يصرفه على منتجات تانية
    """
    real = gross_annual * area_params["purchasing_power"]
    food = real * area_params["food_budget"]
    disposable_yr = real - food
    return {
        "real_annual":        round(real, 2),
        "food_cost_annual":   round(food, 2),
        "disposable_annual":  round(disposable_yr, 2),
        "disposable_monthly": round(disposable_yr / 12, 2),
    }


def enrich_distribution(df: pd.DataFrame, metrics: dict) -> pd.DataFrame:
    """
    يضيف أعمدة monthly_disposable و income_percentile للـ DataFrame.
    بيشتغل على 2026 بس (الـ training data).
    """
    df2026 = df[(df["Year"] == 2026) & (df["Area"].isin(["Urban", "Rural"]))].copy()

    def get_monthly_disp(row):
        params = get_area_params(metrics, row["Area"])
        result = compute_disposable(row["midpoint"], params)
        return result["disposable_monthly"]

    df2026["monthly_disposable"] = df2026.apply(get_monthly_disp, axis=1)
    df2026["income_percentile"] = (df2026["monthly_disposable"] / 8000).clip(0, 1)
    return df2026
