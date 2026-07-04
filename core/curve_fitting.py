"""
core/curve_fitting.py
=====================
Log-Normal Curve Fitting لتوزيع الدخل (2020 و 2026).
بيرسم المنحنيين اللي هيظهروا في شاشة رادار انكماش الطبقات.
"""

import numpy as np
from scipy import stats
import pandas as pd
from core.data_loader import load_distribution, load_macro_metrics


def fit_lognormal(df: pd.DataFrame, year: int, area: str) -> tuple[float, float]:
    """
    بيعمل Weighted Log-Normal Fit لفئة معينة وسنة معينة.

    الـ Weights = Estimate_Income % (نسبة الناس في كل فئة)
    الـ Input   = midpoint لكل bracket

    بيرجع: (mu, sigma) → parameters المنحنى المستمر

    ليه Log-Normal؟
        توزيع الدخل في كل دول العالم بيتبع Log-Normal
        لأن الدخل مينفعش يبقى سلبي، وعنده tail طويل جهة اليمين
        (أغنياء قلة بيشدوا المتوسط الحسابي لفوق).
        الـ median = exp(mu) هو الأدق للتعبير عن "الدخل الوسطي الحقيقي".
    """
    subset = df[(df["Year"] == year) & (df["Area"] == area)].copy()
    weights = subset["Estimate_Income %"].values
    log_mids = np.log(subset["midpoint"].values)

    mu    = np.average(log_mids, weights=weights)
    sigma = np.sqrt(np.average((log_mids - mu) ** 2, weights=weights))
    return mu, sigma


def get_curve_data(area: str = "Urban") -> dict:
    """
    بيرجع كل البيانات اللازمة لرسم المنحنيين فوق بعض.

    Returns:
    {
      x:           numpy array (نطاق الدخل السنوي بالجنيه)
      pdf_2020:    قيم المنحنى الأزرق (السنة الأساس)
      pdf_2026:    قيم المنحنى الأحمر (المزحزح لليمين)
      median_2020: الدخل الوسطي 2020
      median_2026: الدخل الوسطي 2026
      shift_pct:   نسبة الإزاحة (= نسبة التضخم تقريباً)
      mu_2020, sigma_2020, mu_2026, sigma_2026
    }

    الإزاحة لليمين:
        المنحنى الأحمر (2026) مزحزح لليمين لأن الأرقام الاسمية اتضاعفت
        بفعل التضخم. لكن القوة الشرائية الحقيقية انخفضت.
        ده هو جوهر "رادار الانكماش" — الناس بتكسب أكتر بالأرقام
        لكن تقدر تشتري أقل.
    """
    df = load_distribution()

    mu_2020, sigma_2020 = fit_lognormal(df, 2020, area)
    mu_2026, sigma_2026 = fit_lognormal(df, 2026, area)

    # نطاق الـ x: من أقل bracket 2020 لأكبر bracket 2026
    x = np.linspace(5_000, 500_000, 600)

    pdf_2020 = stats.lognorm.pdf(x, s=sigma_2020, scale=np.exp(mu_2020))
    pdf_2026 = stats.lognorm.pdf(x, s=sigma_2026, scale=np.exp(mu_2026))

    median_2020 = np.exp(mu_2020)
    median_2026 = np.exp(mu_2026)
    shift_pct   = (median_2026 - median_2020) / median_2020 * 100

    return {
        "x":           x.tolist(),
        "pdf_2020":    pdf_2020.tolist(),
        "pdf_2026":    pdf_2026.tolist(),
        "median_2020": round(median_2020, 0),
        "median_2026": round(median_2026, 0),
        "shift_pct":   round(shift_pct, 1),
        "mu_2020":     round(mu_2020, 4),
        "sigma_2020":  round(sigma_2020, 4),
        "mu_2026":     round(mu_2026, 4),
        "sigma_2026":  round(sigma_2026, 4),
        "area":        area,
    }


def get_bracket_affordability(
    product_price: float,
    area: str = "Urban",
    purchase_freq_monthly: int = 4,
) -> list[dict]:
    """
    لكل bracket في 2026، بيحسب هل المنتج "في متناول" هذه الفئة أم لا.
    بيرجع list مرتبة من الأفقر للأغنى.

    price_burden = (سعر × تكرار الشراء شهرياً) / monthly_disposable
    """
    df = load_distribution()
    metrics = load_macro_metrics()

    from core.data_loader import get_area_params, compute_disposable
    params = get_area_params(metrics, area)

    subset = df[(df["Year"] == 2026) & (df["Area"] == area)].copy()
    result = []
    for _, row in subset.iterrows():
        disp = compute_disposable(row["midpoint"], params)
        monthly_spend = product_price * purchase_freq_monthly
        burden = monthly_spend / max(disp["disposable_monthly"], 1)

        result.append({
            "bracket":          row["Annual household income"],
            "population_pct":   round(row["Estimate_Income %"] * 100, 2),
            "monthly_disposable": disp["disposable_monthly"],
            "price_burden_pct": round(burden * 100, 1),
            "affordable":       burden < 0.15,  # أقل من 15% من الـ disposable
        })
    return result


if __name__ == "__main__":
    # اختبار سريع
    data = get_curve_data("Urban")
    print(f"Urban | median 2020: {data['median_2020']:,.0f} EGP/yr")
    print(f"Urban | median 2026: {data['median_2026']:,.0f} EGP/yr")
    print(f"Urban | shift:       +{data['shift_pct']}%  (إزاحة اسمية بسبب التضخم)")

    data_r = get_curve_data("Rural")
    print(f"Rural | median 2020: {data_r['median_2020']:,.0f} EGP/yr")
    print(f"Rural | median 2026: {data_r['median_2026']:,.0f} EGP/yr")
