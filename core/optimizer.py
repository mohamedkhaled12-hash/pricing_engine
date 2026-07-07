"""
core/optimizer.py
=================
محرك التسعير الذكي — القلب الفعلي للمشروع.

بياخد مدخلات المصنع ويطلع توصية واحدة من اتنين:
  A) الوزن الأمثل  → يثبت السعر النفسي + يحافظ على هامش الربح
  B) نسبة المقاطعة → لو صمم يرفع السعر من غير تغيير في الوزن
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from core.data_loader import (
    load_distribution, load_macro_metrics,
    get_area_params, compute_disposable, enrich_distribution,
)
from core.ml_model import load_churn_model, predict_churn, FEATURE_COLS


# ─────────────────────────────────────────────
# Data Classes للـ Input والـ Output
# ─────────────────────────────────────────────

@dataclass
class ProductInput:
    """مدخلات المصنع."""
    current_price:    float   # السعر الحالي بالجنيه (السعر النفسي المستهدف)
    current_weight_g: float   # الوزن الحالي بالجرام
    cost_per_gram:    float   # تكلفة الإنتاج لكل جرام (شاملة خامات + تشغيل)
    area:             str     # 'Urban' أو 'Rural'
    purchase_freq:    int = 4 # عدد مرات الشراء شهرياً (افتراضي 4)
    target_margin:    float = 0.25  # هامش الربح المستهدف (25%)


@dataclass
class WeightRecommendation:
    """توصية A: تخفيض الوزن للحفاظ على السعر النفسي."""
    optimal_weight_g:      float
    weight_reduction_pct:  float
    new_cost_total:        float
    new_margin_pct:        float
    price_per_gram_new:    float
    price_per_gram_old:    float
    feasible:              bool    # هل التخفيض منطقي؟ (مش أكتر من 40%)
    warning:               str


@dataclass
class ChurnPrediction:
    """توصية B: التنبؤ بنسبة المقاطعة عند رفع السعر."""
    new_price:             float
    price_increase_pct:    float
    weighted_churn_pct:    float   # نسبة المقاطعة المرجحة للسوق
    at_risk_population_pct: float  # نسبة السكان في الـ brackets المتضررة
    segments_detail:       list    # تفاصيل كل bracket
    ml_churn_prob:         float   # تنبؤ الـ ML Model للـ segment المستهدف
    risk_level:            str
    recommendation:        str


# ─────────────────────────────────────────────
# الـ Optimizer Class الرئيسي
# ─────────────────────────────────────────────

class PricingOptimizer:

    def __init__(self):
        self.df      = load_distribution()
        self.metrics = load_macro_metrics()
        self._model  = None  # lazy load

    @property
    def model(self):
        if self._model is None:
            self._model = load_churn_model()
        return self._model

    # ── A: الوزن الأمثل ──────────────────────────────────────────────────

    def find_optimal_weight(self, product: ProductInput) -> WeightRecommendation:
        """
        يحسب الوزن الأمثل اللي يحافظ على السعر النفسي وهامش الربح.

        المعادلة الأساسية:
        ─────────────────────────────────────────────────
        هامش الربح = (سعر - تكلفة الإنتاج) / سعر
        target_margin = (price - cost_per_gram × weight) / price
        ∴ cost_per_gram × weight = price × (1 - target_margin)
        ∴ optimal_weight = price × (1 - target_margin) / cost_per_gram
        ─────────────────────────────────────────────────

        مثال:
          سعر=25 جنيه، وزن=100 جرام، تكلفة/جرام=0.18 جنيه، هامش مستهدف=30%
          max_cost = 25 × 0.70 = 17.5 جنيه
          optimal_weight = 17.5 / 0.18 = 97.2 جرام (تخفيض 2.8% بس)
        """
        p = product

        # الهامش الحالي
        current_cost    = p.cost_per_gram * p.current_weight_g
        current_margin  = (p.current_price - current_cost) / p.current_price

        # الوزن الأمثل
        max_allowed_cost  = p.current_price * (1 - p.target_margin)
        optimal_weight    = max_allowed_cost / p.cost_per_gram
        reduction_pct     = (1 - optimal_weight / p.current_weight_g) * 100
        new_cost          = p.cost_per_gram * optimal_weight
        new_margin        = (p.current_price - new_cost) / p.current_price

        # التحقق من المنطقية
        feasible = 0 < reduction_pct < 40  # تخفيض أكثر من 40% مش مقبول سوقياً

        if reduction_pct < 0:
            warning = "⚠️ الهامش الحالي أعلى من المستهدف — لا حاجة لتخفيض الوزن."
        elif reduction_pct > 40:
            warning = "🚨 التخفيض المطلوب كبير جداً (>40%) — راجع تكلفة الإنتاج."
        elif reduction_pct < 2:
            warning = "✅ التخفيض بسيط جداً — يمكن تطبيقه بسهولة."
        else:
            warning = "✅ التخفيض في النطاق المقبول."

        return WeightRecommendation(
            optimal_weight_g     = round(optimal_weight, 1),
            weight_reduction_pct = round(reduction_pct, 1),
            new_cost_total       = round(new_cost, 2),
            new_margin_pct       = round(new_margin * 100, 1),
            price_per_gram_new   = round(p.current_price / optimal_weight, 3),
            price_per_gram_old   = round(p.current_price / p.current_weight_g, 3),
            feasible             = feasible,
            warning              = warning,
        )

    # ── B: تنبؤ المقاطعة ─────────────────────────────────────────────────

    def predict_market_churn(
        self,
        product: ProductInput,
        new_price: float,
    ) -> ChurnPrediction:
        """
        يتنبأ بنسبة المقاطعة الكلية للسوق لو السعر اتزاد لـ new_price.

        يشتغل على مستويين:
        1. الـ Rule-Based Segmentation:
           لكل bracket في 2026، يحسب هل العبء السعري يتجاوز العتبة.
           ده بيعطي تصوير تفصيلي للـ segments المتضررة.

        2. الـ ML Model:
           بيحسب churn probability احتمالية للـ bracket الأكثر تأثراً
           (الأكثر شيوعاً في السوق حسب Estimate_Income %).
        """
        p = product
        price_increase_pct = (new_price - p.current_price) / p.current_price
        is_urban = 1 if p.area == "Urban" else 0

        # تجهيز البيانات
        df2026 = enrich_distribution(self.df, self.metrics)
        subset = df2026[df2026["Area"] == p.area].copy()

        segments = []
        for _, row in subset.iterrows():
            monthly_disp   = row["monthly_disposable"]
            income_pct     = row["income_percentile"]

            # العبء السعري بعد الزيادة
            burden = (new_price * p.purchase_freq) / max(monthly_disp, 1)

            # عتبة المقاطعة (تتفاوت حسب الدخل)
            threshold = 0.08 + 0.18 * income_pct

            # ML prediction لهذا الـ segment
            ml_result = predict_churn(
                model              = self.model,
                monthly_disposable = monthly_disp,
                income_percentile  = income_pct,
                base_price         = p.current_price,
                price_increase_pct = price_increase_pct,
                purchase_freq      = p.purchase_freq,
                area_urban         = is_urban,
            )

            at_risk = burden > threshold
            segments.append({
                "bracket":             row["Annual household income"],
                "population_pct":      round(row["Estimate_Income %"] * 100, 2),
                "monthly_disposable":  round(monthly_disp, 0),
                "price_burden_pct":    round(burden * 100, 1),
                "churn_threshold_pct": round(threshold * 100, 1),
                "at_risk":             at_risk,
                "ml_churn_prob":       ml_result["churn_probability"],
                "ml_risk_level":       ml_result["risk_level"],
            })

        # الـ Weighted Churn على مستوى السوق
        seg_df = pd.DataFrame(segments)
        weighted_churn = (
            seg_df["at_risk"] * seg_df["population_pct"]
        ).sum() / 100

        at_risk_pop = seg_df.loc[seg_df["at_risk"], "population_pct"].sum()

        # ML prediction للـ segment الأكثر وزناً
        top_segment = seg_df.loc[seg_df["population_pct"].idxmax()]
        ml_top_prob = top_segment["ml_churn_prob"]

        # الحكم النهائي
        if weighted_churn > 0.35:
            risk, rec = "HIGH", "🚨 قرار الزيادة خطير — ستفقد أكثر من ثلث السوق."
        elif weighted_churn > 0.15:
            risk, rec = "MEDIUM", "⚠️ مقبول بحذر — راقب الـ segments الفقيرة."
        else:
            risk, rec = "LOW", "✅ الزيادة آمنة — العبء السعري في حدوده."

        return ChurnPrediction(
            new_price              = round(new_price, 2),
            price_increase_pct     = round(price_increase_pct * 100, 1),
            weighted_churn_pct     = round(weighted_churn * 100, 1),
            at_risk_population_pct = round(at_risk_pop, 1),
            segments_detail        = segments,
            ml_churn_prob          = round(ml_top_prob * 100, 1),
            risk_level             = risk,
            recommendation         = rec,
        )


if __name__ == "__main__":
    optimizer = PricingOptimizer()

    product = ProductInput(
        current_price    = 25.0,
        current_weight_g = 100.0,
        cost_per_gram    = 0.18,
        area             = "Urban",
        purchase_freq    = 4,
        target_margin    = 0.30,
    )

    print("=== A: Optimal Weight ===")
    w = optimizer.find_optimal_weight(product)
    print(f"  Current:  {product.current_weight_g}g → Optimal: {w.optimal_weight_g}g")
    print(f"  Reduction: {w.weight_reduction_pct}% | New margin: {w.new_margin_pct}%")
    print(f"  {w.warning}")

    print("\n=== B: Churn Prediction (price 25 → 30 EGP) ===")
    churn = optimizer.predict_market_churn(product, new_price=30.0)
    print(f"  Price increase: +{churn.price_increase_pct}%")
    print(f"  Weighted churn: {churn.weighted_churn_pct}%")
    print(f"  At-risk population: {churn.at_risk_population_pct}%")
    print(f"  {churn.recommendation}")
    print("\n  Segment breakdown:")
    for s in churn.segments_detail:
        icon = "🔴" if s["at_risk"] else "🟢"
        print(f"    {icon} {s['bracket']:25s} burden={s['price_burden_pct']}% ML={s['ml_churn_prob']:.0%}")
