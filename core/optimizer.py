"""
core/optimizer.py  —  v3 Production
=====================================
ترقيات على v1:

1. WEIGHT: 3 استراتيجيات (A: ثبّت السعر، B: ثبّت الوزن، C: هجين)
2. CHURN:  ML-weighted بدل binary + revenue impact + break-even
3. SENSITIVITY: curve كاملة 15 نقطة + sweet spot
4. SEGMENTS: composite score (0-100) لكل bracket
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional

from core.data_loader import (
    load_distribution, load_macro_metrics,
    enrich_distribution,
)
from core.ml_model import load_churn_model, predict_churn, FEATURE_COLS


# ─────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────

@dataclass
class ProductInput:
    current_price:    float
    current_weight_g: float
    cost_per_gram:    float
    area:             str
    purchase_freq:    int   = 4
    target_margin:    float = 0.25
    monthly_revenue:  Optional[float] = None


@dataclass
class WeightRecommendation:
    # Strategy A: ثبّت السعر، خفّض الوزن
    optimal_weight_g:      float
    weight_reduction_pct:  float
    new_cost_total:        float
    new_margin_pct:        float
    price_per_gram_new:    float
    price_per_gram_old:    float
    feasible:              bool
    warning:               str
    # Strategy B: ثبّت الوزن، اقبل الهامش الحالي
    strategy_b_margin:     float
    strategy_b_feasible:   bool
    # Strategy C: هجين — نص نص
    hybrid_weight_g:       float
    hybrid_price:          float
    hybrid_margin:         float
    hybrid_recommendation: str


@dataclass
class SegmentScore:
    bracket:            str
    population_pct:     float
    monthly_disposable: float
    affordability:      float   # 0-100
    stability:          float   # 0-100
    market_value:       float   # 0-100
    composite_score:    float
    segment_label:      str


@dataclass
class ChurnPrediction:
    new_price:              float
    price_increase_pct:     float
    # ML-weighted (أدق)
    ml_weighted_churn_pct:  float
    binary_churn_pct:       float
    at_risk_population_pct: float
    segments_detail:        list
    ml_churn_prob:          float
    risk_level:             str
    recommendation:         str
    # Revenue impact
    revenue_loss_estimate:  float
    break_even_price:       float
    price_ceiling:          float


# ─────────────────────────────────────────────────────────
# OPTIMIZER
# ─────────────────────────────────────────────────────────

class PricingOptimizer:

    def __init__(self):
        self.df      = load_distribution()
        self.metrics = load_macro_metrics()
        self._model  = None

    @property
    def model(self):
        if self._model is None:
            self._model = load_churn_model()
        return self._model

    # ── A/B/C: Optimal Weight ─────────────────────────────
    def find_optimal_weight(self, product: ProductInput) -> WeightRecommendation:
        p = product
        current_cost   = p.cost_per_gram * p.current_weight_g
        current_margin = (p.current_price - current_cost) / p.current_price
        margin_gap     = p.target_margin - current_margin

        # Strategy A: ثبّت السعر، خفّض الوزن
        max_cost_A   = p.current_price * (1 - p.target_margin)
        opt_weight_A = max_cost_A / p.cost_per_gram
        reduction_A  = (1 - opt_weight_A / p.current_weight_g) * 100
        new_cost_A   = p.cost_per_gram * opt_weight_A
        new_margin_A = (p.current_price - new_cost_A) / p.current_price
        feasible_A   = 0 < reduction_A < 40

        if   reduction_A < 0:    warning = "⚠️ الهامش الحالي أعلى من المستهدف — لا حاجة لتعديل."
        elif reduction_A > 40:   warning = "🚨 التخفيض أكبر من 40% — راجع تكلفة الإنتاج أو استخدم Strategy C."
        elif reduction_A < 2:    warning = "✅ تخفيض أقل من 2% — سهل ومش ملحوظ."
        elif reduction_A < 10:   warning = "✅ تخفيض معقول في النطاق المقبول سوقياً."
        else:                     warning = "⚠️ تخفيض ملحوظ — فكر في Strategy C (هجين)."

        # Strategy B: ثبّت الوزن
        strategy_b_margin   = current_margin
        strategy_b_feasible = current_margin >= p.target_margin * 0.8

        # Strategy C: هجين 50/50
        if margin_gap > 0:
            price_inc_needed  = p.current_price / (1 - p.target_margin) - p.current_price
            hybrid_price_inc  = price_inc_needed * 0.5
            hybrid_price      = p.current_price + hybrid_price_inc
            max_cost_C        = hybrid_price * (1 - p.target_margin)
            hybrid_weight     = max_cost_C / p.cost_per_gram
            hybrid_cost       = p.cost_per_gram * hybrid_weight
            hybrid_margin     = (hybrid_price - hybrid_cost) / hybrid_price
            hybrid_rec = (
                f"ارفع السعر من {p.current_price:.2f} لـ {hybrid_price:.2f} ج "
                f"(+{hybrid_price_inc/p.current_price*100:.1f}%) "
                f"وخفض الوزن من {p.current_weight_g:.0f}g لـ {hybrid_weight:.1f}g "
                f"← هامش {hybrid_margin*100:.1f}%"
            )
        else:
            hybrid_price  = p.current_price
            hybrid_weight = p.current_weight_g
            hybrid_margin = current_margin
            hybrid_rec    = "✅ الهامش الحالي مناسب — لا حاجة للـ Hybrid."

        return WeightRecommendation(
            optimal_weight_g      = round(opt_weight_A, 1),
            weight_reduction_pct  = round(reduction_A, 1),
            new_cost_total        = round(new_cost_A, 2),
            new_margin_pct        = round(new_margin_A * 100, 1),
            price_per_gram_new    = round(p.current_price / max(opt_weight_A, 0.1), 3),
            price_per_gram_old    = round(p.current_price / p.current_weight_g, 3),
            feasible              = feasible_A,
            warning               = warning,
            strategy_b_margin     = round(strategy_b_margin * 100, 1),
            strategy_b_feasible   = strategy_b_feasible,
            hybrid_weight_g       = round(hybrid_weight, 1),
            hybrid_price          = round(hybrid_price, 2),
            hybrid_margin         = round(hybrid_margin * 100, 1),
            hybrid_recommendation = hybrid_rec,
        )

    # ── Market Segment Scoring ────────────────────────────
    def score_market_segments(self, product: ProductInput) -> List[SegmentScore]:
        """
        Score مركب (0-100) لكل bracket:
          35% × affordability  (القدرة على السعر الحالي)
          35% × stability      (عكس churn prob عند +20%)
          30% × market_value   (الحجم × القدرة)
        """
        df2026   = enrich_distribution(self.df, self.metrics)
        subset   = df2026[df2026["Area"] == product.area].copy()
        is_urban = 1 if product.area == "Urban" else 0

        scores = []
        for _, row in subset.iterrows():
            monthly_disp = row["monthly_disposable"]
            income_pct   = row["income_percentile"]
            pop_pct      = row["Estimate_Income %"] * 100

            burden    = (product.current_price * product.purchase_freq) / max(monthly_disp, 1)
            threshold = 0.08 + 0.18 * income_pct
            afford    = max(0.0, min(100.0, (1 - burden / threshold) * 100))

            ml_res    = predict_churn(
                self.model, monthly_disp, income_pct,
                product.current_price, 0.20, product.purchase_freq, is_urban,
            )
            stability = round((1 - ml_res["churn_probability"]) * 100, 1)
            mkt_val   = round(pop_pct * afford / 100, 1)
            composite = round(0.35 * afford + 0.35 * stability + 0.30 * min(mkt_val * 10, 100), 1)

            if   composite >= 70: label = "⭐ Premium"
            elif composite >= 50: label = "🟢 Core"
            elif composite >= 30: label = "🟡 Sensitive"
            else:                  label = "🔴 At Risk"

            scores.append(SegmentScore(
                bracket            = row["Annual household income"],
                population_pct     = round(pop_pct, 2),
                monthly_disposable = round(monthly_disp, 0),
                affordability      = round(afford, 1),
                stability          = stability,
                market_value       = mkt_val,
                composite_score    = composite,
                segment_label      = label,
            ))

        return sorted(scores, key=lambda s: -s.composite_score)

    # ── Sensitivity Curve ─────────────────────────────────
    def price_sensitivity_curve(
        self,
        product:     ProductInput,
        price_range: tuple = (0.0, 0.80),
        n_points:    int   = 12,
    ) -> pd.DataFrame:
        """
        curve: churn% + revenue_index لكل نسبة زيادة
        sweet_spot: أعلى revenue عند أقل churn
        """
        df2026   = enrich_distribution(self.df, self.metrics)
        subset   = df2026[df2026["Area"] == product.area].copy()
        is_urban = 1 if product.area == "Urban" else 0
        increases = np.linspace(price_range[0], price_range[1], n_points)

        rows = []
        for inc_pct in increases:
            new_price   = product.current_price * (1 + inc_pct)
            total_churn = 0.0; total_pop = 0.0

            for _, row in subset.iterrows():
                ml_res = predict_churn(
                    self.model, row["monthly_disposable"], row["income_percentile"],
                    product.current_price, inc_pct, product.purchase_freq, is_urban,
                )
                pop_w        = row["Estimate_Income %"]
                total_churn += ml_res["churn_probability"] * pop_w
                total_pop   += pop_w

            churn_pct     = (total_churn / total_pop) * 100
            retention_pct = 100 - churn_pct
            revenue_index = (new_price * retention_pct / 100) / product.current_price * 100

            rows.append({
                "price_increase_pct": round(inc_pct * 100, 1),
                "new_price":          round(new_price, 2),
                "churn_pct":          round(churn_pct, 2),
                "retention_pct":      round(retention_pct, 2),
                "revenue_index":      round(revenue_index, 2),
            })

        df_curve = pd.DataFrame(rows)
        sweet    = df_curve["revenue_index"].idxmax()
        df_curve["is_sweet_spot"] = False
        df_curve.loc[sweet, "is_sweet_spot"] = True
        return df_curve

    # ── Churn Prediction ─────────────────────────────────
    def predict_market_churn(
        self,
        product:   ProductInput,
        new_price: float,
    ) -> ChurnPrediction:
        """
        ML-weighted churn = Σ(ML_prob × pop%) / Σ(pop%)
        أدق من binary لأن كل bracket له احتمالية مستمرة.
        """
        p                  = product
        price_increase_pct = (new_price - p.current_price) / p.current_price
        is_urban           = 1 if p.area == "Urban" else 0

        df2026  = enrich_distribution(self.df, self.metrics)
        subset  = df2026[df2026["Area"] == p.area].copy()
        segments= []; total_pop = 0.0

        for _, row in subset.iterrows():
            monthly_disp = row["monthly_disposable"]
            income_pct   = row["income_percentile"]
            pop_w        = row["Estimate_Income %"]
            total_pop   += pop_w

            burden    = (new_price * p.purchase_freq) / max(monthly_disp, 1)
            threshold = 0.08 + 0.18 * income_pct
            at_risk   = burden > threshold

            ml_result = predict_churn(
                self.model, monthly_disp, income_pct,
                p.current_price, price_increase_pct, p.purchase_freq, is_urban,
            )

            segments.append({
                "bracket":             row["Annual household income"],
                "population_pct":      round(pop_w * 100, 2),
                "monthly_disposable":  round(monthly_disp, 0),
                "price_burden_pct":    round(burden * 100, 1),
                "churn_threshold_pct": round(threshold * 100, 1),
                "at_risk":             at_risk,
                "ml_churn_prob":       ml_result["churn_probability"],
                "ml_risk_level":       ml_result["risk_level"],
                "pop_weight":          pop_w,
            })

        seg_df = pd.DataFrame(segments)

        # ML-weighted churn (v3)
        ml_w_churn   = (seg_df["ml_churn_prob"] * seg_df["pop_weight"]).sum() / total_pop
        binary_churn = (seg_df["at_risk"].astype(float) * seg_df["pop_weight"]).sum() / total_pop
        at_risk_pop  = seg_df.loc[seg_df["at_risk"], "population_pct"].sum()

        # Revenue impact
        revenue_loss = ml_w_churn * 100
        if ml_w_churn < 0.99:
            break_even = round(p.current_price / (1 - ml_w_churn), 2)
        else:
            break_even = new_price * 2

        # Price ceiling (churn ≥ 50%)
        curve = self.price_sensitivity_curve(p, n_points=8)
        ceil_rows = curve[curve["churn_pct"] >= 50]
        price_ceiling = float(ceil_rows.iloc[0]["new_price"]) if len(ceil_rows) else float(p.current_price * 1.8)

        # Top segment ML prob
        ml_top = float(seg_df.loc[seg_df["pop_weight"].idxmax(), "ml_churn_prob"])

        churn_pct = ml_w_churn * 100
        if   churn_pct > 35: risk="HIGH";   rec=f"🚨 خطر عالي — {churn_pct:.1f}% مقاطعة. السقف الآمن {price_ceiling:.2f} ج."
        elif churn_pct > 15: risk="MEDIUM"; rec=f"⚠️ مقبول بحذر — {churn_pct:.1f}% مقاطعة. راقب الـ segments الحساسة."
        else:                  risk="LOW";   rec=f"✅ آمن — {churn_pct:.1f}% مقاطعة فقط. الزيادة مقبولة."

        return ChurnPrediction(
            new_price              = round(new_price, 2),
            price_increase_pct     = round(price_increase_pct * 100, 1),
            ml_weighted_churn_pct  = round(churn_pct, 2),
            binary_churn_pct       = round(binary_churn * 100, 2),
            at_risk_population_pct = round(at_risk_pop, 1),
            segments_detail        = segments,
            ml_churn_prob          = round(ml_top * 100, 1),
            risk_level             = risk,
            recommendation         = rec,
            revenue_loss_estimate  = round(revenue_loss, 1),
            break_even_price       = break_even,
            price_ceiling          = round(price_ceiling, 2),
        )


if __name__ == "__main__":
    opt = PricingOptimizer()
    p   = ProductInput(current_price=25, current_weight_g=100,
                       cost_per_gram=0.18, area="Urban", purchase_freq=4, target_margin=0.30)

    w = opt.find_optimal_weight(p)
    print(f"A: {p.current_weight_g}g → {w.optimal_weight_g}g (−{w.weight_reduction_pct}%) margin={w.new_margin_pct}%")
    print(f"B: keep weight → margin={w.strategy_b_margin}%")
    print(f"C: {w.hybrid_recommendation}")

    ch = opt.predict_market_churn(p, 30.0)
    print(f"\nChurn (ML-weighted): {ch.ml_weighted_churn_pct}%")
    print(f"Break-even: {ch.break_even_price} EGP | Ceiling: {ch.price_ceiling} EGP")
    print(ch.recommendation)

    curve = opt.price_sensitivity_curve(p)
    sweet = curve[curve["is_sweet_spot"]].iloc[0]
    print(f"\nSweet spot: +{sweet.price_increase_pct}% → rev_index={sweet.revenue_index:.1f}")
