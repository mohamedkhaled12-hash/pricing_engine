"""
api/main.py
===========
FastAPI Backend — نقطة الوصول الوحيدة للـ Dashboard.

تشغيل:
  cd pricing_engine
  uvicorn api.main:app --reload --port 8000

Endpoints:
  GET  /                          → health check
  GET  /distribution/{area}       → بيانات المنحنيين للرسم
  POST /optimize-weight           → توصية A (الوزن الأمثل)
  POST /predict-churn             → توصية B (نسبة المقاطعة)
  POST /analyze                   → الاتنين مع بعض
  GET  /segments/{area}/{price}   → تفاصيل الـ affordability لكل bracket
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import sys
from pathlib import Path

# عشان الـ imports تشتغل من أي مكان
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.curve_fitting import get_curve_data, get_bracket_affordability
from core.optimizer import PricingOptimizer, ProductInput

app = FastAPI(
    title="Smart Pricing Engine",
    description="محرك التسعير واقتراح الأوزان الذكي — مبني على بيانات توزيع الدخل 2026",
    version="1.0.0",
)

# السماح للـ Streamlit Dashboard بالاتصال
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# تهيئة الـ Optimizer مرة واحدة عند بداية الـ server
optimizer = PricingOptimizer()


# ─────────────────────────────────────────────
# Pydantic Models للـ Request/Response
# ─────────────────────────────────────────────

class ProductRequest(BaseModel):
    current_price:    float = Field(..., gt=0, description="السعر الحالي بالجنيه")
    current_weight_g: float = Field(..., gt=0, description="الوزن الحالي بالجرام")
    cost_per_gram:    float = Field(..., gt=0, description="تكلفة الإنتاج لكل جرام")
    area:             str   = Field("Urban", pattern="^(Urban|Rural)$")
    purchase_freq:    int   = Field(4, ge=1, le=30, description="مرات الشراء شهرياً")
    target_margin:    float = Field(0.25, ge=0.05, le=0.60)

class ChurnRequest(ProductRequest):
    new_price: float = Field(..., gt=0, description="السعر الجديد المقترح")


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/")
def health():
    return {"status": "ok", "engine": "Smart Pricing Engine v1.0"}


@app.get("/distribution/{area}")
def get_distribution(area: str):
    """
    بيانات المنحنيين اللازمة لرسم شاشة رادار الانكماش.
    area: Urban أو Rural
    """
    if area not in ("Urban", "Rural"):
        raise HTTPException(400, "area must be Urban or Rural")
    return get_curve_data(area)


@app.get("/segments/{area}/{price}")
def get_segments(area: str, price: float, purchase_freq: int = 4):
    """
    هل كل bracket قادر على تحمل السعر ده؟
    """
    if area not in ("Urban", "Rural"):
        raise HTTPException(400, "area must be Urban or Rural")
    return get_bracket_affordability(price, area, purchase_freq)


@app.post("/optimize-weight")
def optimize_weight(req: ProductRequest):
    """
    توصية A: الوزن الأمثل للحفاظ على السعر النفسي وهامش الربح.
    """
    product = ProductInput(
        current_price    = req.current_price,
        current_weight_g = req.current_weight_g,
        cost_per_gram    = req.cost_per_gram,
        area             = req.area,
        purchase_freq    = req.purchase_freq,
        target_margin    = req.target_margin,
    )
    result = optimizer.find_optimal_weight(product)
    return {
        "optimal_weight_g":      result.optimal_weight_g,
        "weight_reduction_pct":  result.weight_reduction_pct,
        "new_margin_pct":        result.new_margin_pct,
        "price_per_gram_new":    result.price_per_gram_new,
        "price_per_gram_old":    result.price_per_gram_old,
        "feasible":              result.feasible,
        "warning":               result.warning,
    }


@app.post("/predict-churn")
def predict_churn(req: ChurnRequest):
    """
    توصية B: نسبة المقاطعة المتوقعة عند رفع السعر.
    """
    product = ProductInput(
        current_price    = req.current_price,
        current_weight_g = req.current_weight_g,
        cost_per_gram    = req.cost_per_gram,
        area             = req.area,
        purchase_freq    = req.purchase_freq,
        target_margin    = req.target_margin,
    )
    result = optimizer.predict_market_churn(product, req.new_price)
    return {
        "new_price":               result.new_price,
        "price_increase_pct":      result.price_increase_pct,
        "weighted_churn_pct":      result.weighted_churn_pct,
        "at_risk_population_pct":  result.at_risk_population_pct,
        "ml_churn_prob":           result.ml_churn_prob,
        "risk_level":              result.risk_level,
        "recommendation":          result.recommendation,
        "segments_detail":         result.segments_detail,
    }


@app.post("/analyze")
def full_analysis(req: ChurnRequest):
    """
    الاتنين مع بعض — الـ endpoint الرئيسي للـ Dashboard.
    """
    product = ProductInput(
        current_price    = req.current_price,
        current_weight_g = req.current_weight_g,
        cost_per_gram    = req.cost_per_gram,
        area             = req.area,
        purchase_freq    = req.purchase_freq,
        target_margin    = req.target_margin,
    )

    weight_rec  = optimizer.find_optimal_weight(product)
    churn_pred  = optimizer.predict_market_churn(product, req.new_price)
    segments    = get_bracket_affordability(req.current_price, req.area, req.purchase_freq)

    return {
        "weight_recommendation": {
            "optimal_weight_g":     weight_rec.optimal_weight_g,
            "weight_reduction_pct": weight_rec.weight_reduction_pct,
            "new_margin_pct":       weight_rec.new_margin_pct,
            "feasible":             weight_rec.feasible,
            "warning":              weight_rec.warning,
        },
        "churn_prediction": {
            "new_price":              churn_pred.new_price,
            "price_increase_pct":     churn_pred.price_increase_pct,
            "weighted_churn_pct":     churn_pred.weighted_churn_pct,
            "at_risk_population_pct": churn_pred.at_risk_population_pct,
            "risk_level":             churn_pred.risk_level,
            "recommendation":         churn_pred.recommendation,
            "ml_churn_prob":          churn_pred.ml_churn_prob,
            "segments_detail":        churn_pred.segments_detail,
        },
        "market_affordability": segments,
    }
