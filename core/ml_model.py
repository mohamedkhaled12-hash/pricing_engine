"""
core/ml_model.py
================
بناء وتدريب الـ ML Model للـ Churn Rate Prediction.

ليه محتاجين ML هنا وليس مجرد معادلة؟
===========================================
المعادلة البسيطة (price_burden > threshold → churn) بتفترض
إن كل الناس في نفس الـ bracket بيتصرفوا نفس التصرف.
الـ ML Model بياخد في الاعتبار:
  - تأثير نسبة الزيادة نفسها (مش بس السعر الجديد)
  - تكرار الشراء (منتج ضروري vs. كمالي)
  - الـ income_percentile (حساسية الفئة)
  - تفاعل الـ features مع بعض (interaction effects)

الـ Training Data:
  - مولّدة بشكل برمجي من بيانات 2026 الحقيقية
  - بتمثل سيناريوهات سعرية متعددة لكل bracket
  - الـ label (churned) مبنية على نموذج اقتصادي للمرونة السعرية
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from core.data_loader import (
    load_distribution, load_macro_metrics,
    get_area_params, compute_disposable, enrich_distribution,
)

MODELS_DIR = Path(__file__).parent.parent / "models"
MODEL_PATH  = MODELS_DIR / "churn_model.pkl"

# الـ Features اللي الموديل بيتدرب عليها
FEATURE_COLS = [
    "area_urban",           # 1=حضر, 0=ريف
    "monthly_disposable",   # الدخل المتاح الشهري بالجنيه
    "income_percentile",    # مكانة الفئة (0=أفقر, 1=أغنى)
    "base_price",           # السعر الحالي للمنتج
    "price_increase_pct",   # نسبة الزيادة المقترحة (0.0 – 1.0)
    "purchase_freq",        # تكرار الشراء شهرياً
    "price_burden",         # (new_price × freq) / monthly_disposable
]


def _generate_training_data(n_per_config: int = 80) -> pd.DataFrame:
    """
    توليد بيانات تدريب واقعية من توزيع 2026.

    المنطق الاقتصادي للـ label:
    --------------------------------
    price_burden = ما يدفعه المستهلك على المنتج / دخله المتاح الشهري

    threshold (عتبة المقاطعة) تتفاوت حسب الدخل:
        فقراء  (income_pct≈0): يقاطعون عند 8%  (كل جنيه بحساب)
        أثرياء (income_pct≈1): يقاطعون عند 26% (أكثر تساهلاً)
        المعادلة: threshold = 0.08 + 0.18 × income_percentile

    churn_probability = sigmoid( 5×(burden - threshold) + 2×increase_pct - 1.5 )
    """
    np.random.seed(42)
    df       = load_distribution()
    metrics  = load_macro_metrics()
    df2026   = enrich_distribution(df, metrics)

    PRICE_POINTS = [5, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40, 50, 60, 75, 100, 120, 150]
    AREAS        = ["Urban", "Rural"]
    records      = []

    for area in AREAS:
        subset = df2026[df2026["Area"] == area]
        is_urban = 1 if area == "Urban" else 0

        for _, bracket_row in subset.iterrows():
            monthly_disp    = bracket_row["monthly_disposable"]
            income_pct      = bracket_row["income_percentile"]
            churn_threshold = 0.08 + 0.18 * income_pct

            for base_price in PRICE_POINTS:
                # تكرار الشراء يعتمد على مستوى السعر
                if base_price < 20:
                    freq_options, freq_probs = [4, 6, 8, 10], [0.2, 0.3, 0.3, 0.2]
                elif base_price < 50:
                    freq_options, freq_probs = [2, 3, 4, 6],  [0.25, 0.35, 0.25, 0.15]
                else:
                    freq_options, freq_probs = [1, 2, 3],      [0.4, 0.4, 0.2]

                for _ in range(n_per_config):
                    price_increase = np.random.uniform(0.0, 0.70)
                    new_price      = base_price * (1 + price_increase)
                    freq           = np.random.choice(freq_options, p=freq_probs)
                    burden         = (new_price * freq) / max(monthly_disp, 50)

                    excess         = burden - churn_threshold
                    churn_logit    = 5.0 * excess + 2.0 * price_increase - 1.5
                    churn_prob     = float(np.clip(1 / (1 + np.exp(-churn_logit)), 0.02, 0.98))
                    did_churn      = int(np.random.random() < churn_prob)

                    records.append({
                        "area_urban":         is_urban,
                        "monthly_disposable": round(monthly_disp, 2),
                        "income_percentile":  round(income_pct, 4),
                        "base_price":         base_price,
                        "price_increase_pct": round(price_increase, 4),
                        "purchase_freq":      int(freq),
                        "price_burden":       round(burden, 5),
                        "churned":            did_churn,
                    })

    return pd.DataFrame(records)


def train_churn_model(save: bool = True) -> dict:
    """
    يدرب الـ Gradient Boosting model ويحفظه.
    بيرجع dict فيه الموديل ونتائج الـ Cross-Validation.

    ليه Gradient Boosting؟
    - بيتعامل مع الـ non-linear interactions كويس
    - أفضل من Random Forest على بيانات ذات scale متباين
    - مش محتاج feature scaling (بخلاف Logistic Regression)
    - بيعطي feature importances مفيدة للـ explainability
    """
    print("Generating training data...")
    train_df = _generate_training_data(n_per_config=80)
    print(f"  Dataset: {train_df.shape[0]:,} rows | churn_rate={train_df['churned'].mean():.3f}")

    X = train_df[FEATURE_COLS]
    y = train_df["churned"]

    model = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.10,
        max_depth=4,
        min_samples_leaf=20,
        subsample=0.8,
        random_state=42,
    )

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    print(f"  CV AUC: {auc_scores.mean():.3f} ± {auc_scores.std():.3f}")

    # Train على كل الداتا
    model.fit(X, y)

    importances = dict(zip(FEATURE_COLS, model.feature_importances_.round(3)))
    print("  Feature importances:")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        bar = "█" * int(imp * 30)
        print(f"    {feat:25s}: {imp:.3f} {bar}")

    if save:
        MODELS_DIR.mkdir(exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        print(f"  Model saved → {MODEL_PATH}")

    return {
        "model":        model,
        "cv_auc_mean":  round(auc_scores.mean(), 3),
        "cv_auc_std":   round(auc_scores.std(), 3),
        "importances":  importances,
        "n_samples":    len(train_df),
    }


def load_churn_model() -> GradientBoostingClassifier:
    """يحمل الموديل المحفوظ أو يدربه لو مش موجود."""
    if not MODEL_PATH.exists():
        print("Model not found — training now...")
        result = train_churn_model(save=True)
        return result["model"]
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict_churn(
    model,
    monthly_disposable: float,
    income_percentile: float,
    base_price: float,
    price_increase_pct: float,
    purchase_freq: int,
    area_urban: int = 1,
) -> dict:
    """
    يتنبأ بـ Churn Probability لزبون واحد.

    Returns:
    {
      churn_probability: 0.0 – 1.0
      risk_level:        'LOW' | 'MEDIUM' | 'HIGH'
      price_burden:      نسبة الإنفاق على المنتج من الدخل المتاح
    }
    """
    new_price    = base_price * (1 + price_increase_pct)
    price_burden = (new_price * purchase_freq) / max(monthly_disposable, 1)

    import pandas as _pd
    features = _pd.DataFrame([[
        area_urban, monthly_disposable, income_percentile,
        base_price, price_increase_pct, purchase_freq, price_burden,
    ]], columns=FEATURE_COLS)

    prob = float(model.predict_proba(features)[0][1])

    return {
        "churn_probability": round(prob, 3),
        "risk_level":        "HIGH" if prob > 0.5 else "MEDIUM" if prob > 0.25 else "LOW",
        "price_burden_pct":  round(price_burden * 100, 1),
    }


if __name__ == "__main__":
    result = train_churn_model(save=True)
    print(f"\nModel ready | AUC={result['cv_auc_mean']}")
