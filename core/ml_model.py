"""
core/ml_model.py  —  v3 Production
=====================================
الترقيات على v1:

FEATURE ENGINEERING: 15 features بدل 7
  price_burden_sq, log_monthly_disposable, burden_x_increase,
  burden_x_income, price_elasticity_proxy, necessity_score,
  affordability_ratio, shock_factor

TRAINING DATA: 61,200 صف (22 price points × 200 scenarios)
  نموذج اقتصادي واقعي: threshold متغير بالضرورة + المنطقة + الدخل

ENSEMBLE: XGBoost + LightGBM + GradientBoosting → soft voting
  كل موديل بيغطي ضعف التاني

HYPERPARAMETERS: Optuna tuning (اختياري، 30 trial)

METRICS: AUC + F1 + Brier Score
"""

import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

from core.data_loader import (
    load_distribution, load_macro_metrics,
    get_area_params, compute_disposable, enrich_distribution,
)

MODELS_DIR = Path(__file__).parent.parent / "models"
MODEL_PATH  = MODELS_DIR / "churn_model.pkl"

# ─────────────────────────────────────────────────────────
# FEATURES — 15
# ─────────────────────────────────────────────────────────
FEATURE_COLS = [
    "area_urban",
    "monthly_disposable",
    "income_percentile",
    "base_price",
    "price_increase_pct",
    "purchase_freq",
    "price_burden",
    # Engineered
    "price_burden_sq",
    "log_monthly_disposable",
    "burden_x_increase",
    "burden_x_income",
    "price_elasticity_proxy",
    "necessity_score",
    "affordability_ratio",
    "shock_factor",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """يضيف الـ 8 features المشتقة."""
    df = df.copy()
    threshold = 0.08 + 0.18 * df["income_percentile"]
    df["price_burden_sq"]        = df["price_burden"] ** 2
    df["log_monthly_disposable"] = np.log1p(df["monthly_disposable"])
    df["burden_x_increase"]      = df["price_burden"] * df["price_increase_pct"]
    df["burden_x_income"]        = df["price_burden"] * (1 - df["income_percentile"])
    df["price_elasticity_proxy"] = df["price_increase_pct"] / (df["price_burden"] + 1e-6)
    df["necessity_score"]        = df["purchase_freq"] / (df["base_price"] + 1)
    df["affordability_ratio"]    = df["price_burden"] / (threshold + 1e-6)
    df["shock_factor"]           = df["price_increase_pct"] ** 2
    return df


# ─────────────────────────────────────────────────────────
# TRAINING DATA
# ─────────────────────────────────────────────────────────
def _generate_training_data(n_per_config: int = 200) -> pd.DataFrame:
    np.random.seed(2024)
    df      = load_distribution()
    metrics = load_macro_metrics()
    df2026  = enrich_distribution(df, metrics)

    PRICE_POINTS = [
        3, 5, 7, 8, 10, 12, 14, 15, 18, 20, 22,
        25, 28, 30, 35, 40, 45, 50, 60, 75, 100, 150,
    ]
    records = []

    for area in ["Urban", "Rural"]:
        subset       = df2026[df2026["Area"] == area]
        is_urban     = 1 if area == "Urban" else 0
        area_penalty = 0.0 if area == "Urban" else 0.02

        for _, bracket_row in subset.iterrows():
            monthly_disp = bracket_row["monthly_disposable"]
            income_pct   = bracket_row["income_percentile"]

            for base_price in PRICE_POINTS:
                if base_price < 15:
                    freq_opts, freq_p = [4,6,8,10,12], [0.15,0.25,0.3,0.2,0.1]
                elif base_price < 40:
                    freq_opts, freq_p = [2,3,4,6,8],   [0.2,0.3,0.25,0.15,0.1]
                else:
                    freq_opts, freq_p = [1,2,3,4],     [0.35,0.35,0.2,0.1]

                for _ in range(n_per_config):
                    price_increase = np.random.uniform(0.0, 0.80)
                    new_price      = base_price * (1 + price_increase)
                    freq           = np.random.choice(freq_opts, p=freq_p)
                    burden         = (new_price * freq) / max(monthly_disp, 50)
                    necessity      = freq / (base_price + 1)

                    base_thresh     = 0.08 + 0.18 * income_pct
                    necessity_bonus = min(necessity * 0.5, 0.05)
                    threshold       = base_thresh + necessity_bonus - area_penalty

                    excess      = burden - threshold
                    shock       = price_increase ** 2
                    burden_sq   = burden ** 2
                    churn_logit = (
                        6.0 * excess
                        + 2.5 * price_increase
                        + 1.5 * shock
                        + 0.8 * burden_sq
                        - 1.8
                    )
                    churn_prob = float(np.clip(1 / (1 + np.exp(-churn_logit)), 0.01, 0.99))
                    did_churn  = int(np.random.random() < churn_prob)

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

    return engineer_features(pd.DataFrame(records))


# ─────────────────────────────────────────────────────────
# BUILD MODELS
# ─────────────────────────────────────────────────────────
def _build_estimators(best_xgb_params: dict = None):
    """يبني الـ 3 estimators بـ params الأمثل."""
    estimators = []

    if HAS_XGB:
        params = best_xgb_params or {
            "n_estimators": 400, "max_depth": 5, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 10,
            "gamma": 0.1, "reg_alpha": 0.1, "reg_lambda": 1.0,
            "random_state": 42, "eval_metric": "auc",
            "use_label_encoder": False, "verbosity": 0,
        }
        estimators.append(("xgb", xgb.XGBClassifier(**params)))

    if HAS_LGB:
        estimators.append(("lgb", lgb.LGBMClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            num_leaves=63, subsample=0.8, colsample_bytree=0.8,
            min_child_samples=20, reg_alpha=0.1, reg_lambda=0.5,
            random_state=42, verbose=-1,
        )))

    estimators.append(("gb", GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.08, max_depth=5,
        min_samples_leaf=15, subsample=0.8, random_state=42,
    )))

    # weights: XGB و LGB أقوى → وزن أعلى
    weights = []
    for name, _ in estimators:
        if name in ("xgb", "lgb"):
            weights.append(2)
        else:
            weights.append(1)

    return estimators, weights


def _tune_xgb_optuna(X, y, n_trials: int = 30) -> dict:
    """Optuna: يجرب n_trials combination لـ XGBoost."""
    if not (HAS_OPTUNA and HAS_XGB):
        return {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 200, 600),
            "max_depth":        trial.suggest_int("max_depth", 3, 7),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 30),
            "gamma":            trial.suggest_float("gamma", 0.0, 0.5),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.5, 3.0),
            "random_state": 42, "eval_metric": "auc",
            "use_label_encoder": False, "verbosity": 0,
        }
        m = xgb.XGBClassifier(**params)
        return cross_val_score(m, X, y, cv=cv, scoring="roc_auc").mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# ─────────────────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────────────────
def train_churn_model(save: bool = True, tune: bool = False) -> dict:
    """
    tune=True  → Optuna يضبط XGBoost (~3 دقايق)
    tune=False → params افتراضية جيدة (~90 ثانية)
    """
    print("=" * 52)
    print("  Qystas ML Engine v3 — Ensemble Training")
    print("=" * 52)

    print("\n[1/4] Generating training data...")
    train_df = _generate_training_data(n_per_config=200)
    X = train_df[FEATURE_COLS]
    y = train_df["churned"]
    print(f"      {X.shape[0]:,} rows × {X.shape[1]} features | churn={y.mean():.3f}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ── Optuna tuning (اختياري) ──
    best_xgb_params = None
    if tune and HAS_OPTUNA and HAS_XGB:
        print("\n[2/4] Optuna tuning XGBoost (30 trials)...")
        best_xgb_params = _tune_xgb_optuna(X, y, n_trials=30)
        best_xgb_params.update({
            "random_state": 42, "eval_metric": "auc",
            "use_label_encoder": False, "verbosity": 0,
        })
        print(f"      Best: n_est={best_xgb_params.get('n_estimators')} "
              f"depth={best_xgb_params.get('max_depth')} "
              f"lr={best_xgb_params.get('learning_rate', 0):.4f}")
    else:
        print(f"\n[2/4] Using default hyperparameters "
              f"({'tune=False' if not tune else 'optuna/xgb not available'})...")

    # ── Build + CV each estimator ──
    print("\n[3/4] Cross-validating individual models...")
    estimators, weights = _build_estimators(best_xgb_params)
    for name, m_ in estimators:
        s = cross_val_score(m_, X, y, cv=cv, scoring="roc_auc")
        print(f"      {name.upper():8s} AUC: {s.mean():.4f} ± {s.std():.4f}")

    # ── Ensemble ──
    print("\n[4/4] Ensemble (soft voting)...")
    ensemble = VotingClassifier(estimators=estimators, voting="soft", weights=weights)
    ens_scores = cross_val_score(ensemble, X, y, cv=cv, scoring="roc_auc")
    ens_preds  = cross_val_predict(ensemble, X, y, cv=cv, method="predict_proba")[:, 1]
    ens_binary = (ens_preds > 0.5).astype(int)

    auc_cv   = float(ens_scores.mean())
    auc_full = float(roc_auc_score(y, ens_preds))
    f1       = float(f1_score(y, ens_binary))
    brier    = float(brier_score_loss(y, ens_preds))

    print(f"\n      {'Metric':<22} Value")
    print(f"      {'─'*34}")
    print(f"      {'CV AUC':<22} {auc_cv:.4f} ± {ens_scores.std():.4f}")
    print(f"      {'Full-data AUC':<22} {auc_full:.4f}")
    print(f"      {'F1 Score':<22} {f1:.4f}")
    print(f"      {'Brier Score':<22} {brier:.4f}  ↓ lower=better")

    # ── Fit final ──
    ensemble.fit(X, y)

    # Feature importance (XGB + LGB اللي اتدربوا)
    importances = {}
    for name, m_ in estimators:
        if name == "xgb" and HAS_XGB:
            m_.fit(X, y)
            for feat, imp in zip(FEATURE_COLS, m_.feature_importances_):
                importances[feat] = importances.get(feat, 0) + imp / len(estimators)
        elif name == "lgb" and HAS_LGB:
            m_.fit(X, y)
            norm = m_.feature_importances_ / (m_.feature_importances_.sum() + 1e-9)
            for feat, imp in zip(FEATURE_COLS, norm):
                importances[feat] = importances.get(feat, 0) + imp / len(estimators)
        elif name == "gb":
            m_.fit(X, y)
            for feat, imp in zip(FEATURE_COLS, m_.feature_importances_):
                importances[feat] = importances.get(feat, 0) + imp / len(estimators)

    print("\n      Feature Importances (avg all models):")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        bar = "█" * int(imp * 30)
        print(f"        {feat:30s}: {imp:.4f} {bar}")

    metrics = {
        "cv_auc_mean":  round(auc_cv, 4),
        "cv_auc_std":   round(float(ens_scores.std()), 4),
        "full_auc":     round(auc_full, 4),
        "f1":           round(f1, 4),
        "brier":        round(brier, 4),
    }

    if save:
        MODELS_DIR.mkdir(exist_ok=True)
        payload = {
            "ensemble":     ensemble,
            "feature_cols": FEATURE_COLS,
            "metrics":      metrics,
            "importances":  {k: round(v, 4) for k, v in importances.items()},
            "n_samples":    len(train_df),
        }
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(payload, f)
        print(f"\n      ✅ Saved → {MODEL_PATH}")

    print(f"\n{'='*52}")
    print(f"  v3 Complete | CV AUC={auc_cv:.4f} | F1={f1:.4f}")
    print(f"{'='*52}\n")

    return {"model": ensemble, "n_samples": len(train_df), **metrics,
            "importances": importances}


# ─────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────
def load_churn_model():
    if not MODEL_PATH.exists():
        print("Model not found — training now (90s)...")
        return train_churn_model(save=True, tune=False)["model"]
    with open(MODEL_PATH, "rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and "ensemble" in payload:
        return payload["ensemble"]
    return payload  # backward compat


def get_model_metrics() -> dict:
    if not MODEL_PATH.exists():
        return {}
    with open(MODEL_PATH, "rb") as f:
        payload = pickle.load(f)
    return payload.get("metrics", {}) if isinstance(payload, dict) else {}


# ─────────────────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────────────────
def predict_churn(
    model,
    monthly_disposable: float,
    income_percentile:  float,
    base_price:         float,
    price_increase_pct: float,
    purchase_freq:      int,
    area_urban:         int = 1,
) -> dict:
    new_price    = base_price * (1 + price_increase_pct)
    price_burden = (new_price * purchase_freq) / max(monthly_disposable, 1)
    threshold    = 0.08 + 0.18 * income_percentile
    necessity    = purchase_freq / (base_price + 1)

    row = {
        "area_urban":             area_urban,
        "monthly_disposable":     monthly_disposable,
        "income_percentile":      income_percentile,
        "base_price":             base_price,
        "price_increase_pct":     price_increase_pct,
        "purchase_freq":          purchase_freq,
        "price_burden":           price_burden,
        "price_burden_sq":        price_burden ** 2,
        "log_monthly_disposable": np.log1p(monthly_disposable),
        "burden_x_increase":      price_burden * price_increase_pct,
        "burden_x_income":        price_burden * (1 - income_percentile),
        "price_elasticity_proxy": price_increase_pct / (price_burden + 1e-6),
        "necessity_score":        necessity,
        "affordability_ratio":    price_burden / (threshold + 1e-6),
        "shock_factor":           price_increase_pct ** 2,
    }

    features = pd.DataFrame([row])[FEATURE_COLS]
    prob     = float(model.predict_proba(features)[0][1])

    return {
        "churn_probability": round(prob, 3),
        "risk_level":        "HIGH" if prob > 0.5 else "MEDIUM" if prob > 0.25 else "LOW",
        "price_burden_pct":  round(price_burden * 100, 1),
    }


if __name__ == "__main__":
    result = train_churn_model(save=True, tune=False)
    print(f"AUC={result['cv_auc_mean']:.4f} | F1={result['f1']:.4f}")
