"""
core/ml_model.py  -  v4 Production (fast & reliable)
=====================================================
Why this changed from v3:

v3's train_churn_model() ran Cross-Validation on each estimator
individually (XGBoost, LightGBM, GradientBoosting) AND on the final
VotingClassifier - meaning every model effectively trained 2-3 times.
On 61,200 rows this took several minutes. If the saved model file was
ever missing on Streamlit Cloud, the app tried to redo this heavy
training *inside the live user request*, so the UI sat on
"Initializing AI engine..." until it finished or timed out.

v4 fix:
  1. The default path (train_churn_model()) is now a single
     GradientBoosting model - trains in seconds, not minutes.
  2. The full strong Ensemble (XGB+LGB+GB with optional Optuna) still
     exists, but only as an explicit opt-in:
     train_churn_model(mode="ensemble"). You run this once locally,
     save the resulting file, and commit it to GitHub - it never
     trains during a live app run.
  3. load_churn_model() checks the saved file and if anything is
     wrong (missing/corrupted/old format) it falls back immediately
     to the fast single-model path, never to the heavy ensemble.

Net result: maximum strength (ensemble) is always available for
anyone who wants it locally, but production (Streamlit Cloud) never
depends on it directly - it depends on a pretrained file, and any
emergency fallback is fast by design.
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

FEATURE_COLS = [
    "area_urban",
    "monthly_disposable",
    "income_percentile",
    "base_price",
    "price_increase_pct",
    "purchase_freq",
    "price_burden",
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


def _generate_training_data(n_per_config: int = 40) -> pd.DataFrame:
    """n_per_config: 40 -> ~5,000 rows fast; 200 -> ~61,000 rows ensemble."""
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


def _train_fast_model(n_per_config: int = 40) -> dict:
    """Default model: a single GradientBoosting. Trains in seconds."""
    train_df = _generate_training_data(n_per_config=n_per_config)
    X = train_df[FEATURE_COLS]
    y = train_df["churned"]

    model = GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.1, max_depth=4,
        min_samples_leaf=20, subsample=0.8, random_state=42,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    model.fit(X, y)

    importances = dict(zip(FEATURE_COLS, model.feature_importances_.round(4)))

    metrics = {
        "cv_auc_mean": round(float(scores.mean()), 4),
        "cv_auc_std":  round(float(scores.std()), 4),
        "full_auc":    round(float(roc_auc_score(y, model.predict_proba(X)[:, 1])), 4),
        "f1":          round(float(f1_score(y, model.predict(X))), 4),
        "brier":       round(float(brier_score_loss(y, model.predict_proba(X)[:, 1])), 4),
        "mode":        "fast",
    }

    return {
        "model": model, "n_samples": len(train_df),
        "importances": importances, **metrics,
    }


def _build_estimators(best_xgb_params: dict = None):
    """Strong Ensemble: XGBoost + LightGBM + GradientBoosting."""
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

    weights = [2 if name in ("xgb", "lgb") else 1 for name, _ in estimators]
    return estimators, weights


def _tune_xgb_optuna(X, y, n_trials: int = 30) -> dict:
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

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


def _train_ensemble_model(tune: bool = False, n_per_config: int = 200) -> dict:
    """
    Strongest model. Run this locally only:
        python -c "from core.ml_model import train_churn_model; train_churn_model(mode='ensemble')"
    Takes a few minutes. Commit the resulting file to your repo.
    """
    print("=" * 52)
    print("  Qystas ML Engine - Ensemble Training (local only)")
    print("=" * 52)

    print("\n[1/4] Generating training data...")
    train_df = _generate_training_data(n_per_config=n_per_config)
    X = train_df[FEATURE_COLS]
    y = train_df["churned"]
    print(f"      {X.shape[0]:,} rows x {X.shape[1]} features | churn={y.mean():.3f}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_xgb_params = None
    if tune and HAS_OPTUNA and HAS_XGB:
        print("\n[2/4] Optuna tuning XGBoost (30 trials, 3-fold)...")
        best_xgb_params = _tune_xgb_optuna(X, y, n_trials=30)
        best_xgb_params.update({
            "random_state": 42, "eval_metric": "auc",
            "use_label_encoder": False, "verbosity": 0,
        })
    else:
        print("\n[2/4] Using default hyperparameters (tune=False)...")

    print("\n[3/4] Building ensemble...")
    estimators, weights = _build_estimators(best_xgb_params)

    print("\n[4/4] Cross-validating ensemble + fitting final model...")
    ensemble   = VotingClassifier(estimators=estimators, voting="soft", weights=weights)
    ens_scores = cross_val_score(ensemble, X, y, cv=cv, scoring="roc_auc")
    ens_preds  = cross_val_predict(ensemble, X, y, cv=cv, method="predict_proba")[:, 1]
    ens_binary = (ens_preds > 0.5).astype(int)

    auc_cv   = float(ens_scores.mean())
    auc_full = float(roc_auc_score(y, ens_preds))
    f1       = float(f1_score(y, ens_binary))
    brier    = float(brier_score_loss(y, ens_preds))

    ensemble.fit(X, y)

    importances = {}
    for name, m_ in estimators:
        if name == "xgb" and HAS_XGB:
            for feat, imp in zip(FEATURE_COLS, m_.feature_importances_):
                importances[feat] = importances.get(feat, 0) + imp / len(estimators)
        elif name == "lgb" and HAS_LGB:
            norm = m_.feature_importances_ / (m_.feature_importances_.sum() + 1e-9)
            for feat, imp in zip(FEATURE_COLS, norm):
                importances[feat] = importances.get(feat, 0) + imp / len(estimators)
        elif name == "gb":
            for feat, imp in zip(FEATURE_COLS, m_.feature_importances_):
                importances[feat] = importances.get(feat, 0) + imp / len(estimators)

    print(f"\n      CV AUC: {auc_cv:.4f} +/- {ens_scores.std():.4f}  |  F1: {f1:.4f}  |  Brier: {brier:.4f}")

    metrics = {
        "cv_auc_mean": round(auc_cv, 4), "cv_auc_std": round(float(ens_scores.std()), 4),
        "full_auc": round(auc_full, 4), "f1": round(f1, 4), "brier": round(brier, 4),
        "mode": "ensemble",
    }

    return {"model": ensemble, "n_samples": len(train_df),
            "importances": importances, **metrics}


def train_churn_model(save: bool = True, mode: str = "fast", tune: bool = False) -> dict:
    """
    mode="fast"     (default)  -> single GradientBoosting, seconds. Safe
                       to call inside a live request if ever needed.
    mode="ensemble"             -> XGBoost+LightGBM+GradientBoosting,
                       several minutes. Run locally once, commit the
                       resulting models/churn_model.pkl to your repo.
                       tune=True adds Optuna hyperparameter search.
    """
    if mode == "ensemble":
        result = _train_ensemble_model(tune=tune, n_per_config=200)
    else:
        result = _train_fast_model(n_per_config=40)

    print(f"\n{'='*52}")
    print(f"  Training complete [{result['mode']}] | "
          f"CV AUC={result['cv_auc_mean']:.4f} | F1={result['f1']:.4f} | "
          f"n={result['n_samples']:,}")
    print(f"{'='*52}\n")

    if save:
        MODELS_DIR.mkdir(exist_ok=True)
        payload = {
            "ensemble":     result["model"],
            "feature_cols": FEATURE_COLS,
            "metrics": {k: v for k, v in result.items()
                        if k not in ("model", "importances", "n_samples")},
            "importances":  {k: round(v, 4) for k, v in result["importances"].items()},
            "n_samples":    result["n_samples"],
        }
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(payload, f)
        print(f"      Saved -> {MODEL_PATH}  (mode={result['mode']})")

    return result


def load_churn_model():
    """
    Loads the saved model.
    - valid file present -> instant load
    - missing/corrupted   -> fast fallback (GradientBoosting, seconds only,
      NEVER the heavy ensemble, so the Dashboard never hangs)
    """
    if MODEL_PATH.exists():
        try:
            with open(MODEL_PATH, "rb") as f:
                payload = pickle.load(f)
            if isinstance(payload, dict) and "ensemble" in payload:
                if hasattr(payload["ensemble"], "predict_proba"):
                    return payload["ensemble"]
            elif hasattr(payload, "predict_proba"):
                return payload
        except Exception:
            pass

    print("No valid pretrained model found - training a fast fallback "
          "(single GradientBoosting, seconds not minutes). "
          "For maximum accuracy, run train_churn_model(mode='ensemble') "
          "locally and commit models/churn_model.pkl.")
    return _train_fast_model(n_per_config=40)["model"]


def get_model_metrics() -> dict:
    if not MODEL_PATH.exists():
        return {}
    try:
        with open(MODEL_PATH, "rb") as f:
            payload = pickle.load(f)
        return payload.get("metrics", {}) if isinstance(payload, dict) else {}
    except Exception:
        return {}


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
    import sys
    mode = "ensemble" if "--ensemble" in sys.argv else "fast"
    tune = "--tune" in sys.argv
    result = train_churn_model(save=True, mode=mode, tune=tune)
    print(f"AUC={result['cv_auc_mean']:.4f} | F1={result['f1']:.4f}")
