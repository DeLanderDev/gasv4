"""
Gas Price Predictor - Model Training & Prediction
Ensemble of change-based models with auto-calibrated shrinkage.

Key design decisions:
  1. Predict weekly CHANGE (not absolute price)
  2. Triple ensemble: XGBoost-full + XGBoost-selected + Ridge
  3. Auto-calibrated shrinkage: dampens predictions toward zero
     because models consistently over-predict change magnitude.
     Optimized to maximize the ±$0.02 accuracy rate.
  4. Recent-data weighting: last 40% of data gets 2x weight
"""

import json
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from config import (
    METRICS_FILE,
    MIN_TRAINING_WEEKS,
    MODEL_FILE,
    VALIDATION_WEEKS,
    XGBOOST_PARAMS,
)
from feature_engine import (
    create_features,
    get_feature_columns,
    prepare_prediction_row,
)

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _sample_weights(n: int, recent_frac: float = 0.4) -> np.ndarray:
    """Recent data gets 2× weight; older data ramps from 0.5 → 1.0."""
    w = np.ones(n)
    cut = int(n * (1 - recent_frac))
    w[:cut] = np.linspace(0.5, 1.0, cut)
    w[cut:] = 2.0
    return w


def _top_features(X, y, names, top_n=35):
    """Quick XGBoost to rank features, return top N names."""
    m = XGBRegressor(**{**XGBOOST_PARAMS, "n_estimators": 100})
    m.fit(X[names], y, verbose=False)
    fi = pd.Series(m.feature_importances_, index=names)
    return fi.sort_values(ascending=False).head(top_n).index.tolist()


def _calibrate_shrinkage(
    raw_pred_changes: np.ndarray,
    actual_abs: np.ndarray,
    base_prices: np.ndarray,
    target_cents: float = 0.02,
) -> float:
    """
    Find the shrinkage factor s ∈ [0, 1] that maximizes
    the % of predictions within ±target_cents of actual.
    
    final_change = raw_change × s
    prediction   = base_price + final_change
    """
    best_s = 0.0
    best_rate = 0.0

    for s in np.arange(0.0, 1.01, 0.02):
        dampened = raw_pred_changes * s
        pred_abs = base_prices + dampened
        rate = (np.abs(pred_abs - actual_abs) <= target_cents).mean()
        # Prefer higher shrinkage (more model signal) when tied
        if rate > best_rate or (rate == best_rate and s > best_s):
            best_rate = rate
            best_s = s

    return round(best_s, 2)


# ═══════════════════════════════════════════════════════════════════════════════
#  Model
# ═══════════════════════════════════════════════════════════════════════════════

class GasPriceModel:
    """
    Ensemble gas price model predicting weekly CHANGE with shrinkage.

    Sub-models:
      1. XGBoost on all features
      2. XGBoost on top-35 features (less overfit)
      3. Ridge on top-35 features (linear regularized)

    Pipeline:
      raw_change = 0.40 × M1 + 0.35 × M2 + 0.25 × M3
      final_change = raw_change × shrinkage
      prediction = current_price + final_change
    """

    def __init__(self):
        self.xgb_full: Optional[XGBRegressor] = None
        self.xgb_sel: Optional[XGBRegressor] = None
        self.ridge: Optional[Ridge] = None
        self.scaler: Optional[StandardScaler] = None

        self.all_features: List[str] = []
        self.sel_features: List[str] = []
        self.ew = [0.40, 0.35, 0.25]  # ensemble weights

        self.shrinkage: float = 0.5  # auto-tuned in train()
        self.bias: float = 0.0       # auto-tuned in train()
        self.metrics: Dict = {}
        self.validation_results: Optional[pd.DataFrame] = None
        self.is_trained: bool = False

    # ─── Data Prep ────────────────────────────────────────────────────────

    def _prep(self, df):
        """Create features, return (X, y_change, y_abs, bases, feat_names, feat_df)."""
        fdf = create_features(df)
        names = get_feature_columns(fdf)
        fdf["target_change"] = fdf["target"] - fdf["gas_price"]
        v = fdf[names + ["target_change", "target", "gas_price"]].dropna()
        return (
            v[names], v["target_change"], v["target"],
            v["gas_price"], names, fdf,
        )

    def _raw_ensemble(self, X_row):
        """Get raw ensemble change prediction for one or more rows."""
        p1 = self.xgb_full.predict(X_row[self.all_features])
        p2 = self.xgb_sel.predict(X_row[self.sel_features])
        X_sc = self.scaler.transform(X_row[self.sel_features])
        p3 = self.ridge.predict(X_sc)
        return self.ew[0] * p1 + self.ew[1] * p2 + self.ew[2] * p3

    # ─── Training ─────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame) -> Dict:
        """Train ensemble and auto-calibrate shrinkage."""
        X, y_chg, y_abs, bases, names, fdf = self._prep(df)

        if len(X) < MIN_TRAINING_WEEKS:
            raise ValueError(f"Need {MIN_TRAINING_WEEKS} weeks, got {len(X)}.")

        self.all_features = names
        sw = _sample_weights(len(X))

        # ── Fit 3 models on ALL data ──────────────────────────────────────
        self.xgb_full = XGBRegressor(**XGBOOST_PARAMS)
        self.xgb_full.fit(X, y_chg, sample_weight=sw, verbose=False)

        self.sel_features = _top_features(X, y_chg, names, top_n=35)

        self.xgb_sel = XGBRegressor(**XGBOOST_PARAMS)
        self.xgb_sel.fit(
            X[self.sel_features], y_chg, sample_weight=sw, verbose=False
        )

        self.scaler = StandardScaler()
        Xsc = self.scaler.fit_transform(X[self.sel_features])
        self.ridge = Ridge(alpha=1.0)
        self.ridge.fit(Xsc, y_chg, sample_weight=sw)

        self.is_trained = True

        # ── Calibrate shrinkage on the most recent ~1 year ────────────────
        cal_n = min(52, len(X) // 4)
        cal_sl = slice(len(X) - cal_n, None)
        raw_cal = self._raw_ensemble(X.iloc[cal_sl])
        self.shrinkage = _calibrate_shrinkage(
            raw_cal, y_abs.values[cal_sl], bases.values[cal_sl]
        )

        # ── Calibrate bias (mean residual after shrinkage) ────────────────
        dampened_cal = raw_cal * self.shrinkage
        pred_cal = bases.values[cal_sl] + dampened_cal
        self.bias = float(np.mean(pred_cal - y_abs.values[cal_sl]))

        # ── In-sample metrics (with shrinkage applied) ────────────────────
        raw_all = self._raw_ensemble(X)
        final_chg = raw_all * self.shrinkage - self.bias
        pred_abs_all = bases.values + final_chg

        self.metrics = {
            "mae": mean_absolute_error(y_abs, pred_abs_all),
            "rmse": np.sqrt(mean_squared_error(y_abs, pred_abs_all)),
            "r2": r2_score(y_abs, pred_abs_all),
            "change_mae": mean_absolute_error(y_chg, final_chg),
            "shrinkage": self.shrinkage,
            "bias": round(self.bias, 5),
            "n_samples": len(X),
            "n_features": len(names),
            "n_selected_features": len(self.sel_features),
            "trained_at": datetime.now().isoformat(),
        }
        return self.metrics

    # ─── Walk-Forward Validation ──────────────────────────────────────────

    def walk_forward_validate(self, df: pd.DataFrame) -> Tuple[Dict, pd.DataFrame]:
        """
        Walk-forward validation with per-step shrinkage calibration.
        At each step:
          1. Train ensemble on expanding window
          2. Calibrate shrinkage on last 26 weeks of training data
          3. Predict next week's change with shrinkage applied
        """
        X, y_chg, y_abs, bases, names, fdf = self._prep(df)

        n = len(X)
        test_n = min(VALIDATION_WEEKS, n // 3)
        test_start = n - test_n

        if test_start < MIN_TRAINING_WEEKS:
            raise ValueError("Not enough data for walk-forward validation.")

        # Get dates aligned to valid rows
        valid_idx = fdf[names + ["target_change", "target", "gas_price"]].dropna().index
        date_s = fdf.loc[valid_idx, "date"].reset_index(drop=True)

        preds_abs, acts_abs, preds_chg, acts_chg, dates = [], [], [], [], []

        for i in range(test_start, n):
            Xtr, ytr = X.iloc[:i], y_chg.iloc[:i]
            btr, atr = bases.iloc[:i], y_abs.iloc[:i]
            sw = _sample_weights(len(Xtr))

            # Fit 3 models
            sel = _top_features(Xtr, ytr, names, top_n=35)

            m1 = XGBRegressor(**XGBOOST_PARAMS)
            m1.fit(Xtr, ytr, sample_weight=sw, verbose=False)

            m2 = XGBRegressor(**XGBOOST_PARAMS)
            m2.fit(Xtr[sel], ytr, sample_weight=sw, verbose=False)

            sc = StandardScaler()
            Xsc = sc.fit_transform(Xtr[sel])
            m3 = Ridge(alpha=1.0)
            m3.fit(Xsc, ytr, sample_weight=sw)

            # Raw ensemble on calibration window (last 26 weeks of train)
            cal_n = min(26, len(Xtr) // 4)
            cal_sl = slice(len(Xtr) - cal_n, None)

            p1c = m1.predict(Xtr.iloc[cal_sl])
            p2c = m2.predict(Xtr.iloc[cal_sl][sel])
            p3c = m3.predict(sc.transform(Xtr.iloc[cal_sl][sel]))
            raw_cal = self.ew[0] * p1c + self.ew[1] * p2c + self.ew[2] * p3c

            # Calibrate shrinkage on calibration window
            shrink = _calibrate_shrinkage(
                raw_cal, atr.values[cal_sl], btr.values[cal_sl]
            )

            # Calibrate bias
            damp_cal = raw_cal * shrink
            bias = float(np.mean(btr.values[cal_sl] + damp_cal - atr.values[cal_sl]))

            # Predict test point
            X_test = X.iloc[i:i + 1]
            p1 = m1.predict(X_test)[0]
            p2 = m2.predict(X_test[sel])[0]
            p3 = m3.predict(sc.transform(X_test[sel]))[0]
            raw = self.ew[0] * p1 + self.ew[1] * p2 + self.ew[2] * p3

            final_chg = raw * shrink - bias
            pred = bases.iloc[i] + final_chg

            preds_chg.append(final_chg)
            preds_abs.append(pred)
            acts_chg.append(y_chg.iloc[i])
            acts_abs.append(y_abs.iloc[i])
            dates.append(date_s.iloc[i] if i < len(date_s) else None)

        pa = np.array(preds_abs)
        aa = np.array(acts_abs)
        pc = np.array(preds_chg)
        ac = np.array(acts_chg)
        errors = pa - aa

        # Direction accuracy
        base_arr = np.array([bases.iloc[test_start + j] for j in range(len(aa))])
        act_dir = np.sign(aa - base_arr)
        pre_dir = np.sign(pa - base_arr)
        dir_acc = float(np.mean(act_dir == pre_dir) * 100) if len(aa) > 1 else 50.0

        val_metrics = {
            "val_mae": mean_absolute_error(aa, pa),
            "val_rmse": np.sqrt(mean_squared_error(aa, pa)),
            "val_r2": r2_score(aa, pa) if len(aa) > 1 else 0.0,
            "val_change_mae": mean_absolute_error(ac, pc),
            "val_mean_error": float(np.mean(errors)),
            "val_std_error": float(np.std(errors)),
            "val_median_abs_error": float(np.median(np.abs(errors))),
            "val_95_pct_error": float(np.percentile(np.abs(errors), 95)),
            "val_max_error": float(np.max(np.abs(errors))),
            "val_n_test": len(aa),
            "val_within_2_cents": float(np.mean(np.abs(errors) <= 0.02) * 100),
            "val_within_5_cents": float(np.mean(np.abs(errors) <= 0.05) * 100),
            "val_within_10_cents": float(np.mean(np.abs(errors) <= 0.10) * 100),
            "val_direction_accuracy": dir_acc,
        }

        # Also compute naive baseline (predict no change)
        naive_errors = aa - base_arr
        val_metrics["naive_within_2_cents"] = float(
            np.mean(np.abs(naive_errors) <= 0.02) * 100
        )
        val_metrics["naive_mae"] = float(np.mean(np.abs(naive_errors)))

        self.metrics.update(val_metrics)

        self.validation_results = pd.DataFrame({
            "date": dates,
            "actual": acts_abs,
            "predicted": preds_abs,
            "error": errors.tolist(),
            "abs_error": np.abs(errors).tolist(),
        })

        return val_metrics, self.validation_results

    # ─── Prediction ───────────────────────────────────────────────────────

    def predict_next_week(self, df: pd.DataFrame) -> Dict:
        """Predict next Sunday's gas price using calibrated ensemble."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction.")

        fdf = create_features(df)
        row = prepare_prediction_row(fdf)

        # Fill any missing columns
        for c in self.all_features:
            if c not in row.columns:
                row[c] = 0

        # Raw ensemble change
        raw = self._raw_ensemble(row)[0]

        # Apply shrinkage + bias correction
        final_change = raw * self.shrinkage - self.bias

        current_price = float(df["gas_price"].iloc[-1])
        current_date = df["date"].iloc[-1]
        prediction = current_price + final_change
        pct_change = (final_change / current_price) * 100

        # Confidence intervals
        std_err = self.metrics.get("val_std_error", 0.02)
        ci_68 = (prediction - std_err, prediction + std_err)
        ci_95 = (prediction - 1.96 * std_err, prediction + 1.96 * std_err)

        today = datetime.now()
        days_to_sun = (6 - today.weekday()) % 7
        if days_to_sun == 0:
            days_to_sun = 7
        next_sun = today + timedelta(days=days_to_sun)

        return {
            "prediction": round(prediction, 4),
            "raw_prediction": round(current_price + raw, 4),
            "predicted_change": round(final_change, 4),
            "raw_change": round(raw, 5),
            "current_price": round(current_price, 4),
            "current_date": current_date,
            "predicted_pct_change": round(pct_change, 3),
            "direction": (
                "UP" if final_change > 0.001
                else ("DOWN" if final_change < -0.001 else "FLAT")
            ),
            "ci_68_low": round(ci_68[0], 4),
            "ci_68_high": round(ci_68[1], 4),
            "ci_95_low": round(ci_95[0], 4),
            "ci_95_high": round(ci_95[1], 4),
            "std_error": round(std_err, 4),
            "shrinkage": self.shrinkage,
            "direction_accuracy": round(
                self.metrics.get("val_direction_accuracy", 50), 1
            ),
            "prediction_date": next_sun.strftime("%Y-%m-%d"),
            "prediction_day": next_sun.strftime("%A, %B %d, %Y"),
            "model_1_change": round(float(
                self.xgb_full.predict(row[self.all_features])[0]
            ), 5),
            "model_2_change": round(float(
                self.xgb_sel.predict(row[self.sel_features])[0]
            ), 5),
            "model_3_change": round(float(
                self.ridge.predict(
                    self.scaler.transform(row[self.sel_features])
                )[0]
            ), 5),
        }

    # ─── Feature Importance ───────────────────────────────────────────────

    def get_feature_importance(self, top_n: int = 25) -> pd.DataFrame:
        if not self.is_trained or self.xgb_full is None:
            return pd.DataFrame()
        fi = pd.DataFrame({
            "feature": self.all_features,
            "importance": self.xgb_full.feature_importances_,
        }).sort_values("importance", ascending=False)
        return fi.head(top_n)

    # ─── Persistence ──────────────────────────────────────────────────────

    def save_model(self):
        MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
        if self.xgb_full is not None:
            self.xgb_full.save_model(str(MODEL_FILE))
        if self.metrics:
            safe = {}
            for k, v in self.metrics.items():
                if isinstance(v, (np.floating, np.integer)):
                    safe[k] = float(v)
                elif isinstance(v, (pd.Timestamp, datetime)):
                    safe[k] = str(v)
                else:
                    safe[k] = v
            with open(METRICS_FILE, "w") as f:
                json.dump(safe, f, indent=2, default=str)

    def load_model(self) -> bool:
        if MODEL_FILE.exists() and METRICS_FILE.exists():
            try:
                self.xgb_full = XGBRegressor()
                self.xgb_full.load_model(str(MODEL_FILE))
                with open(METRICS_FILE) as f:
                    self.metrics = json.load(f)
                self.all_features = list(
                    self.xgb_full.get_booster().feature_names or []
                )
                self.is_trained = True
                return True
            except Exception:
                return False
        return False
