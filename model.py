"""
Gas Price Predictor - Hybrid Decomposition Model

Architecture:
  1) STL decomposition on historical gas prices
  2) GRU sequence model forecasts trend + seasonal components
  3) XGBoost forecasts residual component using engineered macro/market features
  4) Ridge meta-learner stacks base forecasts into final absolute price
"""

import json
import pickle
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from config import (
    GRU_PARAMS,
    METRICS_FILE,
    MIN_TRAINING_WEEKS,
    MODEL_BUNDLE_FILE,
    STL_PERIOD,
    VALIDATION_WEEKS,
    XGBOOST_PARAMS,
)
from feature_engine import (
    create_features,
    get_feature_columns,
    prepare_prediction_row,
)

warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None

try:
    from statsmodels.tsa.seasonal import STL
except Exception:  # pragma: no cover
    STL = None


@dataclass
class DecompositionResult:
    trend: np.ndarray
    seasonal: np.ndarray
    resid: np.ndarray


BaseNNModule = nn.Module if nn is not None else object


class GRUForecaster(BaseNNModule):
    """Simple GRU forecaster for 2-channel sequence: [trend, seasonal]."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        if nn is None:
            raise RuntimeError("PyTorch backend unavailable.")
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 2)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


class GasPriceModel:
    """STL + GRU + XGBoost residual model with stacking ensemble."""

    def __init__(self):
        self.sequence_length = int(GRU_PARAMS["sequence_length"])
        self.backend = "gru" if torch is not None else "ridge-seq-fallback"
        self.decomposition_mode = "stl" if STL is not None else "rolling-fallback"

        self.gru: Optional[GRUForecaster] = None
        self.residual_xgb: Optional[XGBRegressor] = None
        self.meta_learner: Optional[Ridge] = None

        self.all_features: List[str] = []
        self.metrics: Dict = {}
        self.validation_results: Optional[pd.DataFrame] = None
        self.is_trained: bool = False

        self.last_components: Optional[DecompositionResult] = None
        self.last_price_series: Optional[np.ndarray] = None

    # ─── Decomposition + Sequence Helpers ───────────────────────────────

    def _decompose(self, gas_prices: pd.Series) -> DecompositionResult:
        if STL is not None:
            stl = STL(gas_prices, period=STL_PERIOD, robust=True)
            result = stl.fit()
            return DecompositionResult(
                trend=result.trend.values,
                seasonal=result.seasonal.values,
                resid=result.resid.values,
            )

        trend = gas_prices.rolling(13, center=True, min_periods=1).mean()
        detrended = gas_prices - trend
        week = np.arange(len(gas_prices)) % 52
        seasonal_map = pd.Series(detrended).groupby(week).transform("mean")
        resid = gas_prices - trend - seasonal_map
        return DecompositionResult(
            trend=trend.values,
            seasonal=seasonal_map.values,
            resid=resid.values,
        )

    def _build_seq_dataset(
        self,
        trend: np.ndarray,
        seasonal: np.ndarray,
        indices: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build supervised windows for rows that predict t+1.
        For feature row at index t, we predict components at t+1.
        """
        X_seq, y_seq, valid_rows = [], [], []
        for t in indices:
            if t - self.sequence_length + 1 < 0:
                continue
            target_idx = t + 1
            if target_idx >= len(trend):
                continue
            seq = np.column_stack([
                trend[t - self.sequence_length + 1:t + 1],
                seasonal[t - self.sequence_length + 1:t + 1],
            ])
            y = np.array([trend[target_idx], seasonal[target_idx]], dtype=np.float32)
            X_seq.append(seq.astype(np.float32))
            y_seq.append(y)
            valid_rows.append(t)

        if not X_seq:
            return np.empty((0, self.sequence_length, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=int)

        return np.array(X_seq), np.array(y_seq), np.array(valid_rows)

    def _fit_gru(self, X_seq: np.ndarray, y_seq: np.ndarray, epochs: Optional[int] = None):
        if torch is None:
            self.backend = "ridge-seq-fallback"
            proxy = Ridge(alpha=1.0)
            proxy.fit(X_seq.reshape(len(X_seq), -1), y_seq)
            return proxy

        self.backend = "gru"
        torch.manual_seed(int(GRU_PARAMS["random_state"]))
        model = GRUForecaster(
            input_size=2,
            hidden_size=int(GRU_PARAMS["hidden_size"]),
            num_layers=int(GRU_PARAMS["num_layers"]),
            dropout=float(GRU_PARAMS["dropout"]),
        )

        ds = TensorDataset(torch.tensor(X_seq), torch.tensor(y_seq))
        loader = DataLoader(ds, batch_size=int(GRU_PARAMS["batch_size"]), shuffle=True)

        opt = torch.optim.Adam(model.parameters(), lr=float(GRU_PARAMS["learning_rate"]))
        loss_fn = nn.HuberLoss()

        n_epochs = int(epochs or GRU_PARAMS["epochs"])
        model.train()
        for _ in range(n_epochs):
            for xb, yb in loader:
                opt.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
        return model

    @staticmethod
    def _predict_gru(model, X_seq: np.ndarray) -> np.ndarray:
        if torch is None or not hasattr(model, "state_dict"):
            return model.predict(X_seq.reshape(len(X_seq), -1))
        model.eval()
        with torch.no_grad():
            return model(torch.tensor(X_seq)).detach().numpy()

    # ─── Data Prep ───────────────────────────────────────────────────────

    def _prep(self, df: pd.DataFrame):
        fdf = create_features(df)
        names = get_feature_columns(fdf)
        valid = fdf[names + ["target", "gas_price"]].dropna().copy()

        # map valid rows back to original index (t predicting t+1)
        valid_idx = valid.index.values
        target = valid["target"].values
        base = valid["gas_price"].values
        return fdf, valid, names, valid_idx, target, base

    # ─── Training ────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame) -> Dict:
        fdf, valid, names, valid_idx, y_abs, bases = self._prep(df)

        if len(valid) < MIN_TRAINING_WEEKS:
            raise ValueError(f"Need {MIN_TRAINING_WEEKS} weeks, got {len(valid)}.")

        self.all_features = names

        # 1) STL decomposition on full observed gas price series
        gas_series = fdf["gas_price"].astype(float)
        comps = self._decompose(gas_series)

        # 2) GRU predicts trend + seasonal for t+1
        X_seq, y_seq, seq_rows = self._build_seq_dataset(comps.trend, comps.seasonal, valid_idx)
        if len(X_seq) < 20:
            raise ValueError("Not enough sequence windows for GRU training.")

        self.gru = self._fit_gru(X_seq, y_seq)
        gru_preds = self._predict_gru(self.gru, X_seq)

        seq_to_valid_pos = {row: pos for pos, row in enumerate(valid_idx)}
        valid_positions = [seq_to_valid_pos[r] for r in seq_rows if r in seq_to_valid_pos]

        vsub = valid.iloc[valid_positions].copy()
        y_sub = y_abs[valid_positions]

        trend_season_pred = gru_preds[:, 0] + gru_preds[:, 1]

        # 3) XGBoost predicts residual target: y - (pred_trend + pred_season)
        residual_target = y_sub - trend_season_pred
        self.residual_xgb = XGBRegressor(**XGBOOST_PARAMS)
        self.residual_xgb.fit(vsub[self.all_features], residual_target, verbose=False)
        residual_pred = self.residual_xgb.predict(vsub[self.all_features])

        # 4) Stacking meta-learner combines components
        base_sum = trend_season_pred + residual_pred
        stack_X = np.column_stack([trend_season_pred, residual_pred, base_sum])
        self.meta_learner = Ridge(alpha=1.0)
        self.meta_learner.fit(stack_X, y_sub)
        pred_abs = self.meta_learner.predict(stack_X)

        pred_change = pred_abs - bases[valid_positions]
        actual_change = y_sub - bases[valid_positions]

        self.last_components = comps
        self.last_price_series = gas_series.values
        self.is_trained = True

        self.metrics = {
            "mae": mean_absolute_error(y_sub, pred_abs),
            "rmse": np.sqrt(mean_squared_error(y_sub, pred_abs)),
            "r2": r2_score(y_sub, pred_abs),
            "change_mae": mean_absolute_error(actual_change, pred_change),
            "n_samples": int(len(vsub)),
            "n_features": int(len(names)),
            "sequence_length": self.sequence_length,
            "trained_at": datetime.now().isoformat(),
            "architecture": f"{self.decomposition_mode}->GRU(trend/seasonal)+XGBoost(residual)+Ridge(meta)",
            "sequence_backend": self.backend,
        }
        return self.metrics

    # ─── Walk-Forward Validation ─────────────────────────────────────────

    def walk_forward_validate(self, df: pd.DataFrame) -> Tuple[Dict, pd.DataFrame]:
        fdf, valid, names, valid_idx, y_abs, bases = self._prep(df)
        n = len(valid)

        test_n = min(VALIDATION_WEEKS, n // 3)
        test_start = n - test_n
        if test_start < MIN_TRAINING_WEEKS:
            raise ValueError("Not enough data for walk-forward validation.")

        date_s = fdf.loc[valid_idx, "date"].reset_index(drop=True)

        preds_abs, acts_abs, preds_chg, acts_chg, dates = [], [], [], [], []

        for i in range(test_start, n):
            train_rows = valid_idx[:i]
            t = valid_idx[i]
            train_df = fdf.iloc[:t + 1].copy()

            gas_series = train_df["gas_price"].astype(float)
            comps = self._decompose(gas_series)

            X_seq, y_seq, seq_rows = self._build_seq_dataset(comps.trend, comps.seasonal, train_rows)
            if len(X_seq) < 15:
                continue

            gru = self._fit_gru(X_seq, y_seq, epochs=40)
            gru_preds = self._predict_gru(gru, X_seq)

            pos_map = {row: pos for pos, row in enumerate(train_rows)}
            train_positions = [pos_map[r] for r in seq_rows if r in pos_map]
            X_train = valid.iloc[:i].iloc[train_positions][names]
            y_train = y_abs[:i][train_positions]

            trend_season_train = gru_preds[:, 0] + gru_preds[:, 1]
            resid_train = y_train - trend_season_train

            xgb = XGBRegressor(**XGBOOST_PARAMS)
            xgb.fit(X_train, resid_train, verbose=False)
            resid_pred_train = xgb.predict(X_train)

            stack_train = np.column_stack([
                trend_season_train,
                resid_pred_train,
                trend_season_train + resid_pred_train,
            ])
            meta = Ridge(alpha=1.0)
            meta.fit(stack_train, y_train)

            # predict point i
            if t - self.sequence_length + 1 < 0:
                continue
            seq_test = np.column_stack([
                comps.trend[t - self.sequence_length + 1:t + 1],
                comps.seasonal[t - self.sequence_length + 1:t + 1],
            ]).astype(np.float32)[None, :, :]
            ts_pred = self._predict_gru(gru, seq_test)[0]
            ts_sum = float(ts_pred[0] + ts_pred[1])

            resid_pred = float(xgb.predict(valid.iloc[i:i + 1][names])[0])
            pred = float(meta.predict(np.array([[ts_sum, resid_pred, ts_sum + resid_pred]]))[0])

            preds_abs.append(pred)
            acts_abs.append(float(y_abs[i]))
            preds_chg.append(pred - float(bases[i]))
            acts_chg.append(float(y_abs[i] - bases[i]))
            dates.append(date_s.iloc[i])

        pa = np.array(preds_abs)
        aa = np.array(acts_abs)
        pc = np.array(preds_chg)
        ac = np.array(acts_chg)
        errors = pa - aa

        if len(pa) == 0:
            raise ValueError("Validation produced no predictions. Try more history.")

        base_arr = np.array([bases[test_start + j] for j in range(len(aa))])
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
            "naive_within_2_cents": float(np.mean(np.abs(aa - base_arr) <= 0.02) * 100),
            "naive_mae": float(np.mean(np.abs(aa - base_arr))),
        }

        self.metrics.update(val_metrics)
        self.validation_results = pd.DataFrame(
            {
                "date": dates,
                "actual": acts_abs,
                "predicted": preds_abs,
                "error": errors.tolist(),
                "abs_error": np.abs(errors).tolist(),
            }
        )

        return val_metrics, self.validation_results

    # ─── Prediction ──────────────────────────────────────────────────────

    def predict_next_week(self, df: pd.DataFrame) -> Dict:
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction.")

        fdf = create_features(df)
        row = prepare_prediction_row(fdf)

        for c in self.all_features:
            if c not in row.columns:
                row[c] = 0

        gas_series = fdf["gas_price"].astype(float)
        comps = self._decompose(gas_series)

        t = len(gas_series) - 1
        if t - self.sequence_length + 1 < 0:
            raise ValueError("Not enough history for sequence prediction.")

        seq = np.column_stack([
            comps.trend[t - self.sequence_length + 1:t + 1],
            comps.seasonal[t - self.sequence_length + 1:t + 1],
        ]).astype(np.float32)[None, :, :]

        ts_pred = self._predict_gru(self.gru, seq)[0]
        trend_pred = float(ts_pred[0])
        seasonal_pred = float(ts_pred[1])
        trend_season_sum = trend_pred + seasonal_pred

        residual_pred = float(self.residual_xgb.predict(row[self.all_features])[0])
        prediction = float(
            self.meta_learner.predict(
                np.array([[trend_season_sum, residual_pred, trend_season_sum + residual_pred]])
            )[0]
        )

        current_price = float(df["gas_price"].iloc[-1])
        final_change = prediction - current_price
        pct_change = (final_change / current_price) * 100

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
            "raw_prediction": round(trend_season_sum + residual_pred, 4),
            "predicted_change": round(final_change, 4),
            "raw_change": round((trend_season_sum + residual_pred) - current_price, 5),
            "current_price": round(current_price, 4),
            "current_date": df["date"].iloc[-1],
            "predicted_pct_change": round(pct_change, 3),
            "direction": "UP" if final_change > 0.001 else ("DOWN" if final_change < -0.001 else "FLAT"),
            "ci_68_low": round(ci_68[0], 4),
            "ci_68_high": round(ci_68[1], 4),
            "ci_95_low": round(ci_95[0], 4),
            "ci_95_high": round(ci_95[1], 4),
            "std_error": round(std_err, 4),
            "shrinkage": "N/A (stacking used)",
            "direction_accuracy": round(self.metrics.get("val_direction_accuracy", 50), 1),
            "prediction_date": next_sun.strftime("%Y-%m-%d"),
            "prediction_day": next_sun.strftime("%A, %B %d, %Y"),
            "model_1_change": round(trend_pred - current_price, 5),
            "model_2_change": round(seasonal_pred, 5),
            "model_3_change": round(residual_pred, 5),
        }

    # ─── Feature Importance ──────────────────────────────────────────────

    def get_feature_importance(self, top_n: int = 25) -> pd.DataFrame:
        if not self.is_trained or self.residual_xgb is None:
            return pd.DataFrame()
        fi = pd.DataFrame(
            {
                "feature": self.all_features,
                "importance": self.residual_xgb.feature_importances_,
            }
        ).sort_values("importance", ascending=False)
        return fi.head(top_n)

    # ─── Persistence ─────────────────────────────────────────────────────

    def save_model(self):
        MODEL_BUNDLE_FILE.parent.mkdir(parents=True, exist_ok=True)

        bundle = {
            "residual_xgb": self.residual_xgb,
            "meta_learner": self.meta_learner,
            "all_features": self.all_features,
            "metrics": self.metrics,
            "is_trained": self.is_trained,
            "sequence_length": self.sequence_length,
            "gru_state_dict": self.gru.state_dict() if (self.gru is not None and hasattr(self.gru, "state_dict")) else None,
            "gru_fallback_model": self.gru if (self.gru is not None and not hasattr(self.gru, "state_dict")) else None,
            "gru_config": {
                "input_size": 2,
                "hidden_size": int(GRU_PARAMS["hidden_size"]),
                "num_layers": int(GRU_PARAMS["num_layers"]),
                "dropout": float(GRU_PARAMS["dropout"]),
            },
        }

        with open(MODEL_BUNDLE_FILE, "wb") as f:
            pickle.dump(bundle, f)

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
        if not MODEL_BUNDLE_FILE.exists():
            return False

        try:
            with open(MODEL_BUNDLE_FILE, "rb") as f:
                bundle = pickle.load(f)

            self.residual_xgb = bundle.get("residual_xgb")
            self.meta_learner = bundle.get("meta_learner")
            self.all_features = bundle.get("all_features", [])
            self.metrics = bundle.get("metrics", {})
            self.is_trained = bundle.get("is_trained", False)
            self.sequence_length = int(bundle.get("sequence_length", GRU_PARAMS["sequence_length"]))

            if torch is not None and bundle.get("gru_state_dict") is not None:
                cfg = bundle.get("gru_config", {})
                self.gru = GRUForecaster(
                    input_size=int(cfg.get("input_size", 2)),
                    hidden_size=int(cfg.get("hidden_size", GRU_PARAMS["hidden_size"])),
                    num_layers=int(cfg.get("num_layers", GRU_PARAMS["num_layers"])),
                    dropout=float(cfg.get("dropout", GRU_PARAMS["dropout"])),
                )
                self.gru.load_state_dict(bundle["gru_state_dict"])
                self.gru.eval()
            else:
                self.gru = bundle.get("gru_fallback_model")

            if METRICS_FILE.exists():
                with open(METRICS_FILE) as f:
                    self.metrics.update(json.load(f))

            return self.is_trained
        except Exception:
            return False
