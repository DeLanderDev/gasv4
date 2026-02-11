"""
Gas Price Predictor - Prediction Logging & Scheduling
Save predictions with timestamps and schedule automatic saves.
"""

import csv
import json
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from config import DATA_DIR

# ─── Paths ────────────────────────────────────────────────────────────────────
PREDICTION_LOG_FILE = DATA_DIR / "prediction_log.csv"
SCHEDULE_CONFIG_FILE = DATA_DIR / "schedule_config.json"

LOG_COLUMNS = [
    "save_timestamp",       # When the prediction was saved
    "prediction_for_date",  # The Sunday being predicted
    "current_price",        # Price at time of prediction
    "current_source",       # EIA or AAA
    "predicted_price",      # The model's predicted price
    "predicted_change",     # Predicted $/gal change
    "direction",            # UP / DOWN / FLAT
    "ci_68_low",
    "ci_68_high",
    "ci_95_low",
    "ci_95_high",
    "shrinkage",
    "model_1_change",
    "model_2_change",
    "model_3_change",
    "actual_price",         # Filled in later when actual is known
    "error",                # Filled in later: predicted - actual
    "abs_error",            # Filled in later: |error|
    "save_type",            # "manual" or "scheduled"
]


# ═══════════════════════════════════════════════════════════════════════════════
#  Prediction Log I/O
# ═══════════════════════════════════════════════════════════════════════════════

def _ensure_log_file():
    """Create the log CSV with headers if it doesn't exist."""
    if not PREDICTION_LOG_FILE.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(PREDICTION_LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(LOG_COLUMNS)


def save_prediction(display: Dict, prediction: Dict, save_type: str = "manual") -> str:
    """
    Save a prediction to the log file.

    Args:
        display: The display dict from app.py (contains current/predicted prices).
        prediction: The raw prediction dict from model.predict_next_week().
        save_type: "manual" or "scheduled".

    Returns:
        Timestamp string of the saved entry.
    """
    _ensure_log_file()
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

    row = {
        "save_timestamp": timestamp,
        "prediction_for_date": display.get("prediction_date", ""),
        "current_price": display.get("current_price", ""),
        "current_source": display.get("current_source", ""),
        "predicted_price": display.get("prediction", ""),
        "predicted_change": display.get("predicted_change", ""),
        "direction": display.get("direction", ""),
        "ci_68_low": display.get("ci_68_low", ""),
        "ci_68_high": display.get("ci_68_high", ""),
        "ci_95_low": display.get("ci_95_low", ""),
        "ci_95_high": display.get("ci_95_high", ""),
        "shrinkage": prediction.get("shrinkage", ""),
        "model_1_change": prediction.get("model_1_change", ""),
        "model_2_change": prediction.get("model_2_change", ""),
        "model_3_change": prediction.get("model_3_change", ""),
        "actual_price": "",
        "error": "",
        "abs_error": "",
        "save_type": save_type,
    }

    with open(PREDICTION_LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        writer.writerow(row)

    return timestamp


def load_prediction_log() -> pd.DataFrame:
    """Load the prediction log as a DataFrame."""
    _ensure_log_file()
    try:
        df = pd.read_csv(PREDICTION_LOG_FILE)
        if df.empty:
            return pd.DataFrame(columns=LOG_COLUMNS)
        # Parse dates
        df["save_timestamp"] = pd.to_datetime(df["save_timestamp"], errors="coerce")
        df["prediction_for_date"] = pd.to_datetime(
            df["prediction_for_date"], errors="coerce"
        )
        return df
    except Exception:
        return pd.DataFrame(columns=LOG_COLUMNS)


def update_actual_prices(actual_prices: Dict[str, float]):
    """
    Update the log with actual prices once they're known.

    Args:
        actual_prices: Dict mapping date strings (YYYY-MM-DD) to actual prices.
    """
    df = load_prediction_log()
    if df.empty:
        return

    for date_str, actual in actual_prices.items():
        date_val = pd.to_datetime(date_str, errors="coerce")
        if pd.isna(date_val):
            continue
        mask = df["prediction_for_date"] == date_val
        if mask.any():
            df.loc[mask, "actual_price"] = actual
            predicted = pd.to_numeric(df.loc[mask, "predicted_price"], errors="coerce")
            df.loc[mask, "error"] = predicted - actual
            df.loc[mask, "abs_error"] = (predicted - actual).abs()

    # Rewrite file
    df.to_csv(PREDICTION_LOG_FILE, index=False)


def backfill_actuals_from_data(data: pd.DataFrame):
    """
    Automatically backfill actual prices from the loaded EIA data.
    Matches prediction_for_date to dates in the historical data.
    """
    log_df = load_prediction_log()
    if log_df.empty:
        return

    needs_fill = log_df["actual_price"].isna() | (log_df["actual_price"] == "")
    if not needs_fill.any():
        return

    data_dates = data[["date", "gas_price"]].copy()
    data_dates["date"] = pd.to_datetime(data_dates["date"])

    updated = False
    for idx in log_df.index[needs_fill]:
        pred_date = log_df.loc[idx, "prediction_for_date"]
        if pd.isna(pred_date):
            continue
        # Find matching date in historical data (within 3 days tolerance)
        time_diffs = (data_dates["date"] - pred_date).abs()
        closest_idx = time_diffs.idxmin()
        if time_diffs[closest_idx] <= timedelta(days=3):
            actual = data_dates.loc[closest_idx, "gas_price"]
            predicted = pd.to_numeric(
                log_df.loc[idx, "predicted_price"], errors="coerce"
            )
            log_df.loc[idx, "actual_price"] = actual
            if not pd.isna(predicted):
                log_df.loc[idx, "error"] = round(predicted - actual, 4)
                log_df.loc[idx, "abs_error"] = round(abs(predicted - actual), 4)
            updated = True

    if updated:
        log_df.to_csv(PREDICTION_LOG_FILE, index=False)


def delete_log_entry(index: int):
    """Delete a specific entry from the log by index."""
    df = load_prediction_log()
    if 0 <= index < len(df):
        df = df.drop(df.index[index]).reset_index(drop=True)
        df.to_csv(PREDICTION_LOG_FILE, index=False)


def clear_log():
    """Delete all entries from the prediction log."""
    _ensure_log_file()
    with open(PREDICTION_LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(LOG_COLUMNS)


# ═══════════════════════════════════════════════════════════════════════════════
#  Schedule Configuration
# ═══════════════════════════════════════════════════════════════════════════════

def save_schedule_config(config: Dict):
    """Save schedule configuration to JSON."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(SCHEDULE_CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2, default=str)


def load_schedule_config() -> Optional[Dict]:
    """Load schedule configuration from JSON."""
    if SCHEDULE_CONFIG_FILE.exists():
        try:
            with open(SCHEDULE_CONFIG_FILE) as f:
                return json.load(f)
        except Exception:
            return None
    return None


def delete_schedule_config():
    """Remove the schedule configuration file."""
    if SCHEDULE_CONFIG_FILE.exists():
        SCHEDULE_CONFIG_FILE.unlink()


# ═══════════════════════════════════════════════════════════════════════════════
#  Background Scheduler
# ═══════════════════════════════════════════════════════════════════════════════

class PredictionScheduler:
    """
    Background thread that saves predictions at scheduled times.
    Designed to run inside a Streamlit session.
    """

    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_save: Optional[datetime] = None
        self.is_running = False

    def start(
        self,
        predict_fn,
        frequency: str,
        times: List[str],
        end_date: str,
    ):
        """
        Start the scheduler.

        Args:
            predict_fn: Callable that returns (display_dict, prediction_dict).
            frequency: "once_per_day" or "twice_per_day".
            times: List of time strings like ["08:00"] or ["08:00", "20:00"].
            end_date: ISO date string for when to stop scheduling.
        """
        self.stop()

        self._stop_event.clear()
        self.is_running = True

        config = {
            "frequency": frequency,
            "times": times,
            "end_date": end_date,
            "started_at": datetime.now().isoformat(),
        }
        save_schedule_config(config)

        self._thread = threading.Thread(
            target=self._run_loop,
            args=(predict_fn, times, end_date),
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        """Stop the scheduler."""
        self._stop_event.set()
        self.is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._thread = None
        delete_schedule_config()

    def _run_loop(self, predict_fn, times: List[str], end_date: str):
        """Main scheduler loop. Checks every 30 seconds if it's time to save."""
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(
            hour=23, minute=59, second=59
        )

        # Track which (date, time_slot) combos have been saved
        saved_slots = set()

        while not self._stop_event.is_set():
            now = datetime.now()

            if now > end_dt:
                self.is_running = False
                delete_schedule_config()
                break

            today_str = now.strftime("%Y-%m-%d")

            for t in times:
                slot_key = f"{today_str}_{t}"
                if slot_key in saved_slots:
                    continue

                try:
                    hour, minute = map(int, t.split(":"))
                except ValueError:
                    continue

                target_time = now.replace(
                    hour=hour, minute=minute, second=0, microsecond=0
                )

                # Save if we're within 2 minutes after the target time
                diff = (now - target_time).total_seconds()
                if 0 <= diff <= 120:
                    try:
                        display_dict, prediction_dict = predict_fn()
                        save_prediction(display_dict, prediction_dict, "scheduled")
                        saved_slots.add(slot_key)
                        self._last_save = now
                    except Exception:
                        pass  # Silently skip on error; will retry next cycle

            # Sleep 30 seconds between checks
            self._stop_event.wait(30)

        self.is_running = False
