"""
Gas Price Predictor - Configuration
All constants, settings, and configurable parameters.
"""

import os
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
MODEL_DIR = APP_DIR / "models"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

CACHE_FILE = DATA_DIR / "cached_data.parquet"
MODEL_FILE = MODEL_DIR / "xgb_model.json"
METRICS_FILE = MODEL_DIR / "metrics.json"

# ─── API Keys ─────────────────────────────────────────────────────────────────
EIA_API_KEY = os.environ.get("EIA_API_KEY", "")
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")

# ─── EIA Series IDs ──────────────────────────────────────────────────────────
EIA_RETAIL_GAS = "PET.EMM_EPMR_PTE_NUS_DPG.W"

# Weekly Petroleum Status Report series (API v2)
EIA_WEEKLY_SERIES = {
    "gasoline_stocks":       "WGTSTUS1",   # Weekly US Total Gasoline Stocks (Kbbl)
    "crude_stocks":          "WCESTUS1",   # Weekly US Ending Crude Oil Stocks (Kbbl)
    "refinery_utilization":  "WPULEUS3",   # Weekly US Refinery Utilization (%)
    "gasoline_production":   "WGFRPUS2",   # Weekly US Gasoline Production (Kbd)
    "gasoline_demand":       "WGFUPUS2",   # Weekly US Gasoline Product Supplied (Kbd)
    "crude_imports":         "WCRIMUS2",   # Weekly US Crude Oil Imports (Kbd)
    "distillate_stocks":     "WDISTUS1",   # Weekly US Distillate Stocks (Kbbl)
}

# ─── Yahoo Finance Tickers ────────────────────────────────────────────────────
YAHOO_TICKERS = {
    "CL=F":      "crude",         # WTI Crude Oil Futures
    "RB=F":      "rbob",          # RBOB Gasoline Futures
    "HO=F":      "heating_oil",   # Heating Oil Futures
    "BZ=F":      "brent",         # Brent Crude Oil Futures
    "NG=F":      "natgas",        # Natural Gas Futures
    "DX-Y.NYB":  "dollar_idx",    # US Dollar Index Futures
    "^GSPC":     "sp500",         # S&P 500 Index
}

# ─── FRED Series IDs ─────────────────────────────────────────────────────────
FRED_SERIES = {
    "DTWEXBGS":    "dollar_broad",    # Trade-Weighted US Dollar Index (Broad)
    "DCOILWTICO":  "fred_wti",        # WTI Crude spot (daily)
    "DHHNGSP":     "henry_hub",       # Henry Hub Natural Gas Spot $/MMBtu
    "DEXUSEU":     "eur_usd",         # EUR/USD Exchange Rate
    "T10Y2Y":      "yield_spread",    # 10Y-2Y Treasury Spread (recession signal)
    "DGS10":       "treasury_10y",    # 10-Year Treasury Yield
    "CPIAUCSL":    "cpi",             # CPI All Urban Consumers (monthly)
    "VIXCLS":      "vix",             # CBOE Volatility Index
    "GASREGW":     "fred_gas",        # Regular Gas Weekly (EIA via FRED)
}

# ─── Feature Engineering ──────────────────────────────────────────────────────
PRICE_LAG_WEEKS = [1, 2, 3, 4]
CRUDE_LAG_WEEKS = [1, 2]
RBOB_LAG_WEEKS = [1, 2]
ROLLING_WINDOWS = [4, 8, 12]

SUMMER_BLEND_START_WEEK = 14
SUMMER_BLEND_END_WEEK = 40
DRIVING_SEASON_START_WEEK = 21
DRIVING_SEASON_END_WEEK = 36

HURRICANE_SEASON_START_WEEK = 22   # ~June 1
HURRICANE_SEASON_END_WEEK = 48     # ~Nov 30
HURRICANE_PEAK_START_WEEK = 33     # Mid-Aug
HURRICANE_PEAK_END_WEEK = 42       # Mid-Oct

# ─── Model Settings ──────────────────────────────────────────────────────────
XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
}

VALIDATION_WEEKS = 26
MIN_TRAINING_WEEKS = 52
MIN_DATA_POINTS = 52
DEFAULT_HISTORY_YEARS = 5
