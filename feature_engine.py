"""
Gas Price Predictor - Feature Engineering
Transforms raw data into ML-ready features for prediction.
"""

import numpy as np
import pandas as pd

from config import (
    CRUDE_LAG_WEEKS,
    DRIVING_SEASON_END_WEEK,
    DRIVING_SEASON_START_WEEK,
    HURRICANE_PEAK_END_WEEK,
    HURRICANE_PEAK_START_WEEK,
    HURRICANE_SEASON_END_WEEK,
    HURRICANE_SEASON_START_WEEK,
    PRICE_LAG_WEEKS,
    RBOB_LAG_WEEKS,
    ROLLING_WINDOWS,
    SUMMER_BLEND_END_WEEK,
    SUMMER_BLEND_START_WEEK,
)


def _safe_col(df: pd.DataFrame, col: str) -> bool:
    """Check if a column exists and has enough non-null data."""
    return col in df.columns and df[col].notna().sum() > 10


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create all predictive features from the combined dataset.
    Gracefully handles missing columns (not all data sources may succeed).
    """
    df = df.copy().sort_values("date").reset_index(drop=True)

    # ─── Target: next week's gas price ────────────────────────────────────
    df["target"] = df["gas_price"].shift(-1)

    # ═══════════════════════════════════════════════════════════════════════
    #  GAS PRICE FEATURES
    # ═══════════════════════════════════════════════════════════════════════
    for lag in PRICE_LAG_WEEKS:
        df[f"gas_lag_{lag}"] = df["gas_price"].shift(lag)

    df["gas_change_1w"] = df["gas_price"] - df["gas_price"].shift(1)
    df["gas_change_2w"] = df["gas_price"] - df["gas_price"].shift(2)
    df["gas_change_4w"] = df["gas_price"] - df["gas_price"].shift(4)
    df["gas_pct_change_1w"] = df["gas_price"].pct_change(1) * 100
    df["gas_pct_change_4w"] = df["gas_price"].pct_change(4) * 100

    for window in ROLLING_WINDOWS:
        df[f"gas_rolling_mean_{window}w"] = df["gas_price"].rolling(window).mean()
        df[f"gas_rolling_std_{window}w"] = df["gas_price"].rolling(window).std()
        df[f"gas_vs_ma_{window}w"] = df["gas_price"] - df[f"gas_rolling_mean_{window}w"]

    df["gas_acceleration"] = df["gas_change_1w"] - df["gas_change_1w"].shift(1)

    # ═══════════════════════════════════════════════════════════════════════
    #  CRUDE OIL FEATURES (WTI from Yahoo)
    # ═══════════════════════════════════════════════════════════════════════
    if _safe_col(df, "crude_price"):
        for lag in CRUDE_LAG_WEEKS:
            df[f"crude_lag_{lag}"] = df["crude_price"].shift(lag)

        df["crude_change_1w"] = df["crude_price"] - df["crude_price"].shift(1)
        df["crude_change_2w"] = df["crude_price"] - df["crude_price"].shift(2)
        df["crude_pct_change_1w"] = df["crude_price"].pct_change(1) * 100
        df["crude_pct_change_4w"] = df["crude_price"].pct_change(4) * 100
        df["crude_rolling_mean_4w"] = df["crude_price"].rolling(4).mean()
        df["crude_rolling_std_4w"] = df["crude_price"].rolling(4).std()
        df["crude_vs_ma_4w"] = df["crude_price"] - df["crude_rolling_mean_4w"]

        # Crude per gallon equivalent (1 barrel = 42 gallons)
        df["crude_per_gallon"] = df["crude_price"] / 42
        df["crack_spread"] = df["gas_price"] - df["crude_per_gallon"]

        # Rockets and feathers (asymmetric price response)
        df["crude_up_1w"] = np.maximum(df["crude_change_1w"], 0)
        df["crude_down_1w"] = np.minimum(df["crude_change_1w"], 0)

        # Crude oil volatility (regime indicator)
        df["crude_volatility_4w"] = df["crude_price"].pct_change().rolling(4).std() * 100
        df["crude_volatility_8w"] = df["crude_price"].pct_change().rolling(8).std() * 100

    # ═══════════════════════════════════════════════════════════════════════
    #  BRENT CRUDE FEATURES
    # ═══════════════════════════════════════════════════════════════════════
    if _safe_col(df, "brent_price") and _safe_col(df, "crude_price"):
        df["brent_wti_spread"] = df["brent_price"] - df["crude_price"]
        df["brent_change_1w"] = df["brent_price"] - df["brent_price"].shift(1)
        df["brent_pct_change_1w"] = df["brent_price"].pct_change(1) * 100
    elif _safe_col(df, "brent_price"):
        df["brent_change_1w"] = df["brent_price"] - df["brent_price"].shift(1)

    # ═══════════════════════════════════════════════════════════════════════
    #  RBOB GASOLINE FUTURES FEATURES
    # ═══════════════════════════════════════════════════════════════════════
    if _safe_col(df, "rbob_price"):
        for lag in RBOB_LAG_WEEKS:
            df[f"rbob_lag_{lag}"] = df["rbob_price"].shift(lag)

        df["rbob_change_1w"] = df["rbob_price"] - df["rbob_price"].shift(1)
        df["rbob_change_2w"] = df["rbob_price"] - df["rbob_price"].shift(2)
        df["rbob_pct_change_1w"] = df["rbob_price"].pct_change(1) * 100

        # RBOB-retail spread (key leading indicator)
        df["rbob_retail_spread"] = df["rbob_price"] - df["gas_price"]
        df["rbob_retail_spread_change"] = (
            df["rbob_retail_spread"] - df["rbob_retail_spread"].shift(1)
        )

    # ═══════════════════════════════════════════════════════════════════════
    #  HEATING OIL & NATURAL GAS
    # ═══════════════════════════════════════════════════════════════════════
    if _safe_col(df, "heating_oil_price"):
        df["heating_change_1w"] = df["heating_oil_price"] - df["heating_oil_price"].shift(1)

    if _safe_col(df, "natgas_price"):
        df["natgas_change_1w"] = df["natgas_price"] - df["natgas_price"].shift(1)
        df["natgas_pct_change_1w"] = df["natgas_price"].pct_change(1) * 100
        df["natgas_rolling_mean_4w"] = df["natgas_price"].rolling(4).mean()

    # ═══════════════════════════════════════════════════════════════════════
    #  US DOLLAR INDEX / CURRENCY FEATURES
    # ═══════════════════════════════════════════════════════════════════════
    # Try Yahoo dollar index first, then FRED broad dollar
    dollar_col = None
    for candidate in ["dollar_idx_price", "dollar_broad"]:
        if _safe_col(df, candidate):
            dollar_col = candidate
            break

    if dollar_col:
        df["dollar_change_1w"] = df[dollar_col] - df[dollar_col].shift(1)
        df["dollar_pct_change_1w"] = df[dollar_col].pct_change(1) * 100
        df["dollar_pct_change_4w"] = df[dollar_col].pct_change(4) * 100
        df["dollar_rolling_mean_4w"] = df[dollar_col].rolling(4).mean()
        df["dollar_vs_ma_4w"] = df[dollar_col] - df["dollar_rolling_mean_4w"]

    # EUR/USD from FRED
    if _safe_col(df, "eur_usd"):
        df["eurusd_change_1w"] = df["eur_usd"] - df["eur_usd"].shift(1)

    # ═══════════════════════════════════════════════════════════════════════
    #  S&P 500 / ECONOMIC HEALTH
    # ═══════════════════════════════════════════════════════════════════════
    if _safe_col(df, "sp500_price"):
        df["sp500_pct_change_1w"] = df["sp500_price"].pct_change(1) * 100
        df["sp500_pct_change_4w"] = df["sp500_price"].pct_change(4) * 100
        df["sp500_rolling_std_4w"] = df["sp500_price"].pct_change().rolling(4).std() * 100

    # ═══════════════════════════════════════════════════════════════════════
    #  VIX (MARKET VOLATILITY / FEAR GAUGE)
    # ═══════════════════════════════════════════════════════════════════════
    if _safe_col(df, "vix"):
        df["vix_change_1w"] = df["vix"] - df["vix"].shift(1)
        df["vix_level"] = df["vix"]  # Raw level is informative
        df["vix_above_20"] = (df["vix"] > 20).astype(int)
        df["vix_above_30"] = (df["vix"] > 30).astype(int)

    # ═══════════════════════════════════════════════════════════════════════
    #  TREASURY / YIELD CURVE (RECESSION SIGNAL)
    # ═══════════════════════════════════════════════════════════════════════
    if _safe_col(df, "yield_spread"):
        df["yield_spread_val"] = df["yield_spread"]
        df["yield_curve_inverted"] = (df["yield_spread"] < 0).astype(int)

    if _safe_col(df, "treasury_10y"):
        df["treasury_change_1w"] = df["treasury_10y"] - df["treasury_10y"].shift(1)

    # ═══════════════════════════════════════════════════════════════════════
    #  CPI / INFLATION
    # ═══════════════════════════════════════════════════════════════════════
    if _safe_col(df, "cpi"):
        df["cpi_pct_change_12w"] = df["cpi"].pct_change(12) * 100  # ~quarterly
        df["cpi_pct_change_52w"] = df["cpi"].pct_change(52) * 100  # ~annual

    # ═══════════════════════════════════════════════════════════════════════
    #  EIA SUPPLY / DEMAND FEATURES
    # ═══════════════════════════════════════════════════════════════════════

    # Gasoline inventories
    if _safe_col(df, "gasoline_stocks"):
        df["gas_stocks_change_1w"] = df["gasoline_stocks"] - df["gasoline_stocks"].shift(1)
        df["gas_stocks_change_4w"] = df["gasoline_stocks"] - df["gasoline_stocks"].shift(4)
        df["gas_stocks_pct_change_1w"] = df["gasoline_stocks"].pct_change(1) * 100

        # Stocks relative to recent average (proxy for "vs 5-year average")
        df["gas_stocks_vs_26w_avg"] = (
            df["gasoline_stocks"] - df["gasoline_stocks"].rolling(26).mean()
        )
        df["gas_stocks_vs_52w_avg"] = (
            df["gasoline_stocks"] - df["gasoline_stocks"].rolling(52).mean()
        )
        # Normalized (% above/below average)
        avg_52 = df["gasoline_stocks"].rolling(52).mean()
        df["gas_stocks_pct_vs_52w"] = ((df["gasoline_stocks"] - avg_52) / avg_52) * 100

    # Crude oil inventories
    if _safe_col(df, "crude_stocks"):
        df["crude_stocks_change_1w"] = df["crude_stocks"] - df["crude_stocks"].shift(1)
        df["crude_stocks_change_4w"] = df["crude_stocks"] - df["crude_stocks"].shift(4)
        avg_52c = df["crude_stocks"].rolling(52).mean()
        df["crude_stocks_pct_vs_52w"] = ((df["crude_stocks"] - avg_52c) / avg_52c) * 100

    # Refinery utilization
    if _safe_col(df, "refinery_utilization"):
        df["refinery_util_val"] = df["refinery_utilization"]
        df["refinery_util_change_1w"] = (
            df["refinery_utilization"] - df["refinery_utilization"].shift(1)
        )
        df["refinery_util_below_90"] = (df["refinery_utilization"] < 90).astype(int)
        df["refinery_util_below_85"] = (df["refinery_utilization"] < 85).astype(int)

    # Gasoline production
    if _safe_col(df, "gasoline_production"):
        df["gas_prod_change_1w"] = (
            df["gasoline_production"] - df["gasoline_production"].shift(1)
        )
        df["gas_prod_pct_change_1w"] = df["gasoline_production"].pct_change(1) * 100

    # Gasoline demand (product supplied)
    if _safe_col(df, "gasoline_demand"):
        df["gas_demand_change_1w"] = df["gasoline_demand"] - df["gasoline_demand"].shift(1)
        df["gas_demand_pct_change_1w"] = df["gasoline_demand"].pct_change(1) * 100
        df["gas_demand_rolling_4w"] = df["gasoline_demand"].rolling(4).mean()

    # Supply-demand balance
    if _safe_col(df, "gasoline_production") and _safe_col(df, "gasoline_demand"):
        df["gas_supply_demand_ratio"] = df["gasoline_production"] / df["gasoline_demand"].replace(0, np.nan)
        df["gas_supply_surplus"] = df["gasoline_production"] - df["gasoline_demand"]

    # Crude imports
    if _safe_col(df, "crude_imports"):
        df["crude_imports_change_1w"] = df["crude_imports"] - df["crude_imports"].shift(1)
        df["crude_imports_pct_change_1w"] = df["crude_imports"].pct_change(1) * 100

    # ═══════════════════════════════════════════════════════════════════════
    #  CALENDAR / SEASONAL FEATURES
    # ═══════════════════════════════════════════════════════════════════════
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter

    # Cyclical encoding
    df["week_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Seasonal binary flags
    df["is_summer_blend"] = (
        (df["week_of_year"] >= SUMMER_BLEND_START_WEEK)
        & (df["week_of_year"] <= SUMMER_BLEND_END_WEEK)
    ).astype(int)

    df["is_driving_season"] = (
        (df["week_of_year"] >= DRIVING_SEASON_START_WEEK)
        & (df["week_of_year"] <= DRIVING_SEASON_END_WEEK)
    ).astype(int)

    df["is_hurricane_season"] = (
        (df["week_of_year"] >= HURRICANE_SEASON_START_WEEK)
        & (df["week_of_year"] <= HURRICANE_SEASON_END_WEEK)
    ).astype(int)

    df["is_hurricane_peak"] = (
        (df["week_of_year"] >= HURRICANE_PEAK_START_WEEK)
        & (df["week_of_year"] <= HURRICANE_PEAK_END_WEEK)
    ).astype(int)

    # Holiday proximity
    holiday_weeks = [1, 21, 27, 36, 47, 52]  # NY, Memorial, July4, Labor, Thanksgiving, Xmas
    df["min_holiday_dist"] = df["week_of_year"].apply(
        lambda w: min(abs(w - hw) for hw in holiday_weeks)
    )

    df["year"] = df["date"].dt.year

    # ═══════════════════════════════════════════════════════════════════════
    #  INTERACTION FEATURES
    # ═══════════════════════════════════════════════════════════════════════
    if "crude_change_1w" in df.columns:
        df["crude_x_summer"] = df["crude_change_1w"] * df["is_summer_blend"]
        df["crude_x_hurricane"] = df["crude_change_1w"] * df["is_hurricane_season"]

    if "rbob_change_1w" in df.columns:
        df["rbob_x_summer"] = df["rbob_change_1w"] * df["is_summer_blend"]

    if "refinery_util_val" in df.columns and "is_hurricane_season" in df.columns:
        df["refinery_x_hurricane"] = df["refinery_util_val"] * df["is_hurricane_season"]

    if "gas_stocks_pct_vs_52w" in df.columns and "is_driving_season" in df.columns:
        df["stocks_x_driving"] = df["gas_stocks_pct_vs_52w"] * df["is_driving_season"]

    # Dollar strength × crude (strong dollar should dampen gas prices)
    if "dollar_pct_change_1w" in df.columns and "crude_change_1w" in df.columns:
        df["dollar_x_crude"] = df["dollar_pct_change_1w"] * df["crude_change_1w"]

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return list of feature column names (exclude identifiers, target, raw source cols)."""
    exclude = {"date", "target", "gas_price", "year"}
    # Exclude raw source columns used only for deriving features
    raw_cols = {
        "crude_price", "rbob_price", "heating_oil_price", "brent_price",
        "natgas_price", "dollar_idx_price", "sp500_price",
        "dollar_broad", "fred_wti", "henry_hub", "eur_usd",
        "fred_gas", "yield_spread", "treasury_10y", "cpi", "vix",
        "gasoline_stocks", "crude_stocks", "refinery_utilization",
        "gasoline_production", "gasoline_demand", "crude_imports",
        "distillate_stocks", "uga_etf_price",
    }
    exclude.update(raw_cols)

    feature_cols = [
        c for c in df.columns
        if c not in exclude
        and df[c].dtype in [np.float64, np.int64, np.int32, float, int]
    ]
    return feature_cols


def prepare_training_data(df: pd.DataFrame):
    """Prepare X and y for model training. Drops NaN rows."""
    feature_cols = get_feature_columns(df)
    train_df = df[feature_cols + ["target"]].dropna()
    X = train_df[feature_cols]
    y = train_df["target"]
    return X, y, feature_cols


def prepare_prediction_row(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare the most recent row for prediction."""
    feature_cols = get_feature_columns(df)
    last_row = df[feature_cols].iloc[-1:]
    if last_row.isnull().any(axis=1).iloc[0]:
        last_row = df[feature_cols].iloc[-2:-1]
    return last_row
