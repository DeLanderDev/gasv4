"""
Gas Price Predictor - Main Application
Streamlit dashboard for predicting US gasoline prices.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from config import DEFAULT_HISTORY_YEARS, MIN_DATA_POINTS
from data_collector import build_combined_dataset
from feature_engine import create_features, get_feature_columns
from model import GasPriceModel
from prediction_log import (
    save_prediction,
    load_prediction_log,
    backfill_actuals_from_data,
    clear_log,
    PredictionScheduler,
    load_schedule_config,
    delete_schedule_config,
)

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="US Gas Price Predictor",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background-color: #0e1117;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 12px 16px;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 8px 16px; border-radius: 4px 4px 0 0; }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⛽ Gas Price Predictor")
    st.markdown("---")

    # ── API Keys ──────────────────────────────────────────────────────────
    st.subheader("🔑 API Keys")

    st.markdown(
        "**EIA** *(required)* — "
        "[Get free key](https://www.eia.gov/opendata/register.php)"
    )
    eia_key = st.text_input("EIA API Key", type="password", key="eia")

    st.markdown(
        "**FRED** *(optional, adds economic data)* — "
        "[Get free key](https://fred.stlouisfed.org/docs/api/api_key.html)"
    )
    fred_key = st.text_input("FRED API Key", type="password", key="fred")

    # ── AAA Price ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("⛽ AAA Current Price")
    st.markdown(
        "Enter today's national average from:  \n"
        "[gasprices.aaa.com](https://gasprices.aaa.com)"
    )
    aaa_price = st.number_input(
        "AAA National Average ($/gal)",
        min_value=0.00, max_value=10.00,
        value=0.000, step=0.001, format="%.3f",
        help="The model predicts the weekly CHANGE using EIA data, "
             "then applies that change to this AAA price.",
    )

    # ── Settings ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("⚙️ Settings")

    history_years = st.slider(
        "History (years)", min_value=2, max_value=10,
        value=DEFAULT_HISTORY_YEARS,
    )

    run_validation = st.checkbox("Run walk-forward validation", value=True)

    # ── About ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("ℹ️ Data Sources")
    st.markdown(
        "**EIA API** — retail gas prices, gasoline & crude inventories, "
        "refinery utilization, production, demand, imports\n\n"
        "**Yahoo Finance** — WTI & Brent crude, RBOB gasoline, "
        "heating oil, natural gas, US Dollar Index, S&P 500\n\n"
        "**FRED** — trade-weighted dollar, treasury yields, "
        "yield curve, VIX, CPI, Henry Hub gas, EUR/USD"
    )

# ─── Main Content ────────────────────────────────────────────────────────────
st.title("🇺🇸 US Gasoline Price Predictor")
st.markdown(
    "*Predicting AAA national average regular gasoline — "
    "every Sunday at 11:59 PM ET*"
)

if not eia_key:
    st.warning(
        "👈 Enter your **free EIA API key** in the sidebar to get started.  \n"
        "Get one at [eia.gov/opendata/register.php]"
        "(https://www.eia.gov/opendata/register.php) — it's instant and free."
    )

    st.markdown("---")
    st.subheader("How It Works")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📊 Data Collection")
        st.markdown(
            "Pulls weekly data from **3 sources**: EIA (prices + supply/demand), "
            "Yahoo Finance (futures & indices), and FRED (economic indicators)."
        )
    with col2:
        st.markdown("### 🔧 Feature Engineering")
        st.markdown(
            "Creates **80+ predictive features** from price lags, momentum, "
            "inventories, refinery utilization, dollar strength, VIX, "
            "yield curve, seasonality, and supply-demand balance."
        )
    with col3:
        st.markdown("### 🤖 XGBoost Prediction")
        st.markdown(
            "Trains a gradient-boosted tree model with walk-forward "
            "validation. Applies predicted change to AAA baseline."
        )
    st.stop()

# ─── Data Loading ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_data(eia: str, fred: str, years: int) -> pd.DataFrame:
    return build_combined_dataset(eia, fred_api_key=fred, years=years)


data = None
with st.spinner("📡 Fetching data from EIA, Yahoo Finance, and FRED..."):
    try:
        data = load_data(eia_key, fred_key, history_years)
    except Exception as e:
        st.error(f"❌ Error fetching data: {str(e)}")
        st.info(
            "**Troubleshooting:**\n"
            "- Verify your EIA API key is correct\n"
            "- Check your internet connection\n"
            "- The EIA or Yahoo Finance API may be temporarily unavailable"
        )
        st.stop()

if data is None or len(data) < MIN_DATA_POINTS:
    st.error(
        f"Not enough data. Got {len(data) if data is not None else 0}, "
        f"need {MIN_DATA_POINTS}."
    )
    st.stop()

# Show data source summary
data_cols = set(data.columns)
eia_supply = sum(
    1 for c in ["gasoline_stocks", "crude_stocks", "refinery_utilization",
                "gasoline_production", "gasoline_demand", "crude_imports"]
    if c in data_cols and data[c].notna().sum() > 10
)
fred_count = sum(
    1 for c in ["dollar_broad", "vix", "yield_spread", "treasury_10y",
                "cpi", "henry_hub", "eur_usd", "fred_wti"]
    if c in data_cols and data[c].notna().sum() > 10
)
yahoo_count = sum(
    1 for c in ["crude_price", "rbob_price", "brent_price", "natgas_price",
                "dollar_idx_price", "sp500_price", "heating_oil_price"]
    if c in data_cols and data[c].notna().sum() > 10
)

source_parts = [f"**EIA**: gas prices + {eia_supply} supply/demand series"]
source_parts.append(f"**Yahoo**: {yahoo_count} market tickers")
if fred_count > 0:
    source_parts.append(f"**FRED**: {fred_count} economic indicators")
else:
    source_parts.append("**FRED**: *not connected — add key for more accuracy*")

st.caption(f"📊 Data loaded: {len(data)} weeks | " + " | ".join(source_parts))

# Data freshness warning
latest_date = data["date"].max()
data_age = (datetime.now() - pd.Timestamp(latest_date)).days
if data_age > 10:
    st.warning(f"⚠️ Most recent data is {data_age} days old.")

# ─── Model Training ──────────────────────────────────────────────────────────
model = GasPriceModel()

with st.spinner("🧠 Training prediction model..."):
    try:
        train_metrics = model.train(data)
    except Exception as e:
        st.error(f"❌ Error training model: {str(e)}")
        st.stop()

val_metrics = None
val_results = None
if run_validation:
    with st.spinner("📐 Running walk-forward validation..."):
        try:
            val_metrics, val_results = model.walk_forward_validate(data)
        except Exception as e:
            st.warning(f"Validation could not be completed: {str(e)}")

# ─── Prediction ───────────────────────────────────────────────────────────────
try:
    prediction = model.predict_next_week(data)
except Exception as e:
    st.error(f"❌ Error generating prediction: {str(e)}")
    st.stop()

# ─── AAA Price Adjustment ────────────────────────────────────────────────────
use_aaa = aaa_price > 0.0
if use_aaa:
    eia_current = prediction["current_price"]
    eia_predicted_change = prediction["predicted_change"]
    eia_aaa_offset = round(aaa_price - eia_current, 4)

    aaa_prediction = round(aaa_price + eia_predicted_change, 4)
    aaa_pct_change = round((eia_predicted_change / aaa_price) * 100, 3)
    aaa_ci_68_low = round(aaa_price + eia_predicted_change - prediction["std_error"], 4)
    aaa_ci_68_high = round(aaa_price + eia_predicted_change + prediction["std_error"], 4)
    aaa_ci_95_low = round(aaa_price + eia_predicted_change - 1.96 * prediction["std_error"], 4)
    aaa_ci_95_high = round(aaa_price + eia_predicted_change + 1.96 * prediction["std_error"], 4)
    aaa_direction = (
        "UP" if eia_predicted_change > 0.001
        else ("DOWN" if eia_predicted_change < -0.001 else "FLAT")
    )

    display = {
        "current_price": aaa_price,
        "current_source": "AAA",
        "prediction": aaa_prediction,
        "predicted_change": eia_predicted_change,
        "predicted_pct_change": aaa_pct_change,
        "direction": aaa_direction,
        "ci_68_low": aaa_ci_68_low, "ci_68_high": aaa_ci_68_high,
        "ci_95_low": aaa_ci_95_low, "ci_95_high": aaa_ci_95_high,
        "prediction_date": prediction["prediction_date"],
        "prediction_day": prediction["prediction_day"],
        "eia_current": eia_current,
        "eia_prediction": prediction["prediction"],
        "eia_aaa_offset": eia_aaa_offset,
    }
else:
    display = {
        "current_price": prediction["current_price"],
        "current_source": "EIA",
        "prediction": prediction["prediction"],
        "predicted_change": prediction["predicted_change"],
        "predicted_pct_change": prediction["predicted_pct_change"],
        "direction": prediction["direction"],
        "ci_68_low": prediction["ci_68_low"], "ci_68_high": prediction["ci_68_high"],
        "ci_95_low": prediction["ci_95_low"], "ci_95_high": prediction["ci_95_high"],
        "prediction_date": prediction["prediction_date"],
        "prediction_day": prediction["prediction_day"],
        "eia_current": prediction["current_price"],
        "eia_prediction": prediction["prediction"],
        "eia_aaa_offset": 0.0,
    }

# ─── Hero Metrics ─────────────────────────────────────────────────────────────
st.markdown("---")

if not use_aaa:
    st.info(
        "💡 **Tip:** Enter the current AAA national average in the sidebar "
        "(from [gasprices.aaa.com](https://gasprices.aaa.com)) for a prediction "
        "calibrated to AAA prices. Currently showing EIA-based prediction."
    )

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label=f"📍 Current Average ({display['current_source']})",
        value=f"${display['current_price']:.3f}",
    )
with col2:
    st.metric(
        label=f"🔮 Predicted: {display['prediction_day']}",
        value=f"${display['prediction']:.3f}",
        delta=f"{display['predicted_change']:+.3f} ({display['predicted_pct_change']:+.2f}%)",
        delta_color="inverse",
    )
with col3:
    st.metric(
        label="📊 68% Confidence Range",
        value=f"${display['ci_68_low']:.3f} – ${display['ci_68_high']:.3f}",
    )
with col4:
    if val_metrics:
        st.metric(label="🎯 Validation MAE", value=f"±${val_metrics['val_mae']:.3f}")
    else:
        st.metric(label="🎯 Training R²", value=f"{train_metrics['r2']:.4f}")

# ─── Summary ──────────────────────────────────────────────────────────────────
st.markdown("---")

dir_color = (
    "#ff6b6b" if display["direction"] == "UP"
    else ("#51cf66" if display["direction"] == "DOWN" else "#ffd43b")
)
summary = (
    f"### Prediction Summary\n"
    f"The model predicts gas prices will go "
    f"**<span style='color:{dir_color}'>{display['predicted_change']:+.3f}/gal "
    f"({display['direction']})</span>** "
    f"by Sunday **{display['prediction_day']}**, "
    f"from ${display['current_price']:.3f} → "
    f"**${display['prediction']:.3f}** per gallon.  \n"
    f"95% confidence interval: "
    f"${display['ci_95_low']:.3f} – ${display['ci_95_high']:.3f}"
)
if use_aaa:
    summary += (
        f"  \n\n*Based on AAA price ${aaa_price:.3f} + model-predicted change "
        f"of {display['predicted_change']:+.3f}. "
        f"EIA/AAA offset: {display['eia_aaa_offset']:+.3f}*"
    )
st.markdown(summary, unsafe_allow_html=True)

# ─── Save Prediction Button ──────────────────────────────────────────────────
save_col1, save_col2 = st.columns([1, 4])
with save_col1:
    if st.button("💾 Save Prediction", type="primary"):
        ts = save_prediction(display, prediction, save_type="manual")
        st.success(f"Prediction saved at {ts}")
with save_col2:
    st.caption(
        "Save the current prediction with a timestamp so you can compare "
        "accuracy across different days and times."
    )

# ─── Auto-backfill actuals from loaded data ──────────────────────────────────
backfill_actuals_from_data(data)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Price & Forecast",
    "🎯 Model Accuracy",
    "🔍 Feature Analysis",
    "🛢️ Supply & Demand",
    "📋 Data Explorer",
    "📝 Prediction Log",
])

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1: Price History
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Historical Gas Prices with Prediction")

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=("US National Average Gas Price ($/gal)", "Weekly Change"),
    )

    fig.add_trace(go.Scatter(
        x=data["date"], y=data["gas_price"], mode="lines",
        name="Actual Price", line=dict(color="#4dabf7", width=2),
    ), row=1, col=1)

    if "crude_price" in data.columns:
        fig.add_trace(go.Scatter(
            x=data["date"], y=data["crude_price"] / 42, mode="lines",
            name="Crude Oil ($/gal equiv)", line=dict(color="#ffa94d", width=1, dash="dot"),
            opacity=0.6,
        ), row=1, col=1)

    if "rbob_price" in data.columns:
        fig.add_trace(go.Scatter(
            x=data["date"], y=data["rbob_price"], mode="lines",
            name="RBOB Wholesale", line=dict(color="#ff6b6b", width=1, dash="dash"),
            opacity=0.6,
        ), row=1, col=1)

    pred_date = pd.Timestamp(display["prediction_date"])
    fig.add_trace(go.Scatter(
        x=[pred_date], y=[display["prediction"]], mode="markers",
        name=f"Prediction ({'AAA' if use_aaa else 'EIA'})",
        marker=dict(color="#51cf66", size=14, symbol="star",
                    line=dict(width=2, color="white")),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=[pred_date, pred_date],
        y=[display["ci_95_low"], display["ci_95_high"]],
        mode="lines", name="95% CI", line=dict(color="#51cf66", width=3),
    ), row=1, col=1)

    weekly_changes = data["gas_price"].diff()
    colors = ["#51cf66" if c <= 0 else "#ff6b6b" for c in weekly_changes]
    fig.add_trace(go.Bar(
        x=data["date"], y=weekly_changes, name="Weekly Change",
        marker_color=colors, showlegend=False,
    ), row=2, col=1)

    fig.update_layout(
        height=650, template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=30, t=60, b=30),
    )
    fig.update_yaxes(title_text="Price ($/gal)", row=1, col=1)
    fig.update_yaxes(title_text="Change ($/gal)", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # Recent trend table
    st.subheader("Recent Price Trends")
    recent = data.tail(8).copy()
    recent["Weekly Change"] = recent["gas_price"].diff()
    recent["% Change"] = recent["gas_price"].pct_change() * 100
    display_df = recent[["date", "gas_price", "Weekly Change", "% Change"]].copy()
    display_df.columns = ["Date", "Price ($/gal)", "Change ($)", "Change (%)"]
    display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m-%d")
    st.dataframe(display_df.iloc[::-1].reset_index(drop=True),
                 use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2: Model Accuracy
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    if val_results is not None and val_metrics is not None:
        st.subheader("Walk-Forward Validation Results")

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("MAE", f"${val_metrics['val_mae']:.4f}")
        c2.metric("RMSE", f"${val_metrics['val_rmse']:.4f}")
        c3.metric("Within ±$0.02", f"{val_metrics['val_within_2_cents']:.1f}%")
        c4.metric("Within ±$0.05", f"{val_metrics['val_within_5_cents']:.1f}%")
        c5.metric("Direction Acc.", f"{val_metrics['val_direction_accuracy']:.1f}%")
        c6.metric(
            "Shrinkage",
            f"{train_metrics.get('shrinkage', 'N/A')}",
            help="Model predictions dampened by this factor. "
                 "Auto-calibrated to maximize ±$0.02 accuracy.",
        )

        # Naive baseline comparison
        naive_2c = val_metrics.get("naive_within_2_cents", 0)
        model_2c = val_metrics["val_within_2_cents"]
        if naive_2c > 0:
            improvement = model_2c - naive_2c
            st.caption(
                f"📐 **Naive baseline** (predict no change): "
                f"±$0.02 = {naive_2c:.1f}%, MAE = ${val_metrics.get('naive_mae', 0):.4f}  |  "
                f"**Model improvement**: {improvement:+.1f} percentage points"
            )

        # Actual vs Predicted
        fig_val = go.Figure()
        fig_val.add_trace(go.Scatter(
            x=val_results["date"], y=val_results["actual"],
            mode="lines", name="Actual", line=dict(color="#4dabf7", width=2),
        ))
        fig_val.add_trace(go.Scatter(
            x=val_results["date"], y=val_results["predicted"],
            mode="lines", name="Predicted", line=dict(color="#ffa94d", width=2, dash="dash"),
        ))
        fig_val.update_layout(
            title="Actual vs Predicted (Out-of-Sample)", height=400,
            template="plotly_dark", yaxis_title="Price ($/gal)",
        )
        st.plotly_chart(fig_val, use_container_width=True)

        # Error charts
        col_a, col_b = st.columns(2)
        with col_a:
            fig_err = go.Figure()
            fig_err.add_trace(go.Histogram(
                x=val_results["error"], nbinsx=30, marker_color="#4dabf7",
            ))
            fig_err.add_vline(x=0, line_dash="dash", line_color="white")
            fig_err.update_layout(
                title="Error Distribution", height=350, template="plotly_dark",
                xaxis_title="Error ($/gal)", yaxis_title="Count",
            )
            st.plotly_chart(fig_err, use_container_width=True)

        with col_b:
            fig_abs = go.Figure()
            fig_abs.add_trace(go.Scatter(
                x=val_results["date"], y=val_results["abs_error"],
                mode="lines+markers", line=dict(color="#ff6b6b", width=1),
                marker=dict(size=4),
            ))
            fig_abs.add_hline(
                y=val_metrics["val_mae"], line_dash="dash", line_color="#ffd43b",
                annotation_text=f"MAE: ${val_metrics['val_mae']:.4f}",
            )
            fig_abs.update_layout(
                title="Absolute Error Over Time", height=350, template="plotly_dark",
                yaxis_title="Abs Error ($/gal)",
            )
            st.plotly_chart(fig_abs, use_container_width=True)

        # Detailed metrics table
        st.subheader("Detailed Metrics")
        metrics_rows = {
            "Mean Absolute Error": f"${val_metrics['val_mae']:.4f}",
            "RMSE": f"${val_metrics['val_rmse']:.4f}",
            "Median Absolute Error": f"${val_metrics['val_median_abs_error']:.4f}",
            "95th Percentile Error": f"${val_metrics['val_95_pct_error']:.4f}",
            "Max Error": f"${val_metrics['val_max_error']:.4f}",
            "R² Score": f"{val_metrics['val_r2']:.4f}",
            "Mean Bias": f"${val_metrics['val_mean_error']:+.4f}",
            "Within ±$0.02": f"{val_metrics['val_within_2_cents']:.1f}%",
            "Within ±$0.05": f"{val_metrics['val_within_5_cents']:.1f}%",
            "Within ±$0.10": f"{val_metrics['val_within_10_cents']:.1f}%",
            "Direction Accuracy": f"{val_metrics['val_direction_accuracy']:.1f}%",
            "Test Weeks": f"{val_metrics['val_n_test']}",
            "Features Used": f"{train_metrics['n_features']}",
            "Auto Shrinkage": f"{train_metrics.get('shrinkage', 'N/A')}",
            "Naive ±$0.02": f"{val_metrics.get('naive_within_2_cents', 0):.1f}%",
            "Naive MAE": f"${val_metrics.get('naive_mae', 0):.4f}",
        }
        st.dataframe(
            pd.DataFrame({"Metric": metrics_rows.keys(), "Value": metrics_rows.values()}),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("Enable **Walk-forward validation** in the sidebar to see accuracy metrics.")


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 3: Feature Analysis
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Feature Importance")
    importance_df = model.get_feature_importance(top_n=25)

    if not importance_df.empty:
        fig_imp = go.Figure()
        fig_imp.add_trace(go.Bar(
            x=importance_df["importance"].values[::-1],
            y=importance_df["feature"].values[::-1],
            orientation="h", marker_color="#4dabf7",
        ))
        fig_imp.update_layout(
            title=f"Top {len(importance_df)} Predictive Features "
                  f"(of {train_metrics['n_features']} total)",
            height=600, template="plotly_dark",
            xaxis_title="Importance (Gain)", margin=dict(l=250),
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    # Feature descriptions
    st.subheader("Feature Descriptions")
    desc_map = {
        "gas_lag_": "Gas price N weeks ago",
        "gas_change_": "Gas price change over N weeks",
        "gas_rolling_mean_": "N-week rolling average",
        "gas_rolling_std_": "N-week price volatility",
        "gas_vs_ma_": "Price vs N-week moving average",
        "gas_acceleration": "Change in rate of price change",
        "crude_": "WTI crude oil ",
        "brent_": "Brent crude oil ",
        "rbob_": "RBOB wholesale gasoline ",
        "crack_spread": "Refinery margin (retail − crude equivalent)",
        "rbob_retail_spread": "Gap between wholesale and retail",
        "natgas_": "Natural gas ",
        "heating_": "Heating oil ",
        "dollar_": "US Dollar Index ",
        "eurusd_": "EUR/USD exchange rate ",
        "sp500_": "S&P 500 index ",
        "vix_": "VIX volatility / fear index ",
        "yield_": "Treasury yield curve ",
        "treasury_": "10-Year Treasury ",
        "cpi_": "Consumer Price Index (inflation) ",
        "gas_stocks_": "Gasoline inventory levels ",
        "crude_stocks_": "Crude oil inventory levels ",
        "refinery_util": "Refinery utilization rate ",
        "gas_prod_": "Gasoline production ",
        "gas_demand_": "Gasoline demand (product supplied) ",
        "gas_supply_": "Gasoline supply-demand balance ",
        "crude_imports_": "Crude oil imports ",
        "week_sin": "Week-of-year (cyclical sine)",
        "week_cos": "Week-of-year (cyclical cosine)",
        "month_sin": "Month (cyclical sine)",
        "month_cos": "Month (cyclical cosine)",
        "is_summer_blend": "Summer gasoline blend period (Apr–Oct)",
        "is_driving_season": "Summer driving season (May–Sep)",
        "is_hurricane_season": "Atlantic hurricane season (Jun–Nov)",
        "is_hurricane_peak": "Peak hurricane season (Aug–Oct)",
        "min_holiday_dist": "Weeks until nearest major holiday",
        "crude_x_": "Crude oil × seasonal interaction",
        "rbob_x_": "RBOB × seasonal interaction",
        "refinery_x_": "Refinery × hurricane season interaction",
        "stocks_x_": "Inventory × driving season interaction",
        "dollar_x_": "Dollar × crude oil interaction",
    }

    if not importance_df.empty:
        rows = []
        for feat in importance_df["feature"].tolist():
            desc = "Derived feature"
            for prefix, d in desc_map.items():
                if feat.startswith(prefix) or feat == prefix:
                    desc = d
                    break
            rows.append({"Feature": feat, "Description": desc})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Correlations
    st.subheader("Top Correlations with Next Week's Price")
    featured_df = create_features(data)
    feature_cols = get_feature_columns(featured_df)
    corr = (
        featured_df[feature_cols + ["target"]]
        .corr()["target"]
        .drop("target")
        .sort_values(key=abs, ascending=False)
        .head(15)
    )
    fig_corr = go.Figure()
    colors_c = ["#51cf66" if v > 0 else "#ff6b6b" for v in corr.values[::-1]]
    fig_corr.add_trace(go.Bar(
        x=corr.values[::-1], y=corr.index[::-1],
        orientation="h", marker_color=colors_c,
    ))
    fig_corr.update_layout(
        title="Top 15 Features Correlated with Next Week's Price",
        height=450, template="plotly_dark", margin=dict(l=250),
    )
    st.plotly_chart(fig_corr, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 4: Supply & Demand Dashboard
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("EIA Supply & Demand Indicators")

    has_supply_data = any(
        c in data.columns and data[c].notna().sum() > 10
        for c in ["gasoline_stocks", "crude_stocks", "refinery_utilization",
                   "gasoline_demand"]
    )

    if not has_supply_data:
        st.warning(
            "No supply/demand data available. This can happen if the EIA "
            "weekly series IDs have changed or the API is temporarily down."
        )
    else:
        # Gasoline & Crude Inventories
        fig_inv = make_subplots(
            rows=2, cols=2, shared_xaxes=True,
            subplot_titles=(
                "Gasoline Stocks (K barrels)", "Crude Oil Stocks (K barrels)",
                "Refinery Utilization (%)", "Gasoline Demand (Kbd)",
            ),
            vertical_spacing=0.12,
        )

        if "gasoline_stocks" in data.columns:
            fig_inv.add_trace(go.Scatter(
                x=data["date"], y=data["gasoline_stocks"],
                line=dict(color="#4dabf7"), showlegend=False,
            ), row=1, col=1)

        if "crude_stocks" in data.columns:
            fig_inv.add_trace(go.Scatter(
                x=data["date"], y=data["crude_stocks"],
                line=dict(color="#ffa94d"), showlegend=False,
            ), row=1, col=2)

        if "refinery_utilization" in data.columns:
            fig_inv.add_trace(go.Scatter(
                x=data["date"], y=data["refinery_utilization"],
                line=dict(color="#51cf66"), showlegend=False,
            ), row=2, col=1)
            fig_inv.add_hline(y=90, line_dash="dash", line_color="red",
                              row=2, col=1)

        if "gasoline_demand" in data.columns:
            fig_inv.add_trace(go.Scatter(
                x=data["date"], y=data["gasoline_demand"],
                line=dict(color="#e599f7"), showlegend=False,
            ), row=2, col=2)

        fig_inv.update_layout(
            height=550, template="plotly_dark",
            margin=dict(l=60, r=30, t=60, b=30),
        )
        st.plotly_chart(fig_inv, use_container_width=True)

    # Market indicators (Dollar, VIX, S&P)
    st.subheader("Market & Economic Indicators")

    has_market = any(
        c in data.columns and data[c].notna().sum() > 10
        for c in ["dollar_idx_price", "dollar_broad", "vix", "sp500_price"]
    )

    if has_market:
        fig_mkt = make_subplots(
            rows=1, cols=3,
            subplot_titles=("US Dollar Index", "VIX (Fear Gauge)", "S&P 500"),
        )

        dollar_col = None
        for c in ["dollar_idx_price", "dollar_broad"]:
            if c in data.columns and data[c].notna().sum() > 10:
                dollar_col = c
                break
        if dollar_col:
            fig_mkt.add_trace(go.Scatter(
                x=data["date"], y=data[dollar_col],
                line=dict(color="#4dabf7"), showlegend=False,
            ), row=1, col=1)

        if "vix" in data.columns and data["vix"].notna().sum() > 10:
            fig_mkt.add_trace(go.Scatter(
                x=data["date"], y=data["vix"],
                line=dict(color="#ffa94d"), showlegend=False,
            ), row=1, col=2)

        if "sp500_price" in data.columns and data["sp500_price"].notna().sum() > 10:
            fig_mkt.add_trace(go.Scatter(
                x=data["date"], y=data["sp500_price"],
                line=dict(color="#51cf66"), showlegend=False,
            ), row=1, col=3)

        fig_mkt.update_layout(height=350, template="plotly_dark")
        st.plotly_chart(fig_mkt, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 5: Data Explorer
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Raw Data Explorer")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Data Points", f"{len(data)} weeks")
    col_b.metric("Columns", f"{len(data.columns)}")
    col_c.metric(
        "Date Range",
        f"{data['date'].min():%Y-%m-%d} to {data['date'].max():%Y-%m-%d}",
    )

    st.dataframe(
        data.sort_values("date", ascending=False).reset_index(drop=True),
        use_container_width=True, hide_index=True,
    )

    csv = data.to_csv(index=False)
    st.download_button(
        "📥 Download Raw Data (CSV)", data=csv,
        file_name=f"gas_price_data_{datetime.now():%Y%m%d}.csv", mime="text/csv",
    )

    # Feature summary
    st.subheader(f"Engineered Features ({train_metrics['n_features']} total)")
    featured_df = create_features(data)
    feature_cols = get_feature_columns(featured_df)
    latest = featured_df.iloc[-1]
    feat_display = pd.DataFrame({
        "Feature": feature_cols,
        "Value": [
            f"{latest[c]:.4f}" if pd.notna(latest[c]) else "N/A"
            for c in feature_cols
        ],
    })
    st.dataframe(feat_display, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 6: Prediction Log & Scheduler
# ═══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.subheader("Saved Predictions")

    log_df = load_prediction_log()

    if log_df.empty:
        st.info(
            "No predictions saved yet. Click **Save Prediction** above "
            "to start tracking predictions over time."
        )
    else:
        # ── Summary metrics ───────────────────────────────────────────────
        has_actuals = log_df["actual_price"].notna() & (log_df["actual_price"] != "")
        n_total = len(log_df)
        n_with_actuals = has_actuals.sum()

        lc1, lc2, lc3, lc4 = st.columns(4)
        lc1.metric("Total Saved", n_total)
        lc2.metric("With Actuals", int(n_with_actuals))

        if n_with_actuals > 0:
            filled = log_df[has_actuals].copy()
            filled["abs_error"] = pd.to_numeric(filled["abs_error"], errors="coerce")
            avg_err = filled["abs_error"].mean()
            within_2c = (filled["abs_error"] <= 0.02).mean() * 100
            lc3.metric("Avg Abs Error", f"${avg_err:.4f}" if not pd.isna(avg_err) else "N/A")
            lc4.metric("Within ±$0.02", f"{within_2c:.1f}%" if not pd.isna(within_2c) else "N/A")
        else:
            lc3.metric("Avg Abs Error", "—")
            lc4.metric("Within ±$0.02", "—")

        # ── Accuracy by day of week and hour ──────────────────────────────
        if n_with_actuals >= 2:
            st.subheader("Accuracy by Save Day & Time")

            filled = log_df[has_actuals].copy()
            filled["abs_error"] = pd.to_numeric(filled["abs_error"], errors="coerce")
            filled["save_dow"] = filled["save_timestamp"].dt.day_name()
            filled["save_hour"] = filled["save_timestamp"].dt.hour

            acc_col1, acc_col2 = st.columns(2)

            with acc_col1:
                by_day = (
                    filled.groupby("save_dow")["abs_error"]
                    .agg(["mean", "count"])
                    .reindex(["Monday", "Tuesday", "Wednesday", "Thursday",
                              "Friday", "Saturday", "Sunday"])
                    .dropna()
                )
                if not by_day.empty:
                    fig_day = go.Figure()
                    fig_day.add_trace(go.Bar(
                        x=by_day.index,
                        y=by_day["mean"],
                        text=[f"n={int(c)}" for c in by_day["count"]],
                        textposition="outside",
                        marker_color="#4dabf7",
                    ))
                    fig_day.update_layout(
                        title="Mean Abs Error by Day of Week (Save Day)",
                        height=350, template="plotly_dark",
                        yaxis_title="Mean Abs Error ($/gal)",
                    )
                    st.plotly_chart(fig_day, use_container_width=True)

            with acc_col2:
                by_hour = (
                    filled.groupby("save_hour")["abs_error"]
                    .agg(["mean", "count"])
                )
                if not by_hour.empty:
                    fig_hour = go.Figure()
                    fig_hour.add_trace(go.Bar(
                        x=[f"{int(h):02d}:00" for h in by_hour.index],
                        y=by_hour["mean"],
                        text=[f"n={int(c)}" for c in by_hour["count"]],
                        textposition="outside",
                        marker_color="#ffa94d",
                    ))
                    fig_hour.update_layout(
                        title="Mean Abs Error by Hour of Day (Save Time)",
                        height=350, template="plotly_dark",
                        yaxis_title="Mean Abs Error ($/gal)",
                    )
                    st.plotly_chart(fig_hour, use_container_width=True)

        # ── Prediction log table ──────────────────────────────────────────
        st.subheader("Prediction History")
        display_log = log_df.copy()
        display_log["save_timestamp"] = display_log["save_timestamp"].dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        display_log["prediction_for_date"] = pd.to_datetime(
            display_log["prediction_for_date"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")
        st.dataframe(
            display_log.iloc[::-1].reset_index(drop=True),
            use_container_width=True, hide_index=True,
        )

        # Download and clear buttons
        dl_col, clr_col, _ = st.columns([1, 1, 3])
        with dl_col:
            csv_data = log_df.to_csv(index=False)
            st.download_button(
                "📥 Download Log (CSV)", data=csv_data,
                file_name=f"prediction_log_{datetime.now():%Y%m%d}.csv",
                mime="text/csv",
            )
        with clr_col:
            if st.button("🗑️ Clear Log"):
                clear_log()
                st.rerun()

    # ── Auto-Scheduler ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Automatic Prediction Scheduler")
    st.markdown(
        "Schedule automatic prediction saves while the app is running. "
        "Predictions are saved at the chosen times so you can later compare "
        "which days and times produce the most accurate results."
    )

    # Initialize scheduler in session state
    if "scheduler" not in st.session_state:
        st.session_state.scheduler = PredictionScheduler()

    scheduler = st.session_state.scheduler

    # Show current schedule status
    existing_config = load_schedule_config()
    if scheduler.is_running:
        st.success(
            f"Scheduler is **running** since "
            f"{existing_config.get('started_at', 'unknown') if existing_config else 'unknown'}."
        )
        if existing_config:
            st.caption(
                f"Frequency: {existing_config.get('frequency', '?').replace('_', ' ')} | "
                f"Times: {', '.join(existing_config.get('times', []))} | "
                f"Ends: {existing_config.get('end_date', '?')}"
            )
        if st.button("⏹️ Stop Scheduler"):
            scheduler.stop()
            st.success("Scheduler stopped.")
            st.rerun()
    else:
        # Schedule configuration form
        with st.form("schedule_form"):
            sched_col1, sched_col2 = st.columns(2)

            with sched_col1:
                frequency = st.selectbox(
                    "Frequency",
                    options=["once_per_day", "twice_per_day"],
                    format_func=lambda x: x.replace("_", " ").title(),
                )
                time_1 = st.time_input(
                    "First save time", value=datetime.strptime("08:00", "%H:%M").time()
                )

            with sched_col2:
                time_2 = st.time_input(
                    "Second save time (for twice per day)",
                    value=datetime.strptime("20:00", "%H:%M").time(),
                )
                end_date = st.date_input(
                    "Run until (end date)",
                    value=datetime.now().date() + timedelta(days=7),
                    min_value=datetime.now().date() + timedelta(days=1),
                )

            duration_days = (end_date - datetime.now().date()).days
            st.caption(
                f"Will save predictions for **{duration_days} day(s)** "
                f"while the app is running."
            )

            submitted = st.form_submit_button("▶️ Start Scheduler", type="primary")

            if submitted:
                times = [time_1.strftime("%H:%M")]
                if frequency == "twice_per_day":
                    times.append(time_2.strftime("%H:%M"))

                def make_prediction():
                    return (display, prediction)

                scheduler.start(
                    predict_fn=make_prediction,
                    frequency=frequency,
                    times=times,
                    end_date=end_date.strftime("%Y-%m-%d"),
                )
                st.success(
                    f"Scheduler started! Saving {'twice' if frequency == 'twice_per_day' else 'once'} "
                    f"daily at {', '.join(times)} until {end_date}."
                )
                st.rerun()

        st.caption(
            "**Note:** The scheduler runs in the background while this app is open in your browser. "
            "If you close the browser tab or stop the Streamlit server, the scheduler will stop. "
            "Saved predictions persist across sessions."
        )


# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 0.85rem;'>"
    "⛽ Gas Price Predictor | Data: EIA + Yahoo Finance + FRED | Model: XGBoost<br>"
    "Predictions are estimates only — not financial advice. "
    "Actual prices may differ due to unforeseen events.<br>"
    f"Last updated: {datetime.now():%Y-%m-%d %H:%M} | "
    f"Data through: {data['date'].max():%Y-%m-%d} | "
    f"Features: {train_metrics['n_features']}"
    "</div>",
    unsafe_allow_html=True,
)
