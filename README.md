# ⛽ US Gas Price Predictor

Predicts the AAA national average regular gasoline price for every Sunday at 11:59 PM ET, using a **decomposition-based hybrid ensemble** with **STL + GRU + XGBoost + stacking** and 100+ engineered features from 3 free data sources.

---

## 🚀 Quick Start

### 1. Get Free API Keys

| Source | Required? | Sign Up |
|--------|-----------|---------|
| **EIA** | ✅ Yes | [eia.gov/opendata/register.php](https://www.eia.gov/opendata/register.php) |
| **FRED** | Optional (adds ~8 economic indicators) | [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html) |

Both are instant and free.

### 2. Install & Run

```bash
cd gas_price_predictor
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

Enter your API key(s) in the sidebar. Optionally enter today's AAA national average from [gasprices.aaa.com](https://gasprices.aaa.com) for AAA-calibrated predictions.

---

## 🤖 Model Architecture

### Decomposition-Based Hybrid Prediction

The model first decomposes weekly gasoline prices into:

- **Trend** (long-run level shifts)
- **Seasonality** (calendar/weekly cyclic effects)
- **Residual** (idiosyncratic shocks)

Then it trains specialized learners for each signal:

| Stage | Model | Role |
|-------|-------|------|
| 1 | STL decomposition | Split price series into trend / seasonal / residual components |
| 2 | GRU sequence model | Forecast next-week trend + seasonal components |
| 3 | XGBoost | Predict residual component from engineered FRED/EIA/Yahoo features |
| 4 | Ridge meta-learner (stacking) | Combine component-level predictions into final price forecast |

This keeps the time-series structure in a recurrent model while still exploiting rich exogenous feature interactions in XGBoost.

### Ensemble Output

```
trend_season_hat = GRU(trend, seasonal history)
residual_hat     = XGBoost(features)
final_price      = MetaLearner([trend_season_hat, residual_hat, trend_season_hat + residual_hat])
```

### Recent-Data Weighting

The most recent 40% of training data gets 2× weight so the model tracks current market conditions.

---

## 📊 Data Sources (All Free)

### EIA API (same key for all)
| Data | Why It Matters |
|------|---------------|
| Retail gas prices (weekly) | Target variable |
| Gasoline inventories | Low stocks → price pressure |
| Crude oil inventories | Supply buffer indicator |
| Refinery utilization | Low utilization → supply drops |
| Gasoline production | Direct supply signal |
| Gasoline demand (product supplied) | Demand side of equation |
| Crude oil imports | Supply disruption warning |

### Yahoo Finance (no key needed)
| Ticker | Data | Why It Matters |
|--------|------|---------------|
| CL=F | WTI Crude Oil | #1 driver of gas prices |
| RB=F | RBOB Gasoline Futures | Leads retail by 1-2 weeks |
| BZ=F | Brent Crude | International oil benchmark |
| HO=F | Heating Oil | Correlated petroleum product |
| NG=F | Natural Gas | Refinery economics |
| DX-Y.NYB | US Dollar Index | Strong dollar → cheaper oil |
| ^GSPC | S&P 500 | Economic health proxy |

### FRED API (optional)
| Series | Data | Why It Matters |
|--------|------|---------------|
| DTWEXBGS | Trade-Weighted Dollar | Broad dollar strength |
| VIXCLS | VIX Volatility Index | Market fear/uncertainty |
| T10Y2Y | Yield Curve Spread | Recession signal |
| DGS10 | 10-Year Treasury | Economic outlook |
| CPIAUCSL | CPI | Inflation context |
| DHHNGSP | Henry Hub Natural Gas | Energy input cost |
| DEXUSEU | EUR/USD Exchange Rate | Currency effects |
| DCOILWTICO | WTI Spot (FRED) | Cross-validation |

---

## 🔧 106 Engineered Features

| Category | Count | Examples |
|----------|-------|---------|
| Gas price signals | 32 | Lags (1-4w), changes, momentum, rolling means/std, acceleration |
| Crude oil signals | 21 | Price lags, volatility, rockets & feathers asymmetry, per-gallon equiv |
| RBOB wholesale | 8 | Price lags, retail spread, spread changes |
| Dollar/currency | 6 | Dollar index change, EUR/USD, broad dollar |
| Inventory levels | 10 | Gasoline/crude stocks vs 26w/52w averages |
| Refinery/supply | 5 | Utilization rate, production changes, low-utilization flags |
| Demand signals | 4 | Product supplied, supply-demand balance/ratio |
| Market indicators | 7 | VIX levels/thresholds, S&P 500, treasury yields, yield curve |
| Seasonal/calendar | 10 | Summer blend, driving season, hurricane season/peak, holidays |
| Interactions | 5 | Crude × season, stocks × driving, dollar × crude, refinery × hurricane |

---

## 🎯 AAA Price Calibration

EIA and AAA report slightly different numbers (different survey methods, ~$0.10 gap). The hybrid approach:

1. Train on EIA historical data (years of consistent history)
2. Predict the weekly **change** in EIA price
3. Apply that change to the AAA price you enter in the sidebar

This gives AAA-accurate predictions without needing scraped AAA history.

---

## 📁 Project Structure

```
gas_price_predictor/
├── app.py              # Streamlit dashboard (5 tabs)
├── data_collector.py   # EIA + Yahoo Finance + FRED data fetching
├── feature_engine.py   # 106-feature engineering pipeline
├── model.py            # STL + GRU + XGBoost residual hybrid + stacking
├── config.py           # All settings, series IDs, and constants
├── requirements.txt    # Python dependencies
└── README.md
```

---

## 📈 Dashboard Tabs

1. **Price & Forecast** — Historical prices with prediction overlay, crude/RBOB comparison, weekly changes
2. **Model Accuracy** — Walk-forward validation results, error distribution, actual vs predicted
3. **Feature Analysis** — Feature importance ranking, correlations with target, feature descriptions
4. **Supply & Demand** — EIA inventory/refinery/demand charts, market indicators (Dollar, VIX, S&P)
5. **Data Explorer** — Raw data table, CSV download, full feature list with current values

---

## 📜 License

MIT License — free for personal and commercial use.
