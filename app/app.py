import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Crypto Price Predictor",
    page_icon="₿",
    layout="wide",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252a3a);
        border: 1px solid #2e3450;
        border-radius: 12px;
        padding: 18px 22px;
        text-align: center;
    }
    .metric-label { color: #7a8299; font-size: 12px; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-value { color: #f0b429; font-size: 26px; font-weight: 700; }
    .metric-sub   { color: #4caf7d; font-size: 12px; margin-top: 4px; }
    .metric-sub-red { color: #f25a5a; font-size: 12px; margin-top: 4px; }
    .section-header {
        color: #f0b429;
        font-size: 16px;
        font-weight: 600;
        border-left: 4px solid #f0b429;
        padding-left: 10px;
        margin: 20px 0 12px;
    }
    .stButton>button {
        background: linear-gradient(135deg, #f0b429, #e09010);
        color: #0e1117;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 10px 28px;
        width: 100%;
    }
    .stButton>button:hover { opacity: 0.9; }
    div[data-testid="stSidebar"] { background-color: #161b27; }
    .forecast-card {
        background: linear-gradient(135deg, #1a2035, #1e2540);
        border: 1px solid #2e3450;
        border-radius: 10px;
        padding: 12px 16px;
        text-align: center;
        margin: 4px;
    }
    .forecast-day { color: #7a8299; font-size: 11px; margin-bottom: 4px; }
    .forecast-price { color: #f0b429; font-size: 17px; font-weight: 600; }
    .forecast-change-up { color: #4caf7d; font-size: 11px; }
    .forecast-change-down { color: #f25a5a; font-size: 11px; }
    .pred-result {
        background: linear-gradient(135deg, #1e2130, #252a3a);
        border: 1px solid #f0b429;
        border-radius: 14px;
        padding: 28px;
        text-align: center;
        margin-top: 16px;
    }
    .signal-bull { background: #1a3a2a; border: 1px solid #4caf7d; border-radius: 8px; padding: 10px 16px; color: #4caf7d; font-weight: 600; }
    .signal-bear { background: #3a1a1a; border: 1px solid #f25a5a; border-radius: 8px; padding: 10px 16px; color: #f25a5a; font-weight: 600; }
    .signal-neu  { background: #2a2a1a; border: 1px solid #f0b429; border-radius: 8px; padding: 10px 16px; color: #f0b429; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ───────────────────────────────────────────────────────────────
FEATURES = ["Open", "High", "Low", "Volume", "MA7", "MA30", "Lag1", "Lag2", "Lag3", "Returns", "RSI"]
MODEL_PATH = "models/model.pkl"
DATA_PATH  = "data/processed_btc_data.csv"

# ─── Helpers ─────────────────────────────────────────────────────────────────
def compute_rsi(series, window=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

@st.cache_data(ttl=300)
def fetch_live_data():
    try:
        import yfinance as yf
        df = yf.download("BTC-USD", period="2y", interval="1d", progress=False)
        df = df.reset_index()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df.rename(columns={"Adj Close": "AdjClose"})
        df["MA7"]     = df["Close"].rolling(7).mean()
        df["MA30"]    = df["Close"].rolling(30).mean()
        df["Lag1"]    = df["Close"].shift(1)
        df["Lag2"]    = df["Close"].shift(2)
        df["Lag3"]    = df["Close"].shift(3)
        df["Returns"] = df["Close"].pct_change()
        df["RSI"]     = compute_rsi(df["Close"])
        df = df.dropna().reset_index(drop=True)
        df["Date"] = pd.to_datetime(df["Date"])
        return df, None
    except Exception as e:
        return None, str(e)

@st.cache_data
def load_csv(path):
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.sort_values("Date").reset_index(drop=True)

@st.cache_resource
def train_model_cached(data_key, model_type, model_path):
    # data_key is just a hashable identifier
    if model_type == "Linear Regression" and os.path.exists(model_path):
        try:
            return joblib.load(model_path), None
        except:
            pass
    return None, None

def train_fresh(df, model_type):
    X = df[FEATURES]
    y = df["Close"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=False)

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc  = scaler.transform(X_te)

    if model_type == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        model = LinearRegression()

    model.fit(X_tr_sc, y_tr)
    preds = model.predict(X_te_sc)

    mae  = mean_absolute_error(y_te, preds)
    rmse = np.sqrt(mean_squared_error(y_te, preds))
    mape = mean_absolute_percentage_error(y_te, preds) * 100

    return model, scaler, {
        "mae": mae, "rmse": rmse, "mape": mape,
        "y_test": y_te.values, "preds": preds,
        "dates": df.iloc[-len(y_te):]["Date"].values,
        "X_te": X_te, "scaler": scaler
    }

def generate_forecast(model, scaler, df, days=7):
    forecast = []
    temp_df = df.copy()
    for i in range(days):
        latest = temp_df.iloc[-1]
        row = {
            "Open":    latest["Close"],
            "High":    latest["Close"] * 1.005,
            "Low":     latest["Close"] * 0.995,
            "Volume":  latest["Volume"],
            "MA7":     temp_df["Close"].tail(7).mean(),
            "MA30":    temp_df["Close"].tail(30).mean(),
            "Lag1":    temp_df["Close"].iloc[-1],
            "Lag2":    temp_df["Close"].iloc[-2],
            "Lag3":    temp_df["Close"].iloc[-3],
            "Returns": (temp_df["Close"].iloc[-1] - temp_df["Close"].iloc[-2]) / temp_df["Close"].iloc[-2],
            "RSI":     latest["RSI"],
        }
        input_df  = pd.DataFrame([[row[f] for f in FEATURES]], columns=FEATURES)
        input_sc  = scaler.transform(input_df)
        pred_price = model.predict(input_sc)[0]

        next_date = temp_df["Date"].iloc[-1] + timedelta(days=1)
        forecast.append({"date": next_date, "price": pred_price})

        new_row = pd.DataFrame([{**row, "Close": pred_price, "Date": next_date, "High": pred_price * 1.005, "Low": pred_price * 0.995}])
        temp_df = pd.concat([temp_df, new_row], ignore_index=True)

    return forecast

def plot_dark(figsize=(12, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#161b27")
    ax.tick_params(colors="#7a8299")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2e3450")
    return fig, ax

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ₿ CryptoPredict")
    st.markdown("---")

    data_source = st.radio("Data Source", ["CSV File", "Live (yfinance)"], index=0)
    model_type  = st.selectbox("Model", ["Linear Regression", "Random Forest"])

    st.markdown("---")
    data_path  = st.text_input("CSV path",   DATA_PATH)
    model_path = st.text_input("Model path", MODEL_PATH)

    st.markdown("---")
    st.markdown("<small style='color:#7a8299'>Hassan AI & Data Lab</small>", unsafe_allow_html=True)

# ─── Load Data ────────────────────────────────────────────────────────────────
if data_source == "Live (yfinance)":
    with st.spinner("Fetching live BTC-USD data..."):
        df, err = fetch_live_data()
    if err or df is None:
        st.error(f"Live fetch failed: {err}. Falling back to CSV.")
        data_source = "CSV File"

if data_source == "CSV File":
    if not os.path.exists(data_path):
        st.error(f"Data file not found: `{data_path}`")
        st.stop()
    df = load_csv(data_path)

last_updated = df["Date"].max().strftime("%d %b %Y") if data_source == "CSV File" else datetime.now().strftime("%d %b %Y %H:%M")

# ─── Train Model ─────────────────────────────────────────────────────────────
with st.spinner(f"Training {model_type}..."):
    model, scaler, metrics = train_fresh(df, model_type)

# ─── Header ──────────────────────────────────────────────────────────────────
col_t, col_u = st.columns([4, 1])
with col_t:
    st.title("₿ Smart Crypto Price Predictor")
    st.caption(f"Bitcoin (BTC-USD) · {model_type} · Data source: {data_source}")
with col_u:
    st.markdown(f"<div style='text-align:right;color:#7a8299;font-size:12px;margin-top:20px;'>Last updated<br><b style='color:#f0b429'>{last_updated}</b></div>", unsafe_allow_html=True)

# ─── KPI Cards ───────────────────────────────────────────────────────────────
latest = df.iloc[-1]
prev   = df.iloc[-2]
chg    = ((latest["Close"] - prev["Close"]) / prev["Close"]) * 100
chg_color = "#4caf7d" if chg >= 0 else "#f25a5a"
arrow  = "▲" if chg >= 0 else "▼"

c1, c2, c3, c4, c5 = st.columns(5)
cards = [
    ("Latest Close",  f"${latest['Close']:,.0f}",  f"{arrow} {abs(chg):.2f}% vs prev", chg >= 0),
    ("RSI (14)",      f"{latest['RSI']:.1f}",       "Overbought" if latest['RSI']>70 else ("Oversold" if latest['RSI']<30 else "Neutral"), None),
    ("MA 7-day",      f"${latest['MA7']:,.0f}",     "Short-term trend", None),
    ("MA 30-day",     f"${latest['MA30']:,.0f}",    "Medium-term trend", None),
    ("Model MAPE",    f"{metrics['mape']:.2f}%",    "Prediction error", None),
]
for col, (label, val, sub, up) in zip([c1,c2,c3,c4,c5], cards):
    sub_class = "metric-sub" if (up is None or up) else "metric-sub-red"
    with col:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>{label}</div>
            <div class='metric-value'>{val}</div>
            <div class='{sub_class}'>{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Tabs ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📈 Price Chart", "🔮 Predict", "📅 7-Day Forecast", "📊 Model Performance"])

# ══ Tab 1: Price Chart ═══════════════════════════════════════════════════════
with tab1:
    st.markdown("<div class='section-header'>BTC-USD Historical Price</div>", unsafe_allow_html=True)

    n_days = st.slider("Show last N days", 30, len(df), 365, step=30)
    df_c = df.tail(n_days)

    # Candlestick
    fig, ax = plot_dark((12, 5))
    for _, row in df_c.iterrows():
        color = "#4caf7d" if row["Close"] >= row["Open"] else "#f25a5a"
        ax.plot([row["Date"], row["Date"]], [row["Low"], row["High"]], color=color, linewidth=0.7)
        ax.bar(row["Date"], abs(row["Close"] - row["Open"]),
               bottom=min(row["Open"], row["Close"]),
               color=color, width=0.7, alpha=0.85)

    ax.plot(df_c["Date"], df_c["MA7"],  color="#5b8df8", linewidth=1.2, linestyle="--", label="MA7",  alpha=0.9)
    ax.plot(df_c["Date"], df_c["MA30"], color="#f0b429", linewidth=1.2, linestyle="--", label="MA30", alpha=0.9)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend(facecolor="#161b27", labelcolor="#ccc", fontsize=9)
    ax.set_ylabel("Price (USD)", color="#7a8299")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # RSI Panel
    st.markdown("<div class='section-header'>RSI (14)</div>", unsafe_allow_html=True)
    fig2, ax2 = plot_dark((12, 2.5))
    ax2.plot(df_c["Date"], df_c["RSI"], color="#a78bfa", linewidth=1.3)
    ax2.axhline(70, color="#f25a5a", linewidth=0.8, linestyle="--", alpha=0.7)
    ax2.axhline(30, color="#4caf7d", linewidth=0.8, linestyle="--", alpha=0.7)
    ax2.fill_between(df_c["Date"], df_c["RSI"], 70, where=(df_c["RSI"] >= 70), alpha=0.15, color="#f25a5a")
    ax2.fill_between(df_c["Date"], df_c["RSI"], 30, where=(df_c["RSI"] <= 30), alpha=0.15, color="#4caf7d")
    ax2.set_ylim(0, 100)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.set_ylabel("RSI", color="#7a8299")
    ax2.text(df_c["Date"].iloc[-1], 72, "Overbought", color="#f25a5a", fontsize=8, ha="right")
    ax2.text(df_c["Date"].iloc[-1], 25, "Oversold",   color="#4caf7d", fontsize=8, ha="right")
    plt.tight_layout()
    st.pyplot(fig2); plt.close()

    # Volume
    st.markdown("<div class='section-header'>Trading Volume</div>", unsafe_allow_html=True)
    fig3, ax3 = plot_dark((12, 2))
    colors_v = ["#4caf7d" if df_c["Close"].iloc[i] >= df_c["Open"].iloc[i] else "#f25a5a" for i in range(len(df_c))]
    ax3.bar(df_c["Date"], df_c["Volume"], color=colors_v, alpha=0.7, width=0.8)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax3.set_ylabel("Volume", color="#7a8299")
    plt.tight_layout()
    st.pyplot(fig3); plt.close()

    # Download button
    csv_data = df_c.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download CSV", csv_data, "btc_data.csv", "text/csv")

# ══ Tab 2: Predict ════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'>Enter Market Data to Predict Close Price</div>", unsafe_allow_html=True)
    st.info("Values auto-filled from latest data. Adjust manually if needed.")

    col_a, col_b = st.columns(2)
    with col_a:
        open_val = st.number_input("Open ($)",        value=float(round(latest["Open"],    2)), step=100.0)
        high_val = st.number_input("High ($)",        value=float(round(latest["High"],    2)), step=100.0)
        low_val  = st.number_input("Low ($)",         value=float(round(latest["Low"],     2)), step=100.0)
        vol_val  = st.number_input("Volume",          value=float(round(latest["Volume"])),     step=1e6, format="%.0f")
        rsi_val  = st.number_input("RSI (0–100)",     value=float(round(latest["RSI"],     2)), min_value=0.0, max_value=100.0)
    with col_b:
        ma7_val  = st.number_input("MA 7-day ($)",    value=float(round(latest["MA7"],     2)), step=100.0)
        ma30_val = st.number_input("MA 30-day ($)",   value=float(round(latest["MA30"],    2)), step=100.0)
        lag1_val = st.number_input("Lag 1 ($)",       value=float(round(latest["Lag1"],    2)), step=100.0)
        lag2_val = st.number_input("Lag 2 ($)",       value=float(round(latest["Lag2"],    2)), step=100.0)
        lag3_val = st.number_input("Lag 3 ($)",       value=float(round(latest["Lag3"],    2)), step=100.0)
        ret_val  = st.number_input("Daily Returns",   value=float(round(latest["Returns"], 5)), format="%.5f")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔮 Predict Close Price"):
        with st.spinner("Calculating prediction..."):
            input_df = pd.DataFrame([[open_val, high_val, low_val, vol_val,
                                  ma7_val, ma30_val, lag1_val, lag2_val,
                                  lag3_val, ret_val, rsi_val]], columns=FEATURES)
            input_sc  = scaler.transform(input_df)
            prediction = model.predict(input_sc)[0]

        diff = prediction - latest["Close"]
        pct  = (diff / latest["Close"]) * 100
        rmse = metrics["rmse"]
        lower = prediction - rmse
        upper = prediction + rmse

        color = "#4caf7d" if diff >= 0 else "#f25a5a"
        arrow_p = "▲" if diff >= 0 else "▼"

        # Signal
        if pct > 1:
            signal_class, signal_text = "signal-bull", "🟢 Bullish Signal — Predicted price above current close"
        elif pct < -1:
            signal_class, signal_text = "signal-bear", "🔴 Bearish Signal — Predicted price below current close"
        else:
            signal_class, signal_text = "signal-neu",  "🟡 Neutral — Predicted price near current close"

        st.markdown(f"""
        <div class='pred-result'>
            <div style='color:#7a8299;font-size:13px;margin-bottom:8px;'>Predicted Next Close Price</div>
            <div style='color:#f0b429;font-size:52px;font-weight:800;'>${prediction:,.2f}</div>
            <div style='color:{color};font-size:16px;margin-top:6px;'>{arrow_p} ${abs(diff):,.2f} ({abs(pct):.2f}%) vs latest close</div>
            <div style='color:#7a8299;font-size:13px;margin-top:12px;'>Confidence Range (±1 RMSE): <b style='color:#ccc'>${lower:,.0f} — ${upper:,.0f}</b></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<div class='{signal_class}'>{signal_text}</div>", unsafe_allow_html=True)
        st.caption("⚠️ Educational purposes only. Not financial advice.")

# ══ Tab 3: 7-Day Forecast ═════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-header'>7-Day Price Forecast</div>", unsafe_allow_html=True)
    st.info("Iterative forecast — each day's prediction feeds into the next.")

    with st.spinner("Generating 7-day forecast..."):
        forecast = generate_forecast(model, scaler, df, days=7)

    # Cards
    cols = st.columns(7)
    base_price = df["Close"].iloc[-1]
    for i, (col, fc) in enumerate(zip(cols, forecast)):
        chg_f = ((fc["price"] - base_price) / base_price) * 100
        chg_class = "forecast-change-up" if chg_f >= 0 else "forecast-change-down"
        arrow_f = "▲" if chg_f >= 0 else "▼"
        day_label = fc["date"].strftime("%a %d")
        with col:
            st.markdown(f"""<div class='forecast-card'>
                <div class='forecast-day'>{day_label}</div>
                <div class='forecast-price'>${fc['price']:,.0f}</div>
                <div class='{chg_class}'>{arrow_f} {abs(chg_f):.1f}%</div>
            </div>""", unsafe_allow_html=True)
        base_price = fc["price"]

    st.markdown("<br>", unsafe_allow_html=True)

    # Forecast chart
    hist_tail = df.tail(30)
    fig4, ax4 = plot_dark((12, 4))
    ax4.plot(hist_tail["Date"], hist_tail["Close"], color="#f0b429", linewidth=1.8, label="Historical")

    f_dates  = [df["Date"].iloc[-1]] + [fc["date"] for fc in forecast]
    f_prices = [df["Close"].iloc[-1]] + [fc["price"] for fc in forecast]
    ax4.plot(f_dates, f_prices, color="#5b8df8", linewidth=1.8, linestyle="--", marker="o", markersize=5, label="Forecast")

    rmse_val = metrics["rmse"]
    ax4.fill_between(f_dates, [p - rmse_val for p in f_prices], [p + rmse_val for p in f_prices],
                     alpha=0.15, color="#5b8df8", label="Confidence band")

    ax4.axvline(df["Date"].iloc[-1], color="#7a8299", linewidth=0.8, linestyle=":")
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax4.legend(facecolor="#161b27", labelcolor="#ccc", fontsize=9)
    ax4.set_ylabel("Price (USD)", color="#7a8299")
    plt.tight_layout()
    st.pyplot(fig4); plt.close()

    st.caption("⚠️ Multi-step forecast uncertainty increases with each day. Educational purposes only.")

# ══ Tab 4: Model Performance ══════════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-header'>Evaluation Metrics</div>", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    for col, (label, val, sub) in zip([m1,m2,m3,m4], [
        ("MAE",   f"${metrics['mae']:,.0f}",       "Mean Absolute Error"),
        ("RMSE",  f"${metrics['rmse']:,.0f}",      "Root Mean Squared Error"),
        ("MAPE",  f"{metrics['mape']:.2f}%",       "Mean Abs % Error"),
        ("Model", model_type,                       "Active model"),
    ]):
        with col:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-label'>{label}</div>
                <div class='metric-value' style='font-size:20px'>{val}</div>
                <div class='metric-sub'>{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Actual vs Predicted
    st.markdown("<div class='section-header'>Actual vs Predicted (Test Set)</div>", unsafe_allow_html=True)
    fig5, ax5 = plot_dark((12, 4))
    ax5.plot(metrics["dates"], metrics["y_test"], color="#f0b429", linewidth=1.5, label="Actual")
    ax5.plot(metrics["dates"], metrics["preds"],  color="#5b8df8", linewidth=1.5, linestyle="--", label="Predicted")
    ax5.fill_between(metrics["dates"], metrics["y_test"], metrics["preds"], alpha=0.1, color="#a78bfa")
    ax5.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax5.legend(facecolor="#161b27", labelcolor="#ccc", fontsize=9)
    ax5.set_ylabel("Price (USD)", color="#7a8299")
    plt.tight_layout()
    st.pyplot(fig5); plt.close()

    # Feature Importance
    st.markdown("<div class='section-header'>Feature Importance</div>", unsafe_allow_html=True)
    fig6, ax6 = plot_dark((10, 3.5))
    if model_type == "Random Forest":
        importance = model.feature_importances_
        colors_fi  = ["#f0b429" if v == max(importance) else "#5b8df8" for v in importance]
    else:
        importance = np.abs(model.coef_)
        colors_fi  = ["#f0b429" if v == max(importance) else "#5b8df8" for v in importance]

    sorted_idx = np.argsort(importance)
    ax6.barh([FEATURES[i] for i in sorted_idx], [importance[i] for i in sorted_idx], color=[colors_fi[i] for i in sorted_idx], alpha=0.85)
    ax6.tick_params(colors="#ccc")
    ax6.set_xlabel("Importance", color="#7a8299")
    plt.tight_layout()
    st.pyplot(fig6); plt.close()

    # Residuals
    st.markdown("<div class='section-header'>Residuals</div>", unsafe_allow_html=True)
    residuals = metrics["y_test"] - metrics["preds"]
    fig7, ax7 = plot_dark((12, 3))
    colors_r = ["#4caf7d" if r >= 0 else "#f25a5a" for r in residuals]
    ax7.bar(range(len(residuals)), residuals, color=colors_r, alpha=0.7)
    ax7.axhline(0, color="#f0b429", linewidth=0.8)
    ax7.set_ylabel("Residual (USD)", color="#7a8299")
    plt.tight_layout()
    st.pyplot(fig7); plt.close()