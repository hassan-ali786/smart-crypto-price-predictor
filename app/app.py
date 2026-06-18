import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

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
    .metric-label { color: #7a8299; font-size: 13px; margin-bottom: 4px; }
    .metric-value { color: #f0b429; font-size: 26px; font-weight: 700; }
    .metric-sub   { color: #4caf7d; font-size: 12px; margin-top: 4px; }
    .section-header {
        color: #f0b429;
        font-size: 18px;
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
</style>
""", unsafe_allow_html=True)

# ─── Helpers ─────────────────────────────────────────────────────────────────

FEATURES = ["Open", "High", "Low", "Volume", "MA7", "MA30", "Lag1", "Lag2", "Lag3", "Returns", "RSI"]
MODEL_PATH = "models/model.pkl"
DATA_PATH  = "data/processed_btc_data.csv"


@st.cache_data
def load_data(path):
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


@st.cache_resource
def load_or_train_model(data_path, model_path):
    """Load saved model if exists, else train fresh."""
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path), None
        except Exception:
            pass

    df = load_data(data_path)
    X = df[FEATURES]
    y = df["Close"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=False)

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc  = scaler.transform(X_te)

    model = LinearRegression()
    model.fit(X_tr_sc, y_tr)

    preds  = model.predict(X_te_sc)
    mae    = mean_absolute_error(y_te, preds)
    rmse   = np.sqrt(mean_squared_error(y_te, preds))

    return model, {"mae": mae, "rmse": rmse, "scaler": scaler,
                   "y_test": y_te.values, "preds": preds,
                   "dates": df.iloc[-len(y_te):]["Date"].values}


def compute_rsi(series, window=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ₿ CryptoPredict")
    st.markdown("---")
    st.markdown("**Model:** Linear Regression")
    st.markdown("**Asset:** Bitcoin (BTC-USD)")
    st.markdown("**Features:** MA7, MA30, RSI, Lag")
    st.markdown("---")

    data_path  = st.text_input("Data path",  DATA_PATH)
    model_path = st.text_input("Model path", MODEL_PATH)

    st.markdown("---")
    st.markdown("<small style='color:#7a8299'>Hassan AI & Data Lab</small>", unsafe_allow_html=True)


# ─── Load data & model ───────────────────────────────────────────────────────
data_ok  = os.path.exists(data_path)
model_ok = os.path.exists(model_path)

if not data_ok:
    st.error(f"Data file not found: `{data_path}`  \nPlace `processed_btc_data.csv` inside `data/` folder.")
    st.stop()

df = load_data(data_path)
model, train_info = load_or_train_model(data_path, model_path)

# ─── Header ──────────────────────────────────────────────────────────────────
st.title("₿ Smart Crypto Price Predictor")
st.caption("Bitcoin closing-price prediction using historical data & technical indicators")

# ─── KPI Cards ───────────────────────────────────────────────────────────────
latest     = df.iloc[-1]
prev        = df.iloc[-2]
price_chg   = ((latest["Close"] - prev["Close"]) / prev["Close"]) * 100

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>Latest Close</div>
        <div class='metric-value'>${latest['Close']:,.0f}</div>
        <div class='metric-sub'>{'▲' if price_chg>=0 else '▼'} {price_chg:.2f}% vs prev day</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>RSI (14)</div>
        <div class='metric-value'>{latest['RSI']:.1f}</div>
        <div class='metric-sub'>{'Overbought' if latest['RSI']>70 else 'Oversold' if latest['RSI']<30 else 'Neutral'}</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>MA 7-day</div>
        <div class='metric-value'>${latest['MA7']:,.0f}</div>
        <div class='metric-sub'>Short-term trend</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>MA 30-day</div>
        <div class='metric-value'>${latest['MA30']:,.0f}</div>
        <div class='metric-sub'>Medium-term trend</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Tabs ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📈 Price Chart", "🔮 Predict", "📊 Model Performance"])

# ── Tab 1: Price Chart ───────────────────────────────────────────────────────
with tab1:
    st.markdown("<div class='section-header'>BTC-USD Historical Price</div>", unsafe_allow_html=True)

    n_days = st.slider("Show last N days", 30, len(df), 365, step=30)
    df_chart = df.tail(n_days)

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#161b27")

    ax.plot(df_chart["Date"], df_chart["Close"],  color="#f0b429", linewidth=1.6, label="Close")
    ax.plot(df_chart["Date"], df_chart["MA7"],    color="#4caf7d", linewidth=1.0, linestyle="--", label="MA7")
    ax.plot(df_chart["Date"], df_chart["MA30"],   color="#5b8df8", linewidth=1.0, linestyle="--", label="MA30")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(colors="#7a8299")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2e3450")
    ax.legend(facecolor="#161b27", labelcolor="#ccc", fontsize=9)
    ax.set_ylabel("Price (USD)", color="#7a8299")
    ax.yaxis.label.set_color("#7a8299")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Volume bar chart
    st.markdown("<div class='section-header'>Trading Volume</div>", unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(figsize=(12, 2))
    fig2.patch.set_facecolor("#0e1117")
    ax2.set_facecolor("#161b27")
    ax2.bar(df_chart["Date"], df_chart["Volume"], color="#5b8df8", alpha=0.7, width=1)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.tick_params(colors="#7a8299")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#2e3450")
    ax2.set_ylabel("Volume", color="#7a8299")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

# ── Tab 2: Predict ───────────────────────────────────────────────────────────
with tab2:
    st.markdown("<div class='section-header'>Enter Market Data to Predict Close Price</div>", unsafe_allow_html=True)

    st.info("Fill in today's market values (or use the auto-filled defaults from latest data).")

    col_a, col_b = st.columns(2)

    with col_a:
        open_val  = st.number_input("Open ($)",   value=float(round(latest["Open"], 2)),   step=100.0)
        high_val  = st.number_input("High ($)",   value=float(round(latest["High"], 2)),   step=100.0)
        low_val   = st.number_input("Low ($)",    value=float(round(latest["Low"],  2)),   step=100.0)
        vol_val   = st.number_input("Volume",     value=float(round(latest["Volume"])),    step=1e6, format="%.0f")
        rsi_val   = st.number_input("RSI (0–100)", value=float(round(latest["RSI"], 2)),  min_value=0.0, max_value=100.0)

    with col_b:
        ma7_val   = st.number_input("MA 7-day ($)",  value=float(round(latest["MA7"],  2)),  step=100.0)
        ma30_val  = st.number_input("MA 30-day ($)", value=float(round(latest["MA30"], 2)),  step=100.0)
        lag1_val  = st.number_input("Lag 1 (yesterday Close $)", value=float(round(latest["Lag1"], 2)), step=100.0)
        lag2_val  = st.number_input("Lag 2 (2 days ago Close $)", value=float(round(latest["Lag2"], 2)), step=100.0)
        lag3_val  = st.number_input("Lag 3 (3 days ago Close $)", value=float(round(latest["Lag3"], 2)), step=100.0)
        ret_val   = st.number_input("Daily Returns (e.g. 0.012)", value=float(round(latest["Returns"], 5)), format="%.5f")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔮 Predict Close Price"):
        input_arr = np.array([[open_val, high_val, low_val, vol_val,
                                ma7_val, ma30_val, lag1_val, lag2_val,
                                lag3_val, ret_val, rsi_val]])
        input_df = pd.DataFrame(input_arr, columns=FEATURES)

        # Scale if we have scaler from training, else predict raw
        if train_info and "scaler" in train_info:
            input_scaled = train_info["scaler"].transform(input_df)
            prediction   = model.predict(input_scaled)[0]
        else:
            prediction = model.predict(input_df)[0]

        diff   = prediction - latest["Close"]
        pct    = (diff / latest["Close"]) * 100
        arrow  = "▲" if diff >= 0 else "▼"
        color  = "#4caf7d" if diff >= 0 else "#f25a5a"

        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#1e2130,#252a3a);
                    border:1px solid #f0b429;border-radius:14px;padding:28px;text-align:center;margin-top:16px;'>
            <div style='color:#7a8299;font-size:14px;margin-bottom:8px;'>Predicted Next Close Price</div>
            <div style='color:#f0b429;font-size:48px;font-weight:800;'>${prediction:,.2f}</div>
            <div style='color:{color};font-size:16px;margin-top:8px;'>{arrow} ${abs(diff):,.2f} ({abs(pct):.2f}%) vs latest close</div>
        </div>
        """, unsafe_allow_html=True)

        st.caption("⚠️ This is an ML model prediction for educational purposes only. Not financial advice.")

# ── Tab 3: Model Performance ─────────────────────────────────────────────────
with tab3:
    st.markdown("<div class='section-header'>Model Evaluation Metrics</div>", unsafe_allow_html=True)

    if train_info:
        mae  = train_info["mae"]
        rmse = train_info["rmse"]
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-label'>MAE</div>
                <div class='metric-value'>${mae:,.0f}</div>
                <div class='metric-sub'>Mean Absolute Error</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-label'>RMSE</div>
                <div class='metric-value'>${rmse:,.0f}</div>
                <div class='metric-sub'>Root Mean Squared Error</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-label'>Model</div>
                <div class='metric-value' style='font-size:18px;'>Linear Regression</div>
                <div class='metric-sub'>Baseline model</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Actual vs Predicted (Test Set)</div>", unsafe_allow_html=True)

        fig3, ax3 = plt.subplots(figsize=(12, 4))
        fig3.patch.set_facecolor("#0e1117")
        ax3.set_facecolor("#161b27")

        dates_arr = train_info["dates"]
        ax3.plot(dates_arr, train_info["y_test"], color="#f0b429", linewidth=1.5, label="Actual")
        ax3.plot(dates_arr, train_info["preds"],  color="#5b8df8", linewidth=1.5, linestyle="--", label="Predicted")

        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax3.tick_params(colors="#7a8299")
        for spine in ax3.spines.values():
            spine.set_edgecolor("#2e3450")
        ax3.legend(facecolor="#161b27", labelcolor="#ccc", fontsize=9)
        ax3.set_ylabel("Price (USD)", color="#7a8299")
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

        # Residuals
        st.markdown("<div class='section-header'>Residuals</div>", unsafe_allow_html=True)
        residuals = train_info["y_test"] - train_info["preds"]
        fig4, ax4 = plt.subplots(figsize=(12, 3))
        fig4.patch.set_facecolor("#0e1117")
        ax4.set_facecolor("#161b27")
        ax4.bar(range(len(residuals)), residuals, color=["#4caf7d" if r >= 0 else "#f25a5a" for r in residuals], alpha=0.7)
        ax4.axhline(0, color="#f0b429", linewidth=0.8)
        ax4.tick_params(colors="#7a8299")
        for spine in ax4.spines.values():
            spine.set_edgecolor("#2e3450")
        ax4.set_ylabel("Residual (USD)", color="#7a8299")
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

    else:
        st.info("Model was loaded from file. Retrain from scratch to see evaluation metrics.")
        st.markdown("Delete `models/model.pkl` and restart to trigger auto-training with metrics.")