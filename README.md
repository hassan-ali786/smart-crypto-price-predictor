# ₿ Smart Crypto Price Predictor

An end-to-end Machine Learning project that predicts Bitcoin (BTC-USD) closing prices using historical market data, technical indicators, and an interactive Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat-square&logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-F7931E?style=flat-square&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## Overview

This project builds a complete ML pipeline for cryptocurrency price prediction — from raw data ingestion and feature engineering to model training, evaluation, and a live interactive UI. The goal is to predict the next Bitcoin closing price using historical OHLCV data combined with technical indicators commonly used in financial analysis.

---

## Demo

> Run locally with `streamlit run app/app.py`

The app has 3 tabs:

- **Price Chart** — historical BTC close price with MA7/MA30 overlay and trading volume
- **Predict** — enter today's market values and get a predicted closing price
- **Model Performance** — MAE, RMSE metrics + actual vs predicted chart + residuals

---

## Project Structure

```
smart-crypto-price-predictor/
│
├── data/
│   └── processed_btc_data.csv      # Preprocessed BTC-USD dataset
│
├── notebooks/
│   ├── eda.ipynb                   # Exploratory Data Analysis
│   ├── features.ipynb              # Feature engineering walkthrough
│   └── model.ipynb                 # Model training & evaluation
│
├── src/
│   ├── preprocess.py               # Data cleaning & feature generation
│   ├── train.py                    # Model training script
│   ├── predict.py                  # Prediction logic
│   ├── data_loader.py              # Data loading utilities
│   └── utils.py                    # Helper functions
│
├── models/
│   └── model.pkl                   # Saved trained model
│
├── app/
│   └── app.py                      # Streamlit frontend
│
├── requirements.txt
└── README.md
```

---

## Features Used

| Feature | Description |
|---|---|
| Open, High, Low | Daily OHLC price values |
| Volume | Daily trading volume |
| MA7 | 7-day moving average |
| MA30 | 30-day moving average |
| Lag1, Lag2, Lag3 | Previous 1, 2, 3 day closing prices |
| Returns | Daily percentage return |
| RSI | Relative Strength Index (14-day) |

---

## ML Pipeline

```
Raw CSV → Preprocessing → Feature Engineering → Train/Test Split (shuffle=False) → Model Training → Evaluation → Streamlit UI
```

**Models compared:**
- Linear Regression (baseline)
- Random Forest Regressor

**Evaluation metrics:** MAE, RMSE

---

## Tech Stack

- **Language:** Python 3.9+
- **Data:** Pandas, NumPy
- **Modeling:** Scikit-learn
- **Visualization:** Matplotlib
- **Frontend:** Streamlit
- **Serialization:** Joblib

---

## Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/hassan-ali786/smart-crypto-price-predictor.git
cd smart-crypto-price-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app/app.py
```

> If `models/model.pkl` is not present, the app will auto-train from the CSV on first launch.

---

## Results

| Model | MAE | RMSE |
|---|---|---|
| Linear Regression | ~$1,200 | ~$1,800 |
| Random Forest | ~$900 | ~$1,400 |

*Results on 20% holdout test set (time-series split)*

---

## Authors

Built as a collaborative ML project.

- **Hassan Ali** — Streamlit frontend (`app/app.py`), UI/UX, app integration
- **Dua Zahra** — Data preprocessing, feature engineering, model training, notebooks

---

## Connect

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Hassan_Ali-0077B5?style=flat-square&logo=linkedin)](https://linkedin.com/in/hassan-ali-data)
[![GitHub](https://img.shields.io/badge/GitHub-hassan--ali786-181717?style=flat-square&logo=github)](https://github.com/hassan-ali786)