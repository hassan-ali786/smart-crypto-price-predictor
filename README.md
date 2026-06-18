# ₿ Smart Crypto Price Predictor

An end-to-end Machine Learning project that predicts Bitcoin (BTC-USD) closing prices using historical market data, technical indicators, and an interactive Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.x-013243?style=flat-square&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-11557C?style=flat-square&logo=python&logoColor=white)
![yfinance](https://img.shields.io/badge/yfinance-0.2+-purple?style=flat-square&logo=yahoo&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)

---

## Overview

This project builds a complete ML pipeline for cryptocurrency price prediction — from raw data ingestion and feature engineering to model training, evaluation, and a live interactive UI. The goal is to predict the next Bitcoin closing price using historical OHLCV data combined with technical indicators commonly used in financial analysis. Two models are compared (Linear Regression and Random Forest), with full evaluation metrics and a 7-day iterative forecast.

---

##  Live Demo :

[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://smart-crypto-price-predictor-flkqnpurgjrv3eqvg2ww3v.streamlit.app/)

---
##  Live Demo :

<video src="https://github.com/user-attachments/assets/d2237d09-3caf-4480-b0e0-f54e577e41dc" width="100%" controls></video>

---

## Features

###  Price Chart Tab
- Interactive **candlestick chart** with MA7 and MA30 overlay
- Dedicated **RSI (14-day) panel** with overbought/oversold zones highlighted
- **Volume bars** color-coded green/red based on candle direction
- Adjustable slider to view last N days of history
- **Download CSV** button for filtered data

###  Predict Tab
- Auto-filled inputs from latest market data
- Predicts next Bitcoin closing price using trained model
- **Confidence range** shown as `Predicted Price ± RMSE`
- **Bullish / Bearish / Neutral signal** badge based on predicted movement

###  7-Day Forecast Tab
- Iterative 7-day forecast — each day's prediction feeds into the next
- Forecast cards with day labels and % change from current price
- Forecast chart with **confidence band** overlay

###  Model Performance Tab
- Evaluation metrics: **MAE, RMSE, MAPE**
- Actual vs Predicted line chart (test set)
- **Feature importance** horizontal bar chart
- Residuals bar chart (green/red)

###  Sidebar Controls
- Toggle between **CSV file** and **Live yfinance data**
- Switch between **Linear Regression** and **Random Forest** models
- Last updated timestamp displayed in header

---

## ML Pipeline

```
Raw CSV / yfinance → Preprocessing → Feature Engineering → Train/Test Split (shuffle=False) → Model Training → Evaluation → Streamlit UI
```

---

## Feature Engineering

| Feature | Description |
|---|---|
| Open, High, Low | Daily OHLC price values |
| Volume | Daily trading volume |
| MA7 | 7-day moving average |
| MA30 | 30-day moving average |
| Lag1, Lag2, Lag3 | Previous 1, 2, 3 day closing prices |
| Returns | Daily percentage return (pct_change) |
| RSI | Relative Strength Index (14-day) |

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.9+ |
| Data Handling | Pandas, NumPy |
| Live Data | yfinance |
| Modeling | Scikit-learn (LinearRegression, RandomForestRegressor) |
| Visualization | Matplotlib |
| Frontend | Streamlit |
| Serialization | Joblib |

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

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| Linear Regression | ~$1,200 | ~$1,800 | ~3.2% |
| Random Forest | ~$900 | ~$1,400 | ~2.4% |

*Results on 20% holdout test set (time-series split, shuffle=False)*

---

## Authors

Built as a collaborative ML project.

- **Hassan Ali** — Streamlit frontend (`app/app.py`), UI/UX, app integration
- **Dua Zahra** — Data preprocessing, feature engineering, model training, notebooks

---

## Connect

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Hassan_Ali-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/hassan-ali-datascientist)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-f0b429?style=flat-square&logo=vercel&logoColor=black)](https://hassanali-portfolio.vercel.app)
[![GitHub](https://img.shields.io/badge/GitHub-hassandatasci-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/hassan-ali786)
