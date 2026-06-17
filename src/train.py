import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)

# =========================
# Metrics
# =========================

def calculate_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred) * 100


# =========================
# Load Data
# =========================

def load_data(path):
    return pd.read_csv("data/processed_btc_data.csv")


# =========================
# Feature Engineering Input
# =========================

def prepare_data(df):

    features = [
        "Open",
        "High",
        "Low",
        "Volume",
        "MA7",
        "MA30",
        "Lag1",
        "Lag2",
        "Lag3",
        "Returns",
        "RSI"
    ]

    X = df[features]
    y = df["Close"]

    return X, y


# =========================
# Train/Test Split
# =========================

def split_data(X, y):

    return train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False
    )


# =========================
# Train Model
# =========================

def train_model(X_train, y_train):

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model


# =========================
# Evaluate Model
# =========================

def evaluate(model, X_test, y_test):

    preds = model.predict(X_test)

    mae = calculate_mae(y_test, preds)
    rmse = calculate_rmse(y_test, preds)
    mape = calculate_mape(y_test, preds)

    return mae, rmse, mape, preds


# =========================
# FEATURE IMPORTANCE
# =========================

def plot_feature_importance(model):

    importance = model.coef_

    features = [
        "Open",
        "High",
        "Low",
        "Volume",
        "MA7",
        "MA14",
        "Lag1",
        "Lag2",
        "Lag3",
        "Returns",
        "RSI"
    ]

    plt.figure(figsize=(10, 5))
    plt.bar(features, importance)
    plt.title("Feature Importance (Linear Regression)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# =========================
# Save Model
# =========================

def save_model(model, path="models/model.pkl"):
    joblib.dump(model, path)


# =========================
# MAIN PIPELINE
# =========================

def main():

    df = load_data("data/processed_btc_data.csv")

    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train)

    mae, rmse, mape, preds = evaluate(model, X_test, y_test)

    # ===== PRINT REPORT INSIDE MAIN =====
    print("\n==============================")
    print("📊 MODEL PERFORMANCE REPORT")
    print("==============================")

    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"MAPE : {mape:.2f}%")

    print("\n==============================")
    print("📌 MODEL SUMMARY")
    print("==============================")
    print("Model Type : Linear Regression")
    print("Data Split : 80% Train / 20% Test (Time Series)")
    print("Target     : Bitcoin Closing Price")
    print("Features   : OHLC + Technical Indicators")

    print("\n==============================\n")

    # FEATURE IMPORTANCE
    plot_feature_importance(model)

    save_model(model)

    print("Model saved successfully → models/model.pkl")


if __name__ == "__main__":
    main()
