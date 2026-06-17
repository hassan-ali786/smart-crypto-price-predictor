import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess_data(df):
    # Handle missing values
    df = df.ffill()
    df = df.dropna()

    # Moving averages
    df['MA7'] = df['Close'].rolling(window=7).mean()
    df['MA14'] = df['Close'].rolling(window=14).mean()

    # Lag features
    df['Lag1'] = df['Close'].shift(1)
    df['Lag2'] = df['Close'].shift(2)
    df['Lag3'] = df['Close'].shift(3)

    # Remove rows created by rolling/shift operations
    df = df.dropna()

    # Features
    features = [
        'Open',
        'High',
        'Low',
        'Volume',
        'MA7',
        'MA14',
        'Lag1',
        'Lag2',
        'Lag3'
    ]

    # Scaling
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    return df, features
