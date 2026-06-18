import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess_data(df):
    # Handle missing values
    df = df.ffill()
    df = df.dropna()

    # Moving averages
    df['MA7'] = df['Close'].rolling(window=7).mean()
    df['MA30'] = df['Close'].rolling(window=30).mean()

    # Lag features
    df['Lag1'] = df['Close'].shift(1)
    df['Lag2'] = df['Close'].shift(2)
    df['Lag3'] = df['Close'].shift(3)

    # Daily returns
    df['Returns'] = df['Close'].pct_change()

    # RSI (14-day)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = (-delta.clip(upper=0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))

    # Remove rows created by rolling/shift operations
    df = df.dropna()

    # Features
    features = [
        'Open',
        'High',
        'Low',
        'Volume',
        'MA7',
        'MA30',
        'Lag1',
        'Lag2',
        'Lag3',
        'Returns',
        'RSI'
    ]

    # Scaling
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    return df, features