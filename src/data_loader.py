import pandas as pd
import yfinance as yf


def load_data(path=None):
    """
    Load dataset from CSV OR fetch live Bitcoin data
    """

    # CASE 1: If file path is given → load CSV
    if path is not None:
        df = pd.read_csv("data/processed_btc_data.csv")
        return df

    # CASE 2: No path → fetch LIVE Bitcoin data
    df = yf.download("BTC-USD", period="5y", interval="1d")
    df.reset_index(inplace=True)

    return df
