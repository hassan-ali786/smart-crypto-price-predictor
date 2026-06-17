import pandas as pd


def load_data(path):
    """
    Loads dataset from CSV file.
    """
    df = pd.read_csv(path)
    return df


def get_data_info(df):
    """
    Optional helper: quick dataset inspection.
    """
    print("Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nMissing values:\n", df.isnull().sum())

    return df.describe()
