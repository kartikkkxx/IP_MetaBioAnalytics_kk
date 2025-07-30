import pandas as pd


def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        return df, "Data loaded successfully."
    except Exception as e:
        return None, f"Failed to load data: {str(e)}"
