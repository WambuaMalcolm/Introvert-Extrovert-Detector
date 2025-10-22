from pathlib import Path

import pandas as pd

# Define these at module level (outside if __name__)
project_root = Path(__file__).resolve().parents[2]
csv_path = project_root / "data" / "raw" / "personality_dataset.csv"


def load_data():
    """Load the raw personality dataset"""
    return pd.read_csv(csv_path)


# This block is only for testing the module directly
if __name__ == "__main__":
    df = load_data()
    print(f"Loaded dataset with shape: {df.shape}")
    print(df.head())
