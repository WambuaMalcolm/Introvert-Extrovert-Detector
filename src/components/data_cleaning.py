import pandas as pd
from pathlib import Path

from src.components.data_ingestion import load_data


def transform_binary(df: pd.DataFrame):
    """Transform Personality column to binary values."""
    df["Personality"] = (
        df["Personality"].str.lower().map({"extrovert": 0, "introvert": 1})
    )
    return df


if __name__ == "__main__":
    # Load raw data
    df = load_data()

    # Apply transformation
    df_cleaned = transform_binary(df)

    # Save to processed folder (ensure directory exists)
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "cleaned.csv"
    df_cleaned.to_csv(output_path, index=False)

    print(f"[INFO] Saved cleaned data → {output_path}")
