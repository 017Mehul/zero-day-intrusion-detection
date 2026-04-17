"""Merge all CICIDS2017 raw CSVs into a single processed.csv for training."""
import glob
import os

import numpy as np
import pandas as pd

files = sorted(glob.glob("data/*.csv"))
print(f"Found {len(files)} files")

dfs = []
for i, f in enumerate(files):
    df = pd.read_csv(f, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df["timestamp"] = i * 100000 + np.arange(len(df))
    df.rename(columns={"Label": "label"}, inplace=True)
    df["label"] = df["label"].astype(str).str.strip()
    name = os.path.basename(f)
    labels = df["label"].unique()[:5].tolist()
    print(f"  {name}: {len(df)} rows, labels: {labels}")
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)
print(f"\nTotal rows: {len(combined)}")
print("Label distribution:")
print(combined["label"].value_counts().to_string())

# Replace inf with nan so imputer handles them
num_cols = combined.select_dtypes(include=[np.number]).columns
combined[num_cols] = combined[num_cols].replace([np.inf, -np.inf], np.nan)

combined.to_csv("data/processed.csv", index=False)
print("\nSaved data/processed.csv")
