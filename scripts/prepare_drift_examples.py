#!/usr/bin/env python3
"""Create example splits and drift scenarios from a CSV with a timestamp and label column.

Usage: python scripts/prepare_drift_examples.py --input data/processed.csv --outdir data/examples
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def chronological_split(df: pd.DataFrame, ts_col: str = "timestamp", test_frac: float = 0.2, val_frac: float = 0.2):
    df = df.sort_values(ts_col).reset_index(drop=True)
    n = len(df)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = n - n_val - n_test
    train = df.iloc[:n_train].copy()
    val = df.iloc[n_train : n_train + n_val].copy()
    test = df.iloc[n_train + n_val :].copy()
    return train, val, test


def make_gradual_drift(test: pd.DataFrame, label_col: str = "label", steps: int = 5):
    # Gradually change class distribution across successive blocks
    n = len(test)
    block = max(1, n // steps)
    out = test.copy()
    labels = out[label_col].unique().tolist()
    for i in range(steps):
        start = i * block
        end = min(n, (i + 1) * block)
        # rotate labels to create distribution shift
        if len(labels) > 1:
            shift = i % len(labels)
            out.iloc[start:end, out.columns.get_loc(label_col)] = np.random.choice(
                labels, size=(end - start), p=_rotated_probs(len(labels), shift)
            )
    return out


def _rotated_probs(k: int, shift: int):
    base = np.ones(k) / k
    return np.roll(base, shift)


def make_sudden_drift(test: pd.DataFrame, label_col: str = "label", unseen_label: str = "NEW_ATTACK"):
    out = test.copy()
    n = len(out)
    start = n // 2
    out.iloc[start:, out.columns.get_loc(label_col)] = unseen_label
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", default="data/examples")
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--ts_col", default="timestamp")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    train, val, test = chronological_split(df, ts_col=args.ts_col)
    train.to_csv(outdir / "train.csv", index=False)
    val.to_csv(outdir / "val.csv", index=False)
    test.to_csv(outdir / "test.csv", index=False)

    gd = make_gradual_drift(test, label_col=args.label_col)
    gd.to_csv(outdir / "test_gradual_drift.csv", index=False)

    sd = make_sudden_drift(test, label_col=args.label_col)
    sd.to_csv(outdir / "test_sudden_drift.csv", index=False)

    print("Wrote examples to", outdir)


if __name__ == "__main__":
    main()
