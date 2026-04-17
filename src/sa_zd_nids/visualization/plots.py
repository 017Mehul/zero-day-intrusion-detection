from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _save(fig, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _load_batch_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {"sample_end", "accuracy", "zero_day_rate", "latency_ms_per_sample", "drift_flag"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in batch metrics {path}: {sorted(missing)}")
    return df


def plot_accuracy_over_time(
    batch_csv_by_experiment: dict[str, str],
    output_path: str | Path,
    drift_source: str = "hybrid_adaptive",
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, path in batch_csv_by_experiment.items():
        df = _load_batch_csv(path)
        ax.plot(df["sample_end"], df["accuracy"], label=name, linewidth=2)

    if drift_source in batch_csv_by_experiment:
        drift_df = _load_batch_csv(batch_csv_by_experiment[drift_source])
        drift_points = drift_df[drift_df["drift_flag"] == True]["sample_end"].tolist()  # noqa: E712
        for d in drift_points:
            ax.axvline(x=d, color="red", alpha=0.2, linewidth=1.2)

    ax.set_title("Accuracy Over Time")
    ax.set_xlabel("Processed Samples")
    ax.set_ylabel("Batch Accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, output_path)


def plot_zero_day_rate_over_time(batch_csv_by_experiment: dict[str, str], output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, path in batch_csv_by_experiment.items():
        df = _load_batch_csv(path)
        ax.plot(df["sample_end"], df["zero_day_rate"], label=name, linewidth=2)
    ax.set_title("Zero-Day Prediction Rate Over Time")
    ax.set_xlabel("Processed Samples")
    ax.set_ylabel("Batch Zero-Day Rate")
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, output_path)


def plot_latency_trend(batch_csv_by_experiment: dict[str, str], output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, path in batch_csv_by_experiment.items():
        df = _load_batch_csv(path)
        ax.plot(df["sample_end"], df["latency_ms_per_sample"], label=name, linewidth=2)
    ax.set_title("Latency Trend")
    ax.set_xlabel("Processed Samples")
    ax.set_ylabel("Latency (ms/sample)")
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, output_path)


def generate_experiment_plots(batch_csv_by_experiment: dict[str, str], out_dir: str | Path) -> dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "accuracy_over_time": str(out_dir / "accuracy_over_time.png"),
        "zero_day_rate_over_time": str(out_dir / "zero_day_rate_over_time.png"),
        "latency_trend": str(out_dir / "latency_trend.png"),
    }
    plot_accuracy_over_time(batch_csv_by_experiment=batch_csv_by_experiment, output_path=paths["accuracy_over_time"])
    plot_zero_day_rate_over_time(batch_csv_by_experiment=batch_csv_by_experiment, output_path=paths["zero_day_rate_over_time"])
    plot_latency_trend(batch_csv_by_experiment=batch_csv_by_experiment, output_path=paths["latency_trend"])
    return paths

