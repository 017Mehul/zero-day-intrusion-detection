from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch


DATA_DIR = Path("data")
OUT_DIR = Path("paper") / "figures"
LABEL_COLUMN = " Label"
BENIGN_LABEL = "BENIGN"
FLOW_DURATION_COLUMN = " Flow Duration"


def _ordered_files() -> list[Path]:
    preferred = [
        "Monday-WorkingHours.pcap_ISCX.csv",
        "Tuesday-WorkingHours.pcap_ISCX.csv",
        "Wednesday-workingHours.pcap_ISCX.csv",
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        "Friday-WorkingHours-Morning.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    ]
    existing = {p.name: p for p in DATA_DIR.glob("*.csv")}
    ordered = [existing[name] for name in preferred if name in existing]
    leftovers = sorted([p for name, p in existing.items() if name not in preferred], key=lambda p: p.name.lower())
    return ordered + leftovers


def _label_counts(path: Path) -> pd.Series:
    chunks = []
    for chunk in pd.read_csv(path, usecols=[LABEL_COLUMN], chunksize=50000):
        counts = chunk[LABEL_COLUMN].astype(str).str.strip().value_counts()
        chunks.append(counts)
    if not chunks:
        return pd.Series(dtype="int64")
    total = chunks[0].copy()
    for counts in chunks[1:]:
        total = total.add(counts, fill_value=0)
    return total.sort_values(ascending=False).astype(int)


def _save(fig: plt.Figure, name: str) -> str:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_capture_volume() -> str:
    files = _ordered_files()
    names: list[str] = []
    benign_counts: list[int] = []
    attack_counts: list[int] = []

    for path in files:
        counts = _label_counts(path)
        benign = int(counts.get(BENIGN_LABEL, 0))
        total = int(counts.sum())
        attack = max(total - benign, 0)
        names.append(path.name.replace(".pcap_ISCX.csv", "").replace(".csv", ""))
        benign_counts.append(benign)
        attack_counts.append(attack)

    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    ax.bar(names, benign_counts, color="#7aa974", label="Benign")
    ax.bar(names, attack_counts, bottom=benign_counts, color="#d95f5f", label="Attack")
    ax.set_title("Traffic Volume Across CICIDS2017 Capture Files")
    ax.set_ylabel("Number of Flows")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    return _save(fig, "capture_volume.png")


def plot_attack_distribution() -> str:
    aggregate = pd.Series(dtype="int64")
    for path in _ordered_files():
        counts = _label_counts(path)
        counts = counts[counts.index != BENIGN_LABEL]
        aggregate = aggregate.add(counts, fill_value=0)

    aggregate = aggregate.sort_values(ascending=False).astype(int)
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.bar(aggregate.index.tolist(), aggregate.values.tolist(), color="#4c78a8")
    ax.set_title("Attack-Class Distribution in the Available Dataset")
    ax.set_ylabel("Number of Flows")
    ax.grid(axis="y", alpha=0.25)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    return _save(fig, "attack_distribution.png")


def plot_attack_ratio_trend() -> str:
    files = _ordered_files()
    names: list[str] = []
    ratios: list[float] = []

    for path in files:
        counts = _label_counts(path)
        benign = int(counts.get(BENIGN_LABEL, 0))
        total = int(counts.sum())
        attack = max(total - benign, 0)
        ratio = (attack / total) if total else 0.0
        names.append(path.name.replace(".pcap_ISCX.csv", "").replace(".csv", ""))
        ratios.append(ratio)

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(names, ratios, marker="o", linewidth=2.2, color="#c44e52")
    ax.fill_between(names, ratios, alpha=0.18, color="#c44e52")
    ax.set_title("Attack Ratio Across Chronological Capture Files")
    ax.set_ylabel("Attack Flow Ratio")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    return _save(fig, "attack_ratio_trend.png")


def _top_attack_labels(limit: int = 5) -> list[str]:
    aggregate = pd.Series(dtype="int64")
    for path in _ordered_files():
        counts = _label_counts(path)
        counts = counts[counts.index != BENIGN_LABEL]
        aggregate = aggregate.add(counts, fill_value=0)
    aggregate = aggregate.sort_values(ascending=False).astype(int)
    return aggregate.head(limit).index.tolist()


def plot_flow_duration_by_class(max_points_per_class: int = 2500) -> str:
    target_labels = [BENIGN_LABEL] + _top_attack_labels(limit=5)
    samples: dict[str, list[float]] = {label: [] for label in target_labels}

    for path in _ordered_files():
        for chunk in pd.read_csv(path, usecols=[LABEL_COLUMN, FLOW_DURATION_COLUMN], chunksize=50000):
            chunk[LABEL_COLUMN] = chunk[LABEL_COLUMN].astype(str).str.strip()
            for label in target_labels:
                current = samples[label]
                remaining = max_points_per_class - len(current)
                if remaining <= 0:
                    continue
                vals = chunk.loc[chunk[LABEL_COLUMN] == label, FLOW_DURATION_COLUMN].astype(float)
                if vals.empty:
                    continue
                if len(vals) > remaining:
                    vals = vals.sample(n=remaining, random_state=42)
                current.extend(vals.tolist())
        if all(len(v) >= max_points_per_class for v in samples.values()):
            break

    labels = [label for label in target_labels if samples[label]]
    data = [[max(v, 0.0) for v in samples[label]] for label in labels]
    log_data = [[float(np.log10(v + 1.0)) for v in series] for series in data]

    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    box = ax.boxplot(log_data, patch_artist=True, labels=labels, showfliers=False)
    colors = ["#7aa974", "#4c78a8", "#f58518", "#e45756", "#72b7b2", "#54a24b"]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_title("Log-Scaled Flow Duration by Benign and Major Attack Classes")
    ax.set_ylabel(r"$\log_{10}(1 + \mathrm{Flow\ Duration})$")
    ax.grid(axis="y", alpha=0.25)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    return _save(fig, "flow_duration_by_class.png")


def _box(ax, xy: tuple[float, float], text: str, width: float = 0.22, height: float = 0.12, fc: str = "#e8f1f8") -> None:
    x, y = xy
    rect = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2,
        edgecolor="#355c7d",
        facecolor=fc,
    )
    ax.add_patch(rect)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=10)


def plot_system_architecture() -> str:
    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _box(ax, (0.04, 0.68), "Traffic Ingestion\nand Preprocessing")
    _box(ax, (0.30, 0.68), "Known-Attack\nClassifier")
    _box(ax, (0.30, 0.38), "Zero-Day\nAutoencoder", fc="#fbe7c6")
    _box(ax, (0.56, 0.68), "Confidence-Gated\nDecision Logic")
    _box(ax, (0.78, 0.68), "Output Labels\nKnown / Benign /\nZERO_DAY", width=0.18)
    _box(ax, (0.56, 0.38), "Drift Detector\nand Adaptive Update", fc="#dff3e4")
    _box(ax, (0.78, 0.38), "Metrics, Logs,\nDeployment Report", width=0.18, fc="#f0e6ff")

    arrow = dict(arrowstyle="->", linewidth=1.6, color="#2f4858")
    ax.annotate("", xy=(0.30, 0.74), xytext=(0.26, 0.74), arrowprops=arrow)
    ax.annotate("", xy=(0.56, 0.74), xytext=(0.52, 0.74), arrowprops=arrow)
    ax.annotate("", xy=(0.78, 0.74), xytext=(0.74, 0.74), arrowprops=arrow)
    ax.annotate("", xy=(0.41, 0.50), xytext=(0.41, 0.68), arrowprops=arrow)
    ax.annotate("", xy=(0.56, 0.44), xytext=(0.52, 0.44), arrowprops=arrow)
    ax.annotate("", xy=(0.78, 0.44), xytext=(0.74, 0.44), arrowprops=arrow)
    ax.annotate("", xy=(0.40, 0.68), xytext=(0.67, 0.50), arrowprops=arrow)
    ax.annotate("", xy=(0.41, 0.38), xytext=(0.66, 0.38), arrowprops=arrow)
    ax.annotate("", xy=(0.41, 0.62), xytext=(0.67, 0.62), arrowprops=arrow)

    ax.text(0.5, 0.94, "SA-ZD-NIDS Hybrid Streaming Architecture", ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(0.56, 0.60, "Low confidence\nroutes to anomaly branch", ha="center", va="center", fontsize=9, color="#444444")
    ax.text(0.56, 0.29, "Drift alarms trigger replay-aware adaptation\nand threshold recalibration", ha="center", va="center", fontsize=9, color="#444444")
    return _save(fig, "system_architecture.png")


def main() -> None:
    paths = {
        "capture_volume": plot_capture_volume(),
        "attack_distribution": plot_attack_distribution(),
        "attack_ratio_trend": plot_attack_ratio_trend(),
        "flow_duration_by_class": plot_flow_duration_by_class(),
        "system_architecture": plot_system_architecture(),
    }
    for key, value in paths.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
