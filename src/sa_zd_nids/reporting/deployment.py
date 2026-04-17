from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from sa_zd_nids.utils.io import write_json


def _file_size_info(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False, "bytes": 0, "mb": 0.0}
    size = p.stat().st_size
    return {"path": str(p), "exists": True, "bytes": int(size), "mb": float(size / (1024 * 1024))}


def collect_model_sizes(models_dir: str | Path = "models") -> dict[str, dict[str, Any]]:
    models_dir = Path(models_dir)
    targets = {
        "classifier": models_dir / "classifier.joblib",
        "autoencoder": models_dir / "autoencoder.pt",
        "preprocessor": models_dir / "preprocessor.joblib",
        "metadata": models_dir / "metadata.json",
    }
    sizes = {name: _file_size_info(path) for name, path in targets.items()}
    sizes["total"] = {
        "bytes": int(sum(v["bytes"] for v in sizes.values())),
        "mb": float(sum(v["mb"] for v in sizes.values())),
    }
    return sizes


def build_deployment_report(
    experiment_summary: dict[str, Any],
    model_sizes: dict[str, Any],
) -> dict[str, Any]:
    exp = experiment_summary.get("experiments", {})
    by_exp = {}
    for name, payload in exp.items():
        m = payload.get("metrics", {})
        by_exp[name] = {
            "accuracy": m.get("accuracy"),
            "f1_macro": m.get("f1_macro"),
            "zero_day_detection_rate": m.get("zero_day_detection_rate"),
            "zero_day_false_positive_rate_benign": m.get("zero_day_false_positive_rate_benign"),
            "avg_latency_ms_per_sample": m.get("avg_latency_ms_per_sample"),
            "avg_cpu_percent": m.get("avg_cpu_percent"),
            "avg_memory_mb": m.get("avg_memory_mb"),
            "adaptation_time_total_sec": m.get("adaptation_time_total_sec"),
            "num_drifts": m.get("num_drifts"),
            "samples": m.get("samples"),
        }

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "model_sizes": model_sizes,
        "experiments": by_exp,
        "notes": [
            "CPU and memory are process-level runtime observations during stream simulation.",
            "Model sizes are filesystem sizes for serialized artifacts.",
        ],
    }


def write_deployment_report(
    report: dict[str, Any],
    json_path: str | Path,
    md_path: str | Path,
) -> None:
    write_json(json_path, report)
    p = Path(md_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Deployment Report",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        "## Model Size",
        "",
        "| Artifact | Exists | Size (bytes) | Size (MB) |",
        "|---|---:|---:|---:|",
    ]
    for key in ("classifier", "autoencoder", "preprocessor", "metadata"):
        row = report["model_sizes"].get(key, {})
        lines.append(
            f"| {key} | {row.get('exists')} | {row.get('bytes')} | {row.get('mb'):.6f} |"
        )
    total = report["model_sizes"].get("total", {})
    lines.extend(
        [
            f"| total | True | {total.get('bytes', 0)} | {float(total.get('mb', 0.0)):.6f} |",
            "",
            "## Experiment Runtime Metrics",
            "",
            "| Experiment | Accuracy | F1 Macro | Zero-Day Detection | Benign ZD FPR | Latency (ms/sample) | CPU (%) | Memory (MB) | Adapt Time (s) | Drifts | Samples |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, m in report.get("experiments", {}).items():
        lines.append(
            "| {name} | {accuracy} | {f1} | {zdr} | {fpr} | {lat} | {cpu} | {mem} | {adapt} | {drifts} | {samples} |".format(
                name=name,
                accuracy=_fmt(m.get("accuracy")),
                f1=_fmt(m.get("f1_macro")),
                zdr=_fmt(m.get("zero_day_detection_rate")),
                fpr=_fmt(m.get("zero_day_false_positive_rate_benign")),
                lat=_fmt(m.get("avg_latency_ms_per_sample")),
                cpu=_fmt(m.get("avg_cpu_percent")),
                mem=_fmt(m.get("avg_memory_mb")),
                adapt=_fmt(m.get("adaptation_time_total_sec")),
                drifts=m.get("num_drifts", 0),
                samples=m.get("samples", 0),
            )
        )
    p.write_text("\n".join(lines), encoding="utf-8")


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.6f}"
    except Exception:
        return str(value)

