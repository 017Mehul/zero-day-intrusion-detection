from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any


def generate_paper_artifacts(
    experiment_summary: dict[str, Any],
    figure_paths: dict[str, str],
    out_dir: str | Path = "paper/artifacts",
) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    results_table = out / "results_table.tex"
    figures_tex = out / "figures.tex"
    summary_md = out / "artifact_summary.md"

    _write_results_table(results_table, experiment_summary)
    _write_figures_tex(figures_tex, figure_paths)
    _write_summary_md(summary_md, experiment_summary, figure_paths)

    return {
        "results_table_tex": str(results_table),
        "figures_tex": str(figures_tex),
        "summary_md": str(summary_md),
    }


def _write_results_table(path: Path, experiment_summary: dict[str, Any]) -> None:
    exp = experiment_summary.get("experiments", {})
    lines = [
        "% Auto-generated table for IEEE paper",
        "\\begin{table}[t]",
        "\\caption{Experiment Summary}",
        "\\label{tab:exp-summary}",
        "\\centering",
        "\\begin{tabular}{lcccc}",
        "\\hline",
        "Method & Accuracy & F1-Macro & ZD Detect. Rate & Latency (ms) \\\\",
        "\\hline",
    ]
    order = ["static_xgb", "static_ae", "hybrid_adaptive"]
    for name in order:
        if name not in exp:
            continue
        m = exp[name].get("metrics", {})
        lines.append(
            f"{name.replace('_', ' ')} & {_f(m.get('accuracy'))} & {_f(m.get('f1_macro'))} & {_f(m.get('zero_day_detection_rate'))} & {_f(m.get('avg_latency_ms_per_sample'))} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}"])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_figures_tex(path: Path, figure_paths: dict[str, str]) -> None:
    lines = [
        "% Auto-generated figure includes",
        "\\begin{figure}[t]",
        "\\centering",
        f"\\includegraphics[width=\\linewidth]{{{_latex_path(figure_paths.get('accuracy_over_time', ''))}}}",
        "\\caption{Accuracy over time with drift markers.}",
        "\\label{fig:accuracy-time}",
        "\\end{figure}",
        "",
        "\\begin{figure}[t]",
        "\\centering",
        f"\\includegraphics[width=\\linewidth]{{{_latex_path(figure_paths.get('zero_day_rate_over_time', ''))}}}",
        "\\caption{Zero-day prediction rate over time.}",
        "\\label{fig:zd-rate}",
        "\\end{figure}",
        "",
        "\\begin{figure}[t]",
        "\\centering",
        f"\\includegraphics[width=\\linewidth]{{{_latex_path(figure_paths.get('latency_trend', ''))}}}",
        "\\caption{Latency trend across stream progression.}",
        "\\label{fig:latency-trend}",
        "\\end{figure}",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_summary_md(path: Path, experiment_summary: dict[str, Any], figure_paths: dict[str, str]) -> None:
    lines = [
        "# Paper Artifacts",
        "",
        f"Generated: {datetime.utcnow().isoformat()}",
        "",
        "## Included",
        "",
        f"- Results table: `{path.parent / 'results_table.tex'}`",
        f"- Figure includes: `{path.parent / 'figures.tex'}`",
        "",
        "## Figures",
        "",
    ]
    for key, val in figure_paths.items():
        lines.append(f"- {key}: `{val}`")
    lines.extend(["", "## Experiments", ""])
    for name, payload in experiment_summary.get("experiments", {}).items():
        m = payload.get("metrics", {})
        lines.append(
            f"- {name}: accuracy={_f(m.get('accuracy'))}, f1={_f(m.get('f1_macro'))}, zero_day_rate={_f(m.get('zero_day_detection_rate'))}"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _f(v: Any) -> str:
    if v is None:
        return "NA"
    try:
        return f"{float(v):.4f}"
    except Exception:
        return str(v)


def _latex_path(path: str) -> str:
    return path.replace("\\", "/")

