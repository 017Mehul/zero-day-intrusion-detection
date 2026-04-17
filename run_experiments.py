from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sa_zd_nids.config import load_config
from sa_zd_nids.experiments.runner import run_all_experiments
from sa_zd_nids.reporting.deployment import build_deployment_report, collect_model_sizes, write_deployment_report
from sa_zd_nids.reporting.paper_artifacts import generate_paper_artifacts
from sa_zd_nids.utils.io import write_json
from sa_zd_nids.visualization.plots import generate_experiment_plots
from sa_zd_nids.visualization.system_diagram import generate_system_diagram_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SA-ZD-NIDS baseline suite and reporting pipeline")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--input", default=None, help="Override input CSV path")
    parser.add_argument("--outdir", default=None, help="Output directory for experiment suite")
    args = parser.parse_args()

    cfg = load_config(args.config)
    input_path = args.input or cfg["data"]["input_path"]
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.outdir) if args.outdir else Path("evaluation") / "experiments" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = run_all_experiments(base_cfg=cfg, input_path=input_path, out_root=out_dir)

    batch_csvs = {name: payload["batch_csv"] for name, payload in summary["experiments"].items()}
    figure_paths = generate_experiment_plots(batch_csv_by_experiment=batch_csvs, out_dir=out_dir / "figures")

    model_sizes = collect_model_sizes("models")
    deploy_report = build_deployment_report(experiment_summary=summary, model_sizes=model_sizes)
    write_deployment_report(
        report=deploy_report,
        json_path=out_dir / "deployment_report.json",
        md_path=out_dir / "deployment_report.md",
    )

    paper_artifacts = generate_paper_artifacts(
        experiment_summary=summary,
        figure_paths=figure_paths,
        out_dir=Path("paper") / "artifacts",
    )
    diagram_paths = generate_system_diagram_files(out_dir=out_dir / "diagrams")

    final_summary = {
        "suite_summary": summary,
        "figure_paths": figure_paths,
        "deployment_report_json": str(out_dir / "deployment_report.json"),
        "deployment_report_md": str(out_dir / "deployment_report.md"),
        "paper_artifacts": paper_artifacts,
        "diagram_paths": diagram_paths,
    }
    write_json(out_dir / "suite_outputs.json", final_summary)

    print("Experiment suite complete")
    print(json.dumps(final_summary, indent=2))


if __name__ == "__main__":
    main()

