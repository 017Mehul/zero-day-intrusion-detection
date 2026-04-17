from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sa_zd_nids.reporting.paper_artifacts import generate_paper_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LaTeX paper artifact files from experiment summary")
    parser.add_argument("--summary", required=True, help="Path to suite summary JSON")
    parser.add_argument("--figures", required=True, help="Path to figures JSON map")
    parser.add_argument("--outdir", default="paper/artifacts")
    args = parser.parse_args()

    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    fig_payload = json.loads(Path(args.figures).read_text(encoding="utf-8"))
    if isinstance(fig_payload, dict) and "figure_paths" in fig_payload:
        figures = fig_payload["figure_paths"]
    else:
        figures = fig_payload
    artifacts = generate_paper_artifacts(experiment_summary=summary, figure_paths=figures, out_dir=args.outdir)
    print(json.dumps(artifacts, indent=2))


if __name__ == "__main__":
    main()
