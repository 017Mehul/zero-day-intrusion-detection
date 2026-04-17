from __future__ import annotations

from pathlib import Path


MERMAID_DIAGRAM = """flowchart LR
    A["Data Ingestion Layer"] --> B["Feature Engineering"]
    B --> C["Known Attack Classifier (XGBoost)"]
    B --> D["Zero-Day Detector (Autoencoder)"]
    C --> E{"Confidence >= Threshold?"}
    E -->|"Yes"| F["Known Class Output"]
    E -->|"No"| D
    D --> G{"Reconstruction Error > Threshold?"}
    G -->|"Yes"| H["ZERO_DAY Alert"]
    G -->|"No"| I["Benign Output"]
    F --> J["Streaming Metrics Logger"]
    H --> J
    I --> J
    J --> K["Drift Detector (ADWIN + Composite Signal)"]
    K -->|"Drift Detected"| L["Adaptive Update Engine"]
    L --> C
    L --> D
    J --> M["Deployment Report + Paper Artifacts"]
"""


def generate_system_diagram_files(out_dir: str | Path = "evaluation/diagrams") -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mmd = out / "system_architecture.mmd"
    md = out / "system_architecture.md"
    mmd.write_text(MERMAID_DIAGRAM, encoding="utf-8")
    md.write_text(
        "\n".join(
            [
                "# SA-ZD-NIDS System Diagram",
                "",
                "```mermaid",
                MERMAID_DIAGRAM,
                "```",
            ]
        ),
        encoding="utf-8",
    )
    return {"mermaid": str(mmd), "markdown": str(md)}

