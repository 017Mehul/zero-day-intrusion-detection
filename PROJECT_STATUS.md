# SA-ZD-NIDS Project Status

## Current state

**Working research / portfolio prototype.** The core training, inference, streaming, drift-detection and serving code is implemented and covered by automated tests.

## Implemented

### Core ML
- Leakage-safe chronological preprocessing with imputation, scaling and optional mutual-information feature selection.
- XGBoost, LightGBM and Random Forest known-attack classifiers.
- PyTorch benign-only autoencoder for zero-day anomaly detection.
- Confidence-gated hybrid prediction.

### Adaptation
- ADWIN-based concept-drift detection.
- Streaming adaptation with recent-window/replay data.
- Validation-gated classifier updates.
- Atomic model and metadata persistence.
- Conservative serving-time pseudo-label adaptation.
- Safe guard against one-class pseudo-label retraining.

### Serving
- FastAPI single, batch and streaming prediction endpoints.
- Health/readiness and drift endpoints.
- Configurable model/config paths.
- Restricted configurable CORS.
- Streamlit dashboard with configurable API URL.
- Mock API for UI-only demonstrations.

### Engineering
- pyproject.toml package configuration.
- Unit/integration/smoke tests.
- GitHub Actions installation, import, test and lint checks.
- Dockerfile, docker-compose and .dockerignore.
- Research-paper and reporting artifacts.

## Reproducibility

The dataset and trained artifacts are intentionally excluded from Git. A fresh checkout requires CICIDS2017/raw network-flow data, preprocessing and model training before the real API can serve predictions.

Benchmark results must be reproduced from the exact dataset, split and configuration used for a particular experiment. Historical 94%/87% figures are not universal guarantees.

## Remaining environment-dependent work

- Live packet capture and production flow-feature extraction for a specific network interface.
- Cross-dataset generalization studies.
- Internet-facing production infrastructure: TLS, authentication, rate limiting, secret management and durable monitoring.
- Large-scale SOC integration and alert routing.

## Verification

The repository CI runs a clean Python 3.10 environment, installs dependencies, installs the package, verifies core imports, runs the full pytest suite and runs flake8.
