# SA-ZD-NIDS

**Self-Adaptive Zero-Day Aware Network Intrusion Detection System (SA-ZD-NIDS)** for tabular network-flow data.

SA-ZD-NIDS is a research and portfolio-oriented NIDS that combines supervised attack classification, benign-only anomaly detection, concept-drift detection, conservative online adaptation, and deployable inference APIs.

> **Status:** Working research/portfolio prototype. The repository contains the implementation and validation paths needed for experimentation and deployment, while environment-specific live-traffic and production-infrastructure validation must be performed in the target environment.

## Key capabilities

- **Known-attack classification** using XGBoost, LightGBM, or Random Forest.
- **Zero-day / unknown-attack detection** using a PyTorch autoencoder trained on benign traffic.
- **Concept-drift detection** using ADWIN and a composite streaming signal.
- **Drift-triggered adaptation** with validation-gated classifier updates and atomic artifact persistence.
- **Serving-time adaptation** using conservative high-confidence pseudo-labels and anomaly-threshold recalibration.
- **Leakage-safe preprocessing** with chronological splitting and reuse of the fitted inference pipeline.
- **FastAPI inference service** for single, batch, and streaming predictions.
- **Streamlit dashboard** for demonstration and monitoring.
- **Optuna-based tuning**, evaluation metrics, reporting, and research artifacts.
- **Docker deployment** for the API and dashboard.
- **API hardening** with optional API-key authentication, rate limiting, configurable CORS, and security headers.
- **Live-traffic capture path** through optional PCAP capture.
- **Cross-dataset validation** against independent labeled datasets with compatible feature schemas.
- **CI** covering installation, imports, tests, and linting.

## Architecture

```text
Network-flow data
       |
       v
Leakage-safe preprocessing
(imputation -> optional MI selection -> scaling)
       |
       +--------------------+
       |                    |
       v                    v
Known-attack model     Autoencoder
XGBoost/RF/LGBM        benign-only training
       |                    |
       +--------+-----------+
                v
      confidence / anomaly decision
                |
                v
        streaming + ADWIN
                |
         drift detected?
           /          \
         no            yes
         |              |
         |       validation-gated
         |       adaptation/recalibration
         v              v
              metrics + model artifacts
```

## Repository structure

```text
.
├── config.yaml
├── train.py
├── stream.py
├── adapt.py
├── mock_api.py
├── requirements.txt
├── requirements-dev.txt
├── requirements-capture.txt
├── Dockerfile
├── docker-compose.yml
├── scripts/
│   ├── prepare_data.py
│   ├── capture_packets.py
│   └── validate_external_dataset.py
├── src/sa_zd_nids/
│   ├── data/
│   ├── models/
│   ├── drift/
│   ├── streaming/
│   ├── evaluation/
│   ├── optimization/
│   ├── reporting/
│   ├── utils/
│   └── visualization/
├── deployment/
│   ├── api/
│   └── frontend/
├── tests/
└── .github/workflows/
```

## Reproducible setup

### 1. Install

Create a virtual environment and install the project dependencies:

```bash
python -m venv .venv

# Windows
.venv\\Scripts\\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
```

### 2. Prepare CICIDS2017

Place the raw CICIDS2017 CSV files under `data/`, then run:

```bash
python scripts/prepare_data.py
```

This produces the processed dataset used by the training pipeline.

### 3. Train

```bash
python train.py --config config.yaml
```

Training creates the model artifacts required by inference:

- `models/classifier.joblib`
- `models/autoencoder.pt`
- `models/preprocessor.joblib`
- `models/metadata.json`

Datasets and generated model artifacts are intentionally gitignored.

### 4. Run streaming evaluation

```bash
python stream.py --config config.yaml
```

The streaming pipeline evaluates drift detection and adaptation behavior and writes metrics/event logs to the configured output paths.

### 5. Run the API

```bash
python -m uvicorn deployment.api.main:app --host 0.0.0.0 --port 8000
```

Main endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Service information |
| GET | `/api/v1/health` | Health check |
| GET | `/api/v1/ready` | Model readiness |
| GET | `/api/v1/drift` | Current drift state |
| POST | `/api/v1/predict` | Single-flow prediction |
| POST | `/api/v1/predict/batch` | Batch prediction |
| POST | `/api/v1/predict/stream` | Streaming prediction |
| GET | `/api/v1/predict/stats` | Prediction statistics |

The dashboard API URL can be configured with `SA_ZD_NIDS_API_URL`.

### Example API request

The exact feature fields depend on the trained model and dataset configuration. A request therefore follows the model's feature schema rather than assuming a universal network-flow format:

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '<JSON matching the trained feature schema>'
```

For a deployed service with API-key authentication enabled, add:

```text
X-API-Key: <your-api-key>
```

### Demo mode

For a UI-only demonstration without trained model artifacts:

```bash
python -m uvicorn mock_api:app --host 0.0.0.0 --port 8000
python -m streamlit run deployment/frontend/app.py
```

## Docker deployment

After generating model artifacts locally:

```bash
docker compose up --build
```

The API is exposed on port `8000` and the dashboard on port `8501`. The `models/` directory is mounted read-only into the API container.

For an internet-facing deployment, place the service behind a TLS-terminating reverse proxy or load balancer and use external authentication, durable distributed rate limiting, secret management, centralized logs/metrics, and alerting.

## Testing and CI

Run the local test suite:

```bash
python -m pytest -vv
python -m flake8 --max-line-length=200 --extend-ignore=E203,W503 src tests
```

GitHub Actions runs the project's installation, import checks, tests, and linting on pushes to `main` and pull requests.

## Evaluation and research metrics

The repository includes metrics for:

### Classification

- Accuracy
- Macro precision
- Macro recall
- Macro F1

### Zero-day detection

- Unseen-attack detection rate
- Benign false-positive rate

### Streaming/adaptation

- Drift count
- Drift latency
- Adaptation time
- Inference latency
- CPU and memory usage

### Reproducibility

Reported values depend on the exact dataset, preprocessing configuration, chronological split, selected features, classifier, and random state. The repository therefore does **not** claim a universal accuracy or zero-day detection percentage.

Run the training and streaming pipelines to generate results for the specific experiment.

## Real-world validation and deployment

### Live network traffic capture

Optional packet capture provides a real ingestion starting point:

```bash
pip install -r requirements-capture.txt
python scripts/capture_packets.py --interface "Wi-Fi" --seconds 60 --output data/live.pcap
```

The capture script produces a PCAP file. It does **not** pretend that raw packets are directly equivalent to CICIDS2017 flow features.

The next step is to use a flow extractor such as **CICFlowMeter** or **Zeek** to convert packets into the exact feature schema expected by the trained model.

On systems that require elevated packet-capture permissions, run the capture tool with the appropriate administrator/root privileges and install the required packet-capture driver.

### Cross-dataset validation

Validate a trained model on an independent labeled dataset:

```bash
python scripts/validate_external_dataset.py --data data/external.csv
```

The validation pipeline:

1. Loads the fitted training preprocessor.
2. Checks that the external dataset contains the required feature columns.
3. Reuses the trained classifier and autoencoder.
4. Does not refit the preprocessing pipeline on the external dataset.
5. Computes classification and zero-day metrics.
6. Writes the evaluation output to the configured JSON path.

An external dataset with a different feature schema requires an explicit feature adapter. Columns should not be silently guessed or reordered without validation.

### Production API hardening

Optional API protection can be enabled with environment variables:

```bash
export SA_ZD_NIDS_API_KEY="change-me"
export SA_ZD_NIDS_RATE_LIMIT_PER_MINUTE=120
```

On Windows PowerShell:

```powershell
$env:SA_ZD_NIDS_API_KEY="change-me"
$env:SA_ZD_NIDS_RATE_LIMIT_PER_MINUTE="120"
```

CORS is configurable through `SA_ZD_NIDS_ALLOWED_ORIGINS`.

The built-in controls provide a baseline for a deployed prototype. A production internet-facing service should additionally use TLS, a proper authentication/authorization layer, a distributed rate limiter, a managed secret store, centralized observability, and alerting.

### Adaptation safety

Serving-time adaptation is deliberately conservative and uses high-confidence pseudo-labels. For authoritative model updates, use `adapt.py` with a validated recent labeled dataset rather than treating automatically generated pseudo-labels as ground truth.

## Limitations and responsible claims

SA-ZD-NIDS should be interpreted as a research/portfolio prototype rather than a drop-in enterprise SOC appliance.

The following are intentionally environment-dependent or require additional validation:

- Live packet-to-flow feature extraction.
- Performance on an independent dataset.
- Cross-network/generalization performance.
- Production-scale distributed serving.
- TLS termination and enterprise identity/authentication.
- Durable distributed rate limiting.
- Long-term monitoring, alerting, and model governance.
- Hardware- and network-specific throughput/latency.

These limitations are explicit so that reported experiments remain reproducible and claims remain defensible.

## Project status

**Implemented:** core detection pipeline, zero-day anomaly path, ADWIN drift detection, adaptation logic, FastAPI service, Streamlit dashboard, Docker deployment, API hardening, live-capture entry point, external-dataset validation path, tests, linting, and CI configuration.

**Requires environment-specific validation:** actual live traffic capture and flow extraction, independent-dataset experiments, production infrastructure, and measured deployment performance.

## Citation

```bibtex
@software{sa_zd_nids,
  title={SA-ZD-NIDS: Self-Adaptive Zero-Day Aware Network Intrusion Detection System},
  author={Mehul Gupta},
  year={2026},
  url={https://github.com/017Mehul/zero-day-intrusion-detection}
}
```

Built by **Mehul Gupta**.
