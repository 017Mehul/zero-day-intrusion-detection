# SA-ZD-NIDS

Self-Adaptive Zero-Day Aware Network Intrusion Detection System for tabular network-flow data.

## What is implemented

SA-ZD-NIDS combines:

- **Known-attack classification** with XGBoost, LightGBM or Random Forest.
- **Zero-day anomaly detection** with a PyTorch autoencoder trained on benign traffic.
- **Concept-drift detection** with ADWIN and a composite streaming signal.
- **Drift-triggered adaptation** in the offline streaming engine, with validation-gated classifier updates and atomic artifact persistence.
- **Serving-time adaptation** using conservative high-confidence pseudo-labels plus anomaly-threshold recalibration.
- **FastAPI inference API** for single, batch and streaming predictions.
- **Streamlit dashboard** for demonstration and monitoring.
- **Optuna tuning, evaluation metrics, reporting and research artifacts**.
- **CI** covering package installation, imports, the full test suite and linting.

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

## Reproducible setup

### 1. Install

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
```

### 2. Prepare CICIDS2017

Put the raw CSV files under `data/`, then run:

```bash
python scripts/prepare_data.py
```

This creates `data/processed.csv`.

### 3. Train

```bash
python train.py --config config.yaml
```

Training creates:

- `models/classifier.joblib`
- `models/autoencoder.pt`
- `models/preprocessor.joblib`
- `models/metadata.json`

Model artifacts and datasets are intentionally gitignored.

### 4. Run the streaming evaluation

```bash
python stream.py --config config.yaml
```

Metrics and event logs are written to the configured `logs/` paths.

### 5. Run the API

```bash
python -m uvicorn deployment.api.main:app --host 0.0.0.0 --port 8000
```

Useful endpoints:

- `GET /`
- `GET /api/v1/health`
- `GET /api/v1/ready`
- `GET /api/v1/drift`
- `POST /api/v1/predict`
- `POST /api/v1/predict/batch`
- `POST /api/v1/predict/stream`
- `GET /api/v1/predict/stats`

The dashboard API URL is configurable with `SA_ZD_NIDS_API_URL`.

### Demo mode

The repository also contains `mock_api.py` for UI-only demonstrations without trained model artifacts:

```bash
python -m uvicorn mock_api:app --host 0.0.0.0 --port 8000
python -m streamlit run deployment/frontend/app.py
```

## Docker

After generating model artifacts locally:

```bash
docker compose up --build
```

The API is exposed on port 8000 and the dashboard on port 8501. The `models/` directory is mounted read-only into the API container.

## Testing

Run:

```bash
python -m pytest -vv
python -m flake8 --max-line-length=200 --extend-ignore=E203,W503 src tests
```

GitHub Actions runs the same install/import/test/lint pipeline on every push to `main` and pull request.

## Evaluation

The repository includes classification and zero-day metrics:

- accuracy
- macro precision
- macro recall
- macro F1
- unseen-attack detection rate
- benign false-positive rate
- drift count and drift latency
- adaptation time
- inference latency
- CPU and memory usage

**Important:** benchmark values such as accuracy or zero-day detection rate depend on the exact dataset, preprocessing, split and model configuration. The README does not claim a universal 94%/87% result; reproduce `train.py` and `stream.py` to obtain results for your run.

## Real-world validation and deployment

### Live traffic capture

Optional packet capture provides a real ingestion starting point:

```bash
pip install -r requirements-capture.txt
python scripts/capture_packets.py --interface "Wi-Fi" --seconds 60 --output data/live.pcap
```

The capture intentionally stops at PCAP. A flow extractor such as CICFlowMeter or Zeek must convert packets into the exact feature schema used by the trained model; raw packets cannot safely be mapped to CICIDS2017 features by guesswork.

### Cross-dataset validation

Validate a trained model on an independent labeled dataset with the same feature schema:

```bash
python scripts/validate_external_dataset.py --data data/external.csv
```

The script reuses the fitted training preprocessor and reports classification and zero-day metrics without fitting on the external dataset. Different feature schemas require an explicit adapter rather than silent column guessing.

### Production API hardening

The API supports optional `X-API-Key` authentication and a per-process rate limit:

```bash
export SA_ZD_NIDS_API_KEY="change-me"
export SA_ZD_NIDS_RATE_LIMIT_PER_MINUTE=120
```

CORS remains configurable with `SA_ZD_NIDS_ALLOWED_ORIGINS`. For internet-facing deployments, terminate TLS at a reverse proxy/load balancer and use a durable distributed rate limiter, authentication/authorization service, secret manager, logs/metrics backend and alerting. The built-in controls are a safe baseline, not a replacement for a production gateway.

The system's automatic serving adaptation is deliberately conservative and uses high-confidence pseudo-labels. For authoritative labeled model updates, use `adapt.py` with a validated recent labeled dataset.

## Project status

This is a **working research/portfolio prototype** with explicit live-capture, cross-dataset-validation and API-hardening paths. It is not a claim of a production SOC/NIDS appliance until those environment-specific integrations and infrastructure controls are deployed.

## Citation

```bibtex
@software{sa_zd_nids,
  title={SA-ZD-NIDS: Self-Adaptive Zero-Day Aware Network Intrusion Detection System},
  author={Mehul Gupta},
  year={2026},
  url={https://github.com/017Mehul/zero-day-intrusion-detection}
}
```

Built by Mehul Gupta.
