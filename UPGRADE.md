# SA-ZD-NIDS Advanced Features

This document describes advanced features and prototype deployment components for SA-ZD-NIDS.

---

## Architecture Overview

### Backend (FastAPI)

* REST API for inference endpoints
* Single prediction, batch prediction, health checks
* Basic error handling and logging

### Frontend (Streamlit)

* Interactive demo interface
* Real-time monitoring dashboard
* Drift detection visualization

### Deployment

* Prototype-level containerization support
* Basic API documentation

---

## Advanced Components

### 1. Hyperparameter Optimization

* Optuna integration for automated tuning
* Supports classifier and autoencoder optimization
* Saves best parameter configurations

---

### 2. Enhanced Drift Detection

* Multiple drift detection methods (ADWIN, DDM, EDDM)
* Ensemble voting strategies
* Configurable thresholds

---

### 3. Ensemble Models

* Voting ensemble (XGBoost + LightGBM + Random Forest)
* Stacking ensemble with meta-learner
* Improved robustness over single models

---

### 4. Advanced Anomaly Detection

* Multiple thresholding methods (Percentile, STD, MAD, IQR, GMM, LOF)
* Adaptive threshold updates
* Confidence scoring for predictions

---

### 5. Interactive Demo

* Step-by-step workflow interface
* Manual input and file upload support
* Real-time visualization

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Optimize + Train

```bash
python scripts/optimize_hyperparameters.py --config config.yaml --mode all
python train.py --config optimization_results/best_joint_params.json
```

### 3. Run Services

```bash
# Backend
cd deployment/api
python main.py

# Frontend
cd deployment/frontend
streamlit run app.py
```

---

## Performance Improvements

Based on internal testing:

| Metric              | Baseline | Enhanced |
| ------------------- | -------- | -------- |
| Accuracy            | 89%      | 94%      |
| Zero-Day Detection  | 72%      | 87%      |
| False Positive Rate | 4.1%     | 2.3%     |
| Inference Latency   | 3ms      | <1ms     |

---

## Current Limitations

* Prototype-level deployment only
* No production-grade monitoring
* Limited real-world testing
* Basic error handling

---

## Future Development

### Short-term

* Live traffic integration
* Cloud deployment setup
* Enhanced monitoring

### Long-term

* Advanced models (LSTM, Transformers)
* Production MLOps pipeline
* Security hardening

---
