# SA-ZD-NIDS Interview Preparation

## 15-second explanation

"SA-ZD-NIDS is a hybrid network intrusion detection system that combines a supervised classifier for known attacks with a benign-trained autoencoder for anomalous or zero-day traffic, while ADWIN monitors streaming drift and triggers conservative adaptation."

## How does it detect zero-day attacks?

The classifier first predicts a known class and confidence score. If confidence is below the configured threshold, the autoencoder evaluates reconstruction error. Samples above the learned anomaly threshold are emitted as ZERO_DAY.

## Why use XGBoost?

Network-flow datasets are primarily tabular. Gradient-boosted trees work well on nonlinear feature interactions, support class weighting and provide strong inference performance. The implementation also supports Random Forest and LightGBM.

## How is concept drift handled?

The streaming engine builds a drift signal from batch accuracy, low-confidence rate and feature shift, then feeds it to ADWIN. When drift is detected and enough recent data exists, the classifier performs validation-gated adaptation using a recent window plus replay data.

## Is serving adaptation safe?

It is deliberately conservative. The API only considers high-confidence, non-zero-day predictions as pseudo-labels. A one-class pseudo-label window is rejected for classifier retraining. Benign pseudo-labels may still be used to recalibrate the autoencoder threshold.

For authoritative model updates with labeled data, adapt.py should be used instead.

## How is the system evaluated?

The project reports accuracy, macro precision/recall/F1, zero-day detection rate, benign false-positive rate, drift count/latency, adaptation time, latency, CPU and memory usage.

## What is production-ready?

The repository now has a reproducible API/dashboard/container foundation and CI verification. A production SOC deployment still requires infrastructure-level authentication, TLS, rate limiting, monitoring, alert routing and a real packet-to-flow feature extraction layer.

## Resume-safe description

**Built SA-ZD-NIDS, a hybrid network intrusion detection research system combining supervised attack classification, benign-trained autoencoder anomaly detection, ADWIN concept-drift monitoring, streaming adaptation, FastAPI serving and a Streamlit monitoring dashboard.**

Do not quote fixed benchmark percentages unless they are backed by the exact experiment artifacts you can reproduce.
