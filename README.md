# SA-ZD-NIDS

A network intrusion detection system that learns to spot both known and unknown cyber attacks.

## What it does

I built this to solve three common problems with traditional intrusion detection:
- **Zero-day attacks** - New attacks that signature-based systems miss
- **Concept drift** - Attack patterns that change over time
- **Imbalanced data** - Very few attack samples compared to normal traffic

## How it works

The system uses a hybrid approach:
- **XGBoost** to detect known attack patterns
- **Autoencoder** to spot unusual traffic (zero-day attacks)
- **Drift detection** to automatically adapt when patterns change
- **Streaming engine** to process data in batches

## Key features

- Detects both known and unknown attacks
- Automatically adapts to new attack patterns
- Runs on GPU for faster training
- Shows which features matter most for decisions
- Includes a demo dashboard

## Tech I used

- **ML models**: XGBoost, PyTorch, Scikit-learn
- **Drift detection**: River library
- **Data tools**: Pandas, NumPy
- **Visualization**: Matplotlib, Plotly
- **Optimization**: Optuna for tuning
- **Demo**: FastAPI + Streamlit

## Getting started

```bash
# Install what you need
pip install -r requirements.txt

# Train the models
python train.py --config config.yaml

# Run the streaming demo
python stream.py --config config.yaml

# Try the experiments
python run_experiments.py --config config.yaml
```

## GUI Dashboard

The system includes an interactive web dashboard for real-time monitoring and analysis:

### Quick Start (Demo Mode)

```bash
# Start the mock API backend
python -m uvicorn mock_api:app --host 0.0.0.0 --port 8000

# In a new terminal, start the GUI frontend
python -m streamlit run deployment/frontend/app.py

# Access the dashboard at http://localhost:8501
```

### Full Setup (With Trained Models)

```bash
# First train the models
python train.py --config config.yaml

# Start the production API backend
python deployment/api/main.py

# In a new terminal, start the GUI frontend
python -m streamlit run deployment/frontend/app.py

# Access the dashboard at http://localhost:8501
```

### GUI Features

- **Demo Interface** - Interactive network traffic analysis with manual input
- **System Monitoring** - Real-time performance metrics and health checks
- **Drift Analysis** - Concept drift visualization and statistics
- **Batch Prediction** - Upload CSV files for bulk analysis

## Configuration

The main settings are in `config.yaml`:
- Which classifier to use and its parameters
- Autoencoder architecture
- When to trigger drift detection
- Batch size for streaming

## How well it works

On my test data:
- About 94% accuracy overall
- Catches ~87% of zero-day attacks
- False positive rate around 2.3%
- Predictions take less than 1ms

## Project structure

```
sa-zd-nids/
|-- src/sa_zd_nids/          # Main code
|-- deployment/              # API and dashboard
|-- configs/                 # Configuration files
|-- data/                    # Training data (gitignored)
|-- notebooks/               # Jupyter experiments
|-- tests/                   # Unit tests
|-- scripts/                 # Helper scripts
|-- train.py                 # Train models
|-- stream.py                # Run streaming demo
|-- run_experiments.py       # Run all experiments
|-- README.md                # This file
|-- UPGRADE.md               # Advanced features
|-- INTERVIEW_PREP.md        # Interview questions
|-- PROJECT_STATUS.md        # Current status
```

## More docs

- **UPGRADE.md** - Advanced features and deployment setup
- **INTERVIEW_PREP.md** - Questions I get asked in interviews
- **PROJECT_STATUS.md** - What's done and what's next

## Citation

If you use this in your work:

```bibtex
@software{sa_zd_nids,
  title={SA-ZD-NIDS: Self-Adaptive Zero-Day Aware Network Intrusion Detection System},
  author={Mehul Gupta},
  year={2024},
  url={https://github.com/yourusername/sa-zd-nids}
}
```

---

**Built by Mehul Gupta**
