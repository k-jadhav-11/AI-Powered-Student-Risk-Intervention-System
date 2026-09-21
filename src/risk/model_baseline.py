# src/risk/model_baseline.py
"""
STUB -- baseline model training for the risk pipeline.

The actual training logic (CatBoost baseline for UCI, XGBoost + GridSearchCV
for OULAD, train/test split, the metrics that end up in
models/risk_model_{dataset}_v1_summary.json) lives in:
- notebooks/uci_notebooks/model_training_uci.ipynb
- notebooks/oulad_notebooks/model_training_oulad.ipynb
Both write their trained model to models/risk_model_{dataset}_v1.joblib,
which risk_queries.load_model() reads -- retrain by re-running the
notebook end-to-end, not by editing this file.

This file exists for structural parity with the target src/risk/ layout.
"""
