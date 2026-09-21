# src/risk/tuning.py
"""
STUB -- hyperparameter tuning for the risk pipeline.

OULAD's XGBoost model was tuned with GridSearchCV -- see
models/risk_model_oulad_v1_summary.json's tuning_method and best_params
fields -- inline in notebooks/oulad_notebooks/model_training_oulad.ipynb.
UCI's CatBoost model is an untuned baseline (see
models/risk_model_uci_v1_summary.json's selected_model:
"CatBoost (baseline, untuned)"), so there was no tuning step to extract for
that dataset.

This file exists for structural parity with the target src/risk/ layout.
Fill it in if UCI gets a tuning pass in a future sprint, or if OULAD's
GridSearchCV step needs to be re-run outside the notebook.
"""
