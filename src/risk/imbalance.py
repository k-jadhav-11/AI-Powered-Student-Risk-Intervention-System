# src/risk/imbalance.py
"""
STUB -- class-imbalance handling for the risk pipeline.

Both datasets have meaningfully more not-at-risk than at-risk students, and
each training notebook (notebooks/{uci,oulad}_notebooks/model_training_*.ipynb)
handles that inline -- class_weights for CatBoost (see
models/risk_model_uci_v1_summary.json's model_params.class_weights), and
threshold selection off the precision-recall curve rather than a naive 0.5
cutoff (see risk_queries.RISK_THRESHOLD_UCI / RISK_THRESHOLD_OULAD, each
derived from its own model's PR curve, not hardcoded).

This file exists for structural parity with the target src/risk/ layout --
imbalance handling wasn't factored out into its own reusable module because
each dataset's approach is different enough (class_weights vs. threshold
tuning) that a shared abstraction wasn't worth building for two call sites.
"""
