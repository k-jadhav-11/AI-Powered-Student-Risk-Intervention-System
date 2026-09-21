# src/risk/feature_engineering.py
"""
STUB -- feature engineering for the risk pipeline.

The real feature-engineering logic (ordinal encodings, engagement-trend
computation, one-hot encoding, the exact column order risk_queries.py's
UCI_FEATURE_COLUMNS / OULAD_FEATURE_COLUMNS depend on) lives in:
- notebooks/uci_notebooks/feature_eng_uci.ipynb
- notebooks/oulad_notebooks/feature_engineering_oulad.ipynb
and produces data/processed/{dataset}_feature_matrix.csv, which
risk_queries.load_feature_matrix() reads.

Partial exception: the OULAD ordinal-encoding constants and
_imd_band_to_numeric() helper WERE promoted into model_gru.py, because
build_oulad_record_from_raw() (used by the live Student Time-Series Risk
Explorer and by evaluation.run_live_sampled_comparison()) needs to
reproduce that exact encoding at inference time, on demand, for a single
real student -- not just once, offline, for the whole training set. That
logic intentionally lives with model_gru.py rather than being duplicated
here.

This file exists for structural parity with the target src/risk/ layout.
"""
