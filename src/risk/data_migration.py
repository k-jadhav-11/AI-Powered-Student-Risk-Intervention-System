# src/risk/data_migration.py
"""
STUB -- data migration / ingestion for the risk pipeline.

The logic this module name implies (loading raw UCI/OULAD CSVs into
data/raw/, or into a database, ahead of feature engineering) currently
lives only as manual steps at the top of the EDA notebooks
(notebooks/uci_notebooks/uci_eda.ipynb, notebooks/oulad_notebooks/eda_oulad.ipynb)
-- it was never promoted into an importable module, since Sprint 3 never
needed to run it more than once per dataset.

This file exists for structural parity with the target src/risk/ layout.
Fill it in if/when the ingestion step needs to be re-run programmatically
(e.g. a new OULAD export, or automating retraining per the Sprint 3 doc's
"Limitations & Future Work" section) rather than by hand.
"""
