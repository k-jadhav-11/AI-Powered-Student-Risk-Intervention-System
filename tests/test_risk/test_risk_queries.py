# tests/test_risk/test_risk_queries.py
"""src/risk/risk_queries.py -- model loading, prediction, and raw-CSV
access. Genuinely real tests (not stubs), exercised against the trained
models and processed CSVs already on disk."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.risk import risk_queries


@pytest.mark.parametrize("dataset", ["uci", "oulad"])
def test_load_model_returns_a_fitted_model(dataset):
    model = risk_queries.load_model(dataset)
    assert hasattr(model, "predict_proba")


@pytest.mark.parametrize("dataset", ["uci", "oulad"])
def test_get_model_class_name_matches_loaded_model(dataset):
    """This exists specifically because the OULAD dashboard used to show
    'GridSearchCV' (the tuning method) instead of the real model class --
    get_model_class_name() reads it off the loaded object instead of a
    possibly-inconsistent summary JSON field, precisely to prevent that."""
    model = risk_queries.load_model(dataset)
    assert risk_queries.get_model_class_name(dataset) == type(model).__name__


@pytest.mark.parametrize("dataset", ["uci", "oulad"])
def test_load_risk_scores_has_expected_columns(dataset):
    df = risk_queries.load_risk_scores(dataset)
    assert set(df.columns) >= {"risk_score", "predicted_label", "risk_level"}
    assert df["risk_score"].between(0, 1).all()


def test_predict_risk_matches_precomputed_scores_for_uci():
    """A fresh predict_risk() call on the training feature matrix should
    reproduce the same risk_score already saved in uci_risk_scores.csv."""
    model = risk_queries.load_model("uci")
    feature_matrix = risk_queries.load_feature_matrix("uci")
    row = feature_matrix.iloc[[0]][risk_queries.UCI_FEATURE_COLUMNS]
    fresh = risk_queries.predict_risk(model, row, risk_queries.RISK_THRESHOLD_UCI)
    precomputed = risk_queries.load_risk_scores("uci").iloc[0]
    assert fresh["risk_score"].iloc[0] == pytest.approx(precomputed["risk_score"], abs=1e-4)


def test_build_single_record_enforces_column_order():
    record = {col: 0 for col in risk_queries.UCI_FEATURE_COLUMNS}
    df = risk_queries.build_single_record(record, risk_queries.UCI_FEATURE_COLUMNS)
    assert list(df.columns) == risk_queries.UCI_FEATURE_COLUMNS


def test_build_single_record_raises_on_missing_field():
    incomplete = {col: 0 for col in risk_queries.UCI_FEATURE_COLUMNS[:-1]}  # drop the last one
    with pytest.raises(ValueError):
        risk_queries.build_single_record(incomplete, risk_queries.UCI_FEATURE_COLUMNS)
