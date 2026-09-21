# tests/test_explainability.py
"""src/explainability.py -- SHAP-based per-student factor explanations. No
LLM/network needed; this only exercises the trained CatBoost/XGBoost models
already on disk."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import explainability
from src.risk import risk_queries


@pytest.mark.parametrize("dataset", ["uci", "oulad"])
def test_shap_top_factors_are_real_columns(dataset):
    """Every factor SHAP returns must be a real column in that dataset's
    feature matrix -- never a name the explainer invented."""
    real_columns = set(pd.read_csv(f"data/processed/{dataset}_feature_matrix.csv").columns) - {"at_risk"}
    feature_cols = risk_queries.UCI_FEATURE_COLUMNS if dataset == "uci" else risk_queries.OULAD_FEATURE_COLUMNS
    feature_matrix = risk_queries.load_feature_matrix(dataset)
    row = feature_matrix.iloc[[0]][feature_cols]
    result = explainability.explain_prediction(dataset, row, top_n=5)
    for factor in result["top_factors"]:
        assert factor in real_columns, f"{factor!r} is not a real {dataset} feature column"


def test_explain_prediction_returns_requested_top_n():
    feature_matrix = risk_queries.load_feature_matrix("uci")
    row = feature_matrix.iloc[[0]][risk_queries.UCI_FEATURE_COLUMNS]
    result = explainability.explain_prediction("uci", row, top_n=3)
    assert len(result["top_factors"]) == 3
    assert len(result["factor_directions"]) == 3
    assert len(result["factor_values"]) == 3


def test_explain_prediction_directions_are_valid_labels():
    feature_matrix = risk_queries.load_feature_matrix("oulad")
    row = feature_matrix.iloc[[0]][risk_queries.OULAD_FEATURE_COLUMNS]
    result = explainability.explain_prediction("oulad", row, top_n=5)
    assert all(d in ("increases risk", "decreases risk") for d in result["factor_directions"])


def test_explain_prediction_rejects_multi_row_input():
    """explain_prediction() is documented to require exactly one row --
    verify it actually enforces that rather than silently explaining only
    the first row of a larger batch."""
    feature_matrix = risk_queries.load_feature_matrix("uci")
    rows = feature_matrix.iloc[0:3][risk_queries.UCI_FEATURE_COLUMNS]
    with pytest.raises(ValueError):
        explainability.explain_prediction("uci", rows)


def test_explainer_is_cached_across_calls():
    """The second explain_prediction() call for the same dataset should
    reuse the cached TreeExplainer rather than rebuilding it -- this is
    what keeps per-student explanation fast enough for a chat UI."""
    feature_matrix = risk_queries.load_feature_matrix("oulad")
    row = feature_matrix.iloc[[1]][risk_queries.OULAD_FEATURE_COLUMNS]
    explainability.explain_prediction("oulad", row)  # warms the cache
    assert "oulad" in explainability._EXPLAINER_CACHE
