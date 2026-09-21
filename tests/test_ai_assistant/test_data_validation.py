# tests/test_ai_assistant/test_data_validation.py
"""Master Prompt Section 29 -- "Data Validation Tests", each bullet as its
own test. No LLM/network needed -- these check that the grounding LAYER
(tools + underlying Sprint 3 data) is internally consistent, independent of
whether any LLM ever narrates it correctly.

Two of these (risk_level threshold consistency, GRU attribution dimension
match) were genuine gaps -- not covered by any earlier test file -- found
while auditing against this exact checklist, not assumed covered.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai_assistant import tools
from src.risk import model_gru, risk_queries


# "Every returned student ID exists."
def test_returned_student_id_exists_in_source_data():
    result = tools.get_student_risk("uci", "10")
    raw = risk_queries.load_risk_scores("uci")
    assert int(result["student_id"]) in raw.index


# "Every returned feature exists in the model's feature matrix."
@pytest.mark.parametrize("dataset", ["uci", "oulad"])
def test_returned_shap_features_exist_in_feature_matrix(dataset):
    real_columns = set(pd.read_csv(f"data/processed/{dataset}_feature_matrix.csv").columns) - {"at_risk"}
    result = tools.get_student_shap(dataset, "0")
    all_factors = result["academic_factors"] + result["engagement_factors"] + result["contextual_factors"]
    for f in all_factors:
        assert f["feature"] in real_columns


# "Every SHAP feature corresponds to an actual model input."
@pytest.mark.parametrize("dataset", ["uci", "oulad"])
def test_shap_features_are_actual_model_inputs(dataset):
    model_input_cols = set(risk_queries.UCI_FEATURE_COLUMNS if dataset == "uci" else risk_queries.OULAD_FEATURE_COLUMNS)
    result = tools.get_student_shap(dataset, "0")
    all_factors = result["academic_factors"] + result["engagement_factors"] + result["contextual_factors"]
    for f in all_factors:
        assert f["feature"] in model_input_cols


# "Risk score exactly matches the source prediction."
def test_risk_score_exactly_matches_source_csv():
    raw = pd.read_csv("data/processed/oulad_risk_scores.csv")
    for idx in [0, 50, 500]:
        result = tools.get_student_risk("oulad", str(idx))
        assert result["risk_score"] == pytest.approx(float(raw.loc[idx, "risk_score"]))


# "Risk level exactly matches the existing threshold logic."
@pytest.mark.parametrize("dataset", ["uci", "oulad"])
def test_risk_level_matches_threshold_buckets(dataset):
    """risk_level must be Low/Medium/High exactly per the 0.3/0.6 bucket
    boundaries risk_queries.predict_risk()'s bucket() helper uses
    (p < 0.3 -> Low, p < 0.6 -> Medium, else High -- left-inclusive, so a
    score of exactly 0.3 is Medium, not Low).

    One known, harmless exception, found while writing this test (not
    hidden): risk_score is rounded to 4 decimals before being written to
    data/processed/{dataset}_risk_scores.csv (see predict_risk()'s
    .round(4)), but risk_level was computed from the unrounded score at
    generation time. For a score landing extremely close to a 0.3/0.6
    boundary, the rounded-then-reloaded value can fall on the opposite
    side of that boundary from the original computation -- a floating-point
    storage artifact, not a logic bug, and it can only ever affect a value
    within float64 precision of exactly 0.3 or 0.6. This test allows
    mismatches ONLY when the stored score is that close to a boundary, and
    fails on anything else.
    """
    raw = pd.read_csv(f"data/processed/{dataset}_risk_scores.csv")
    expected_level = pd.cut(
        raw["risk_score"], bins=[-0.01, 0.3, 0.6, 1.01], labels=["Low", "Medium", "High"], right=False,
    )
    mismatches = raw[raw["risk_level"] != expected_level.astype(str)]

    boundary_eps = 1e-9
    non_boundary_mismatches = mismatches[
        (mismatches["risk_score"] - 0.3).abs().gt(boundary_eps)
        & (mismatches["risk_score"] - 0.6).abs().gt(boundary_eps)
    ]
    assert len(non_boundary_mismatches) == 0, (
        f"{len(non_boundary_mismatches)} rows have a risk_level mismatch NOT explained by "
        f"the known float64 boundary-rounding artifact: {non_boundary_mismatches}"
    )


# "Model name matches the actual loaded model."
@pytest.mark.parametrize("dataset", ["uci", "oulad"])
def test_model_name_matches_loaded_model_class(dataset):
    model = risk_queries.load_model(dataset)
    result = tools.get_model_information(dataset)
    assert result["model_name"] == type(model).__name__ or result["model_name"] == risk_queries.get_model_class_name(dataset)


# "Model metrics come from the existing summary JSON."
@pytest.mark.parametrize("dataset", ["uci", "oulad"])
def test_model_metrics_come_from_real_summary_json(dataset):
    import json
    with open(f"models/risk_model_{dataset}_v1_summary.json") as f:
        summary = json.load(f)
    result = tools.get_model_information(dataset)
    assert result["model_metrics"]["precision"] == summary["test_precision_at_risk"]
    assert result["model_metrics"]["recall"] == summary["test_recall_at_risk"]
    assert result["model_metrics"]["f1"] == summary["test_f1_at_risk"]


# "GRU attribution dimensions match the sequence dimensions."
def test_gru_attribution_dimensions_match_sequence():
    """Every temporal_factor's week must be within [0, max_weeks), and
    every feature must be one of the GRU's only two real inputs -- an
    attribution outside the sequence's actual shape would mean the
    explainer is describing data that was never fed to the model."""
    _, gru_cfg = model_gru.load_gru_model("oulad")
    directory = model_gru.load_oulad_student_directory(sample_per_outcome=1)
    sid = int(directory.iloc[0]["id_student"])
    result = tools.get_student_gru_explanation(sid)
    for tf in result["temporal_factors"]:
        assert 0 <= tf["week"] < gru_cfg["max_weeks"]
        assert tf["feature"] in gru_cfg["feature_cols"]
