# tests/test_ai_assistant/test_granular_tools.py
"""Individual tests for the granular Sprint 4 tools (Master Prompt Section
10-16) -- get_student_risk, get_student_shap, get_model_information,
search_interventions, get_student_engagement, get_student_gru_explanation.
No LLM/network needed; verified against real Sprint 3 data."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai_assistant import tools
from src.analytics.exceptions import StudentNotFoundError
from src.risk import model_gru, risk_queries


def test_get_student_risk_matches_real_csv():
    result = tools.get_student_risk("uci", "1")
    raw = pd.read_csv("data/processed/uci_risk_scores.csv")
    assert result["risk_score"] == pytest.approx(float(raw.loc[0, "risk_score"]))
    assert result["risk_level"] == raw.loc[0, "risk_level"]
    assert "model_name" in result and result["model_name"]


def test_get_student_risk_unknown_id_raises():
    with pytest.raises(StudentNotFoundError):
        tools.get_student_risk("uci", "999999999")


def test_get_student_shap_returns_only_real_columns():
    real_columns = set(pd.read_csv("data/processed/oulad_feature_matrix.csv").columns) - {"at_risk"}
    result = tools.get_student_shap("oulad", "1")
    all_factors = result["academic_factors"] + result["engagement_factors"] + result["contextual_factors"]
    assert len(all_factors) > 0
    for f in all_factors:
        assert f["feature"] in real_columns


def test_get_student_shap_no_risk_score_leaks_in():
    """get_student_shap is supposed to be surgical -- no risk_score/level
    field should appear in its output (that's get_student_risk's job)."""
    result = tools.get_student_shap("uci", "1")
    assert "risk_score" not in result
    assert "risk_level" not in result


def test_get_model_information_matches_configured_threshold_metrics():
    performance = risk_queries.get_model_performance("uci")
    result = tools.get_model_information("uci")
    assert result["model_metrics"]["precision"] == performance["precision"]
    assert result["model_metrics"]["recall"] == performance["recall"]


def test_get_model_information_bad_dataset_raises():
    with pytest.raises(ValueError):
        tools.get_model_information("not_a_real_dataset")


def test_search_interventions_returns_real_source_ids():
    from ai_assistant import vectorstore
    real_ids = {item["id"] for item in vectorstore.STARTER_REMEDIAL_CONTENT}
    result = tools.search_interventions(["failures", "absences"], top_k=2)
    assert len(result["interventions"]) > 0
    for item in result["interventions"]:
        assert item["source_id"] in real_ids


def test_get_student_engagement_real_oulad_student():
    directory = model_gru.load_oulad_student_directory(sample_per_outcome=1)
    sid = int(directory.iloc[0]["id_student"])
    result = tools.get_student_engagement(sid)
    assert result["num_weeks_active"] > 0
    assert result["engagement_trend_direction"] in ("rising", "falling")


def test_get_student_engagement_zero_activity_raises():
    with pytest.raises(StudentNotFoundError):
        tools.get_student_engagement(999999999)


def test_get_student_gru_explanation_temporal_factors_reference_real_features():
    directory = model_gru.load_oulad_student_directory(sample_per_outcome=1)
    sid = int(directory.iloc[0]["id_student"])
    result = tools.get_student_gru_explanation(sid)
    assert result["gru_risk_level"] in ("Low", "Medium", "High")
    for tf in result["temporal_factors"]:
        assert tf["feature"] in ("total_clicks", "days_active")
        assert 0 <= tf["week"] < 40


def test_composed_get_risk_explanation_matches_granular_calls():
    """get_risk_explanation() must produce numbers identical to calling the
    granular tools separately -- it's supposed to be pure composition."""
    risk = tools.get_student_risk("uci", "5")
    full = tools.get_risk_explanation("uci", "5")
    assert full["risk_score"] == risk["risk_score"]
    assert full["risk_level"] == risk["risk_level"]
    assert full["model_name"] == risk["model_name"]
