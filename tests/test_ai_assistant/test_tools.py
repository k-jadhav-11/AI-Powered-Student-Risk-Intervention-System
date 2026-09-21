# tests/test_ai_assistant/test_tools.py
"""get_risk_explanation() / get_cohort_summary() -- the grounding layer's
hallucination guards and exact-match checks against real Sprint 3 output.
No LLM/network needed."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai_assistant import tools
from src.analytics.exceptions import StudentNotFoundError


def test_get_risk_explanation_known_uci_student():
    """Matches data/processed/uci_risk_scores.csv exactly for a known row."""
    result = tools.get_risk_explanation("uci", "0")
    assert result["risk_score"] == 0.9827
    assert result["risk_level"] == "High"
    assert result["predicted_label"] == "at_risk"
    assert result["dataset"] == "uci"


def test_get_risk_explanation_known_oulad_student():
    """Matches data/processed/oulad_risk_scores.csv exactly for a known row."""
    result = tools.get_risk_explanation("oulad", "1")
    assert result["risk_score"] == 0.9902
    assert result["risk_level"] == "High"
    assert result["predicted_label"] == "at_risk"
    assert result["dataset"] == "oulad"


@pytest.mark.parametrize("dataset", ["uci", "oulad"])
def test_get_risk_explanation_unknown_student_raises(dataset):
    """Raises StudentNotFoundError, doesn't hallucinate a fallback answer."""
    with pytest.raises(StudentNotFoundError):
        tools.get_risk_explanation(dataset, "999999999")


def test_get_risk_explanation_non_numeric_id_raises():
    with pytest.raises(StudentNotFoundError):
        tools.get_risk_explanation("uci", "not_a_number")


def test_get_risk_explanation_bad_dataset_raises():
    with pytest.raises(ValueError):
        tools.get_risk_explanation("not_a_real_dataset", "0")


def test_no_hallucinated_risk_score():
    """The score tools.py returns always matches the CSV -- never invented,
    never recomputed by this layer."""
    raw = pd.read_csv("data/processed/oulad_risk_scores.csv")
    for idx in [0, 5, 100, 1000]:
        result = tools.get_risk_explanation("oulad", str(idx))
        assert result["risk_score"] == pytest.approx(float(raw.loc[idx, "risk_score"]))
        assert result["risk_level"] == raw.loc[idx, "risk_level"]


def test_get_cohort_summary_counts_match_real_data():
    raw = pd.read_csv("data/processed/oulad_risk_scores.csv")
    result = tools.get_cohort_summary("oulad")
    assert result["total_students"] == len(raw)
    assert result["risk_level_counts"] == raw["risk_level"].value_counts().to_dict()


def test_get_cohort_summary_filter_only_returns_that_level():
    result = tools.get_cohort_summary("oulad", risk_level="High", limit=20)
    raw = pd.read_csv("data/processed/oulad_risk_scores.csv")
    for sid in result["top_student_ids"]:
        assert raw.loc[int(sid), "risk_level"] == "High"
