# tests/test_ai_assistant/test_schemas.py
"""Pydantic schema validation for ai_assistant/schemas.py. No LLM/network needed."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai_assistant import tools
from ai_assistant.schemas import CohortQuestion, RiskExplanation


def test_structured_output_schema_validation():
    """Every get_risk_explanation() response parses as RiskExplanation --
    tools.py already builds one internally, this re-validates the dict it
    returns to catch any drift between the two."""
    for dataset, idx in [("uci", "10"), ("oulad", "50")]:
        result = tools.get_risk_explanation(dataset, idx)
        validated = RiskExplanation(**result)  # raises if the shape is wrong
        assert validated.student_id == idx
        assert 0.0 <= validated.risk_score <= 1.0
        all_factors = validated.academic_factors + validated.engagement_factors + validated.contextual_factors
        assert len(all_factors) > 0  # SHAP always returns at least one top factor


def test_cohort_question_rejects_invalid_risk_level():
    with pytest.raises(Exception):  # pydantic.ValidationError
        CohortQuestion(dataset="uci", risk_level="Extremely High")


def test_cohort_question_defaults():
    q = CohortQuestion(dataset="oulad")
    assert q.risk_level is None
    assert q.limit == 10


def test_every_factor_has_real_grounded_fields():
    """Each RiskFactor in a real response must carry a real feature name,
    a real contribution, and a non-empty explanation -- never a partially
    populated factor."""
    result = tools.get_risk_explanation("uci", "0")
    validated = RiskExplanation(**result)
    for factor in validated.academic_factors + validated.engagement_factors + validated.contextual_factors:
        assert factor.feature
        assert factor.explanation
        assert factor.direction in ("increases_risk", "decreases_risk")
