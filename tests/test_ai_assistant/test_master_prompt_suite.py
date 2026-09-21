# tests/test_ai_assistant/test_master_prompt_suite.py
"""Master Prompt Section 28 -- "Tool-Calling Tests", using the exact
function names the doc specifies, for direct traceability between that
document's requirements and this codebase's test coverage.

test_function_calling_accuracy() and test_function_calling_declines() are
about whether a live LLM correctly chooses to call (or not call) a tool for
a given question -- that's fundamentally live-LLM-dependent and can't be
verified offline. They're defined here (not skipped/omitted) but marked
xfail-if-unconfigured with a clear reason, and actually run for real
against whatever LLM_PROVIDER is configured in .env when one is -- see
tests/test_ai_assistant/test_hallucination_live.py, which is where the
substantive version of this behavior is exercised in detail.

Every other test below runs for real, offline, right now, against the
actual Sprint 3 data and models.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai_assistant import llm_client, tools
from ai_assistant.schemas import RiskExplanation
from src.analytics.exceptions import StudentNotFoundError
from src.risk import model_gru


@pytest.mark.skipif(not llm_client.is_configured(), reason="needs a configured LLM provider (.env)")
def test_function_calling_accuracy():
    """Live: a student-specific question should result in the agent
    actually calling a tool (not answering from nowhere). See
    test_hallucination_live.py for the detailed live scenarios this
    exercises in full."""
    from ai_assistant import agent
    result = agent.run_agent("What is the risk score for student 5 in the UCI dataset?", {})
    assert result["tool_calls_made"], "expected at least one tool call for a student-specific question"


@pytest.mark.skipif(not llm_client.is_configured(), reason="needs a configured LLM provider (.env)")
def test_function_calling_declines():
    """Live: a general, non-student-specific question should NOT trigger a
    tool call -- this is the Master Prompt's Section 27 Test 5 scenario."""
    from ai_assistant import agent
    result = agent.run_agent("What is SHAP?", {})
    assert result["tool_calls_made"] == [], (
        f"expected no tool calls for a general question, got {result['tool_calls_made']}"
    )


def test_get_risk_known_student():
    result = tools.get_student_risk("uci", "0")
    raw = pd.read_csv("data/processed/uci_risk_scores.csv")
    assert result["risk_score"] == pytest.approx(float(raw.loc[0, "risk_score"]))
    assert result["risk_level"] == raw.loc[0, "risk_level"]


def test_get_risk_unknown_student():
    with pytest.raises(StudentNotFoundError):
        tools.get_student_risk("uci", "999999999")


@pytest.mark.parametrize("dataset", ["uci", "oulad"])
def test_shap_top_factors_are_real_columns(dataset):
    real_columns = set(pd.read_csv(f"data/processed/{dataset}_feature_matrix.csv").columns) - {"at_risk"}
    result = tools.get_student_shap(dataset, "0")
    all_factors = result["academic_factors"] + result["engagement_factors"] + result["contextual_factors"]
    for f in all_factors:
        assert f["feature"] in real_columns


def test_gru_explanation_known_student():
    directory = model_gru.load_oulad_student_directory(sample_per_outcome=1)
    sid = int(directory.iloc[0]["id_student"])
    result = tools.get_student_gru_explanation(sid)
    assert result["gru_risk_level"] in ("Low", "Medium", "High")
    assert len(result["temporal_factors"]) > 0


def test_no_hallucinated_risk_score():
    raw = pd.read_csv("data/processed/oulad_risk_scores.csv")
    for idx in [0, 5, 100]:
        result = tools.get_student_risk("oulad", str(idx))
        assert result["risk_score"] == pytest.approx(float(raw.loc[idx, "risk_score"]))


def test_no_hallucinated_shap_factor():
    """No SHAP factor tools.py returns is ever a name that isn't a real
    trained feature column -- checked across several students, not one."""
    real_columns = set(pd.read_csv("data/processed/uci_feature_matrix.csv").columns) - {"at_risk"}
    for idx in ["0", "5", "10", "20"]:
        result = tools.get_student_shap("uci", idx)
        all_factors = result["academic_factors"] + result["engagement_factors"] + result["contextual_factors"]
        for f in all_factors:
            assert f["feature"] in real_columns


def test_structured_output_validation():
    """Every field of a real get_risk_explanation() response validates
    against the full RiskExplanation schema, including the richer
    academic/engagement/contextual split and model_metrics."""
    result = tools.get_risk_explanation("uci", "0")
    validated = RiskExplanation(**result)
    assert 0.0 <= validated.risk_score <= 1.0
    assert validated.model_metrics.precision > 0
