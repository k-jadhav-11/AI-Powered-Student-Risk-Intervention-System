# tests/test_ai_assistant/test_chains.py
"""ai_assistant/chains.py -- the explanation chain's offline fallback path. Tested
WITHOUT a configured LLM provider on purpose: the whole point of the
template fallback is that a real, grounded answer exists even when no LLM
is reachable, so that's exactly the condition to test under."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai_assistant import chains, tools


def test_explain_risk_chain_falls_back_to_grounded_template(monkeypatch):
    """With no LLM configured, the chain must still produce a real answer,
    and that answer must contain the exact risk score -- never a
    placeholder, never silently empty."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    result = tools.get_risk_explanation("uci", "0")
    answer = chains.explain_risk_chain("Why is this student at risk?", result)
    assert answer["source"] == "template"
    assert "0.983" in answer["answer"]
    assert (result['academic_factors'] + result['engagement_factors'] + result['contextual_factors'])[0]['explanation'] in answer["answer"]


def test_explain_cohort_chain_falls_back_to_grounded_template(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    result = tools.get_cohort_summary("oulad", risk_level="High", limit=3)
    answer = chains.explain_cohort_chain("Which students are highest risk?", result)
    assert answer["source"] == "template"
    for sid in result["top_student_ids"]:
        assert sid in answer["answer"]


def test_chain_result_shape_is_consistent():
    """Both chain functions must always return the same {answer, source,
    provider} shape, whether or not an LLM was used -- the Streamlit pages
    depend on that consistency to render the source badge."""
    result = tools.get_risk_explanation("uci", "0")
    answer = chains.explain_risk_chain("test question", result)
    assert set(answer.keys()) == {"answer", "source", "provider"}
    assert answer["source"] in ("llm", "template")


def test_explain_risk_chain_stream_yields_grounded_text_offline(monkeypatch):
    """Streaming variant, offline: must yield the same grounded facts as
    the non-streaming template, just as an iterator of chunks instead of
    one string."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    result = tools.get_risk_explanation("uci", "0")
    status = {}
    chunks = list(chains.explain_risk_chain_stream("Why is this student at risk?", result, status))
    full_answer = "".join(chunks)
    assert status["source"] == "template"
    assert "0.983" in full_answer
    assert (result['academic_factors'] + result['engagement_factors'] + result['contextual_factors'])[0]['explanation'] in full_answer


def test_explain_gru_timeseries_chain_falls_back_to_grounded_template(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    fake_context = {
        "student_id": "12345", "gru_risk_score": 0.8, "gru_risk_level": "High",
        "xgboost_risk_score": 0.75, "xgboost_risk_level": "High", "models_agree": True,
        "last_active_week": 20, "first_high_risk_week": 8, "actual_final_result": "Withdrawn",
    }
    answer = chains.explain_gru_timeseries_chain("How is this student doing?", fake_context)
    assert answer["source"] == "template"
    assert "12345" in answer["answer"]
    assert "0.800" in answer["answer"] or "0.8" in answer["answer"]
    assert "Withdrawn" in answer["answer"]


def test_streaming_status_dict_reports_template_when_unconfigured(monkeypatch):
    """The status dict, not a pre-check of is_configured(), is what the UI
    trusts for the source badge -- this is the regression test for that
    design (see chains.py's explain_risk_chain_stream docstring)."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    result = tools.get_risk_explanation("uci", "0")
    status = {}
    list(chains.explain_risk_chain_stream("test", result, status))
    assert status == {"source": "template"}
