import pandas as pd

from backend.routers import agent as agent_router
from backend.schemas import BatchInterventionRequest


def test_batch_filters_then_returns_the_requested_offset(monkeypatch):
    risk_data = pd.DataFrame(
        {"risk_level": ["High", "Low", "High", "High", "Low", "High", "High", "High", "Low"]},
        index=range(9),
    )
    monkeypatch.setattr(agent_router.risk_queries, "load_risk_scores", lambda _dataset: risk_data)
    monkeypatch.setattr(
        agent_router.ai_tools,
        "resolve_student_id_from_row_index",
        lambda _dataset, row_index: str(row_index),
    )
    monkeypatch.setattr(
        agent_router.ai_tools,
        "get_risk_explanation",
        lambda dataset, row_index: {
            "dataset": dataset,
            "risk_level": "High",
            "risk_score": 0.9,
            "display_student_id": f"UCI-{row_index}",
        },
    )
    monkeypatch.setattr(
        agent_router.intervention_graph,
        "run_intervention",
        lambda dataset, student_id, *_args, **_kwargs: {
            "student_id": student_id,
            "display_student_id": f"UCI-{student_id}",
            "row_index": int(student_id),
            "lookup_by": "batch",
            "dataset": dataset,
            "risk_level": "High",
            "decision": "escalate",
            "rationale": "Review required.",
            "drafted_message": None,
            "resources_used": [],
            "escalation_reason": "Borderline confidence.",
            "mentor_brief": "Student briefing.",
            "router_confidence": 0.2,
            "agent_trace": ["escalate_node"],
            "source": "template",
        },
    )
    monkeypatch.setattr(agent_router.intervention_audit, "log_decision", lambda *_args, **_kwargs: {})

    response = agent_router.run_batch(
        BatchInterventionRequest(dataset="uci", risk_level="High", limit=2, offset=2)
    )

    assert response.total_students == 6
    assert response.offset == 2
    assert response.limit == 2
    assert [result.student_id for result in response.results] == ["4", "6"]
    assert response.results[0].mentor_brief == "Student briefing."
    assert response.results[0].display_student_id == "UCI-4"
