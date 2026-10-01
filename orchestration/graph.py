# orchestration/graph.py
"""
LangGraph Intervention Agent for student intervention decisions.

Routes students to nudge, resource, or escalate branches using deterministic
risk signals, router confidence, and optional model-agreement checks.
"""
from __future__ import annotations

from typing import Literal, TypedDict

from langgraph.graph import StateGraph
from langsmith import traceable

from ai_assistant import rerank, vectorstore
from ai_assistant.schemas import InterventionDecision
from src.risk import risk_queries

import logging

from . import briefing, crew

logger = logging.getLogger(__name__)

# ============================================================
# Router Configuration
# ============================================================

# Confidence below this threshold is treated as borderline.
CONFIDENCE_FLOOR = 0.15

Branch = Literal["nudge_node", "resource_node", "escalate_node"]

# Knowledge-base entries written as advice a STUDENT can act on themselves.
# Every other entry in vectorstore.STARTER_REMEDIAL_CONTENT is guidance for
# staff (counselor referrals, family/health/alcohol notes, persona notes,
# "prioritize this student for a mentor"...) and must never be sent to a
# student. Staff guidance still reaches the human mentor through the
# RiskExplanation's mitigation_strategies and the escalation briefing.
# New KB entries default to staff-only until added here -- the safe default.
STUDENT_FACING_RESOURCE_IDS = frozenset(
    {"rem_001", "rem_002", "rem_003", "rem_004", "rem_006", "rem_007"}
)


def _staff_only_resource_ids() -> set[str]:
    """IDs of every knowledge-base entry that is not student-facing."""
    return {item["id"] for item in vectorstore.STARTER_REMEDIAL_CONTENT} - STUDENT_FACING_RESOURCE_IDS

# ============================================================
# Intervention State
# ============================================================
class InterventionState(TypedDict, total=False):
    """Store inputs, routing results, and intervention outputs."""

    dataset: str
    student_id: str
    risk_explanation: dict  # ai_assistant.schemas.RiskExplanation.model_dump()

    oulad_id_student: int | None
    models_agree: bool  # absent entirely unless a real OULAD id_student was supplied

    # populated as the graph runs
    router_confidence: float
    decision: Branch | None
    rationale: str
    escalation_reason: str | None
    drafted_message: str | None
    resources_used: list[str]
    source: Literal["crew", "template"]
    agent_trace: list[str]
    mentor_brief: str | None  # escalations only

# ============================================================
# Router Confidence
# ============================================================

def compute_router_confidence(risk_score: float, dataset: str) -> float:
    """
    Calculate router confidence from distance to the dataset threshold.
    Returns a normalized value from 0 to 1, not the model probability.
    """
    threshold = (
        risk_queries.RISK_THRESHOLD_UCI
        if dataset == "uci"
        else risk_queries.RISK_THRESHOLD_OULAD
    )
    if risk_score < threshold:
        return (
            (threshold - risk_score) / threshold
            if threshold > 0
            else 0.0
        )

    return (
        (risk_score - threshold) / (1 - threshold)
        if threshold < 1
        else 1.0
    )


def _confidence_rationale(risk_score: float, dataset: str, confidence: float) -> str:
    """Create a plain-language explanation of router confidence."""
    threshold = risk_queries.RISK_THRESHOLD_UCI if dataset == "uci" else risk_queries.RISK_THRESHOLD_OULAD
    
    position = (
        "close to the model's own decision boundary"
        if confidence < CONFIDENCE_FLOOR
        else "reasonably far from the model's own decision boundary"
    )
    
    return (
        f"risk_score={risk_score:.3f} vs. this dataset's decision "
        f"threshold={threshold:.3f} "
        f"(router_confidence={confidence:.3f}); this student's score "
        f"sits {position}."
    )

# ============================================================
# Compounding Risk Detection
# ============================================================

def _has_compounding_risk(r: dict) -> bool:
    """Check whether risk-increasing factors span at least two factor groups."""

    buckets_with_increasing_risk = sum(
        1 for bucket in ("academic_factors", "engagement_factors", "contextual_factors")
        if any(f["direction"] == "increases_risk" for f in (r.get(bucket) or []))
    )
    return buckets_with_increasing_risk >= 2

# ============================================================
# Severity Classification
# ============================================================

def classify_severity(state: InterventionState) -> Branch:
    """
    Select the intervention branch using conservative routing rules.
    Disagreement and borderline confidence are escalated first.
    """

    r = state["risk_explanation"]

    # 1. Escalate when GRU and XGBoost disagree.
    if state.get("models_agree") is False:
        return "escalate_node"

    # 2. Escalate when the risk score is too close to the threshold.
    confidence = compute_router_confidence(r["risk_score"], r["dataset"])
    if confidence < CONFIDENCE_FLOOR:
        return "escalate_node"

    # 3. Escalate high-risk students with compounding risk.
    if r["risk_level"] == "High" and _has_compounding_risk(r):
        return "escalate_node"
    # 4. High risk receives contextual resources.
    if r["risk_level"] == "High":
        return "resource_node"
    # 5. Lower-risk cases receive a nudge.
    return "nudge_node"

# ============================================================
# Classification Node
# ============================================================

@traceable(name="classify", run_type="chain")
def _classify_and_record(state: InterventionState) -> InterventionState:
    """Classify the student and record routing details in the graph state."""
    r = state["risk_explanation"]
    confidence = compute_router_confidence(r["risk_score"], r["dataset"])
    branch = classify_severity(state)

    trace = list(state.get("agent_trace") or []) + ["classify"]
    escalation_reason = None
    if branch == "escalate_node":
        if state.get("models_agree") is False:
            escalation_reason = "GRU/XGBoost disagree on this student's risk"
        elif confidence < CONFIDENCE_FLOOR:
            escalation_reason = "borderline router confidence (risk_score close to the decision threshold)"
        else:
            escalation_reason = "compounding risk across multiple factor categories (academic + engagement + contextual)"

    return {
        **state,
        "router_confidence": confidence,
        "decision": branch,
        "rationale": _confidence_rationale(r["risk_score"], r["dataset"], confidence)
        + f" Persona: {r.get('persona_name', 'unknown')}.",
        "escalation_reason": escalation_reason,
        "agent_trace": trace,
    }

# ============================================================
# Nudge Node
# ============================================================

@traceable(name="nudge_node", run_type="chain")
def nudge_node(state: InterventionState) -> InterventionState:
    """Create a short student nudge using the CrewAI drafting layer."""

    r = state["risk_explanation"]
    draft = crew.draft_intervention(r, branch="nudge", resources=None)
    trace = list(state.get("agent_trace") or []) + ["nudge_node"]
    return {
        **state,
        "drafted_message": draft["message"],
        "resources_used": [],
        "source": draft["source"],
        "agent_trace": trace,
    }

# ============================================================
# Resource Node
# ============================================================

@traceable(name="resource_node", run_type="chain")
def resource_node(state: InterventionState) -> InterventionState:
    """
    Select relevant remedial resources and create a contextualized message.
    Uses exact factor matching first and semantic reranking when needed.
    """
    r = state["risk_explanation"]
    all_factors = r["academic_factors"] + r["engagement_factors"] + r["contextual_factors"]
    top_factors = [f["feature"] for f in all_factors]

    # Use SHAP magnitude to prioritize stronger factors when
    # multiple resources match the same number of factors.
    factor_weights = {f["feature"]: abs(f["contribution"]) for f in all_factors}

    # Only student-facing entries may be sent to a student (see
    # STUDENT_FACING_RESOURCE_IDS); staff-directed entries are excluded from
    # both the exact-match step and the semantic re-ranking below.
    staff_only_ids = _staff_only_resource_ids()

    # Prefer resources with exact factor-tag matches.
    ranked_matches = rerank.rank_exact_matches(
        top_factors, exclude_ids=staff_only_ids, weights=factor_weights
    )
    if len(ranked_matches) >= 2:
        top_two = ranked_matches[:2]
        resources = [item["text"] for item in top_two]
        source_ids = [item["id"] for item in top_two]
        match_types = ["exact match"] * 2
    else:
        # Fill missing resource slots using semantic reranking.
        exact_ids = {
            item["id"]
            for item in ranked_matches
        }

        remaining_slots = 2 - len(ranked_matches)

        reranked = rerank.rerank_remaining_after_tag_match(
            top_factors,
            exact_ids | staff_only_ids,
            top_k=remaining_slots,
        )

        resources = (
            [item["text"] for item in ranked_matches]
            + reranked["suggestions"]
        )

        source_ids = (
            [item["id"] for item in ranked_matches]
            + reranked["source_ids"]
        )

        match_types = (
            ["exact match"] * len(ranked_matches)
            + ["re-ranked"] * len(reranked["suggestions"])
        )

    draft = crew.draft_intervention(r, branch="resource", resources=resources)
    trace = list(state.get("agent_trace") or []) + ["resource_node"]
    # Preserve match type in the existing resources_used field.
    resources_used = [
        f"{text} (source: {source_id}, {match_type})"
        for text, source_id, match_type in zip(
            resources,
            source_ids,
            match_types,
        )
    ]
    
    return {
        **state,
        "drafted_message": draft["message"],
        "resources_used": resources_used,
        "source": draft["source"],
        "agent_trace": trace,
    }

# ============================================================
# Escalation Node
# ============================================================

@traceable(name="escalate_node", run_type="chain")
def escalate_node(state: InterventionState) -> InterventionState:
    """Route the case to human review. No message goes to the student, but a
    deterministic briefing (factors, persona, guidance) is prepared for the mentor."""

    trace = list(state.get("agent_trace") or []) + ["escalate_node"]
    try:
        mentor_brief = briefing.build_mentor_brief(
            state["risk_explanation"],
            state.get("escalation_reason"),
            state.get("router_confidence"),
        )
    except Exception:  # the escalation itself must never fail because the brief did
        logger.exception("Could not build mentor briefing")
        mentor_brief = None
    return {
        **state,
        "mentor_brief": mentor_brief,
        "drafted_message": None,
        "resources_used": [],
        "source": "template",
        "agent_trace": trace,
    }

# ============================================================
# Build Graph
# ============================================================

def build_graph():
    """Build and return the uncompiled intervention StateGraph."""
    graph = StateGraph(InterventionState)

    graph.add_node("classify", _classify_and_record)
    graph.add_node("nudge_node", nudge_node)
    graph.add_node("resource_node", resource_node)
    graph.add_node("escalate_node", escalate_node)

    graph.add_conditional_edges(
        "classify",
        lambda state: state["decision"],
        {"nudge_node": "nudge_node", "resource_node": "resource_node", "escalate_node": "escalate_node"},
    )
    graph.set_entry_point("classify")
    graph.set_finish_point("nudge_node")
    graph.set_finish_point("resource_node")
    graph.set_finish_point("escalate_node")

    return graph

# ============================================================
# Compiled Graph Cache
# ============================================================
_COMPILED_GRAPH = None


def _get_compiled_graph():
    """Process-wide compiled-graph cache, same pattern as
    ai_assistant.rag.build_retriever()'s _RETRIEVER_CACHE -- LangGraph's
    .compile() step has real (if small) fixed cost, no reason to pay it on
    every run_intervention() call."""
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        _COMPILED_GRAPH = build_graph().compile()
    return _COMPILED_GRAPH

# ============================================================
# Run Intervention Agent
# ============================================================
@traceable(name="Intervention Agent", run_type="chain")
def run_intervention(
    dataset: str,
    student_id: str,
    risk_explanation: dict,
    oulad_id_student: int | None = None,
    models_agree: bool | None = None,
) -> dict:
    """
    Run the LangGraph Intervention Agent for one student.
    Supports row-index and real-student-ID lookup modes.
    Returns the resolved student ID, row index, and lookup method.
    """

    if models_agree is not None and oulad_id_student is None:
        raise ValueError(
            "models_agree was supplied without oulad_id_student -- "
            "the disagreement check requires the real OULAD id_student."
        )

    initial_state: InterventionState = {
        "dataset": dataset,
        "student_id": student_id,
        "risk_explanation": risk_explanation,
        "oulad_id_student": oulad_id_student,
        "agent_trace": [],
    }
    if models_agree is not None:
        initial_state["models_agree"] = models_agree

    final_state = _get_compiled_graph().invoke(initial_state)

    # Convert graph node name to the public decision value.
    branch_label = final_state["decision"].removesuffix("_node")

    # Build the API/audit response from the final graph state.
    decision = InterventionDecision(
        student_id=student_id,
        
        # Pass through the display ID supplied by the risk explanation.
        # ID resolution remains outside this graph.
        display_student_id=risk_explanation.get("display_student_id"),
        dataset=dataset,
        risk_level=risk_explanation["risk_level"],
        decision=branch_label,
        rationale=final_state["rationale"],
        drafted_message=final_state.get("drafted_message"),
        resources_used=final_state.get("resources_used") or [],
        escalation_reason=final_state.get("escalation_reason"),
        router_confidence=final_state["router_confidence"],
        oulad_id_student=oulad_id_student,
        agent_trace=final_state["agent_trace"],
        source=final_state.get("source", "template"),
        mentor_brief=final_state.get("mentor_brief"),
    )
    return decision.model_dump()