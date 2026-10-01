# backend/routers/agent.py
"""
Expose the Intervention Agent through FastAPI.
Handles student ID resolution consistently across run, batch, and audit endpoints.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from ai_assistant import agent as explain_agent
from ai_assistant import llm_client
from ai_assistant import memory as genai_memory
from ai_assistant import tools as ai_tools

from orchestration import audit as intervention_audit
from orchestration import graph as intervention_graph

from src.analytics.exceptions import (
    DataNotFoundError,
    StudentNotFoundError,
)
from src.risk import risk_queries

from ..schemas import (
    AuditLogEntry,
    AuditLogPage,
    BatchInterventionRequest,
    BatchInterventionResponse,
    ChatHistoryEntry,
    ChatHistoryPage,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    InterventionResponse,
    RunInterventionRequest,
)


router = APIRouter(
    prefix="/agent",
    tags=["Intervention Agent"],
)


# ============================================================
# Logger
# ============================================================

_terminal = logging.getLogger("uvicorn.error")


# ============================================================
# Health
# ============================================================

@router.get(
    "/health",
    response_model=HealthResponse,
)
def health():
    try:
        backend_label = (
            intervention_audit.get_storage_backend_label()
        )
        status = "ok"

    except Exception as e:
        backend_label = f"unavailable ({e})"
        status = "degraded (jsonl fallback active)"

    return HealthResponse(
        status=status,
        storage_backend=backend_label,
    )


# ============================================================
# Single Student Intervention
# ============================================================

@router.post(
    "/run",
    response_model=InterventionResponse,
)
def run_intervention(
    payload: RunInterventionRequest,
):
    """
Run the LangGraph Intervention Agent for one student.
Supports row-index and real-student-ID lookup modes.
Returns the resolved student ID, row index, and lookup method.
"""
    # ========================================================
    # 1. Resolve student identifier
    # ========================================================

    try:

        if payload.real_student_id:

            # ------------------------------------------------
            # REAL STUDENT ID MODE
            # ------------------------------------------------
            # Example:
            #
            # 39655
            #   ↓
            # internal dataframe index = 110
            #   ↓
            # user-facing row index = 111
            # ------------------------------------------------

            internal_row_index = (
                ai_tools.resolve_row_index_from_student_id(
                    payload.dataset,
                    payload.student_id,
                )
            )

            row_index = internal_row_index + 1

            resolved_student_id = (
                ai_tools.resolve_student_id_from_row_index(
                    payload.dataset,
                    row_index,
                )
            )

            lookup_by = "real_student_id"

        else:

            # ------------------------------------------------
            # ROW INDEX MODE
            # ------------------------------------------------
            # Example:
            #
            # 111
            #   ↓
            # internal dataframe index = 110
            #   ↓
            # real student ID = 39655
            # ------------------------------------------------

            row_index = int(
                payload.student_id
            )

            if row_index < 1:
                raise ValueError(
                    "row index must be greater than or equal to 1"
                )

            resolved_student_id = (
                ai_tools.resolve_student_id_from_row_index(
                    payload.dataset,
                    row_index,
                )
            )

            lookup_by = "row_index"

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except (
        StudentNotFoundError,
        DataNotFoundError,
    ) as e:

        raise HTTPException(
            status_code=404,
            detail=str(e),
        )

    # ========================================================
    # 2. Get risk explanation
    # ========================================================
    #
    # IMPORTANT:
    #
    # get_risk_explanation() works with the 1-based row index.
    #
    # Therefore:
    #
    #     row_index = 111
    #     real student ID = 39655
    #
    # We pass:
    #
    #     "111"
    #
    # NOT:
    #
    #     "39655"
    # ========================================================

    try:

        risk_explanation = (
            ai_tools.get_risk_explanation(
                payload.dataset,
                str(row_index),
            )
        )

    except (
        StudentNotFoundError,
        DataNotFoundError,
    ) as e:

        raise HTTPException(
            status_code=404,
            detail=str(e),
        )

    # ========================================================
    # 3. Optional GRU/XGBoost disagreement check
    # ========================================================

    models_agree = None

    if (
        payload.check_disagreement
        and payload.dataset == "oulad"
    ):

        try:

            # GRU uses REAL OULAD student ID.
            gru_context, _, _ = (
                ai_tools.get_gru_timeseries_context(
                    int(resolved_student_id)
                )
            )

            models_agree = (
                gru_context.get("models_agree")
            )

        except StudentNotFoundError as e:

            raise HTTPException(
                status_code=404,
                detail=(
                    f"Disagreement check failed: {e}"
                ),
            )

    # ========================================================
    # 4. Run intervention agent
    # ========================================================

    try:

        decision = (
            intervention_graph.run_intervention(
                payload.dataset,

                # IMPORTANT:
                # Intervention receives REAL student ID.
                resolved_student_id,

                risk_explanation,

                oulad_id_student=(
                    int(resolved_student_id)
                    if (
                        payload.dataset == "oulad"
                        and payload.check_disagreement
                    )
                    else None
                ),

                models_agree=models_agree,
            )
        )

    except (
        StudentNotFoundError,
        DataNotFoundError,
    ) as e:

        raise HTTPException(
            status_code=404,
            detail=str(e),
        )

    # ========================================================
    # 5. Normalize response/audit fields
    # ========================================================
    #
    # IMPORTANT:
    #
    # student_id = REAL student ID
    # row_index  = 1-based row index
    # lookup_by   = lookup mode
    # ========================================================

    decision["student_id"] = str(
        resolved_student_id
    )

    decision["row_index"] = int(
        row_index
    )

    decision["lookup_by"] = lookup_by

    # ========================================================
    # 6. Audit log
    # ========================================================

    intervention_audit.log_decision(
        decision,
        run_id=(
            f"api:single:"
            f"{payload.dataset}:"
            f"{resolved_student_id}"
        ),
    )

    # ========================================================
    # 7. Return response
    # ========================================================

    return InterventionResponse(
        **decision
    )


# ============================================================
# Batch Intervention
# ============================================================

@router.post(
    "/batch",
    response_model=BatchInterventionResponse,
)
def run_batch(
    payload: BatchInterventionRequest,
):
    """
    Run the Intervention Agent across a cohort slice.

    For every successful batch record:

        student_id = REAL student ID
        row_index  = 1-based dataframe row index
        lookup_by  = "batch"
    """

    # ========================================================
    # 1. Create batch run ID
    # ========================================================

    run_id = (
        f"api:batch:"
        f"{payload.dataset}:"
        f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
    )

    results = []
    errors = []

    # ========================================================
    # 2. Load risk data
    # ========================================================

    try:

        risk_df = (
            risk_queries.load_risk_scores(
                payload.dataset
            )
        )

    except DataNotFoundError as e:

        raise HTTPException(
            status_code=404,
            detail=str(e),
        )

    # ========================================================
    # 3. Filter by risk level
    # ========================================================

    if payload.risk_level:

        if "risk_level" not in risk_df.columns:

            raise HTTPException(
                status_code=500,
                detail=(
                    f"'risk_level' column not found "
                    f"in {payload.dataset} risk data."
                ),
            )

        cohort_df = risk_df[
            risk_df["risk_level"]
            .astype(str)
            .str.strip()
            .str.lower()
            == payload.risk_level.lower()
        ]

    else:

        cohort_df = risk_df

    # ========================================================
    # 4. Limit cohort
    # ========================================================

    cohort_df = cohort_df.head(
        payload.limit
    )

    # ========================================================
    # 5. Process each student
    # ========================================================

    for internal_idx, row in cohort_df.iterrows():

        # Reset every iteration.
        resolved_student_id = None
        row_index = None

        try:

            # ==================================================
            # 5.1 Convert dataframe index -> row index
            # ==================================================

            row_index = (
                int(internal_idx) + 1
            )

            # ==================================================
            # 5.2 Convert row index -> REAL student ID
            # ==================================================

            resolved_student_id = (
                ai_tools.resolve_student_id_from_row_index(
                    payload.dataset,
                    row_index,
                )
            )

            # ==================================================
            # 5.3 Get risk explanation
            # ==================================================

            risk_explanation = (
                ai_tools.get_risk_explanation(
                    payload.dataset,
                    str(row_index),
                )
            )

            # ==================================================
            # 5.4 OULAD GRU/XGBoost agreement
            # ==================================================

            models_agree = None

            if payload.dataset == "oulad":

                try:

                    gru_context, _, _ = (
                        ai_tools.get_gru_timeseries_context(
                            int(resolved_student_id)
                        )
                    )

                    models_agree = (
                        gru_context.get(
                            "models_agree"
                        )
                    )

                except StudentNotFoundError:

                    # Do not fail the complete batch if
                    # GRU context is unavailable.
                    models_agree = None

            # ==================================================
            # 5.5 Run intervention agent
            # ==================================================

            decision = (
                intervention_graph.run_intervention(
                    payload.dataset,

                    # REAL student ID goes to intervention.
                    resolved_student_id,

                    risk_explanation,

                    oulad_id_student=(
                        int(resolved_student_id)
                        if payload.dataset == "oulad"
                        else None
                    ),

                    models_agree=models_agree,
                )
            )

            # ==================================================
            # 5.6 Normalize response fields
            # ==================================================

            # REAL student ID
            decision["student_id"] = str(
                resolved_student_id
            )

            # 1-based dataframe row
            decision["row_index"] = int(
                row_index
            )

            # Batch lookup mode
            decision["lookup_by"] = "batch"

            # ==================================================
            # 5.7 Save audit record
            # ==================================================

            intervention_audit.log_decision(
                decision,
                run_id=run_id,
            )

            # ==================================================
            # 5.8 Validate response
            # ==================================================

            results.append(
                InterventionResponse(
                    **decision
                )
            )

        except (
            StudentNotFoundError,
            DataNotFoundError,
            ValueError,
        ) as e:

            # --------------------------------------------------
            # If the real ID was resolved, report it.
            #
            # Otherwise report the row index.
            # --------------------------------------------------

            error_student_id = (
                str(resolved_student_id)
                if resolved_student_id is not None
                else str(
                    row_index
                    if row_index is not None
                    else int(internal_idx) + 1
                )
            )

            errors.append(
                {
                    "student_id": error_student_id,
                    "row_index": row_index,
                    "error": str(e),
                }
            )

    # ========================================================
    # 6. Return batch response
    # ========================================================

    return BatchInterventionResponse(
        run_id=run_id,
        results=results,
        errors=errors,
    )


# ============================================================
# Audit Log
# ============================================================

@router.get(
    "/audit-log",
    response_model=AuditLogPage,
)
def get_audit_log(
    limit: int = 50,
    dataset: str | None = None,
    decision: str | None = None,
):
    """
    Read the intervention audit trail.

    Current records contain:

        student_id = REAL student ID
        row_index  = 1-based row index
        lookup_by  = row_index / real_student_id / batch

    Older records created before lookup_by was introduced do not
    contain that field. Those records are returned as:

        lookup_by = "legacy"
    """

    # ========================================================
    # 1. Read audit records
    # ========================================================

    entries = (
        intervention_audit.read_audit_log(
            limit=None
        )
    )

    # ========================================================
    # 2. Filter dataset
    # ========================================================

    if dataset:

        entries = [
            e
            for e in entries
            if e.get("dataset") == dataset
        ]

    # ========================================================
    # 3. Filter decision
    # ========================================================

    if decision:

        entries = [
            e
            for e in entries
            if e.get("decision") == decision
        ]

    # ========================================================
    # 4. Apply limit AFTER filters
    # ========================================================

    entries = entries[:limit]

    # ========================================================
    # 5. Determine storage backend
    # ========================================================

    if entries:

        storage_backend = entries[0].get(
            "storage_backend",
            intervention_audit.get_storage_backend_label(),
        )

    else:

        storage_backend = (
            intervention_audit.get_storage_backend_label()
        )

    # ========================================================
    # 6. Normalize audit records
    # ========================================================

    audit_entries = []

    for e in entries:

        # ----------------------------------------------------
        # lookup_by
        # ----------------------------------------------------
        #
        # New records:
        #     row_index
        #     real_student_id
        #     batch
        #
        # Old records:
        #     missing
        #
        # We cannot reconstruct the original lookup mode from
        # old records reliably, so mark them as legacy.
        # ----------------------------------------------------

        lookup_by = e.get(
            "lookup_by",
            "legacy",
        )

        # ----------------------------------------------------
        # Create clean dictionary
        # ----------------------------------------------------

        entry_data = {
            k: v
            for k, v in e.items()
            if k not in (
                "storage_backend",
                "id",
            )
        }

        # ----------------------------------------------------
        # Ensure lookup_by always exists
        # ----------------------------------------------------

        entry_data["lookup_by"] = (
            lookup_by
        )

        # ----------------------------------------------------
        # Ensure row_index exists
        # ----------------------------------------------------
        #
        # Historical records may not have it.
        # AuditLogEntry allows None.
        # ----------------------------------------------------

        if "row_index" not in entry_data:

            entry_data["row_index"] = None

        # ----------------------------------------------------
        # Ensure student_id exists
        # ----------------------------------------------------

        if "student_id" not in entry_data:

            entry_data["student_id"] = ""

        # ----------------------------------------------------
        # Explicit ID
        # ----------------------------------------------------

        entry_data["id"] = e.get(
            "id",
            -1,
        )

        # ----------------------------------------------------
        # Build Pydantic response
        # ----------------------------------------------------

        audit_entries.append(
            AuditLogEntry(
                **entry_data
            )
        )

    # ========================================================
    # 7. Return response
    # ========================================================

    return AuditLogPage(
        entries=audit_entries,
        total_returned=len(audit_entries),
        storage_backend=storage_backend,
    )


# ============================================================
# Chat
# ============================================================

@router.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(
    payload: ChatRequest,
):
    """
    Stateful chat endpoint.

    First request:

        POST /agent/chat

        {
            "question": "Hello"
        }

    Server generates:

        chat-xxxxxxxx

    Subsequent requests:

        {
            "question": "What about the student?",
            "session_id": "chat-xxxxxxxx"
        }
    """

    session_state: dict = {}

    session_key = "api_chat"

    # ========================================================
    # 1. Get or create session ID
    # ========================================================

    session_id = payload.session_id

    if not session_id:

        session_id = (
            f"chat-{uuid.uuid4().hex[:8]}"
        )

        _terminal.info(
            "[/agent/chat] Created new session_id=%s",
            session_id,
        )

    # ========================================================
    # 2. Load previous conversation
    # ========================================================

    genai_memory.load_history_into_session(
        session_state,
        session_id,
        session_key,
    )

    # ========================================================
    # 3. Run agent
    # ========================================================

    result = explain_agent.run_agent(
        payload.question,
        session_state,
        session_key,
        provider=payload.provider,
    )

    # ========================================================
    # 4. Normalize answer
    # ========================================================

    answer = (
        llm_client._extract_chunk_text(
            result["answer"]
        )
    )

    # ========================================================
    # 5. Save turn
    # ========================================================

    genai_memory.save_turn_to_db(
        session_id,
        session_key,
        payload.question,
        answer,
    )

    # ========================================================
    # 6. Build current history
    # ========================================================

    full_history = [
        {
            "role": (
                "human"
                if m.type == "human"
                else "ai"
            ),
            "content": (
                llm_client._extract_chunk_text(
                    m.content
                )
            ),
        }
        for m in genai_memory.get_session_history(
            session_state,
            session_key,
        ).messages
    ]

    # ========================================================
    # 7. Logging
    # ========================================================

    _terminal.info(
        "[/agent/chat] session_id=%s",
        session_id,
    )

    # ========================================================
    # 8. Response
    # ========================================================

    return ChatResponse(
        answer=answer,
        source=result["source"],
        tool_calls_made=result[
            "tool_calls_made"
        ],
        session_id=session_id,
        history=full_history,
    )


# ============================================================
# Chat History
# ============================================================

@router.get(
    "/chat/history",
    response_model=ChatHistoryPage,
)
def get_chat_history(
    session_id: str,
    session_key: str = "api_chat",
):
    """
    Read saved chat messages for a session.
    """

    from backend.database import get_session
    from backend.models_db import ChatMessage

    with get_session() as session:

        rows = (
            session.query(ChatMessage)
            .filter(
                ChatMessage.teacher_id
                == session_id,
                ChatMessage.session_key
                == session_key,
            )
            .order_by(
                ChatMessage.created_at.asc()
            )
            .all()
        )

    return ChatHistoryPage(
        session_id=session_id,
        messages=[
            ChatHistoryEntry(
                role=r.role,
                content=r.content,
                created_at=r.created_at.isoformat(),
            )
            for r in rows
        ],
    )