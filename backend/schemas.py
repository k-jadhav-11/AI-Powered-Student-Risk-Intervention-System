# backend/schemas.py

from __future__ import annotations

from pydantic import BaseModel, Field


# ============================================================
# Run Intervention Request
# ============================================================

class RunInterventionRequest(BaseModel):
    """
    Request to run an intervention for one student.

    real_student_id=False
        student_id is interpreted as a 1-based row index.

    real_student_id=True
        student_id is interpreted as the actual student ID.
    """

    dataset: str = Field(
        pattern="^(uci|oulad)$"
    )

    student_id: str

    real_student_id: bool = False

    check_disagreement: bool = False


# ============================================================
# Intervention Response
# ============================================================

class InterventionResponse(BaseModel):
    """
    Intervention result.

    Identifier convention:

        student_id
            REAL student ID.

        row_index
            1-based dataframe row index.

        lookup_by
            How the student was selected.
    """

    id: int | None = None

    # REAL student ID
    student_id: str

    # 1-based dataframe row index
    row_index: int | None = None

    # row_index / real_student_id / batch / legacy
    lookup_by: str

    dataset: str

    risk_level: str

    decision: str

    rationale: str

    drafted_message: str | None

    resources_used: list[str]

    escalation_reason: str | None

    mentor_brief: str | None = None

    router_confidence: float

    agent_trace: list[str]

    source: str


# ============================================================
# Batch Intervention Request
# ============================================================

class BatchInterventionRequest(BaseModel):
    """
    Request to run interventions across a cohort slice.
    """

    dataset: str = Field(
        pattern="^(uci|oulad)$"
    )

    risk_level: str | None = Field(
        default=None,
        pattern="^(Low|Medium|High)$",
    )

    limit: int = Field(
        default=5,
        ge=1,
        le=50,
    )


# ============================================================
# Batch Intervention Response
# ============================================================

class BatchInterventionResponse(BaseModel):
    """
    Response containing successful intervention results
    and per-student errors.
    """

    run_id: str

    results: list[
        InterventionResponse
    ]

    errors: list[dict]


# ============================================================
# Audit Log Entry
# ============================================================

class AuditLogEntry(
    InterventionResponse
):
    """
    Persisted intervention audit record.

    New records contain:

        student_id
            REAL student ID.

        row_index
            1-based dataframe row index.

        lookup_by
            row_index / real_student_id / batch.

    Older records may contain:

        lookup_by = "legacy"

    because the old records were created before lookup_by
    was introduced.
    """

    id: int

    logged_at: str

    run_id: str | None


# ============================================================
# Audit Log Page
# ============================================================

class AuditLogPage(BaseModel):
    """
    Paginated collection of intervention audit records.
    """

    entries: list[
        AuditLogEntry
    ]

    total_returned: int

    storage_backend: str


# ============================================================
# Health Response
# ============================================================

class HealthResponse(BaseModel):
    """
    Service health information.
    """

    status: str

    storage_backend: str


# ============================================================
# Chat Message Input
# ============================================================

class ChatMessageIn(BaseModel):
    """
    One caller-supplied chat message.

    role must be:

        human
        ai
    """

    role: str = Field(
        pattern="^(human|ai)$"
    )

    content: str


# ============================================================
# Chat Request
# ============================================================

class ChatRequest(BaseModel):
    """
    Request to the Intervention Agent chat endpoint.

    session_id
        Optional persistent conversation identifier.

    history
        Optional backward-compatible history.

    provider
        Optional LLM provider.
    """

    question: str

    session_id: str | None = None

    history: list[
        ChatMessageIn
    ] = Field(
        default_factory=list
    )

    provider: str | None = None


# ============================================================
# Chat Response
# ============================================================

class ChatResponse(BaseModel):
    """
    Response from the Intervention Agent chat endpoint.
    """

    answer: str

    source: str

    tool_calls_made: list[str]

    session_id: str | None = None

    history: list[
        ChatMessageIn
    ] = Field(
        default_factory=list
    )


# ============================================================
# Chat History Entry
# ============================================================

class ChatHistoryEntry(BaseModel):
    """
    One persisted chat message.
    """

    role: str

    content: str

    created_at: str


# ============================================================
# Chat History Page
# ============================================================

class ChatHistoryPage(BaseModel):
    """
    All persisted messages associated with a session.
    """

    session_id: str

    messages: list[
        ChatHistoryEntry
    ]