# backend/models_db.py
"""
Define the SQLAlchemy ORM model for the Intervention Agent audit trail.
Store validated InterventionDecision fields with audit metadata.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base


class InterventionLog(Base):
    __tablename__ = "intervention_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    logged_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    run_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)

    student_id: Mapped[str] = mapped_column(String(64), index=True)
    display_student_id: Mapped[str | None] = mapped_column(String(64), nullable=True)  
    dataset: Mapped[str] = mapped_column(String(16), index=True)
    risk_level: Mapped[str] = mapped_column(String(16), index=True)

    decision: Mapped[str] = mapped_column(String(16), index=True)  # "nudge" | "resource" | "escalate"
    rationale: Mapped[str] = mapped_column(Text)

    drafted_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    resources_used: Mapped[list] = mapped_column(JSON, default=list)

    escalation_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    mentor_brief: Mapped[str | None] = mapped_column(Text, nullable=True)

    router_confidence: Mapped[float] = mapped_column(Float)
    oulad_id_student: Mapped[int | None] = mapped_column(Integer, nullable=True)

    agent_trace: Mapped[list] = mapped_column(JSON, default=list)
    source: Mapped[str] = mapped_column(String(16))  # "crew" | "template"

    def to_dict(self) -> dict:
        """Match the JSONL audit entry structure across both storage backends.
            Keep the storage layer interchangeable for all downstream callers."""
        return {
            "logged_at": self.logged_at.isoformat() if self.logged_at else None,
            "run_id": self.run_id,
            "student_id": self.student_id,
            "display_student_id": self.display_student_id,
            "dataset": self.dataset,
            "risk_level": self.risk_level,
            "decision": self.decision,
            "rationale": self.rationale,
            "drafted_message": self.drafted_message,
            "resources_used": self.resources_used or [],
            "escalation_reason": self.escalation_reason,
            "mentor_brief": self.mentor_brief,
            "router_confidence": self.router_confidence,
            "oulad_id_student": self.oulad_id_student,
            "agent_trace": self.agent_trace or [],
            "source": self.source,
        }


class ChatMessage(Base):
    """
Store chat history for returning teachers across sessions.
Use a free-text teacher ID; authentication and multi-tenant security are out of scope.
"""
    __tablename__ = "chat_message"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)

    teacher_id: Mapped[str] = mapped_column(String(128), index=True)  # free-text identifier, not an authenticated user
    session_key: Mapped[str] = mapped_column(String(64), index=True)  # which chat panel: "ia_chat", "ai_agent_chat", etc.

    role: Mapped[str] = mapped_column(String(16))  # "human" | "ai"
    content: Mapped[str] = mapped_column(Text)

    def to_dict(self) -> dict:
        return {
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "teacher_id": self.teacher_id,
            "session_key": self.session_key,
            "role": self.role,
            "content": self.content,
        }