# orchestration/audit.py
"""
Manage audit logging for Intervention Agent decisions.
Use PostgreSQL as the primary store with JSONL as a fallback.
Expose storage backend information for consistent audit records.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
LOG_PATH = os.path.join(LOG_DIR, "interventions.jsonl")

# db backend imports are deferred to avoid circular imports and to allow the module to be imported even if the database is unreachable.
def _db_backend_label() -> str:
    from backend.database import DATABASE_URL, IS_SQLITE_FALLBACK
    if IS_SQLITE_FALLBACK:
        return "sqlite (DATABASE_URL not set)"
    # Never echo the connection string itself (credentials live in it) --
    # just confirm a real PostgreSQL URL is configured.
    return "postgresql" if DATABASE_URL.startswith("postgresql") else "database (non-default DATABASE_URL)"


def get_storage_backend_label() -> str:
    """Public wrapper around _db_backend_label() for external callers
    (backend/routers/agent.py's /agent/health) that just want to know
    which backend is active without reaching into a private function."""
    return _db_backend_label()


def _log_decision_db(decision: dict, run_id: str | None) -> dict:
    from backend.database import get_session, init_db
    from backend.models_db import InterventionLog

    init_db()  # idempotent -- see backend/database.py's init_db() docstring
    session = get_session()
    try:
        row = InterventionLog(
            run_id=run_id,
            student_id=decision["student_id"],
            display_student_id=decision.get("display_student_id"),
            dataset=decision["dataset"],
            risk_level=decision["risk_level"],
            decision=decision["decision"],
            rationale=decision["rationale"],
            drafted_message=decision.get("drafted_message"),
            resources_used=decision.get("resources_used") or [],
            escalation_reason=decision.get("escalation_reason"),
            mentor_brief=decision.get("mentor_brief"),
            router_confidence=decision["router_confidence"],
            oulad_id_student=decision.get("oulad_id_student"),
            agent_trace=decision.get("agent_trace") or [],
            source=decision["source"],
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        entry = row.to_dict()
        entry["id"] = row.id
        entry["storage_backend"] = _db_backend_label()
        return entry
    finally:
        session.close()


def _log_decision_jsonl(decision: dict, run_id: str | None) -> dict:
    os.makedirs(LOG_DIR, exist_ok=True)
    entry = {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        **decision,
        "storage_backend": "jsonl-fallback (database unreachable)",
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def log_decision(decision: dict, run_id: str | None = None) -> dict:
    """Persist Intervention decisions to PostgreSQL with JSONL fallback.
    Return the stored entry with its timestamp, run ID, and storage backend.
    """
    try:
        return _log_decision_db(decision, run_id)
    except Exception:
        return _log_decision_jsonl(decision, run_id)


def _read_audit_log_db(limit: int | None) -> list[dict]:
    from backend.database import get_session, init_db
    from backend.models_db import InterventionLog

    init_db()
    session = get_session()
    try:
        query = session.query(InterventionLog).order_by(InterventionLog.id.desc())
        if limit:
            query = query.limit(limit)
        backend_label = _db_backend_label()
        entries = []
        for row in query.all():
            entry = row.to_dict()
            entry["id"] = row.id
            entry["storage_backend"] = backend_label
            entries.append(entry)
        return entries
    finally:
        session.close()


def _read_audit_log_jsonl(limit: int | None) -> list[dict]:
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH, encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    entries = [json.loads(line) for line in lines]
    entries.reverse()  # most recent first
    return entries[:limit] if limit else entries


def read_audit_log(limit: int | None = None) -> list[dict]:
    """Read audit decisions from PostgreSQL with JSONL fallback.
    Return recent decisions first, or an empty list when no records exist.
    """
    try:
        return _read_audit_log_db(limit)
    except Exception:
        return _read_audit_log_jsonl(limit)


def backfill_display_student_ids() -> int:
    """One-off repair for rows logged before display_student_id was persisted."""
    from backend.database import get_session, init_db
    from backend.models_db import InterventionLog
    from ai_assistant import tools as ai_tools

    init_db()
    session = get_session()
    updated = 0
    try:
        rows = (
            session.query(InterventionLog)
            .filter(InterventionLog.display_student_id.is_(None))
            .all()
        )
        for row in rows:
            try:
                if (row.run_id or "").startswith("api:"):
                    # API runs already store the real ID in student_id
                    row.display_student_id = str(row.student_id)
                else:
                    # Streamlit runs store the row index -> resolve to real ID
                    row.display_student_id = ai_tools.resolve_student_id_from_row_index(
                        row.dataset, row.student_id
                    )
                updated += 1
            except Exception:
                continue
        session.commit()
        return updated
    finally:
        session.close()