# backend/schemas_sprint2.py
"""
Pydantic schemas for the Sprint 2 (student marks analytics) REST API.
Mirrors the style of backend/schemas.py: plain BaseModel classes, no ORM
mode needed since every response is built from a sanitized dict/list.
"""

from __future__ import annotations

from pydantic import BaseModel


# ============================================================
# Health
# ============================================================

class Sprint2HealthResponse(BaseModel):
    """Modeled after agent /health's response shape (status + a backend label)."""

    status: str

    storage_backend: str


# ============================================================
# Simple list responses
# ============================================================

class ClassListResponse(BaseModel):
    classes: list[str]


class SubjectListResponse(BaseModel):
    subjects: list[str]


# ============================================================
# Generic "table" response -- list-of-dicts, safe for an HTML table.
# ============================================================

class TableResponse(BaseModel):
    """Generic JSON-safe table: list of row dicts, NaN already replaced with None."""

    rows: list[dict]


class TopBottomResponse(BaseModel):
    top: list[dict]
    bottom: list[dict]


# ============================================================
# Pivot table (wide matrix -> columns + rows, JSON-safe)
# ============================================================

class PivotTableResponse(BaseModel):
    columns: list[str]
    rows: list[dict]


# ============================================================
# Student report
# ============================================================

class StudentReportResponse(BaseModel):
    student_id: int
    name: str
    roll_number: str
    class_name: str | None
    scores: list[dict]
    student_average: float | None
    class_average: float | None
    zscore: float | None
    trend: str


# ============================================================
# Class comparison (overall + by-subject, or either alone)
# ============================================================

class ClassComparisonResponse(BaseModel):
    overall: list[dict] | None = None
    by_subject: PivotTableResponse | None = None


# ============================================================
# Load data (seed_all passthrough)
# ============================================================

class LoadDataResponse(BaseModel):
    students: dict
    subjects: dict
    marks: dict
