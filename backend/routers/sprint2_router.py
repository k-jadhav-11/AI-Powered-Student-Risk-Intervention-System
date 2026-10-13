# backend/routers/sprint2.py
"""
REST layer for Sprint 2 -- the MySQL-backed student marks analytics module.

This router is a thin wrapper: every endpoint opens its own connection
(get_connection), calls the EXISTING src/analytics/queries.py +
src/analytics/analytics.py functions exactly as src/cli.py already composes
them, sanitizes the resulting DataFrame(s) to JSON-safe structures, and
closes the connection in a finally block. No query/analytics logic is
reimplemented here.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from src.analytics import analytics, db_manager, queries
from src.analytics.exceptions import (
    DataNotFoundError,
    InvalidCSVError,
    StudentNotFoundError,
)

from ..schemas_sprint2 import (
    ClassComparisonResponse,
    ClassListResponse,
    LoadDataResponse,
    PivotTableResponse,
    Sprint2HealthResponse,
    StudentReportResponse,
    SubjectListResponse,
    TableResponse,
    TopBottomResponse,
)

router = APIRouter(
    prefix="/sprint2",
    tags=["Student Marks Analytics"],
)

# Default data directory, same as src/cli.py's DATA_DIR (project_root/data).
# This file lives at backend/routers/sprint2.py, so two levels up is the
# project root, matching src/cli.py's os.path.join(dirname(__file__), "..", "data").
import os as _os

_DATA_DIR_DEFAULT = _os.path.normpath(
    _os.path.join(_os.path.dirname(__file__), "..", "..", "data")
)


# ============================================================
# JSON-safety helpers
# ============================================================


def _sanitize_value(v: Any) -> Any:
    """Convert a single scalar to something json.dumps can handle."""
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        f = float(v)
        return None if math.isnan(f) else f
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def df_to_records(df: pd.DataFrame) -> list[dict]:
    """
    Convert a DataFrame to a list-of-dicts that is safe to json.dumps:
    NaN -> None, numpy int64/float64 -> native int/float.
    """
    if df is None or df.empty:
        return []
    safe = df.where(pd.notnull(df), None)
    records = safe.to_dict(orient="records")
    return [{k: _sanitize_value(v) for k, v in row.items()} for row in records]


def series_to_records(s: pd.Series, key_name: str, value_name: str) -> list[dict]:
    if s is None or s.empty:
        return []
    return [
        {key_name: _sanitize_value(k), value_name: _sanitize_value(v)}
        for k, v in s.items()
    ]


def pivot_to_response(pivot_df: pd.DataFrame) -> PivotTableResponse:
    """Wide student/class x subject matrix -> {columns, rows}, index folded into rows."""
    if pivot_df is None or pivot_df.empty:
        return PivotTableResponse(columns=[], rows=[])
    reset = pivot_df.reset_index()
    columns = [str(c) for c in reset.columns]
    rows = df_to_records(reset)
    return PivotTableResponse(columns=columns, rows=rows)


# ============================================================
# Connection helper
# ============================================================


def _get_conn():
    try:
        return db_manager.get_connection()
    except Exception as e:  # pragma: no cover - surfaced as 500 below
        raise HTTPException(status_code=500, detail=f"Could not connect to database: {e}")


# ============================================================
# 1. Health
# ============================================================


@router.get("/health", response_model=Sprint2HealthResponse)
def health():
    """DB connectivity check -- can we get_connection() and has_data()?"""
    try:
        conn = db_manager.get_connection()
        try:
            has_data = db_manager.has_data(conn)
        finally:
            conn.close()
        return Sprint2HealthResponse(
            status="ok",
            storage_backend=f"mysql (has_data={has_data})",
        )
    except Exception as e:
        return Sprint2HealthResponse(
            status=f"degraded ({e})",
            storage_backend="unavailable",
        )


# ============================================================
# 2. Classes
# ============================================================


@router.get("/classes", response_model=ClassListResponse)
def get_classes():
    conn = _get_conn()
    try:
        classes = queries.get_available_classes(conn)
    finally:
        conn.close()
    return ClassListResponse(classes=classes)


# ============================================================
# 3. Subjects
# ============================================================


@router.get("/subjects", response_model=SubjectListResponse)
def get_subjects():
    conn = _get_conn()
    try:
        subjects = queries.get_available_subjects(conn)
    finally:
        conn.close()
    return SubjectListResponse(subjects=subjects)


# ============================================================
# 4. Class average by subject
# ============================================================


@router.get("/class-average", response_model=TableResponse)
def class_average(class_name: str | None = None):
    conn = _get_conn()
    try:
        df = queries.get_class_average_by_subject(conn, class_name=class_name)
    finally:
        conn.close()
    if df.empty:
        raise HTTPException(status_code=404, detail="No matching data found.")
    return TableResponse(rows=df_to_records(df))


# ============================================================
# 5. Top / bottom performers
# ============================================================


@router.get("/top-bottom", response_model=TopBottomResponse)
def top_bottom(
    n: int = 5,
    subject: str | None = None,
    class_name: str | None = None,
):
    conn = _get_conn()
    try:
        top, bottom = queries.get_top_bottom_students(
            conn, n=n, subject=subject, class_name=class_name
        )
    finally:
        conn.close()
    if top.empty and bottom.empty:
        raise HTTPException(status_code=404, detail="No matching data found.")
    return TopBottomResponse(top=df_to_records(top), bottom=df_to_records(bottom))


# ============================================================
# 6. Weak subjects
# ============================================================


@router.get("/weak-subjects", response_model=TableResponse)
def weak_subjects(
    class_name: str | None = None,
    avg_threshold: float = 55,
    fail_threshold: float = 25,
    pass_mark: float = 35,
):
    conn = _get_conn()
    try:
        avg_df = queries.get_class_average_by_subject(conn, class_name=class_name)
        fail_df = queries.get_subject_failure_rates(
            conn, pass_mark=pass_mark, class_name=class_name
        )
    finally:
        conn.close()
    if avg_df.empty:
        raise HTTPException(status_code=404, detail="No matching data found.")
    weak_df = analytics.identify_weak_subjects(
        avg_df, fail_df, avg_threshold=avg_threshold, fail_threshold=fail_threshold
    )
    return TableResponse(rows=df_to_records(weak_df))


# ============================================================
# 7. Grade distribution
# ============================================================


@router.get("/grade-distribution", response_model=TableResponse)
def grade_distribution(class_name: str | None = None, subject: str | None = None):
    conn = _get_conn()
    try:
        scores_df = queries.get_all_scores(conn, subject=subject, class_name=class_name)
    finally:
        conn.close()
    if scores_df.empty:
        raise HTTPException(status_code=404, detail="No matching data found.")
    band_counts = analytics.compute_grade_bands(scores_df["score"].values)
    rows = series_to_records(band_counts, "grade_band", "count")
    return TableResponse(rows=rows)


# ============================================================
# 8. Student report
# ============================================================


@router.get("/student-report", response_model=StudentReportResponse)
def student_report(search: str = Query(..., description="Student name or roll number"), class_name: str | None = None):
    conn = _get_conn()
    try:
        matches = queries.find_student(conn, search, class_name=class_name)
        if matches.empty:
            raise StudentNotFoundError(f"No student found matching '{search}'.")
        if len(matches) > 1:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": f"Multiple students match '{search}'; narrow the search.",
                    "matches": df_to_records(matches),
                },
            )

        student = matches.iloc[0]
        student_id = int(student["student_id"])

        scores_df = queries.get_student_scores_over_time(conn, student_id)
        if scores_df.empty:
            raise DataNotFoundError(f"{student['name']} has no recorded marks yet.")

        class_scores = queries.get_all_scores(conn, class_name=student["class_name"])
        stats = analytics.class_stats(class_scores["score"].values)
        student_avg = float(scores_df["score"].mean())
        zscore = analytics.compute_zscore(student_avg, stats["mean"], stats["std"])

        per_test = scores_df.groupby("test_name")["score"].mean().reindex(
            scores_df["test_name"].unique()
        )
        trend = analytics.classify_trend(per_test.values)
    except StudentNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except DataNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    finally:
        conn.close()

    return StudentReportResponse(
        student_id=student_id,
        name=str(student["name"]),
        roll_number=str(student["roll_number"]),
        class_name=student["class_name"],
        scores=df_to_records(scores_df),
        student_average=round(student_avg, 2),
        class_average=round(stats["mean"], 2) if stats["mean"] is not None else None,
        zscore=zscore,
        trend=trend,
    )


# ============================================================
# 9. Pivot table
# ============================================================


@router.get("/pivot-table", response_model=PivotTableResponse)
def pivot_table(class_name: str | None = None):
    conn = _get_conn()
    try:
        scores_df = queries.get_all_scores(conn, class_name=class_name)
    finally:
        conn.close()
    if scores_df.empty:
        raise HTTPException(status_code=404, detail="No matching data found.")
    pivot = analytics.build_pivot_table(scores_df)
    return pivot_to_response(pivot)


# ============================================================
# 10. Percentile ranking
# ============================================================


@router.get("/percentile-ranking", response_model=TableResponse)
def percentile_ranking(class_name: str | None = None, subject: str | None = None):
    conn = _get_conn()
    try:
        avg_df = queries.get_student_averages(conn, subject=subject, class_name=class_name)
    finally:
        conn.close()
    if avg_df.empty:
        raise HTTPException(status_code=404, detail="No matching data found.")
    ranked = analytics.add_percentile_ranks(avg_df)
    return TableResponse(rows=df_to_records(ranked))


# ============================================================
# 11. Subject strengths
# ============================================================


@router.get("/subject-strengths", response_model=TableResponse)
def subject_strengths(class_name: str | None = None):
    conn = _get_conn()
    try:
        scores_df = queries.get_all_scores(conn, class_name=class_name)
    finally:
        conn.close()
    if scores_df.empty:
        raise HTTPException(status_code=404, detail="No matching data found.")
    pivot = analytics.build_pivot_table(scores_df)
    strengths = analytics.find_subject_strengths(pivot)
    return TableResponse(rows=df_to_records(strengths))


# ============================================================
# 12. Improvement leaderboard
# ============================================================


@router.get("/improvement-leaderboard", response_model=TableResponse)
def improvement_leaderboard(class_name: str | None = None, subject: str | None = None):
    conn = _get_conn()
    try:
        scores_df = queries.get_all_scores(conn, subject=subject, class_name=class_name)
    finally:
        conn.close()
    if scores_df.empty:
        raise HTTPException(status_code=404, detail="No matching data found.")
    improve_df = analytics.compute_improvement_by_student(scores_df)
    return TableResponse(rows=df_to_records(improve_df))


# ============================================================
# 13. Consistency
# ============================================================


@router.get("/consistency", response_model=TableResponse)
def consistency(class_name: str | None = None, subject: str | None = None):
    conn = _get_conn()
    try:
        scores_df = queries.get_all_scores(conn, subject=subject, class_name=class_name)
    finally:
        conn.close()
    if scores_df.empty:
        raise HTTPException(status_code=404, detail="No matching data found.")
    cons_df = analytics.compute_consistency_by_student(scores_df)
    return TableResponse(rows=df_to_records(cons_df))


# ============================================================
# 14. Outliers
# ============================================================


@router.get("/outliers", response_model=TableResponse)
def outliers(
    class_name: str | None = None,
    subject: str | None = None,
    threshold: float = 2.0,
):
    conn = _get_conn()
    try:
        avg_df = queries.get_student_averages(conn, subject=subject, class_name=class_name)
    finally:
        conn.close()
    if avg_df.empty:
        raise HTTPException(status_code=404, detail="No matching data found.")
    outliers_df = analytics.detect_outliers(avg_df, threshold=threshold)
    return TableResponse(rows=df_to_records(outliers_df))


# ============================================================
# 15. Class comparison
# ============================================================


@router.get("/class-comparison", response_model=ClassComparisonResponse)
def class_comparison(
    subject: str | None = None,
    by: str = Query("both", pattern="^(overall|subject|both)$"),
):
    conn = _get_conn()
    try:
        scores_df = queries.get_all_scores(conn, subject=subject)
    finally:
        conn.close()
    if scores_df.empty:
        raise HTTPException(status_code=404, detail="No matching data found.")

    overall = None
    by_subject = None
    if by in ("overall", "both"):
        overall_df = analytics.compare_classes(scores_df)
        overall = df_to_records(overall_df)
    if by in ("subject", "both"):
        by_subject_df = analytics.compare_classes_by_subject(scores_df)
        by_subject = pivot_to_response(by_subject_df)

    return ClassComparisonResponse(overall=overall, by_subject=by_subject)


# ============================================================
# 16. Pass rate
# ============================================================


@router.get("/pass-rate", response_model=TableResponse)
def pass_rate(
    class_name: str | None = None,
    group_by: str = Query("subject", pattern="^(subject|class_name)$"),
    pass_mark: float = 35,
):
    conn = _get_conn()
    try:
        scores_df = queries.get_all_scores(conn, class_name=class_name)
    finally:
        conn.close()
    if scores_df.empty:
        raise HTTPException(status_code=404, detail="No matching data found.")
    pass_df = analytics.compute_pass_rate(scores_df, pass_mark=pass_mark, group_by=group_by)
    return TableResponse(rows=df_to_records(pass_df))


# ============================================================
# 17. Subject top performers
# ============================================================


@router.get("/subject-top-performers", response_model=TableResponse)
def subject_top_performers(class_name: str | None = None, top_n: int = 1):
    conn = _get_conn()
    try:
        scores_df = queries.get_all_scores(conn, class_name=class_name)
    finally:
        conn.close()
    if scores_df.empty:
        raise HTTPException(status_code=404, detail="No matching data found.")
    top_df = analytics.rank_subject_performers(scores_df, top_n=top_n)
    return TableResponse(rows=df_to_records(top_df))


# ============================================================
# 18. Load data (seed_all) -- the only write/POST path
# ============================================================


@router.post("/load-data", response_model=LoadDataResponse)
def load_data(data_dir: str | None = None):
    """
    Wraps the CLI's "Load Data from CSV" option (db_manager.seed_all).

    seed_all's own loaders (load_students_csv/load_subjects_csv/load_marks_csv)
    already skip duplicates on their unique constraints (IntegrityError is
    caught and counted as `skipped`), so calling this endpoint again is safe
    and will not double-insert. No extra guard is added here beyond that
    existing behaviour, per the project's own README note on this.
    """
    target_dir = data_dir or _DATA_DIR_DEFAULT
    if not _os.path.isdir(target_dir):
        raise HTTPException(status_code=400, detail=f"data_dir not found: {target_dir}")

    conn = _get_conn()
    try:
        try:
            summary = db_manager.seed_all(conn, target_dir)
        except InvalidCSVError as e:
            raise HTTPException(status_code=400, detail=str(e))
    finally:
        conn.close()

    return LoadDataResponse(**summary)
