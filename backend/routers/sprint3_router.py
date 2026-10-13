# backend/routers/sprint3.py
"""
Sprint 3 REST layer -- ML/DL risk prediction (XGBoost/CatBoost for UCI,
XGBoost+GRU for OULAD), wrapping the EXISTING Sprint 3 functions in
src/risk/risk_queries.py, src/risk/model_gru.py, src/risk/clustering.py and
the clean, DB-free, Streamlit-ready wrapper functions in ai_assistant/tools.py.

No model-loading, feature-engineering, scoring or clustering logic is
reimplemented here. Every endpoint:
  1. validates its `dataset` path param (uci/oulad) the same way
     ai_assistant/tools.py's own functions do,
  2. calls straight into the existing function(s),
  3. translates StudentNotFoundError/DataNotFoundError/ValueError into
     HTTPException the same way backend/routers/agent.py already does,
  4. runs the result through schemas_sprint3.sanitize() before returning,
     since most of this data originates in pandas DataFrames/Series or
     numpy/torch computations.

ID convention: identical to agent.py / ai_assistant/tools.py. For the
tabular (UCI/CatBoost, OULAD/XGBoost) endpoints, `student_id` is the
1-based ROW INDEX shown on the Risk Dashboard page (ai_tools._resolve_row_index
subtracts 1 to get the dataframe index). For the OULAD GRU endpoints,
`id_student` is the REAL OULAD id_student from the raw CSVs -- a
different, larger ID space, never zero/one-based. This file invents no
third ID scheme. Persona and clustering rows expose `row_index` separately
from the real/display student identifier.
"""
from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, Body, HTTPException, Query
from pydantic import ValidationError

from ai_assistant import tools as ai_tools
from src.analytics.exceptions import DataNotFoundError, StudentNotFoundError
from src.risk import clustering, model_gru, risk_queries

from ..schemas_sprint3 import OuladPredictRequest, UCIPredictRequest, sanitize

router = APIRouter(
    prefix="/sprint3",
    tags=["Sprint 3 - Risk Models"],
)

_DATASETS = ("uci", "oulad")


def _check_dataset(dataset: str) -> None:
    if dataset not in _DATASETS:
        raise HTTPException(
            status_code=400,
            detail=f"dataset must be 'uci' or 'oulad', got {dataset!r}",
        )


# ============================================================
# 01. Health
# ============================================================

@router.get("/health")
def sprint3_health():
    """Can models/data load for both datasets? Tries load_model_summary()
    for uci and oulad independently and reports per-dataset ok/error, so a
    problem with one dataset's artifacts doesn't hide the other's status."""
    datasets = {}
    overall_ok = True
    for ds in _DATASETS:
        try:
            summary = risk_queries.load_model_summary(ds)
            datasets[ds] = {
                "ok": True,
                "model_class": summary.get("model_class") or summary.get("selected_model"),
            }
        except Exception as e:  # noqa: BLE001 -- health check must never 500
            overall_ok = False
            datasets[ds] = {"ok": False, "error": str(e)}

    return {
        "status": "ok" if overall_ok else "degraded",
        "datasets": datasets,
    }


# ============================================================
# 02. Risk scores (Risk Dashboard table)
# ============================================================

@router.get("/{dataset}/risk-scores")
def risk_scores(
    dataset: str,
    risk_level: list[str] | None = Query(default=None),
    limit: int = 50,
    offset: int = 0,
):
    _check_dataset(dataset)
    if limit < 1 or limit > 500:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 500")
    if offset < 0:
        raise HTTPException(status_code=400, detail="offset must be >= 0")

    try:
        df = risk_queries.load_risk_scores(dataset)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if risk_level is not None:
        normalized_levels = {level.strip().lower() for level in risk_level if level.strip()}
        invalid_levels = normalized_levels - {"low", "medium", "high"}
        if invalid_levels:
            raise HTTPException(
                status_code=400,
                detail=f"risk_level must contain only Low, Medium, or High; got {sorted(invalid_levels)}",
            )
        df = df[df["risk_level"].astype(str).str.strip().str.lower().isin(normalized_levels)]

    total_matching = len(df)
    risk_level_counts = {
        str(level): int(count)
        for level, count in df["risk_level"].value_counts().items()
    }
    page = df.iloc[offset: offset + limit]

    rows = []
    for idx, row in page.iterrows():
        record = sanitize(row)
        record["row_index"] = int(idx) + 1  # 1-based, matches Risk Dashboard / agent.py convention
        record["display_student_id"] = ai_tools._display_student_id(dataset, idx, row)
        rows.append(record)

    return {
        "dataset": dataset,
        "total_matching": total_matching,
        "risk_level_counts": risk_level_counts,
        "limit": limit,
        "offset": offset,
        "rows": rows,
    }


# ============================================================
# 03. Model info
# ============================================================

@router.get("/{dataset}/model-info")
def model_info(dataset: str):
    _check_dataset(dataset)
    try:
        result = ai_tools.get_model_information(dataset)
        return {**result, **result["model_metrics"]}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ============================================================
# 04. Predict -- New Student
# ============================================================

@router.post("/{dataset}/predict")
def predict(dataset: str, payload: dict = Body(...)):
    _check_dataset(dataset)

    schema_cls = UCIPredictRequest if dataset == "uci" else OuladPredictRequest
    try:
        validated = schema_cls(**payload)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.errors())

    feature_columns = (
        risk_queries.UCI_FEATURE_COLUMNS if dataset == "uci" else risk_queries.OULAD_FEATURE_COLUMNS
    )
    threshold = (
        risk_queries.RISK_THRESHOLD_UCI if dataset == "uci" else risk_queries.RISK_THRESHOLD_OULAD
    )

    try:
        X_new = risk_queries.build_single_record(validated.to_feature_dict(), feature_columns)
        model = risk_queries.load_model(dataset)
        result = risk_queries.predict_risk(model, X_new, threshold)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    row = result.iloc[0]
    return {
        "dataset": dataset,
        "risk_score": float(row["risk_score"]),
        "predicted_label": row["predicted_label"],
        "risk_level": row["risk_level"],
        "threshold_used": threshold,
    }


# ============================================================
# 05. Student risk
# ============================================================

@router.get("/{dataset}/student/{student_id}/risk")
def student_risk(dataset: str, student_id: str):
    _check_dataset(dataset)
    try:
        return ai_tools.get_student_risk(dataset, student_id)
    except StudentNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except (ValueError, DataNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================
# 06. Student SHAP
# ============================================================

@router.get("/{dataset}/student/{student_id}/shap")
def student_shap(dataset: str, student_id: str, top_n: int = 5):
    _check_dataset(dataset)
    if top_n < 1 or top_n > 25:
        raise HTTPException(status_code=400, detail="top_n must be between 1 and 25")

    try:
        result = ai_tools.get_student_shap(dataset, student_id, top_n=top_n)
    except StudentNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except (ValueError, DataNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    # _raw_top_factors is an internal-only field tools.py attaches for
    # search_interventions()'s consumption -- not part of the public shape.
    result.pop("_raw_top_factors", None)
    return result


# ============================================================
# 07. Cohort summary
# ============================================================

@router.get("/{dataset}/cohort-summary")
def cohort_summary(dataset: str, risk_level: str | None = None, limit: int = 10):
    _check_dataset(dataset)
    try:
        return ai_tools.get_cohort_summary(dataset, risk_level=risk_level, limit=limit)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.errors())
    except (ValueError, DataNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================
# 08. OULAD GRU time-series (Student Time-Series Risk Explorer)
# ============================================================

@router.get("/oulad/gru/{id_student}/timeseries")
def gru_timeseries(id_student: int):
    try:
        context, weekly_df, trajectory_df = ai_tools.get_gru_timeseries_context(id_student)
    except StudentNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except (ValueError, DataNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        **sanitize(context),
        "weekly_engagement": sanitize(weekly_df),
        "risk_trajectory": sanitize(trajectory_df),
    }


# ============================================================
# 09. OULAD student directory (picker dropdown)
# ============================================================

@router.get("/oulad/gru/student-directory")
def gru_student_directory(sample_per_outcome: int = 20):
    if sample_per_outcome < 1 or sample_per_outcome > 200:
        raise HTTPException(status_code=400, detail="sample_per_outcome must be between 1 and 200")

    try:
        directory = model_gru.load_oulad_student_directory(sample_per_outcome=sample_per_outcome)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {"students": sanitize(directory)}


# ============================================================
# 10. Personas
# ============================================================

@router.get("/{dataset}/personas")
def personas(dataset: str, limit: int = 100, offset: int = 0):
    _check_dataset(dataset)
    if limit < 1 or limit > 1000:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 1000")
    if offset < 0:
        raise HTTPException(status_code=400, detail="offset must be >= 0")

    try:
        df = risk_queries.load_personas(dataset)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        identifiers = risk_queries.load_persona_student_ids(dataset)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))

    total = len(df)
    start = min(offset, total)
    stop = min(offset + limit, total)
    page = df.iloc[start:stop].copy()
    id_page = identifiers.iloc[start:stop]
    page.insert(0, "row_index", id_page["row_index"].to_numpy())
    if dataset == "oulad":
        page.insert(1, "id_student", id_page["id_student"].to_numpy())
    else:
        page.insert(1, "display_student_id", id_page["display_student_id"].to_numpy())
    return {
        "dataset": dataset,
        "total": total,
        "limit": limit,
        "offset": offset,
        "rows": sanitize(page),
    }


# ============================================================
# 11. Clustering (Clustering Explorer)
# ============================================================

_CENTROID_VIEW_COLS = {
    "uci": clustering.UCI_CENTROID_VIEW_COLS,
    "oulad": clustering.OULAD_CENTROID_VIEW_COLS,
}


@router.get("/{dataset}/clustering/optimal-k")
def clustering_optimal_k(dataset: str):
    """find_optimal_k()'s inertia (elbow) + silhouette score per k, for the
    Clustering Explorer's k-selection chart -- numeric data only, no Figure."""
    _check_dataset(dataset)
    try:
        feature_df = risk_queries.load_feature_matrix(dataset)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    results = clustering.find_optimal_k(feature_df)
    # find_optimal_k() returns {"inertia": {k: val}, "silhouette": {k: val}}
    # with int keys -- JSON object keys must be strings.
    return {
        "dataset": dataset,
        "inertia": {str(k): v for k, v in results["inertia"].items()},
        "silhouette": {str(k): v for k, v in results["silhouette"].items()},
    }


@router.get("/{dataset}/clustering")
def clustering_run(dataset: str, k: int = 3):
    _check_dataset(dataset)
    if k < 2 or k > 10:
        raise HTTPException(status_code=400, detail="k must be between 2 and 10")

    try:
        feature_df = risk_queries.load_feature_matrix(dataset)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    clustered = clustering.run_kmeans(feature_df, k)
    centroids = clustering.cluster_centroid_table(clustered, _CENTROID_VIEW_COLS[dataset])

    # Lightweight per-row (cluster_id, pc1, pc2) points -- the numeric data
    # behind plot_cluster_scatter(), for the frontend to render its own
    # scatter chart instead of a matplotlib image.
    points = clustered[["cluster_id", "pc1", "pc2"]].reset_index().rename(columns={"index": "row_index"})
    try:
        identifiers = risk_queries.load_persona_student_ids(dataset)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    if len(points) != len(identifiers):
        raise HTTPException(
            status_code=503,
            detail=f"{dataset} clustering and student ID row counts differ; IDs cannot be attached.",
        )
    points["row_index"] = identifiers["row_index"].to_numpy()
    if dataset == "oulad":
        points.insert(1, "id_student", identifiers["id_student"].to_numpy())
    else:
        points.insert(
            1,
            "display_student_id",
            identifiers["display_student_id"].to_numpy(),
        )

    return {
        "dataset": dataset,
        "k": k,
        "centroids": sanitize(centroids.reset_index()),
        "points": sanitize(points),
    }


# ============================================================
# 12. Model performance
# ============================================================

@router.get("/{dataset}/model-performance")
def model_performance(dataset: str):
    """Evaluate the trained model on its held-out split at the configured threshold."""
    _check_dataset(dataset)

    try:
        return risk_queries.get_model_performance(dataset)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
