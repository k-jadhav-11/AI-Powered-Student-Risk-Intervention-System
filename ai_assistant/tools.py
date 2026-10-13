"""
function-calling tools.

These tools allow the AI agent to access student risk predictions,
SHAP explanations, engagement data, GRU explanations, model metrics,
cohort information, and RAG-based intervention suggestions.

Each tool has a specific responsibility so the agent can call only
the information required for a user's question.

Important:
- Tools do not create or modify model predictions.
- Risk scores come from the existing trained models.
- SHAP values come from the explainability pipeline.
- GRU explanations come from the GRU explainability pipeline.
- Recommendations come from the curated RAG knowledge base.
- Student-specific answers must remain grounded in actual tool output.

get_risk_explanation() and get_gru_timeseries_context() are higher-level
composing functions. They combine multiple granular tools into one
complete response for existing non-agentic pages and test harnesses.
"""

from src import explainability, gru_explainability
from src.risk import risk_queries, model_gru
from src.analytics.exceptions import StudentNotFoundError

from typing import Literal
import pandas as pd

from . import feature_metadata, rag
from .schemas import (
    CohortQuestion,
    GRUTimeSeriesContext,
    ModelMetrics,
    RiskExplanation,
    RiskFactor,
    TemporalFactor,
)


# ============================================================
# Shared Student ID Validation
# ============================================================

def _resolve_row_index(dataset: str, student_id: str) -> int:
    """Validates a dataset and resolves a 1-based student ID to a row index.
    Raises StudentNotFoundError when the student cannot be resolved."""
    if dataset not in ("uci", "oulad"):
        raise ValueError(
            f"dataset must be 'uci' or 'oulad', got {dataset!r}"
        )

    try:
        display_num = int(student_id)
    except (TypeError, ValueError):
        raise StudentNotFoundError(
            f"'{student_id}' isn't a valid student_id for {dataset} -- expected "
            f"the row number shown on the Risk Dashboard — {dataset.upper()} page, "
            f"starting at 1 (e.g. '43')."
        )

    idx = display_num - 1
    risk_df = risk_queries.load_risk_scores(dataset)

    if idx not in risk_df.index:
        raise StudentNotFoundError(
            f"student {display_num} not found in {dataset} risk scores "
            f"(valid range: 1-{risk_df.index.max() + 1})."
        )

    return idx


def resolve_student_id_from_row_index(
    dataset: str,
    row_index: int,
) -> str:
    """Resolves a user-facing 1-based row index to the student's display ID.
    Supports datasets where the real student ID differs from the row number."""
    if dataset not in ("uci", "oulad"):
        raise ValueError(
            f"Unsupported dataset: {dataset}"
        )

    try:
        display_num = int(row_index)
    except (TypeError, ValueError):
        raise StudentNotFoundError(
            f"Invalid row index: {row_index}"
        )

    if display_num < 1:
        raise StudentNotFoundError(
            f"Row index must be >= 1. Got: {display_num}"
        )

    idx = display_num - 1
    risk_df = risk_queries.load_risk_scores(dataset)

    if idx not in risk_df.index:
        raise StudentNotFoundError(
            f"Student row index {display_num} not found in {dataset} dataset."
        )

    row = risk_df.loc[idx]

    return _display_student_id(
        dataset,
        idx,
        row,
    )


def resolve_row_index_from_student_id(
    dataset: str,
    student_id: str,
) -> int:
    """Resolves a real student ID to its internal dataframe row index.
    Supports both real and display student ID columns."""
    if dataset not in ("uci", "oulad"):
        raise ValueError(
            f"Unsupported dataset: {dataset}"
        )

    target = str(student_id).strip()

    if not target:
        raise StudentNotFoundError(
            "Student ID cannot be empty."
        )

    risk_df = risk_queries.load_risk_scores(dataset)

    for col in ("id_student", "display_student_id"):
        if col not in risk_df.columns:
            continue

        series = risk_df[col]

        try:
            numeric_target = int(target)
        except (TypeError, ValueError):
            numeric_target = None

        if (
            numeric_target is not None
            and pd.api.types.is_numeric_dtype(series)
        ):
            matches = series == numeric_target
        else:
            matches = (
                series.astype(str).str.strip() == target
            )

        if matches.any():
            matching_indexes = risk_df.index[matches]

            if len(matching_indexes) > 0:
                return int(matching_indexes[0])

    raise StudentNotFoundError(
        f"Student ID '{student_id}' not found in {dataset} dataset."
    )

    
# ============================================================
# Display Student ID Helper
# ============================================================

def _display_student_id(dataset: str, idx: int, row: pd.Series) -> str:
    """The real-world-facing ID for one student -- separate from `idx`,
    the row index every tool actually looks up by."""
    for col in ("id_student", "display_student_id"):
        if col in row.index and pd.notna(row[col]):
            val = row[col]
            return str(int(val)) if col == "id_student" else str(val)
    return f"{dataset}-{idx}"


# ============================================================
# 01. Get Student Risk
# ============================================================

def get_student_risk(dataset: Literal["uci", "oulad"], student_id: str) -> dict:
    """
    Return the basic risk prediction for one student.

    This tool provides only the student's risk score, risk level,
    predicted label, and model name. It intentionally does not return
    SHAP factors, persona information, or recommendations, making it
    suitable for simple questions such as "Is this student at risk?".
    """

    # Validate and resolve the row-index student ID
    idx = _resolve_row_index(dataset, student_id)

    # Load the student's risk prediction
    row = risk_queries.load_risk_scores(dataset).loc[idx]

    # Return only the basic risk information
    return {
        "student_id": str(idx + 1),
        "display_student_id": _display_student_id(dataset, idx, row),
        "dataset": dataset,
        "risk_score": float(row["risk_score"]),
        "risk_level": row["risk_level"],
        "predicted_label": row["predicted_label"],
        "model_name": risk_queries.get_model_class_name(dataset),
    }


# ============================================================
# Categorize SHAP Factors
# ============================================================

def _categorize_factors(dataset: str, factors: dict) -> dict:
    """
    Split the SHAP explanation into academic, engagement, and contextual
    factor groups using the real feature metadata definitions.

    The function preserves the actual feature values and SHAP contributions
    returned by explainability.explain_prediction(). It only reorganizes the
    factors and creates their teacher-friendly explanations; it does not
    calculate or invent new model evidence.
    """

    # Create empty buckets for the three feature categories
    buckets = {
        "academic": [],
        "engagement": [],
        "contextual": [],
    }

    # Get the real SHAP values when available
    shap_values = factors.get("shap_values")

    # Fallback for an older explainability.py without shap_values
    if shap_values is None:
        shap_values = [
            1.0 if d == "increases risk" else -1.0
            for d in factors["factor_directions"]
        ]

    # Process every returned SHAP factor
    for feature, direction_str, value, shap_value in zip(
        factors["top_factors"],
        factors["factor_directions"],
        factors["factor_values"],
        shap_values,
    ):

        # Convert direction text into the schema's expected values
        direction = (
            "increases_risk"
            if direction_str == "increases risk"
            else "decreases_risk"
        )

        # Get the feature's category from feature_metadata.py
        category = feature_metadata.get_category(dataset, feature)

        # Create a validated RiskFactor object
        buckets[category].append(
            RiskFactor(
                feature=feature,
                value=value,
                contribution=shap_value,
                direction=direction,
                explanation=feature_metadata.build_factor_explanation(
                    dataset,
                    feature,
                    value,
                    direction,
                ),
            )
        )

    return buckets


# ============================================================
# 02. Get Student SHAP Explanation
# ============================================================

def get_student_shap(
    dataset: Literal["uci", "oulad"],
    student_id: str,
    top_n: int = 5,
) -> dict:
    """
    Return the student's top SHAP-derived risk factors.

    The factors are divided into academic, engagement, and contextual
    categories. This tool is intended for "why" questions and does not
    return the student's risk score, model metrics, persona, or
    intervention recommendations.
    """

    # Validate and resolve the student's row index
    idx = _resolve_row_index(dataset, student_id)

    # Select the correct feature columns for the dataset
    feature_cols = (
        risk_queries.UCI_FEATURE_COLUMNS
        if dataset == "uci"
        else risk_queries.OULAD_FEATURE_COLUMNS
    )

    # Load the student's feature row
    feature_row = (
        risk_queries
        .load_feature_matrix(dataset)
        .loc[[idx]][feature_cols]
    )

    # Generate the student's SHAP explanation
    factors = explainability.explain_prediction(
        dataset,
        feature_row,
        top_n=top_n,
    )

    # Organize factors into the three categories
    buckets = _categorize_factors(dataset, factors)

    # Return structured SHAP information
    return {
        "student_id": str(idx + 1),
        "dataset": dataset,
        "academic_factors": [
            f.model_dump()
            for f in buckets["academic"]
        ],
        "engagement_factors": [
            f.model_dump()
            for f in buckets["engagement"]
        ],
        "contextual_factors": [
            f.model_dump()
            for f in buckets["contextual"]
        ],
        # Internal value used by search_interventions()
        "_raw_top_factors": factors["top_factors"],
    }


# ============================================================
# 03. Get Model Information
# ============================================================

def get_model_information(dataset: Literal["uci", "oulad"]) -> dict:
    """
    Return the trained model name and held-out metrics at the configured threshold.
    """
    if dataset not in ("uci", "oulad"):
        raise ValueError(
            f"dataset must be 'uci' or 'oulad', got {dataset!r}"
        )

    performance = risk_queries.get_model_performance(dataset)
    return {
        "dataset": dataset,
        "model_name": performance["model_name"],
        "threshold_used": performance["threshold_used"],
        "model_metrics": ModelMetrics(
            precision=performance["precision"],
            recall=performance["recall"],
            f1=performance["f1"],
            roc_auc=performance["roc_auc"],
        ).model_dump(),
    }





# ============================================================
# 04. Search Interventions
# ============================================================

def search_interventions(
    risk_factors: list[str],
    top_k: int = 2,
) -> dict:
    """
    Search the curated intervention knowledge base for recommendations
    related to real SHAP risk factors.

    This is the RAG tool used when the agent needs to answer "What should
    I do?" questions. The supplied risk_factors should come from an actual
    SHAP result, and the returned suggestions include their source IDs so
    recommendations remain traceable to the knowledge base.
    """

    # Retrieve relevant intervention content using RAG
    result = rag.retrieve_remedial_content(
        risk_factors,
        k=top_k,
    )

    # Return interventions together with their source IDs
    return {
        "risk_factors_searched": risk_factors,
        "interventions": [
            {
                "suggestion": text,
                "source_id": sid,
            }
            for text, sid in zip(
                result["suggestions"],
                result["source_ids"],
            )
        ],
        "retrieval_method": result["backend"],
    }


# ============================================================
# Get Persona Context
# ============================================================

def _get_persona_context(dataset: str, idx: int) -> dict:
    """
    Retrieve the persona assigned to a student and summarize that persona.

    The function finds the student's persona, calculates the average risk
    score for students in the same persona, and returns the number of
    students in that group. It is used as supporting context for the
    complete student risk explanation.
    """

    # Load the persona assignments
    personas = risk_queries.load_personas(dataset)

    # Get this student's persona
    persona_name = personas.loc[idx, "persona_name"]

    # Select all students belonging to the same persona
    persona_group = personas[
        personas["persona_name"] == persona_name
    ]

    # Return persona-level context
    return {
        "persona_name": persona_name,
        "persona_avg_risk_score": float(
            persona_group["risk_score"].mean()
        ),
        "persona_student_count": int(
            len(persona_group)
        ),
    }


# ============================================================
# Get Complete Risk Explanation
# ============================================================

def get_risk_explanation(
    dataset: str,
    student_id: str,
) -> dict:
    """
    Build the complete RiskExplanation for one student.

    This is a composing wrapper that combines the student's risk prediction,
    SHAP factors, model metrics, persona context, and RAG-based intervention
    suggestions. It performs no new prediction or explanation logic; each
    piece of information comes from the existing granular tools.
    """

    # Validate and resolve the student's row index
    idx = _resolve_row_index(dataset, student_id)

    # Get the basic risk prediction
    risk = get_student_risk(
        dataset,
        student_id,
    )

    # Get the student's top SHAP factors
    shap = get_student_shap(
        dataset,
        student_id,
        top_n=5,
    )

    # Get model name and evaluation metrics
    model_info = get_model_information(dataset)

    # Get the student's persona information
    persona = _get_persona_context(
        dataset,
        idx,
    )

    # Search interventions using the actual SHAP factors
    rag_result = search_interventions(
        shap["_raw_top_factors"],
        top_k=2,
    )

    # Build and validate the complete RiskExplanation object
    return RiskExplanation(
        student_id=risk["student_id"],
        display_student_id=risk["display_student_id"],
        dataset=dataset,
        risk_score=risk["risk_score"],
        risk_level=risk["risk_level"],
        predicted_label=risk["predicted_label"],
        model_name=model_info["model_name"],
        model_metrics=ModelMetrics(
            **model_info["model_metrics"]
        ),
        academic_factors=[
            RiskFactor(**f)
            for f in shap["academic_factors"]
        ],
        engagement_factors=[
            RiskFactor(**f)
            for f in shap["engagement_factors"]
        ],
        contextual_factors=[
            RiskFactor(**f)
            for f in shap["contextual_factors"]
        ],
        persona_name=persona["persona_name"],
        persona_avg_risk_score=persona["persona_avg_risk_score"],
        persona_student_count=persona["persona_student_count"],
        mitigation_strategies=[
            i["suggestion"]
            for i in rag_result["interventions"]
        ],
        evidence_sources=[
            i["source_id"]
            for i in rag_result["interventions"]
        ],
    ).model_dump()


# ============================================================
# 05. Get Cohort Summary
# ============================================================

def get_cohort_summary(
    dataset: Literal["uci", "oulad"],
    risk_level: str | None = None,
    limit: int = 10,
) -> dict:
    """
    Return an aggregate summary of students in a dataset.

    The tool can optionally filter students by Low, Medium, or High risk
    and returns the total student count, risk-level distribution, and
    highest-risk students within the selected group. Input validation is
    handled by CohortQuestion so the agent cannot request unsupported
    risk levels or invalid limits.
    """

    # Validate cohort query parameters
    question = CohortQuestion(
        dataset=dataset,
        risk_level=risk_level,
        limit=limit,
    )

    # Load all student risk scores
    risk_df = risk_queries.load_risk_scores(
        question.dataset
    )

    # Apply the optional risk-level filter
    filtered = (
        risk_df
        if question.risk_level is None
        else risk_df[
            risk_df["risk_level"] == question.risk_level
        ]
    )

    # Count students in each risk category
    counts = (
        risk_df["risk_level"]
        .value_counts()
        .to_dict()
    )

    # Sort selected students by highest risk
    top = (
        filtered
        .sort_values("risk_score", ascending=False)
        .head(question.limit)
    )

    # Return cohort-level information
    return {
        "dataset": question.dataset,
        "total_students": int(len(risk_df)),
        "risk_level_counts": counts,
        "filter_applied": question.risk_level,
        "top_student_ids": [str(i + 1) for i in top.index],
        "top_display_student_ids": [
            _display_student_id(question.dataset, i, top.loc[i])
            for i in top.index
        ],
        "top_risk_scores": [
            round(float(s), 4)
            for s in top["risk_score"]
        ],
    }


# ============================================================
# 06.Get Student Engagement
# ============================================================

def get_student_engagement(id_student: int) -> dict:
    """
    Return the weekly engagement summary for a real OULAD student.

    The tool reports total clicks, active days and weeks, engagement trend,
    last active week, and a disclosed first-significant-decline heuristic.
    It uses the real OULAD id_student identifier, which is different from
    the row-index student ID used by the tabular risk tools.
    """

    # Load the GRU configuration
    _, gru_cfg = model_gru.load_gru_model("oulad")

    # Build the student's weekly engagement data
    weekly = model_gru.build_student_weekly_engagement(
        id_student,
        max_weeks=gru_cfg["max_weeks"],
    )

    # Keep only weeks with recorded activity
    active = (
        weekly[weekly["total_clicks"] > 0]
        .sort_values("week_number")
        .reset_index(drop=True)
    )

    # Make sure the student has recorded engagement data
    if active.empty:
        raise StudentNotFoundError(
            f"OULAD student {id_student} has zero recorded VLE activity -- nothing to summarize."
        )

    # Store the first significant decline week
    decline_week = None

    # Store previous active-week click values
    running_clicks = []

    # Check weekly engagement for a sustained decline
    for i, row in active.iterrows():

        if running_clicks:

            # Calculate the mean clicks from previous active weeks
            running_mean = (
                sum(running_clicks) / len(running_clicks)
            )

            # Check whether the current week is below half the running mean
            is_low = (
                row["total_clicks"] < running_mean / 2
            )

            # Check whether the next active week is also below the threshold
            next_also_low = (
                i + 1 < len(active)
                and active.loc[i + 1, "total_clicks"]
                < running_mean / 2
            )

            # Record the first sustained decline
            if (
                is_low
                and next_also_low
                and decline_week is None
            ):
                decline_week = int(
                    row["week_number"]
                )

        # Add current week's clicks to the running history
        running_clicks.append(
            row["total_clicks"]
        )

    # Determine the overall simple trend direction
    trend_direction = (
        "rising"
        if (
            len(active) >= 2
            and active["total_clicks"].iloc[-1]
            > active["total_clicks"].iloc[0]
        )
        else "falling"
    )

    # Return the engagement summary
    return {
        "student_id": str(id_student),
        "total_clicks": int(
            active["total_clicks"].sum()
        ),
        "total_days_active": int(
            active["days_active"].sum()
        ),
        "num_weeks_active": int(
            len(active)
        ),
        "last_active_week": int(
            active["week_number"].iloc[-1]
        ),
        "engagement_trend_direction": trend_direction,
        "first_significant_decline_week": decline_week,
        "_weekly_chart_data": weekly[
            ["week_number", "total_clicks", "days_active"]
        ].to_dict(orient="records"),
    }


# ============================================================
# 07. Get Student GRU Explanation
# ============================================================

def get_student_gru_explanation(
    id_student: int,
) -> dict:
    """
    Return the GRU risk prediction and temporal attribution for a real
    OULAD student.

    The GRU score and risk level come from the existing GRU model, while
    temporal factors come from Integrated Gradients. The result explains
    which weeks and which GRU input features (total_clicks or days_active)
    contributed most to the prediction.
    """

    # Load the trained GRU model and configuration
    gru_model, gru_cfg = model_gru.load_gru_model("oulad")

    # Build weekly engagement data
    weekly = model_gru.build_student_weekly_engagement(
        id_student,
        max_weeks=gru_cfg["max_weeks"],
    )

    # Make sure the student has recorded VLE activity
    if weekly["total_clicks"].sum() == 0:
        raise StudentNotFoundError(
            f"OULAD student {id_student} has zero recorded VLE activity -- nothing to explain."
        )

    # Calculate the GRU risk prediction
    gru_result = model_gru.score_student_gru(
        gru_model,
        gru_cfg,
        weekly,
    )

    # Calculate temporal Integrated Gradients attribution
    attribution = gru_explainability.explain_gru_prediction(
        gru_model,
        gru_cfg,
        weekly,
        top_n=5,
    )

    # Return the GRU explanation
    return {
        "student_id": str(id_student),
        "gru_risk_score": gru_result["risk_score"],
        "gru_risk_level": gru_result["risk_level"],
        "temporal_factors": attribution["temporal_factors"],
        "feature_totals": attribution["feature_totals"],
    }


# ============================================================
# 08. Get GRU Time-Series Context
# ============================================================

def get_gru_timeseries_context(
    id_student: int,
) -> dict:
    """
    Build the complete time-series context for a real OULAD student.

    This combines the GRU explanation, engagement summary, GRU risk
    trajectory, and a fresh XGBoost tabular prediction for the same
    real student. It also returns the weekly and trajectory DataFrames
    required by the Student Time-Series Risk Explorer to render its chart.
    """

    # Load the GRU model and configuration
    gru_model, gru_cfg = model_gru.load_gru_model("oulad")

    # Build the student's weekly engagement sequence
    weekly = model_gru.build_student_weekly_engagement(
        id_student,
        max_weeks=gru_cfg["max_weeks"],
    )

    # Make sure the student has usable engagement data
    if weekly["total_clicks"].sum() == 0:
        raise StudentNotFoundError(
            f"OULAD student {id_student} has zero recorded VLE activity -- nothing to build a "
            f"time series from. Try a student sampled from the Student Time-Series Risk Explorer page."
        )

    # Get the GRU explanation
    gru_explanation = get_student_gru_explanation(
        id_student
    )

    # Calculate the GRU risk trajectory
    trajectory, first_high_risk_week = (
        model_gru.compute_gru_risk_trajectory(
            gru_model,
            gru_cfg,
            weekly,
        )
    )

    # Load the trained XGBoost model
    xgb_model = risk_queries.load_model("oulad")

    # Build the tabular record from the same real OULAD student
    record, actual_final_result = (
        model_gru.build_oulad_record_from_raw(
            id_student,
            weekly,
        )
    )

    # Convert the record into the expected feature format
    X_new = risk_queries.build_single_record(
        record,
        risk_queries.OULAD_FEATURE_COLUMNS,
    )

    # Generate the XGBoost risk prediction
    xgb_result = risk_queries.predict_risk(
        xgb_model,
        X_new,
        risk_queries.RISK_THRESHOLD_OULAD,
    )

    # Get the single prediction row
    xgb_row = xgb_result.iloc[0]

    # Convert the GRU score into the GRU predicted label
    gru_predicted_label = (
        "at_risk"
        if (
            gru_explanation["gru_risk_score"]
            >= risk_queries.RISK_THRESHOLD_OULAD_GRU
        )
        else "not_at_risk"
    )

    # Build the validated GRU time-series context
    context = GRUTimeSeriesContext(
        student_id=str(id_student),
        gru_risk_score=gru_explanation["gru_risk_score"],
        gru_risk_level=gru_explanation["gru_risk_level"],
        xgboost_risk_score=float(
            xgb_row["risk_score"]
        ),
        xgboost_risk_level=xgb_row["risk_level"],
        models_agree=(
            gru_predicted_label
            == xgb_row["predicted_label"]
        ),
        last_active_week=int(
            weekly.loc[
                weekly["total_clicks"] > 0,
                "week_number",
            ].max()
        ),
        first_high_risk_week=first_high_risk_week,
        actual_final_result=actual_final_result,
        temporal_factors=[
            TemporalFactor(**tf)
            for tf in gru_explanation["temporal_factors"]
        ],
    ).model_dump()

    # Return the grounded context and chart data
    return context, weekly, trajectory