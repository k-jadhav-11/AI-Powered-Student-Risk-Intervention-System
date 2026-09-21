"""
The structured output contract for Sprint 4.

Defines the Pydantic schemas used to keep risk predictions,
SHAP explanations, GRU time-series results, and cohort questions
in a consistent and validated structure.
"""

from typing import Literal
from pydantic import BaseModel, Field


# ============================================================
# Risk Factor Schema
# ============================================================

class RiskFactor(BaseModel):
    """
    Represents one SHAP-based factor contributing to a student's
    risk prediction. Stores the real feature, value, contribution,
    direction, and a teacher-friendly explanation.
    """

    # Real feature/column name from the feature matrix
    feature: str
    value: float
    contribution: float
    direction: Literal["increases_risk", "decreases_risk"]
    explanation: str


# ============================================================
# Temporal Factor Schema
# ============================================================

class TemporalFactor(BaseModel):
    """
    Represents one week's contribution to the GRU time-series
    prediction. The attribution shows how a specific weekly
    feature contributed to the predicted risk.
    """

    week: int
    feature: Literal["total_clicks", "days_active"]
    attribution: float
    direction: Literal["increases_risk", "decreases_risk"]


# ============================================================
# Model Metrics Schema
# ============================================================

class ModelMetrics(BaseModel):
    """
    Stores the evaluation metrics of the trained risk model.
    These values come from the model summary and are not
    calculated or estimated by the LLM.
    """

    precision: float
    recall: float
    f1: float
    roc_auc: float | None = None


# ============================================================
# Risk Explanation Schema
# ============================================================

class RiskExplanation(BaseModel):
    """
    Stores the complete risk explanation for one student.
    It combines the prediction, model metrics, SHAP factors,
    persona information, and RAG-based intervention suggestions.
    """

    # Row-index student identifier used by the risk dashboard
    student_id: str

    display_student_id: str

    dataset: Literal["uci", "oulad"]

    risk_score: float = Field(ge=0.0, le=1.0)

    risk_level: Literal["Low", "Medium", "High"]

    predicted_label: Literal["at_risk", "not_at_risk"]

    model_name: str

    model_metrics: ModelMetrics

    academic_factors: list[RiskFactor] = Field(default_factory=list)

    engagement_factors: list[RiskFactor] = Field(default_factory=list)

    contextual_factors: list[RiskFactor] = Field(default_factory=list)

    persona_name: str

    persona_avg_risk_score: float = Field(ge=0.0, le=1.0)

    persona_student_count: int

    mitigation_strategies: list[str] = Field(default_factory=list)

    evidence_sources: list[str] = Field(default_factory=list)


# ============================================================
# GRU Time-Series Context Schema
# ============================================================

class GRUTimeSeriesContext(BaseModel):
    """
    Stores the real OULAD student's GRU time-series prediction context.
    It keeps GRU and XGBoost predictions separate and can also store
    weekly temporal factors from Integrated Gradients.
    """
    student_id: str

    gru_risk_score: float = Field(ge=0.0, le=1.0)

    gru_risk_level: Literal["Low", "Medium", "High"]

    xgboost_risk_score: float = Field(ge=0.0, le=1.0)

    xgboost_risk_level: Literal["Low", "Medium", "High"]

    models_agree: bool

    last_active_week: int

    first_high_risk_week: int | None

    actual_final_result: str | None

    temporal_factors: list[TemporalFactor] = Field(default_factory=list)


# ============================================================
# Cohort Question Schema
# ============================================================

class CohortQuestion(BaseModel):
    """
    Represents the parsed intent of a cohort-level question.
    It identifies the dataset, optional risk-level filter,
    and maximum number of students to return.
    """

    dataset: Literal["uci", "oulad"]
    risk_level: Literal["Low", "Medium", "High"] | None = None
    limit: int = Field(default=10, ge=1, le=100)