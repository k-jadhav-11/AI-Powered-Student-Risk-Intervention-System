# src/explainability.py

import pandas as pd
import shap

from src.risk import risk_queries


# ============================================================
# SHAP Explainer Cache
# ============================================================

_EXPLAINER_CACHE: dict[str, shap.TreeExplainer] = {}


# ============================================================
# Get / Create SHAP Explainer
# ============================================================

def _get_explainer(dataset: str) -> shap.TreeExplainer:
    """Returns a cached SHAP explainer for the dataset.
    Loads the trained model and creates the explainer when needed."""

    if dataset not in _EXPLAINER_CACHE:
        model = risk_queries.load_model(dataset)
        _EXPLAINER_CACHE[dataset] = shap.TreeExplainer(model)

    return _EXPLAINER_CACHE[dataset]


# ============================================================
# Explain Student Prediction
# ============================================================

def explain_prediction(
    dataset: str,
    student_row: pd.DataFrame,
    top_n: int = 5,
) -> dict:
    """Explains one student's risk prediction using SHAP.
    Returns the top factors, directions, values, and SHAP contributions."""

    if len(student_row) != 1:
        raise ValueError(
            f"Expected exactly one student row, got {len(student_row)}"
        )

    if dataset not in {"uci", "oulad"}:
        raise ValueError(
            f"Unsupported dataset: {dataset}. "
            "Expected 'uci' or 'oulad'."
        )

    explainer = _get_explainer(dataset)

    shap_values = explainer.shap_values(student_row)

    ranked_factors = sorted(
        zip(student_row.columns, shap_values[0]),
        key=lambda item: abs(item[1]),
        reverse=True,
    )[:top_n]

    return {
        "top_factors": [
            feature
            for feature, _ in ranked_factors
        ],
        "factor_directions": [
            "increases risk" if value > 0 else "decreases risk"
            for _, value in ranked_factors
        ],
        "factor_values": [
            float(student_row.iloc[0][feature])
            for feature, _ in ranked_factors
        ],
        "shap_values": [
            float(value)
            for _, value in ranked_factors
        ],
    }

