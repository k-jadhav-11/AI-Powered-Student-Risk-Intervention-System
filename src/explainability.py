"""
Per-student risk explanation using SHAP.

Uses SHAP TreeExplainer to explain predictions from:
    - UCI  -> CatBoost model
    - OULAD -> XGBoost model

The module does not retrain models. It loads the same trained models
used by risk_queries.py and identifies the top factors contributing
to an individual student's prediction.
"""

import pandas as pd
import shap

from src.risk import risk_queries


# ============================================================
# SHAP Explainer Cache
# ============================================================

# Cache one SHAP explainer per dataset.

_EXPLAINER_CACHE: dict[str, shap.TreeExplainer] = {}


# ============================================================
# Get / Create SHAP Explainer
# ============================================================

def _get_explainer(dataset: str) -> shap.TreeExplainer:
    """
    Get a cached SHAP explainer for the requested dataset.

    If an explainer does not exist, load the corresponding trained
    model and create a TreeExplainer once.
    """

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
    """
    Explain the risk prediction for one student.

    Parameters
    ----------
    dataset:
        Dataset name: "uci" or "oulad".

    student_row:
        One-row DataFrame containing the student's model features.

    top_n:
        Number of most important SHAP factors to return.

    Returns
    -------
    dict:
        Contains:
        - top_factors
        - factor_directions
        - factor_values
    """

    
    # Validate input
    if len(student_row) != 1:
        raise ValueError(
            f"Expected exactly one student row, got {len(student_row)}"
        )

    if dataset not in {"uci", "oulad"}:
        raise ValueError(
            f"Unsupported dataset: {dataset}. "
            "Expected 'uci' or 'oulad'."
        )

    # Get cached SHAP explainer
    explainer = _get_explainer(dataset)

    # Calculate SHAP values
    shap_values = explainer.shap_values(student_row)

    # Rank features by absolute SHAP contribution
    ranked_factors = sorted(
        zip(student_row.columns, shap_values[0]),
        key=lambda item: abs(item[1]),
        reverse=True,
    )[:top_n]

    # Format explanation
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
    }