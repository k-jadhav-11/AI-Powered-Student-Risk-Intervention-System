# src/gru_explainability.py
"""
GRU sequence explainability using Captum Integrated Gradients.
The GRU model works with weekly engagement sequences
Integrated Gradients explains the GRU prediction by identifying
which week and which engagement feature contributed to the risk score.
Features:
    - total_clicks
    - days_active
"""

import numpy as np
import torch
from captum.attr import IntegratedGradients

from src.risk import model_gru


# ============================================================
# Baseline Creation
# ============================================================

def _self_average_baseline(X: torch.Tensor) -> torch.Tensor:
    """
    Create a baseline using the student's own average behavior.
    The student's mean value for each feature is calculated across
    all weeks and repeated for every week.
    This provides a personalized baseline instead of comparing the
    student against an artificial all-zero sequence.
    """

    mean_per_feature = X.mean(dim=1, keepdim=True)

    return mean_per_feature.expand_as(X)


# ============================================================
# GRU Prediction Explanation
# ============================================================

def explain_gru_prediction(
    model,
    cfg: dict,
    weekly_df,
    top_n: int = 5,
    n_steps: int = 50,
) -> dict:
    """
    Explain one student's GRU risk prediction.

    Uses Integrated Gradients to identify the most important
    week-feature combinations contributing to the prediction.

    Parameters
    ----------
    model: Trained GRU model.
    cfg: GRU configuration containing feature columns.
    weekly_df: Student's weekly engagement data.
    top_n: Number of most important week-feature combinations.
    n_steps: Number of Integrated Gradients interpolation steps.
    
    Returns
    -------
    dict:
        temporal_factors:
            Top week-feature contributions.

        feature_totals:
            Total attribution for each feature.

        week_totals:
            Total attribution for each week.

        baseline_score:
            GRU prediction for the personalized baseline.
    """

    # Convert weekly data to GRU input tensor
    X = model_gru._weekly_df_to_gru_tensor(
        weekly_df,
        cfg,
    )

    
    # Create personalized baseline
    baseline = _self_average_baseline(X)

    # Calculate baseline prediction
    with torch.no_grad():
        baseline_score = float(model(baseline).item())

    # Calculate Integrated Gradients
    ig = IntegratedGradients(model)

    attributions = ig.attribute(
        X,
        baselines=baseline,
        n_steps=n_steps,
    )

    attr = (
        attributions
        .squeeze(0)
        .detach()
        .numpy()
    )

    # Get feature names
    feature_names = cfg["feature_cols"]

    # Create week-feature attribution list
    factors = []

    for week in range(attr.shape[0]):
        for feature_idx, feature_name in enumerate(feature_names):

            value = float(attr[week, feature_idx])

            factors.append(
                (
                    week,
                    feature_name,
                    value,
                )
            )

    # Rank by absolute attribution
    ranked = sorted(
        factors,
        key=lambda item: abs(item[2]),
        reverse=True,
    )[:top_n]

    # Format temporal factors
    temporal_factors = [
        {
            "week": week,
            "feature": feature_name,
            "attribution": round(value, 5),
            "direction": (
                "increases_risk"
                if value > 0
                else "decreases_risk"
            ),
        }
        for week, feature_name, value in ranked
    ]

    # Aggregate attribution by feature
    feature_totals = {
        feature_name: float(attr[:, idx].sum())
        for idx, feature_name in enumerate(feature_names)
    }

    # Aggregate attribution by week
    week_totals = {
        int(week): float(attr[week].sum())
        for week in range(attr.shape[0])
    }

    # Return explanation
    return {
        "temporal_factors": temporal_factors,
        "feature_totals": feature_totals,
        "week_totals": week_totals,
        "baseline_score": round(baseline_score, 4),
    }