# src/risk/evaluation.py
"""
Sprint 3 -- model evaluation and comparison.

get_test_split() (reconstructing a held-out split for a live PR-curve chart)
lives in risk_queries.py since it's a simple data-loading operation with no
cross-model comparison logic. This module is for genuinely comparative
evaluation: run_live_sampled_comparison() scores the SAME sampled OULAD
students with both the GRU (sequence) and XGBoost (tabular) models and
reports precision/recall/F1 for each, live -- a smaller-sample stand-in for
the frozen baseline_vs_gru_oulad.png comparison chart.
"""
import pandas as pd

from .risk_queries import OULAD_FEATURE_COLUMNS, RISK_THRESHOLD_OULAD, build_single_record, predict_risk
from .model_gru import build_weekly_engagement_batch, build_oulad_record_from_raw, \
    load_oulad_student_directory, score_students_gru_batch

def run_live_sampled_comparison(gru_model, gru_cfg, xgb_model, sample_per_outcome: int = 15, seed: int = 7):
    """A live, sample-sized stand-in for the frozen baseline_vs_gru_oulad.png:
    samples real students across all 4 outcomes, builds both models' inputs
    from the same raw data in one batched pass, scores both, and returns a
    tidy comparison table plus precision/recall/F1 for each model against
    the real final_result labels on that sample. Deliberately sample-sized
    (not the full ~32K-student population like the frozen chart) so it stays
    fast enough to run on a page load.
    """
    directory = load_oulad_student_directory(sample_per_outcome=sample_per_outcome, seed=seed)
    ids = directory["id_student"].tolist()

    weekly_batch = build_weekly_engagement_batch(ids, max_weeks=gru_cfg["max_weeks"])
    gru_scores = score_students_gru_batch(gru_model, gru_cfg, weekly_batch)

    xgb_rows = []
    for sid in ids:
        try:
            record, _ = build_oulad_record_from_raw(sid, weekly_batch[sid])
            X_new = build_single_record(record, OULAD_FEATURE_COLUMNS)
            xgb_pred = predict_risk(xgb_model, X_new, RISK_THRESHOLD_OULAD).iloc[0]
            xgb_rows.append({
                "id_student": sid,
                "xgb_risk_score": xgb_pred["risk_score"],
                "xgb_predicted_label": xgb_pred["predicted_label"],
            })
        except ValueError:
            xgb_rows.append({"id_student": sid, "xgb_risk_score": None, "xgb_predicted_label": None})
    xgb_scores = pd.DataFrame(xgb_rows)

    comparison = (
        directory.merge(gru_scores.rename(columns={
            "risk_score": "gru_risk_score", "predicted_label": "gru_predicted_label"}), on="id_student")
        .merge(xgb_scores, on="id_student")
    )
    comparison["actual_at_risk"] = comparison["final_result"].isin(["Fail", "Withdrawn"])

    def _prf(pred_col):
        pred = comparison[pred_col] == "at_risk"
        actual = comparison["actual_at_risk"]
        tp = int((pred & actual).sum())
        fp = int((pred & ~actual).sum())
        fn = int((~pred & actual).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}

    metrics = {"GRU (sequence)": _prf("gru_predicted_label"), "XGBoost (tabular)": _prf("xgb_predicted_label")}
    return comparison, metrics
