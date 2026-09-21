# tests/test_gru_explainability.py
"""src/gru_explainability.py -- Captum Integrated Gradients over the real
trained OULAD GRU. No LLM/network needed; exercised against real students
via src/risk/model_gru.py's existing (unchanged) inference functions."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import gru_explainability
from src.risk import model_gru


@pytest.fixture(scope="module")
def gru_model_and_cfg():
    return model_gru.load_gru_model("oulad")


@pytest.fixture(scope="module")
def real_student_weekly(gru_model_and_cfg):
    _, cfg = gru_model_and_cfg
    directory = model_gru.load_oulad_student_directory(sample_per_outcome=1)
    sid = int(directory.iloc[0]["id_student"])
    return sid, model_gru.build_student_weekly_engagement(sid, max_weeks=cfg["max_weeks"])


def test_explain_gru_prediction_returns_expected_shape(gru_model_and_cfg, real_student_weekly):
    model, cfg = gru_model_and_cfg
    _, weekly = real_student_weekly
    result = gru_explainability.explain_gru_prediction(model, cfg, weekly, top_n=5)
    assert len(result["temporal_factors"]) == 5
    assert set(result["feature_totals"].keys()) == {"total_clicks", "days_active"}
    assert len(result["week_totals"]) == cfg["max_weeks"]
    assert 0.0 <= result["baseline_score"] <= 1.0


def test_temporal_factors_reference_real_weeks_and_features(gru_model_and_cfg, real_student_weekly):
    model, cfg = gru_model_and_cfg
    _, weekly = real_student_weekly
    result = gru_explainability.explain_gru_prediction(model, cfg, weekly, top_n=10)
    for factor in result["temporal_factors"]:
        assert 0 <= factor["week"] < cfg["max_weeks"]
        assert factor["feature"] in cfg["feature_cols"]


def test_direction_matches_attribution_sign(gru_model_and_cfg, real_student_weekly):
    model, cfg = gru_model_and_cfg
    _, weekly = real_student_weekly
    result = gru_explainability.explain_gru_prediction(model, cfg, weekly, top_n=10)
    for factor in result["temporal_factors"]:
        if factor["attribution"] > 0:
            assert factor["direction"] == "increases_risk"
        elif factor["attribution"] < 0:
            assert factor["direction"] == "decreases_risk"


def test_temporal_factors_ranked_by_absolute_magnitude(gru_model_and_cfg, real_student_weekly):
    model, cfg = gru_model_and_cfg
    _, weekly = real_student_weekly
    result = gru_explainability.explain_gru_prediction(model, cfg, weekly, top_n=10)
    magnitudes = [abs(f["attribution"]) for f in result["temporal_factors"]]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_self_average_baseline_is_closer_to_real_score_than_zero_baseline(gru_model_and_cfg, real_student_weekly):
    """Regression test for the documented baseline-choice finding: an
    all-zero baseline scores near-maximal risk (~0.999) for this model
    because a fully-silent sequence looks like a genuine early withdrawal
    in training data, making it a poor 'neutral' reference point. The
    self-average baseline this module actually uses must be meaningfully
    closer to each student's real score."""
    model, cfg = gru_model_and_cfg
    _, weekly = real_student_weekly
    X = model_gru._weekly_df_to_gru_tensor(weekly, cfg)
    with torch.no_grad():
        real_score = model(X).item()
        zero_score = model(torch.zeros_like(X)).item()
    result = gru_explainability.explain_gru_prediction(model, cfg, weekly)
    self_avg_score = result["baseline_score"]
    assert abs(self_avg_score - real_score) < abs(zero_score - real_score)


def test_explain_gru_prediction_runs_fast_enough_for_interactive_use(gru_model_and_cfg, real_student_weekly):
    """Must be fast enough for a chat response, not just eventually correct."""
    import time
    model, cfg = gru_model_and_cfg
    _, weekly = real_student_weekly
    t0 = time.time()
    gru_explainability.explain_gru_prediction(model, cfg, weekly)
    assert time.time() - t0 < 5.0


def test_high_risk_student_shows_net_positive_feature_totals():
    """Directional sanity check on real, clearly at-risk students: a
    student who actually failed/withdrew and scores near-maximal risk
    should, in the majority of cases, show attribution weighted toward
    increasing risk overall relative to their own steadier self-average
    baseline -- not systematically the reverse."""
    model, cfg = model_gru.load_gru_model("oulad")
    directory = model_gru.load_oulad_student_directory(sample_per_outcome=8)
    fail_students = directory[directory["final_result"].isin(["Fail", "Withdrawn"])]

    checked, majority_positive = 0, 0
    for _, row in fail_students.iterrows():
        sid = int(row["id_student"])
        weekly = model_gru.build_student_weekly_engagement(sid, max_weeks=cfg["max_weeks"])
        if weekly["total_clicks"].sum() == 0:
            continue
        score = model_gru.score_student_gru(model, cfg, weekly)
        if score["risk_score"] < 0.9:  # only check clearly high-risk cases
            continue
        result = gru_explainability.explain_gru_prediction(model, cfg, weekly)
        total_attribution = sum(result["feature_totals"].values())
        checked += 1
        if total_attribution > -0.05:  # allow near-zero / mildly negative, just not strongly opposite
            majority_positive += 1

    assert checked > 0, "no clearly high-risk students found in this sample to check"
    assert majority_positive / checked >= 0.5, (
        f"only {majority_positive}/{checked} clearly high-risk students showed "
        f"risk-increasing attribution overall -- expected a majority"
    )
