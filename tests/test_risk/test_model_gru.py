# tests/test_risk/test_model_gru.py
"""src/risk/model_gru.py -- GRU checkpoint loading, live weekly-engagement
construction from raw OULAD data, and the trajectory replay logic. Exercised
against the real trained checkpoint and real studentVle.csv."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.risk import model_gru


def test_load_gru_model_returns_model_and_config():
    model, cfg = model_gru.load_gru_model("oulad")
    assert hasattr(model, "forward")
    assert {"max_weeks", "feature_cols", "log1p_clicks"} <= set(cfg.keys())


def test_build_student_weekly_engagement_shape():
    _, cfg = model_gru.load_gru_model("oulad")
    directory = model_gru.load_oulad_student_directory(sample_per_outcome=1)
    sid = int(directory.iloc[0]["id_student"])
    weekly = model_gru.build_student_weekly_engagement(sid, max_weeks=cfg["max_weeks"])
    assert len(weekly) == cfg["max_weeks"]
    assert {"week_number", "total_clicks", "days_active"} <= set(weekly.columns)


def test_load_oulad_student_directory_excludes_zero_activity_students():
    """Regression test: earlier versions of the sampler could hand out a
    student with zero recorded VLE activity, which the Student
    Time-Series Risk Explorer could do nothing with."""
    active_ids = model_gru.get_active_student_ids()
    directory = model_gru.load_oulad_student_directory(sample_per_outcome=5)
    assert directory["id_student"].isin(active_ids).all()
