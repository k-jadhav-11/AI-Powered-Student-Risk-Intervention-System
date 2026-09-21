# tests/test_risk/test_evaluation.py
"""src/risk/evaluation.py -- the live GRU-vs-XGBoost sampled comparison.
Kept small (sample_per_outcome=3) so this test runs quickly; the module
itself is used with a larger sample from the Streamlit UI."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.risk import evaluation, model_gru, risk_queries


def test_run_live_sampled_comparison_returns_metrics_for_both_models():
    gru_model, gru_cfg = model_gru.load_gru_model("oulad")
    xgb_model = risk_queries.load_model("oulad")
    comparison, metrics = evaluation.run_live_sampled_comparison(
        gru_model, gru_cfg, xgb_model, sample_per_outcome=3,
    )
    assert len(comparison) == 3 * 4  # 4 outcomes
    assert set(metrics.keys()) == {"GRU (sequence)", "XGBoost (tabular)"}
    for m in metrics.values():
        assert 0.0 <= m["precision"] <= 1.0
        assert 0.0 <= m["recall"] <= 1.0
        assert 0.0 <= m["f1"] <= 1.0
