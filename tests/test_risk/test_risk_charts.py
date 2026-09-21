# tests/test_risk/test_risk_charts.py
"""src/risk/risk_charts.py -- pure plotting helpers. Verifies each function
returns a real matplotlib Figure without raising, given realistic inputs;
doesn't inspect pixel content (that's what visual review is for)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.risk import risk_charts, risk_queries


def test_plot_risk_distribution_returns_figure():
    predictions = risk_queries.load_risk_scores("uci")
    fig = risk_charts.plot_risk_distribution(predictions, threshold=risk_queries.RISK_THRESHOLD_UCI)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_risk_level_counts_returns_figure():
    predictions = risk_queries.load_risk_scores("oulad")
    fig = risk_charts.plot_risk_level_counts(predictions)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_precision_recall_returns_figure():
    X_train, X_test, y_train, y_test = risk_queries.get_test_split("uci")
    model = risk_queries.load_model("uci")
    y_scores = model.predict_proba(X_test)[:, 1]
    fig = risk_charts.plot_precision_recall(y_test, y_scores, threshold=risk_queries.RISK_THRESHOLD_UCI)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
