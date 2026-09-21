# tests/test_risk/test_clustering.py
"""src/risk/clustering.py -- live K-Means. Runs the real algorithm on a
small slice of real feature data (not the full ~26K OULAD rows, to keep
this test fast) rather than mocking sklearn."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.risk import clustering, risk_queries


def test_find_optimal_k_returns_inertia_and_silhouette():
    feature_matrix = risk_queries.load_feature_matrix("uci").sample(100, random_state=42)
    result = clustering.find_optimal_k(feature_matrix, k_range=range(2, 4))
    assert set(result["inertia"].keys()) == {2, 3}
    assert set(result["silhouette"].keys()) == {2, 3}


def test_run_kmeans_adds_cluster_and_pca_columns():
    feature_matrix = risk_queries.load_feature_matrix("uci").sample(100, random_state=42)
    clustered = clustering.run_kmeans(feature_matrix, k=3)
    assert "cluster_id" in clustered.columns
    assert {"pc1", "pc2"} <= set(clustered.columns)
    assert clustered["cluster_id"].nunique() <= 3
