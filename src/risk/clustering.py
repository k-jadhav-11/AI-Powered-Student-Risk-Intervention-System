# src/risk/clustering.py
"""
Sprint 3 -- live K-Means clustering (originally
notebooks/shared_notebooks/clustering_performance_personas.ipynb's
find_optimal_k() / assign_personas(), made interactive).

Unlike the tuned XGBoost/CatBoost/GRU models, K-Means on these feature
matrices is fast enough (well under a second) to actually run on a
Streamlit page load rather than only serving a frozen precomputed result --
so unlike risk_queries.py, this module computes fresh on every call.
"""
import pandas as pd
import matplotlib.pyplot as plt

from .risk_queries import RANDOM_STATE

# --------------------------------------------------------------------------
# Live K-Means clustering -- unlike the tuned XGBoost/CatBoost/GRU models,
# K-Means on these feature matrices is fast enough (well under a second) to
# actually run on a Streamlit page load rather than only serving a frozen
# precomputed result. This is what clustering_performance_personas.ipynb's
# find_optimal_k() / assign_personas() do, made interactive.
# --------------------------------------------------------------------------
UCI_CENTROID_VIEW_COLS = ["failures", "absences", "studytime", "g1", "g2"]
OULAD_CENTROID_VIEW_COLS = ["total_clicks_sum", "days_active_sum", "num_weeks_active",
                            "pct_assessments_submitted", "engagement_trend"]


def find_optimal_k(feature_df: pd.DataFrame, k_range=range(2, 8)) -> dict:
    """Same method as clustering_performance_personas.ipynb: inertia (elbow)
    + silhouette score for each k, on standardized numeric features."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    num_df = feature_df.select_dtypes(include="number").drop(columns=["at_risk"], errors="ignore")
    X = StandardScaler().fit_transform(num_df.fillna(0))

    inertia, silhouette = {}, {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X)
        inertia[k] = float(km.inertia_)
        if len(set(labels)) > 1:
            silhouette[k] = float(silhouette_score(X, labels))
    return {"inertia": inertia, "silhouette": silhouette}


def run_kmeans(feature_df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Fits K-Means live with the chosen k and returns feature_df + a
    cluster_id column + 2D PCA coordinates (for the scatter plot)."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    num_df = feature_df.select_dtypes(include="number").drop(columns=["at_risk"], errors="ignore")
    X = StandardScaler().fit_transform(num_df.fillna(0))

    labels = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit_predict(X)
    coords = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X)

    result = feature_df.copy()
    result["cluster_id"] = labels
    result["pc1"] = coords[:, 0]
    result["pc2"] = coords[:, 1]
    return result


def plot_elbow_silhouette(k_results: dict) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ks = list(k_results["inertia"].keys())
    axes[0].plot(ks, list(k_results["inertia"].values()), marker="o", color="#4C72B0")
    axes[0].set_title("Elbow method (inertia)")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("inertia")

    sk = list(k_results["silhouette"].keys())
    axes[1].plot(sk, list(k_results["silhouette"].values()), marker="o", color="#DD8452")
    axes[1].set_title("Silhouette score")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("silhouette score")
    fig.tight_layout()
    return fig


def plot_cluster_scatter(clustered_df: pd.DataFrame, title="K-Means clusters (PCA projection)") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(clustered_df["pc1"], clustered_df["pc2"], c=clustered_df["cluster_id"],
                          cmap="tab10", alpha=0.5, s=15)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    legend1 = ax.legend(*scatter.legend_elements(), title="cluster", loc="best", fontsize=8)
    ax.add_artist(legend1)
    fig.tight_layout()
    return fig


def cluster_centroid_table(clustered_df: pd.DataFrame, view_cols: list) -> pd.DataFrame:
    """Mean of a few interpretable raw features + at_risk rate per cluster --
    what you'd actually read to decide what to NAME each cluster."""
    cols = [c for c in view_cols if c in clustered_df.columns]
    agg_cols = cols + (["at_risk"] if "at_risk" in clustered_df.columns else [])
    centroids = clustered_df.groupby("cluster_id")[agg_cols].mean().round(2)
    centroids["n_students"] = clustered_df["cluster_id"].value_counts().sort_index()
    return centroids

