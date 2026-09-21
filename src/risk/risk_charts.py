# src/risk/risk_charts.py
"""
Sprint 3 -- shared plotting helpers (originally notebooks/shared_notebooks/risk_charts.ipynb).

Pure presentation: every function here takes already-computed data (a
DataFrame, a metrics dict) and returns a matplotlib Figure for
st.pyplot() -- none of these functions load a model, hit disk, or run
inference themselves. That happens in risk_queries.py / model_gru.py /
evaluation.py; this module only draws what it's handed.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve

# --------------------------------------------------------------------------
# Charts -- same functions as risk_charts.ipynb, adapted to return a Figure
# for st.pyplot() rather than plt.show()
# --------------------------------------------------------------------------
def plot_precision_recall(y_true, y_scores, threshold: float, title="Precision/Recall Curve") -> plt.Figure:
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, lw=2, color="#5C6BC0", label="PR curve")

    if len(thresholds) > 0:
        idx = (abs(thresholds - threshold)).argmin()
        ax.scatter(recall[idx], precision[idx], color="#EF5350", s=80, zorder=5,
                   label=f"threshold={threshold:.3f}")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_risk_distribution(predictions_df: pd.DataFrame, threshold: float, title="Risk Score Distribution") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(predictions_df["risk_score"], bins=20, color="#5C6BC0", edgecolor="white", alpha=0.8)
    ax.axvline(threshold, color="#EF5350", linestyle="--", lw=2, label=f"threshold={threshold:.3f}")
    ax.set_xlabel("Risk Score")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_risk_level_counts(predictions_df: pd.DataFrame, title="Risk Level") -> plt.Figure:
    order = ["Low", "Medium", "High"]
    counts = predictions_df["risk_level"].value_counts().reindex(order).fillna(0)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(order, counts.values, color=["#55A868", "#DD8452", "#C44E52"])
    ax.set_ylabel("students")
    ax.set_title(title)
    for i, v in enumerate(counts.values):
        ax.text(i, v + max(counts.values) * 0.01, f"{int(v):,}", ha="center")
    fig.tight_layout()
    return fig


def plot_persona_risk_boxplot(personas_df: pd.DataFrame, title="Risk Score by Persona") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=personas_df, x="persona_name", y="risk_score", hue="persona_name",
                ax=ax, legend=False)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    return fig



def plot_live_sampled_comparison(metrics: dict, n_students: int,
                                  title="Live sampled comparison (this run)") -> plt.Figure:
    models = list(metrics.keys())
    fig, ax = plt.subplots(figsize=(7, 4.5))
    width = 0.25
    x = np.arange(len(models))
    for i, metric_name in enumerate(["precision", "recall", "f1"]):
        values = [metrics[m][metric_name] for m in models]
        ax.bar(x + (i - 1) * width, values, width, label=metric_name.capitalize())
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"{title}  (n={n_students} students, vs. actual Fail/Withdrawn)")
    ax.legend()
    ax.grid(True, alpha=0.25, axis="y")
    fig.tight_layout()
    return fig


def plot_student_engagement_and_risk(
    weekly_df: pd.DataFrame,
    trajectory_df: pd.DataFrame,
    threshold: float,
    first_high_risk_week=None,
    event_week=None,
    title: str = "Weekly Engagement & GRU Risk Trajectory",
) -> plt.Figure:
    """The per-student time-series chart: top panel is raw weekly VLE
    engagement (clicks + active days), bottom panel is the GRU's risk score
    replayed week-by-week, with the model's threshold and the first
    high-risk week (and the actual withdrawal week, if known) marked."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax1.bar(weekly_df["week_number"], weekly_df["total_clicks"], color="#5C6BC0",
            alpha=0.85, label="Total clicks")
    ax1b = ax1.twinx()
    ax1b.plot(weekly_df["week_number"], weekly_df["days_active"], color="#EF5350",
               marker="o", markersize=3, label="Days active")
    ax1.set_ylabel("Total clicks / week")
    ax1b.set_ylabel("Days active / week")
    ax1.set_title("Weekly VLE engagement")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.25)

    # Risk scores mostly live in a narrow high band (0.85-1.0) for a long
    # stretch of most trajectories -- a side effect of the zero-padding
    # artifact documented above. On a plain 0-to-1 axis, real, meaningful
    # movement within that band (e.g. 0.995 -> 0.91) is visually
    # indistinguishable from a flat line, which reads as "the score isn't
    # responding to this student's actual data" even when it demonstrably
    # is. A logit-scale y-axis stretches both the near-0 and near-1 ends,
    # making that real movement visible instead of compressed away.
    plot_scores = trajectory_df["risk_score"].clip(1e-3, 1 - 1e-3)
    ax2.plot(trajectory_df["week_number"], plot_scores, color="#5C6BC0",
              lw=2, marker="o", markersize=3, label="GRU risk score")
    ax2.axhline(threshold, color="#C44E52", linestyle="--", lw=1.5, label=f"threshold={threshold:.2f}")
    if first_high_risk_week is not None:
        ax2.axvline(first_high_risk_week, color="#55A868", linestyle=":", lw=2,
                     label=f"first high-risk week (illustrative) = {first_high_risk_week}")
    if event_week is not None:
        ax2.axvline(event_week, color="#8172B2", linestyle=":", lw=2,
                     label=f"actual withdrawal week = {event_week}")
    ax2.set_yscale("logit")
    ax2.set_ylim(1e-3, 1 - 1e-3)
    ax2.set_xlabel("Week number")
    ax2.set_ylabel("GRU risk score (logit scale)")
    ax2.set_title("GRU risk score as the course progresses (week-by-week replay) — logit scale")
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(True, alpha=0.25, which="both")

    fig.suptitle(title)
    fig.tight_layout()
    return fig
