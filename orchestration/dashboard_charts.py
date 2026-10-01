# orchestration/dashboard_charts.py
"""
Sprint 5 -- Teacher Dashboard charts. Matplotlib, same
fig/ax/plt.subplots/tight_layout convention as src/charts.py (Sprint 2's
chart module) -- each function returns the Figure object; the calling
Streamlit page decides whether/how to render it (st.pyplot(fig)), same
split of responsibility Sprint 2 already established.

Every chart here reads directly from orchestration.audit.read_audit_log()'s
real entries -- no synthetic or placeholder data. An empty audit log
produces an empty-but-valid chart with an on-chart "no data yet" label
rather than a crash, matching this project's "graceful, honest, no
crash" convention.
"""
from __future__ import annotations

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_DECISION_COLORS = {"nudge": "#4CAF50", "resource": "#FFA726", "escalate": "#E53935"}
_RISK_COLORS = {"Low": "#4CAF50", "Medium": "#FFA726", "High": "#E53935"}


def _empty_chart(title: str, message: str = "No intervention decisions logged yet"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=11, color="gray", transform=ax.transAxes)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    return fig


def plot_decision_distribution(df: pd.DataFrame, title: str = "Intervention Decisions by Branch"):
    """Bar chart of nudge / resource / escalate counts -- the single
    fastest "is the agent behaving conservatively or aggressively" read a
    teacher or admin can get."""
    if df.empty:
        return _empty_chart(title)
    counts = df["decision"].value_counts().reindex(["nudge", "resource", "escalate"]).fillna(0)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [_DECISION_COLORS.get(d, "#999999") for d in counts.index]
    ax.bar(counts.index, counts.values, color=colors)
    ax.set_ylabel("Number of Decisions")
    ax.set_title(title)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.05 * max(counts.values.max(), 1), str(int(v)), ha="center", fontsize=9)
    fig.tight_layout()
    return fig


def plot_risk_level_breakdown(df: pd.DataFrame, title: str = "Decisions by Risk Level"):
    """Horizontal bar chart of Low/Medium/High counts among logged
    decisions -- shows whether the agent is mostly being run against
    already-High-risk students or a broader mix."""
    if df.empty:
        return _empty_chart(title)
    counts = df["risk_level"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [_RISK_COLORS.get(r, "#999999") for r in counts.index]
    ax.barh(counts.index, counts.values, color=colors)
    ax.set_xlabel("Number of Decisions")
    ax.set_title(title)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def plot_decisions_over_time(df: pd.DataFrame, title: str = "Intervention Decisions Over Time"):
    """Daily decision count, one line per branch -- surfaces a spike (a
    big batch cohort run) or a gap (nothing logged in N days) at a
    glance. Uses logged_at's calendar date; a single day with mixed
    single-student and batch runs still reads as one bar per branch."""
    if df.empty:
        return _empty_chart(title)
    plot_df = df.copy()
    plot_df["date"] = pd.to_datetime(plot_df["logged_at"]).dt.date
    pivot = plot_df.groupby(["date", "decision"]).size().unstack(fill_value=0)
    for branch in ("nudge", "resource", "escalate"):
        if branch not in pivot.columns:
            pivot[branch] = 0
    pivot = pivot[["nudge", "resource", "escalate"]].sort_index()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for branch in ("nudge", "resource", "escalate"):
        ax.plot(pivot.index, pivot[branch], marker="o", label=branch, color=_DECISION_COLORS[branch])
    ax.set_ylabel("Decisions")
    ax.set_title(title)
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    return fig


def plot_confidence_gauge(confidence: float, floor: float, title: str = "Router Confidence"):
    """A single-value horizontal gauge for one decision's
    router_confidence against orchestration.graph.CONFIDENCE_FLOOR --
    the number that actually decides escalation, made visible instead of
    reported as a bare float. The red zone below `floor` is exactly the
    region classify_severity() escalates on; this chart draws the same
    line the router itself uses, not an illustrative approximation."""
    fig, ax = plt.subplots(figsize=(7, 1.6))
    ax.barh([0], [1.0], color="#3A3F4B", height=0.5)
    ax.barh([0], [floor], color="#E53935", height=0.5, alpha=0.55)
    color = "#4CAF50" if confidence >= floor else "#E53935"
    ax.barh([0], [confidence], color=color, height=0.5)
    ax.axvline(floor, color="#E53935", linestyle="--", linewidth=1)
    ax.text(floor, 0.42, f"escalation floor {floor:.2f}", color="#E53935", fontsize=8, ha="left", va="bottom")
    ax.text(confidence, -0.42, f"{confidence:.3f}", color=color, fontsize=10, ha="center", va="top", fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([])
    ax.set_xlabel("Confidence (distance from the dataset's decision threshold)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_student_factors(risk_explanation: dict, title: str | None = None):
    """Horizontal bar chart of a single student's real SHAP-derived
    factors (all three RiskExplanation buckets combined), signed by
    direction so 'increases risk' and 'decreases risk' read as opposite
    bars around zero -- the same contribution values the drafted
    message's prose is built from, shown as data instead of only as a
    sentence buried in that prose."""
    rows = []
    for bucket, label in (("academic_factors", "academic"), ("engagement_factors", "engagement"),
                           ("contextual_factors", "contextual")):
        for f in risk_explanation.get(bucket) or []:
            rows.append({"feature": f["feature"], "contribution": f["contribution"], "bucket": label})
    if not rows:
        return _empty_chart(title or "SHAP Factors", "No factors returned for this student")

    factor_df = pd.DataFrame(rows).sort_values("contribution")
    colors = ["#E53935" if c > 0 else "#4CAF50" for c in factor_df["contribution"]]
    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.5 * len(factor_df))))
    ax.barh(factor_df["feature"], factor_df["contribution"], color=colors)
    ax.axvline(0, color="#666", linewidth=0.8)
    ax.set_xlabel("SHAP contribution (red = increases risk, green = decreases risk)")
    ax.set_title(title or "What's Driving This Student's Risk Score")
    fig.tight_layout()
    return fig


def plot_escalation_rate_by_dataset(df: pd.DataFrame, title: str = "Escalation Rate by Dataset"):
    """What fraction of decisions escalate to a human, per dataset --
    Section 6.1's conservative-by-construction design means this should
    normally be a small minority; a high or rising rate here is itself a
    signal worth a teacher's attention (frequent GRU/XGBoost disagreement,
    or many borderline scores)."""
    if df.empty:
        return _empty_chart(title)
    rates = df.groupby("dataset")["decision"].apply(lambda s: (s == "escalate").mean() * 100)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(rates.index, rates.values, color="#E53935")
    ax.set_ylabel("Escalation Rate (%)")
    ax.set_ylim(0, max(100, rates.values.max() * 1.2 if len(rates) else 100))
    ax.set_title(title)
    for i, v in enumerate(rates.values):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=9)
    fig.tight_layout()
    return fig
