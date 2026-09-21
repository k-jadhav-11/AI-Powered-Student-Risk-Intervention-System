# src/charts.py
"""
Matplotlib chart generation. Each function returns the Figure object so the
CLI layer can decide whether to show it (plt.show) and/or export it (reports.py).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # safe default for headless/terminal use; CLI can switch to show()
import matplotlib.pyplot as plt

# 1. Classical bar chart of average score by subject, with the actual value labeled on top of each bar
def plot_class_average_bar(df: pd.DataFrame, title: str = "Class Average by Subject"):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df["subject"], df["average_score"], color="#4C72B0")
    ax.set_ylabel("Average Score")
    ax.set_title(title)
    ax.set_ylim(0, 100)
    for i, v in enumerate(df["average_score"]):
        ax.text(i, v + 1, str(v), ha="center", fontsize=9)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    return fig

# 2. Horizontal bar chart of top/bottom performers, with color-coded bars
def plot_ranked_bar(top_df: pd.DataFrame, bottom_df: pd.DataFrame, title: str = "Top / Bottom Performers"):
    fig, ax = plt.subplots(figsize=(8, 6))
    combined = pd.concat([
        top_df.assign(group="Top"),
        bottom_df.assign(group="Bottom"),
    ])
    colors = combined["group"].map({"Top": "#4CAF50", "Bottom": "#F44336"})
    ax.barh(combined["name"], combined["average_score"], color=colors)
    ax.set_xlabel("Average Score")
    ax.set_title(title)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig

# 3. Bar chart of subject averages, with weak subjects highlighted in red
def plot_weak_subjects(df: pd.DataFrame, title: str = "Subject Averages (Weak Subjects Highlighted)"):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = df["is_weak"].map({True: "#F44336", False: "#4C72B0"})
    ax.bar(df["subject"], df["average_score"], color=colors)
    ax.set_ylabel("Average Score")
    ax.set_title(title)
    ax.set_ylim(0, 100)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    return fig

# 4. Histogram of scores, with vertical lines for grade boundaries
def plot_grade_histogram(scores: np.ndarray, title: str = "Grade Distribution"):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores, bins=10, range=(0, 100), color="#8172B2", edgecolor="black")
    for boundary in [40, 60, 75, 90]:
        ax.axvline(boundary, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Score")
    ax.set_ylabel("Number of Records")
    ax.set_title(title)
    fig.tight_layout()
    return fig

# 5. Line chart of a student's score trend over time, with class average as a reference line
def plot_student_trend(df: pd.DataFrame, student_name: str, class_avg: float):
    """df must have columns: test_name (ordered), score."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["test_name"], df["score"], marker="o", color="#4C72B0", label=student_name)
    ax.axhline(class_avg, color="gray", linestyle="--", label=f"Class Avg ({class_avg:.1f})")
    ax.set_ylabel("Score")
    ax.set_title(f"Score Trend — {student_name}")
    ax.set_ylim(0, 100)
    ax.legend()
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Additional chart types, paired with the newer analytics functions
# ---------------------------------------------------------------------------

# 6. Horizontal bar chart of each student's percentile rank, colored red->green by percentile
def plot_percentile_ranks(df: pd.DataFrame, title: str = "Percentile Ranking", limit: int = 30):
    
    plot_df = df.sort_values("percentile_rank", ascending=False).head(limit)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(plot_df))))
    colors = plt.cm.RdYlGn(plot_df["percentile_rank"].values / 100)
    ax.barh(plot_df["name"], plot_df["percentile_rank"], color=colors)
    ax.set_xlabel("Percentile Rank")
    ax.set_xlim(0, 100)
    ax.set_title(title)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig

# 7. Grouped bar chart of each student's strongest vs weakest subject scores
def plot_subject_strengths(df: pd.DataFrame, title: str = "Subject Strengths (Strongest vs Weakest)", limit: int = 20):
    """
    Grouped bar chart: each student's strongest-subject score next to their
    weakest-subject score. df from analytics.find_subject_strengths.
    Each bar is labeled with its subject name (vertical text inside the bar)
    so it's clear *which* subject is strongest/weakest, not just the score.
    """
    plot_df = df.head(limit)
    x = np.arange(len(plot_df))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(9, 0.5 * len(plot_df)), 6))
    strong_bars = ax.bar(x - width / 2, plot_df["strongest_score"], width, label="Strongest Subject", color="#4CAF50")
    weak_bars = ax.bar(x + width / 2, plot_df["weakest_score"], width, label="Weakest Subject", color="#F44336")

    def _label_bars(bars, subjects):
        for bar, subject in zip(bars, subjects):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height / 2,
                subject,
                ha="center",
                va="center",
                rotation=90,
                fontsize=7,
                color="white",
                fontweight="bold",
            )

    _label_bars(strong_bars, plot_df["strongest_subject"])
    _label_bars(weak_bars, plot_df["weakest_subject"])

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["student_name"], rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, max(100, plot_df[["strongest_score", "weakest_score"]].max().max() + 10))
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig

# 8. Diverging horizontal bar chart of improvement/decline from first to last test
def plot_improvement(df: pd.DataFrame, title: str = "Improvement Leaderboard", limit: int = 15):
    """
    Diverging horizontal bar chart: biggest gains (green) and biggest declines (red),
    from first test to last test. df from analytics.compute_improvement_by_student
    (already sorted by improvement, highest first).
    """
    top_gain = df.head(limit)
    top_decline = df.tail(limit).sort_values("improvement")
    combined = pd.concat([top_gain, top_decline]).drop_duplicates(subset="student_name")
    combined = combined.sort_values("improvement")

    colors = ["#4CAF50" if v >= 0 else "#F44336" for v in combined["improvement"]]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(combined))))
    ax.barh(combined["student_name"], combined["improvement"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Score Change (Last Test − First Test)")
    ax.set_title(title)
    fig.tight_layout()
    return fig

# 9. Scatter plot of average score vs. score variability (std dev), to visualize consistency
def plot_consistency_scatter(df: pd.DataFrame, title: str = "Consistency: Average Score vs Variability"):
    """
    Scatter plot: average score (x) vs. score variability/std dev (y).
    Bottom-right (high average, low std) = steady top performer.
    Top area (high std) = unpredictable, regardless of average.
    df from analytics.compute_consistency_by_student.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["average_score"], df["consistency_std"], alpha=0.7, color="#5B8FF9", edgecolors="white", s=70)
    ax.set_xlabel("Average Score")
    ax.set_ylabel("Consistency (Std Dev — lower = more consistent)")
    ax.set_title(title)
    fig.tight_layout()
    return fig

# 10. Scatter plot of average score vs. z-score, with flagged outliers highlighted
def plot_outliers_scatter(full_df: pd.DataFrame, outliers_df: pd.DataFrame, title: str = "Outlier Detection"):
    """
    Scatter plot of every student's average score vs. z-score, with flagged outliers
    highlighted in color on top of the rest of the class in gray.
    full_df from analytics.add_zscore_column; outliers_df from analytics.detect_outliers.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(full_df["average_score"], full_df["zscore"], color="#B0BEC5", alpha=0.6, label="Normal", s=50)

    if not outliers_df.empty:
        high = outliers_df[outliers_df["outlier_type"] == "High Outlier"]
        low = outliers_df[outliers_df["outlier_type"] == "Low Outlier"]
        if not high.empty:
            ax.scatter(high["average_score"], high["zscore"], color="#4CAF50", s=110, edgecolors="black", label="High Outlier")
        if not low.empty:
            ax.scatter(low["average_score"], low["zscore"], color="#F44336", s=110, edgecolors="black", label="Low Outlier")

    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Average Score")
    ax.set_ylabel("Z-score")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig

# 11. Heatmap of average score by class x subject, to visualize strengths/weaknesses
def plot_class_comparison_heatmap(pivot_df: pd.DataFrame, title: str = "Class vs Subject Comparison"):
    """
    Heatmap of average score by class (rows) x subject (columns), so a single glance
    shows e.g. "9th grade struggles specifically in Math" rather than just an overall average.
    pivot_df from analytics.compare_classes_by_subject.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    data = pivot_df.values.astype(float)
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(range(len(pivot_df.columns)))
    ax.set_xticklabels(pivot_df.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot_df.index)))
    ax.set_yticklabels(pivot_df.index)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center", color="black", fontsize=9)

    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Average Score")
    fig.tight_layout()
    return fig

# 12. Bar chart of pass rate, color-coded (green >=60%, amber 40-59%, red <40%)
def plot_pass_rate_bar(df: pd.DataFrame, title: str = "Pass Rate"):
    """
    Bar chart of pass rate, color-coded (green >=60%, amber 40-59%, red <40%).
    df from analytics.compute_pass_rate -- works whether grouped by subject or by class,
    since it just uses whatever the first column is as the x-axis label.
    """
    label_col = df.columns[0]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4CAF50" if v >= 60 else ("#FFC107" if v >= 40 else "#F44336") for v in df["pass_rate_pct"]]
    ax.bar(df[label_col].astype(str), df["pass_rate_pct"], color=colors)
    ax.set_ylabel("Pass Rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title(title)
    for i, v in enumerate(df["pass_rate_pct"]):
        ax.text(i, v + 1, f"{v}%", ha="center", fontsize=9)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    return fig

# 13. Bar chart with one bar per (subject, student) pair, labeled with both
def plot_subject_top_performers(df: pd.DataFrame, title: str = "Top Performers by Subject"):
    """
    Bar chart with one bar per (subject, student) pair, labeled with both --
    e.g. "who is the best in Math", "who is the best in History" at a glance.
    df from analytics.rank_subject_performers.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [f"{row.student_name}\n({row.subject})" for row in df.itertuples()]
    ax.bar(labels, df["score"], color="#5B8FF9")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 100)
    ax.set_title(title)
    plt.xticks(rotation=30, ha="right", fontsize=8)
    fig.tight_layout()
    return fig
