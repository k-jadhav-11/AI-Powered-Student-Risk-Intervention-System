"""
Statistical and transformation logic built on Pandas/NumPy, sitting between
raw SQL query results and the CLI/chart layers.
"""
import numpy as np
import pandas as pd

DEFAULT_GRADE_BANDS = {
    "A (90-100)": (90, 100),
    "B (75-89)": (75, 89),
    "C (60-74)": (60, 74),
    "D (40-59)": (40, 59),
    "F (<40)": (0, 39),
}

# 1. Weak Subject Detection
def identify_weak_subjects(
    avg_df: pd.DataFrame,
    fail_df: pd.DataFrame,
    avg_threshold: float = 55,
    fail_threshold: float = 25,
) -> pd.DataFrame:
   
    merged = avg_df.merge(fail_df, on="subject", how="left")
    merged["failure_rate_pct"] = merged["failure_rate_pct"].fillna(0)
    merged["is_weak"] = (merged["average_score"] < avg_threshold) | (
        merged["failure_rate_pct"] > fail_threshold
    )
    return merged.sort_values("average_score")

# 2. Find Grade Band counts for a set of scores, e.g. "how many students got A/B/C/D/F".
def compute_grade_bands(scores: np.ndarray, bands: dict = None) -> pd.Series:
    if bands is None:
        bands = DEFAULT_GRADE_BANDS

    scores = np.asarray(scores, dtype=float)
    counts = {}
    for label, (low, high) in bands.items():
        counts[label] = int(np.sum((scores >= low) & (scores <= high)))
    return pd.Series(counts)

# 3. Standard z-score
def compute_zscore(value: float, mean: float, std: float) -> float:
    if std == 0 or np.isnan(std):
        return 0.0
    return round(float((value - mean) / std), 2)

# 4. Find Trend
def classify_trend(scores: np.ndarray) -> str:
    """
    Fit a simple linear trend (NumPy polyfit) over a score sequence
    and classify the slope as Improving / Declining / Stable.
    Requires at least 2 data points.
    """
    scores = np.asarray(scores, dtype=float)
    if len(scores) < 2:
        return "Not enough data"

    x = np.arange(len(scores))
    slope, _intercept = np.polyfit(x, scores, 1)

    if slope > 1.5:
        return "Improving"
    elif slope < -1.5:
        return "Declining"
    return "Stable"

# 5. Build the Pivot Table
def build_pivot_table(marks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a student x subject matrix of average scores using pandas pivot_table.
    marks_df must have columns: student_name, subject, score.
    """
    if marks_df.empty:
        return pd.DataFrame()

    pivot = pd.pivot_table(
        marks_df,
        index="student_name",
        columns="subject",
        values="score",
        aggfunc="mean",
    ).round(1)
    return pivot

# 6. Class Stats
def class_stats(scores: np.ndarray) -> dict:
    """Mean and standard deviation for a set of scores, used as the z-score baseline."""
    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0:
        return {"mean": 0.0, "std": 0.0}
    return {"mean": float(np.mean(scores)), "std": float(np.std(scores))}


# ---------------------------------------------------------------------------
# Additional analytics
# ---------------------------------------------------------------------------


def compute_percentile_rank(value: float, all_values: np.ndarray) -> float:
    """
    Percentage of values at or below `value` within `all_values`.
    e.g. 90.0 means this value is at or above 90% of the group ("top 10%").
    """
    all_values = np.asarray(all_values, dtype=float)
    if len(all_values) == 0:
        return 0.0
    return round(float(np.sum(all_values <= value) / len(all_values) * 100), 2)


def add_percentile_ranks(df: pd.DataFrame, value_col: str = "average_score") -> pd.DataFrame:
    """
    Add a `percentile_rank` column to a dataframe of per-student averages
    (e.g. the output of queries.get_student_averages), so a teacher can see
    where each student stands relative to the group ("top 10%", etc.).
    """
    if df.empty:
        return df.assign(percentile_rank=pd.Series(dtype=float))

    values = df[value_col].values.astype(float)
    result = df.copy()
    result["percentile_rank"] = [compute_percentile_rank(v, values) for v in values]
    return result.sort_values("percentile_rank", ascending=False)


def find_subject_strengths(pivot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a student x subject pivot table (see build_pivot_table), identify
    each student's strongest and weakest subject -- useful for personalized feedback.
    """
    if pivot_df.empty:
        return pd.DataFrame(
            columns=["student_name", "strongest_subject", "strongest_score",
                     "weakest_subject", "weakest_score"]
        )

    numeric_pivot = pivot_df.astype(float)
    return pd.DataFrame({
        "student_name": numeric_pivot.index,
        "strongest_subject": numeric_pivot.idxmax(axis=1).values,
        "strongest_score": numeric_pivot.max(axis=1).values,
        "weakest_subject": numeric_pivot.idxmin(axis=1).values,
        "weakest_score": numeric_pivot.min(axis=1).values,
    }).reset_index(drop=True)


def compute_improvement_by_student(scores_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare each student's earliest vs latest test average (ordered by test_date)
    to measure improvement. scores_df must have columns: student_name, test_date, score
    (e.g. the output of queries.get_all_scores).
    """
    if scores_df.empty:
        return pd.DataFrame(columns=["student_name", "first_score", "last_score", "improvement"])

    df = scores_df.copy()
    df["test_date"] = pd.to_datetime(df["test_date"])
    per_test = df.groupby(["student_name", "test_date"])["score"].mean().reset_index()

    rows = []
    for name, group in per_test.groupby("student_name"):
        group = group.sort_values("test_date")
        first_score = float(group["score"].iloc[0])
        last_score = float(group["score"].iloc[-1])
        rows.append({
            "student_name": name,
            "first_score": round(first_score, 2),
            "last_score": round(last_score, 2),
            "improvement": round(last_score - first_score, 2),
        })
    return pd.DataFrame(rows).sort_values("improvement", ascending=False).reset_index(drop=True)


def compute_consistency(scores: np.ndarray) -> float:
    """
    Standard deviation of a single student's scores across tests.
    Lower = more consistent; higher = more erratic performance.
    """
    scores = np.asarray(scores, dtype=float)
    if len(scores) < 2:
        return 0.0
    return round(float(np.std(scores)), 2)


def compute_consistency_by_student(scores_df: pd.DataFrame) -> pd.DataFrame:
    """
    Standard deviation of scores per student across all their tests, to distinguish
    a "steady performer" (low std) from someone "good but unstable" (high std).
    scores_df must have columns: student_name, score.
    """
    if scores_df.empty:
        return pd.DataFrame(columns=["student_name", "average_score", "consistency_std"])

    grouped = scores_df.groupby("student_name")["score"].agg(
        average_score="mean",
        consistency_std=lambda s: np.std(s.values, ddof=0),
    ).reset_index()
    grouped["average_score"] = grouped["average_score"].round(2)
    grouped["consistency_std"] = grouped["consistency_std"].round(2)
    return grouped.sort_values("consistency_std").reset_index(drop=True)


def add_zscore_column(df: pd.DataFrame, value_col: str = "average_score") -> pd.DataFrame:
    """
    Add a `zscore` column for every row in df, computed against the whole group in df
    (population std). Unlike detect_outliers, this keeps every row -- useful for charting
    the full population with outliers highlighted on top of it.
    """
    if df.empty:
        return df.assign(zscore=pd.Series(dtype=float))

    values = df[value_col].values.astype(float)
    mean, std = float(np.mean(values)), float(np.std(values))
    result = df.copy()
    result["zscore"] = 0.0 if std == 0 else np.round((values - mean) / std, 2)
    return result


def detect_outliers(values_df: pd.DataFrame, value_col: str = "average_score", threshold: float = 2.0) -> pd.DataFrame:
    """
    Flag rows whose z-score magnitude exceeds `threshold` as statistical outliers --
    unusually high performers (talent spotting) or unusually low ones (need attention).
    """
    empty_result = values_df.iloc[0:0].copy()
    if "zscore" not in empty_result.columns:
        empty_result["zscore"] = pd.Series(dtype=float)
        empty_result["outlier_type"] = pd.Series(dtype=str)
    if values_df.empty:
        return empty_result

    df = add_zscore_column(values_df, value_col)
    df["outlier_type"] = np.select(
        [df["zscore"] >= threshold, df["zscore"] <= -threshold],
        ["High Outlier", "Low Outlier"],
        default="Normal",
    )
    return df[df["outlier_type"] != "Normal"].sort_values("zscore", ascending=False).reset_index(drop=True)


def compare_classes(scores_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare overall average performance across classes/grades.
    scores_df must have columns: class_name, score (e.g. queries.get_all_scores()
    called with no class_name filter, so it spans every grade).
    """
    if scores_df.empty:
        return pd.DataFrame(columns=["class_name", "average_score"])

    result = scores_df.groupby("class_name")["score"].mean().round(2).reset_index()
    return result.rename(columns={"score": "average_score"}).sort_values(
        "average_score", ascending=False
    ).reset_index(drop=True)


def compare_classes_by_subject(scores_df: pd.DataFrame) -> pd.DataFrame:
    """
    Class x subject matrix of average scores, for spotting e.g. "9th grade struggles
    in Math specifically" rather than just an overall class average.
    """
    if scores_df.empty:
        return pd.DataFrame()
    return pd.pivot_table(
        scores_df, index="class_name", columns="subject", values="score", aggfunc="mean"
    ).round(2)


def compute_pass_rate(scores_df: pd.DataFrame, pass_mark: float = 35, group_by: str = "subject") -> pd.DataFrame:
    """
    Percentage of records at or above pass_mark, grouped by `group_by`
    (default "subject"; pass "class_name" for a per-grade pass rate instead).
    Complements get_subject_failure_rates (pass_rate_pct + failure_rate_pct sum to 100).
    """
    if scores_df.empty:
        return pd.DataFrame(columns=[group_by, "pass_rate_pct"])

    df = scores_df.copy()
    df["passed"] = df["score"] >= pass_mark
    result = df.groupby(group_by)["passed"].mean().mul(100).round(2).reset_index()
    return result.rename(columns={"passed": "pass_rate_pct"}).sort_values(
        "pass_rate_pct", ascending=False
    ).reset_index(drop=True)


def rank_subject_performers(scores_df: pd.DataFrame, top_n: int = 1) -> pd.DataFrame:
    """
    For each subject, return the top_n student(s) by average score --
    "who is the best in Math", "who is the best in History", etc.
    """
    if scores_df.empty:
        return pd.DataFrame(columns=["subject", "student_name", "score"])

    avg = scores_df.groupby(["subject", "student_name"])["score"].mean().round(2).reset_index()
    return (
        avg.sort_values(["subject", "score"], ascending=[True, False])
        .groupby("subject")
        .head(top_n)
        .reset_index(drop=True)
    )
