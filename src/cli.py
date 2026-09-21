# src/cli.py
"""
Menu-driven terminal interface tying together db_manager, queries,
analytics, charts, and reports.
"""
import os

import numpy as np
import pandas as pd

from src.analytics import db_manager, queries, analytics, charts, reports
from src.analytics.exceptions import DataNotFoundError, StudentNotFoundError, InvalidCSVError

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")

MENU_TEXT = """
==================================================
 STUDENT PERFORMANCE ANALYTICS CLI
==================================================
1. Load Data from CSV
2. Class Average Report (per subject)
3. Top/Bottom Performers
4. Subject-wise Weak Area Detection
5. Grade Distribution
6. Individual Student Report
7. Pivot Table (Subject vs Student)
8. Percentile Ranking
9. Subject Strengths (Best vs Weakest)
10. Improvement Leaderboard
11. Consistency Analysis
12. Outlier Detection
13. Class Comparison (Grade vs Grade)
14. Pass Rate Summary
15. Subject Top Performers
16. Export Last Report
0. Exit
--------------------------------------------------
"""


def print_table(df: pd.DataFrame) -> None:
    if df.empty:
        print("(no data)")
    else:
        print(df.to_string(index=False))


def require_data(conn) -> bool:
    if not db_manager.has_data(conn):
        print("\n No data found. Please run option 1 (Load Data) first.\n")
        return False
    return True


def prompt_class_filter(conn) -> str | None:
    """
    Show the available classes and ask which one to scope the report to.
    Returns None for "All" (school-wide), or the chosen class_name string.
    """
    classes = queries.get_available_classes(conn)
    if not classes:
        return None

    print(f"\nAvailable classes: {', '.join(classes)} (or leave blank for All)")
    choice = input("Filter by class? ").strip()
    if not choice:
        return None
    if choice not in classes:
        print(f"'{choice}' not recognized — showing all classes instead.")
        return None
    return choice


def prompt_subject_filter(conn) -> str | None:
    """
    Show the available subjects and ask which one to scope the report to.
    Matches case-insensitively and falls back to a substring match (e.g. "math" ->
    "Mathematics") so a near-miss doesn't silently return "No matching data found."
    Returns None for "overall" (all subjects), or the resolved canonical subject name.
    """
    subjects = queries.get_available_subjects(conn)
    if not subjects:
        return None

    print(f"\nAvailable subjects: {', '.join(subjects)} (or leave blank for overall)")

    choice = input("Filter by subject? (leave blank for overall): ").strip()
    if not choice:
        return None

    # exact, case-insensitive match first
    for s in subjects:
        if s.lower() == choice.lower():
            return s

    # fall back to a substring match ("math" -> "Mathematics")
    partial_matches = [s for s in subjects if choice.lower() in s.lower()]
    if len(partial_matches) == 1:
        print(f"Matched subject: {partial_matches[0]}")
        return partial_matches[0]
    elif len(partial_matches) > 1:
        print(f"'{choice}' matches multiple subjects ({', '.join(partial_matches)}) — please be more specific.")
        return None

    print(f"'{choice}' not recognized — showing overall (all subjects) instead.")
    return None


#  Exporting reports (tables and charts) is optional, so we ask the user if they want to save the last generated report.
def prompt_export(df: pd.DataFrame = None, fig=None, name_hint: str = "report") -> dict | None:

    choice = input("\nExport this report? (y/n): ").strip().lower()
    if choice != "y":
        return None

    exported = {}
    if df is not None:
        path = reports.export_dataframe_csv(df, name_hint, REPORTS_DIR)
        exported["csv"] = path
        print(f"Saved: {path}")
    if fig is not None:
        path = reports.export_chart_png(fig, name_hint, REPORTS_DIR)
        exported["png"] = path
        print(f"Saved: {path}")
    return exported


# ------------------------------ handlers --------------------------------- 

# Each handler corresponds to a menu option and is responsible for:
# 1. Prompting for any additional filters (class, subject, etc.)
# 2. Calling the appropriate query/analytics function(s)
# 3. Printing the resulting table to the console
# 4. Generating a chart (if applicable)


# Handler for seeding the database with CSV data
def handle_seed(conn, state: dict) -> None:
    try:
        summary = db_manager.seed_all(conn, DATA_DIR)
        print("\nSeeding complete:")
        for table, result in summary.items():
            print(f"  {table}: {result}")
    except (InvalidCSVError, FileNotFoundError) as e:
        print(f"\nError loading data: {e}")


# Handler for displaying class average by subject, optionally filtered by class
def handle_class_average(conn, state: dict) -> None:
    if not require_data(conn):
        return
    class_name = prompt_class_filter(conn)

    df = queries.get_class_average_by_subject(conn, class_name=class_name)
    label = class_name or "All Classes"
    print(f"\nClass Average by Subject — {label}:")
    print_table(df)

    fig = charts.plot_class_average_bar(df, title=f"Class Average by Subject — {label}")
    state["last_df"] = df
    state["last_fig"] = fig
    prompt_export(df, fig, f"class_average_{label}")


# Handler for displaying top and bottom performing students
def handle_top_bottom(conn, state: dict) -> None:
    if not require_data(conn):
        return
    class_name = prompt_class_filter(conn)
    subject = prompt_subject_filter(conn)

    try:
        n = int(input("How many top/bottom students to show? [default 5]: ") or 5)
    except ValueError:
        n = 5

    top, bottom = queries.get_top_bottom_students(conn, n=n, subject=subject, class_name=class_name)
    if top.empty:
        print("\nNo matching data found.")
        return

    label = class_name or "All Classes"
    print(f"\nTop {n} — {label}" + (f" ({subject})" if subject else "") + ":")
    print_table(top)
    print(f"\nBottom {n} — {label}" + (f" ({subject})" if subject else "") + ":")
    print_table(bottom)

    title = f"Top/Bottom Performers — {label}" + (f" — {subject}" if subject else "")
    fig = charts.plot_ranked_bar(top, bottom, title=title)
    combined = pd.concat([top.assign(group="Top"), bottom.assign(group="Bottom")])
    state["last_df"] = combined
    state["last_fig"] = fig
    prompt_export(combined, fig, f"top_bottom_{label}")

# Handler for identifying weak subjects based on average scores and failure rates
def handle_weak_subjects(conn, state: dict) -> None:
    if not require_data(conn):
        return
    class_name = prompt_class_filter(conn)

    avg_df = queries.get_class_average_by_subject(conn, class_name=class_name)
    fail_df = queries.get_subject_failure_rates(conn, pass_mark=35, class_name=class_name)
    weak_df = analytics.identify_weak_subjects(avg_df, fail_df)

    label = class_name or "All Classes"
    print(f"\nSubject-wise Averages & Weak Area Flags — {label}:")
    print_table(weak_df)

    fig = charts.plot_weak_subjects(weak_df, title=f"Subject Averages — {label} (Weak Subjects Highlighted)")
    state["last_df"] = weak_df
    state["last_fig"] = fig
    prompt_export(weak_df, fig, f"weak_subjects_{label}")

# Handler for displaying grade distribution as a histogram and table
def handle_grade_distribution(conn, state: dict) -> None:
    if not require_data(conn):
        return
    class_name = prompt_class_filter(conn)
    subject = prompt_subject_filter(conn)

    scores_df = queries.get_all_scores(conn, subject=subject, class_name=class_name)
    if scores_df.empty:
        print("\nNo matching data found.")
        return

    label = class_name or "All Classes"
    band_counts = analytics.compute_grade_bands(scores_df["score"].values)
    print(f"\nGrade Distribution — {label}" + (f" ({subject})" if subject else "") + ":")
    print_table(band_counts.reset_index().rename(columns={"index": "grade_band", 0: "count"}))

    title = f"Grade Distribution — {label}" + (f" — {subject}" if subject else "")
    fig = charts.plot_grade_histogram(scores_df["score"].values, title=title)
    state["last_df"] = band_counts.reset_index()
    state["last_fig"] = fig
    prompt_export(state["last_df"], fig, f"grade_distribution_{label}")

# Handler for generating an individual student report, including scores, averages, z-scores, and trends
def handle_student_report(conn, state: dict) -> None:
    if not require_data(conn):
        return
    search_term = input("Enter student name or roll number: ").strip()
    matches = queries.find_student(conn, search_term)

    if matches.empty:
        print(f"\nNo student found matching '{search_term}'.")
        return
    if len(matches) > 1:
        print("\nMultiple matches found:")
        print_table(matches)
        return

    student = matches.iloc[0]
    student_id = int(student["student_id"])

    scores_df = queries.get_student_scores_over_time(conn, student_id)
    if scores_df.empty:
        print(f"\n{student['name']} has no recorded marks yet.")
        return

    # Z-score is computed against the student's own class/grade, not the whole school.
    class_scores = queries.get_all_scores(conn, class_name=student["class_name"])
    stats = analytics.class_stats(class_scores["score"].values)
    student_avg = scores_df["score"].mean()
    zscore = analytics.compute_zscore(student_avg, stats["mean"], stats["std"])

    per_test = scores_df.groupby("test_name")["score"].mean().reindex(
        scores_df["test_name"].unique()
    )
    trend = analytics.classify_trend(per_test.values)

    print(f"\n--- Student Report: {student['name']} ({student['roll_number']}, Class {student['class_name']}) ---")
    print_table(scores_df)
    print(f"\nStudent average:      {student_avg:.2f}")
    print(f"Class {student['class_name']} average: {stats['mean']:.2f}")
    print(f"Z-score (vs own class): {zscore}")
    print(f"Trend:                 {trend}")
    if zscore < -1:
        print("Flag: Consistently underperforming relative to class.")

    trend_df = pd.DataFrame({"test_name": per_test.index, "score": per_test.values})
    fig = charts.plot_student_trend(trend_df, student["name"], stats["mean"])
    state["last_df"] = scores_df
    state["last_fig"] = fig
    prompt_export(scores_df, fig, f"student_report_{student['roll_number']}")

# Handler for generating a pivot table of subjects vs students, showing average scores
def handle_pivot_table(conn, state: dict) -> None:
    if not require_data(conn):
        return
    class_name = prompt_class_filter(conn)

    scores_df = queries.get_all_scores(conn, class_name=class_name)
    pivot = analytics.build_pivot_table(scores_df)

    label = class_name or "All Classes"
    print(f"\nSubject vs Student Pivot Table — {label} (average scores):")
    print_table(pivot.reset_index())

    state["last_df"] = pivot.reset_index()
    state["last_fig"] = None
    prompt_export(state["last_df"], None, f"pivot_table_{label}")

# Handler for computing percentile ranks of students based on average scores
def handle_percentile_ranking(conn, state: dict) -> None:
    if not require_data(conn):
        return
    class_name = prompt_class_filter(conn)
    subject = prompt_subject_filter(conn)

    avg_df = queries.get_student_averages(conn, subject=subject, class_name=class_name)
    if avg_df.empty:
        print("\nNo matching data found.")
        return

    label = class_name or "All Classes"
    ranked = analytics.add_percentile_ranks(avg_df)
    print(f"\nPercentile Ranking — {label}" + (f" ({subject})" if subject else "") + ":")
    print_table(ranked)

    title = f"Percentile Ranking — {label}" + (f" — {subject}" if subject else "")
    fig = charts.plot_percentile_ranks(ranked, title=title)
    state["last_df"] = ranked
    state["last_fig"] = fig
    prompt_export(ranked, fig, f"percentile_ranking_{label}")

# Handler for analyzing subject strengths and weaknesses across the school or a specific class
def handle_subject_strengths(conn, state: dict) -> None:
    if not require_data(conn):
        return
    class_name = prompt_class_filter(conn)

    scores_df = queries.get_all_scores(conn, class_name=class_name)
    if scores_df.empty:
        print("\nNo matching data found.")
        return

    label = class_name or "All Classes"
    pivot = analytics.build_pivot_table(scores_df)
    strengths = analytics.find_subject_strengths(pivot)
    print(f"\nSubject Strengths (Best vs Weakest) — {label}:")
    print_table(strengths)

    fig = charts.plot_subject_strengths(strengths, title=f"Subject Strengths — {label}")
    state["last_df"] = strengths
    state["last_fig"] = fig
    prompt_export(strengths, fig, f"subject_strengths_{label}")

# Handler for computing improvement trends of students over time, highlighting those with the most positive change
def handle_improvement(conn, state: dict) -> None:
    if not require_data(conn):
        return
    class_name = prompt_class_filter(conn)
    subject = prompt_subject_filter(conn)

    scores_df = queries.get_all_scores(conn, subject=subject, class_name=class_name)
    if scores_df.empty:
        print("\nNo matching data found.")
        return

    label = class_name or "All Classes"
    improve_df = analytics.compute_improvement_by_student(scores_df)
    print(f"\nImprovement Leaderboard — {label}" + (f" ({subject})" if subject else "") + ":")
    print_table(improve_df)

    title = f"Improvement Leaderboard — {label}" + (f" — {subject}" if subject else "")
    fig = charts.plot_improvement(improve_df, title=title)
    state["last_df"] = improve_df
    state["last_fig"] = fig
    prompt_export(improve_df, fig, f"improvement_{label}")

# Handler for analyzing consistency of student performance across subjects, identifying those with the most stable scores
def handle_consistency(conn, state: dict) -> None:
    if not require_data(conn):
        return
    class_name = prompt_class_filter(conn)
    subject = prompt_subject_filter(conn)

    scores_df = queries.get_all_scores(conn, subject=subject, class_name=class_name)
    if scores_df.empty:
        print("\nNo matching data found.")
        return

    label = class_name or "All Classes"
    cons_df = analytics.compute_consistency_by_student(scores_df)
    print(f"\nConsistency Analysis — {label}" + (f" ({subject})" if subject else "") + ":")
    print_table(cons_df)

    title = f"Consistency — {label}" + (f" — {subject}" if subject else "")
    fig = charts.plot_consistency_scatter(cons_df, title=title)
    state["last_df"] = cons_df
    state["last_fig"] = fig
    prompt_export(cons_df, fig, f"consistency_{label}")

# Handler for detecting outliers in student performance based on z-scores, highlighting those who significantly deviate from the mean
def handle_outlier_detection(conn, state: dict) -> None:
    if not require_data(conn):
        return
    class_name = prompt_class_filter(conn)
    subject = prompt_subject_filter(conn)
    try:
        threshold = float(input("Outlier sensitivity (z-score threshold) [default 2.0]: ") or 2.0)
    except ValueError:
        threshold = 2.0

    avg_df = queries.get_student_averages(conn, subject=subject, class_name=class_name)
    if avg_df.empty:
        print("\nNo matching data found.")
        return

    label = class_name or "All Classes"
    full_df = analytics.add_zscore_column(avg_df)
    outliers = analytics.detect_outliers(avg_df, threshold=threshold)

    print(f"\nOutlier Detection — {label}" + (f" ({subject})" if subject else "") + f" (threshold={threshold}):")
    if outliers.empty:
        print("No outliers found at this threshold.")
    else:
        print_table(outliers)

    title = f"Outlier Detection — {label}" + (f" — {subject}" if subject else "")
    fig = charts.plot_outliers_scatter(full_df, outliers, title=title)
    state["last_df"] = outliers
    state["last_fig"] = fig
    prompt_export(outliers, fig, f"outliers_{label}")

# Handler for comparing performance across different classes
def handle_class_comparison(conn, state: dict) -> None:
    if not require_data(conn):
        return
    subject = prompt_subject_filter(conn)

    scores_df = queries.get_all_scores(conn, subject=subject)
    if scores_df.empty:
        print("\nNo matching data found.")
        return

    overall = analytics.compare_classes(scores_df)
    by_subject = analytics.compare_classes_by_subject(scores_df)

    print("\nClass Comparison — Overall Average:")
    print_table(overall)
    print("\nClass Comparison — By Subject:")
    print_table(by_subject.reset_index())

    title = "Class vs Subject Comparison" + (f" — {subject}" if subject else "")
    fig = charts.plot_class_comparison_heatmap(by_subject, title=title)
    state["last_df"] = by_subject.reset_index()
    state["last_fig"] = fig
    prompt_export(by_subject.reset_index(), fig, "class_comparison")

# Handler for computing pass rates by subject or class, allowing the user to choose the grouping
def handle_pass_rate(conn, state: dict) -> None:
    if not require_data(conn):
        return
    class_name = prompt_class_filter(conn)

    group_by = "subject"
    if not class_name:
        choice = input("Group pass rate by (1) Subject [default] or (2) Class/Grade: ").strip()
        if choice == "2":
            group_by = "class_name"

    scores_df = queries.get_all_scores(conn, class_name=class_name)
    if scores_df.empty:
        print("\nNo matching data found.")
        return

    label = class_name or "All Classes"
    pass_df = analytics.compute_pass_rate(scores_df, pass_mark=35, group_by=group_by)
    print(f"\nPass Rate Summary — {label} (grouped by {group_by}):")
    print_table(pass_df)

    fig = charts.plot_pass_rate_bar(pass_df, title=f"Pass Rate — {label}")
    state["last_df"] = pass_df
    state["last_fig"] = fig
    prompt_export(pass_df, fig, f"pass_rate_{label}")

# Handler for displaying the top N performers in each subject, allowing the user to specify how many to show
def handle_subject_top_performers(conn, state: dict) -> None:
    if not require_data(conn):
        return
    class_name = prompt_class_filter(conn)
    try:
        top_n = int(input("How many top performers per subject? [default 1]: ") or 1)
    except ValueError:
        top_n = 1

    scores_df = queries.get_all_scores(conn, class_name=class_name)
    if scores_df.empty:
        print("\nNo matching data found.")
        return

    label = class_name or "All Classes"
    top_df = analytics.rank_subject_performers(scores_df, top_n=top_n)
    print(f"\nTop {top_n} Performer(s) by Subject — {label}:")
    print_table(top_df)

    fig = charts.plot_subject_top_performers(top_df, title=f"Top Performers by Subject — {label}")
    state["last_df"] = top_df
    state["last_fig"] = fig
    prompt_export(top_df, fig, f"subject_top_performers_{label}")

# Handler for exporting the last generated report
def handle_export_last(conn, state: dict) -> None:
    if state.get("last_df") is None and state.get("last_fig") is None:
        print("\nNo report generated yet this session.")
        return
    prompt_export(state.get("last_df"), state.get("last_fig"), "last_report")


HANDLERS = {
    "1": handle_seed,
    "2": handle_class_average,
    "3": handle_top_bottom,
    "4": handle_weak_subjects,
    "5": handle_grade_distribution,
    "6": handle_student_report,
    "7": handle_pivot_table,
    "8": handle_percentile_ranking,
    "9": handle_subject_strengths,
    "10": handle_improvement,
    "11": handle_consistency,
    "12": handle_outlier_detection,
    "13": handle_class_comparison,
    "14": handle_pass_rate,
    "15": handle_subject_top_performers,
    "16": handle_export_last,
}

# ---------------------------------------------------------------- main loop
def run() -> None:
    db_manager.init_db()
    conn = db_manager.get_connection()
    state = {"last_df": None, "last_fig": None}

    try:
        while True:
            print(MENU_TEXT)
            choice = input("Select an option: ").strip()

            if choice == "0":
                print("Goodbye!")
                break

            handler = HANDLERS.get(choice)
            if handler is None:
                print("\nInvalid option. Please choose a number from the menu.")
                continue

            try:
                handler(conn, state)
            except (DataNotFoundError, StudentNotFoundError) as e:
                print(f"\nError: {e}")
            except Exception as e:  # last-resort guard so the CLI never crashes
                print(f"\nUnexpected error: {e}")
    finally:
        conn.close()
