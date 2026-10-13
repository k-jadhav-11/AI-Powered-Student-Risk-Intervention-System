"""
streamlit_app.py — Streamlit UI for the Student Performance Analytics platform.

This is a NEW presentation layer only. It imports src/db_manager.py,
src/queries.py, src/analytics.py, src/charts.py, src/reports.py exactly as
they are — none of that code was touched. src/cli.py and main.py still work
unchanged, so the terminal CLI is not removed, just joined by a second way
to use the same backend.

Run with:
    streamlit run streamlit_app.py
"""
import base64
import io
import json
import os

import numpy as np
import pandas as pd
import streamlit as st

from src.analytics import db_manager, queries, analytics, charts
from src.analytics.exceptions import InvalidCSVError, DataNotFoundError, StudentNotFoundError
from src import risk as rs
from src.risk import risk_queries
from ai_assistant import memory as genai_memory, llm_client, agent, rag as ai_rag
from ai_assistant import tools as ai_tools
from orchestration import audit as intervention_audit
from orchestration import dashboard_charts as intervention_charts
from orchestration import graph as intervention_graph

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")

st.set_page_config(page_title="Student Performance Analytics", layout="wide")

# --------------------------------------------------------------- visual polish
# One shared stylesheet, injected once per run, so every Sprint 2-5 page
# inherits the same spacing/card/badge treatment instead of each page
# hand-rolling its own st.markdown(unsafe_allow_html=True) snippets. Pure
# CSS only -- no page's Python logic changes, so this is safe to add
# without touching any page function's behavior, including the newer
# _render_embedded_chat_expander()/_render_intervention_chat_panel() split.
_CUSTOM_CSS = """
<style>
/* Tighter vertical rhythm -- Streamlit's default block spacing reads as
   very loose once a page has more than 4-5 stacked sections. */
div.block-container { padding-top: 2rem; padding-bottom: 3rem; }
div[data-testid="stVerticalBlock"] > div { margin-bottom: 0.15rem; }

/* Metrics as elevated cards instead of bare numbers -- used heavily on
   the Teacher Dashboard and the single-student risk lookup. */
div[data-testid="stMetric"] {
    background: var(--secondary-background-color);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 0.9rem 1rem 0.7rem 1rem;
}
div[data-testid="stMetricLabel"] { opacity: 0.75; }

/* Expanders ("Example messages", "Full decision detail", and the new
   embedded "Ask the Intervention Agent about this page" panel) as
   bordered cards matching the metric/dataframe language below, instead
   of Streamlit's near-invisible default expander. */
div[data-testid="stExpander"] {
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
}

/* Sidebar: a visible divider under the branding block, and a touch more
   breathing room around the module/report radios. */
section[data-testid="stSidebar"] div.block-container { padding-top: 1.5rem; }
section[data-testid="stSidebar"] hr { margin: 0.6rem 0 1rem 0; }

/* Buttons: rounded to match the card language above; subtle lift on
   hover so primary actions ("Run Intervention Agent", "Run batch")
   read as clearly clickable. */
div.stButton > button, div.stDownloadButton > button {
    border-radius: 8px;
    transition: transform 0.05s ease-in-out;
}
div.stButton > button:hover, div.stDownloadButton > button:hover { transform: translateY(-1px); }

/* Dataframes/tables: rounded container so they match the card language
   instead of sitting as a hard-edged rectangle. */
div[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

/* Decision pills (nudge / resource / escalate) -- see _decision_pill_html
   below. Colors intentionally match the emoji they replace so anyone
   used to the old 🟢/🟡/🔴 badges reads these the same way at a glance. */
.decision-pill {
    display: inline-block; padding: 0.25rem 0.75rem; border-radius: 999px;
    font-weight: 600; font-size: 0.95rem; letter-spacing: 0.01em;
}
.decision-pill.nudge     { background: rgba(46, 204, 113, 0.16); color: #2ECC71; }
.decision-pill.resource  { background: rgba(243, 156, 18, 0.16); color: #F39C12; }
.decision-pill.escalate  { background: rgba(231, 76, 60, 0.16);  color: #E74C3C; }

/* Chat bubbles: a little extra padding so multi-paragraph answers don't
   feel cramped against the bubble edge -- matters more now that chat
   also renders inside the narrower embedded expander panels. */
div[data-testid="stChatMessage"] { padding: 0.3rem 0.4rem; }

/* Independent scrolling for the Single Student / Cohort Batch / Teacher
   Dashboard pages' layout -- st.columns() alone shares ONE page-level
   scrollbar, so a long chat thread on the right pushed the whole page
   (including the main column) down with it. Scoped to just these two
   containers (via st.container(key=...), NOT a blanket
   `div[data-testid="column"]` rule) so ordinary st.columns() usage
   elsewhere -- the metric rows, the dataset/decision filter row, the
   cohort nav buttons -- is completely unaffected.
   min-height: 0 is the actual fix, not height/overflow alone: these
   containers are flex ITEMS (Streamlit lays out column content with
   flexbox), and a flex item's default min-height is "auto", which means
   it refuses to shrink below its own content size no matter what height
   or overflow you set on it -- the classic reason "I set overflow-y:auto
   but nothing scrolls, it just grows the whole page instead" happens.
   Forcing min-height: 0 lets the box actually respect its height and
   scroll internally instead. !important because Streamlit's own
   generated styles on these elements otherwise win the specificity tie. */
.st-key-ia_main_col, .st-key-ia_chat_col {
    height: calc(100vh - 7rem) !important;
    max-height: calc(100vh - 7rem) !important;
    min-height: 0 !important;
    overflow-y: auto !important;
    padding-right: 0.75rem;
    scrollbar-gutter: stable;
}
/* The column div wrapping each keyed container also needs min-height: 0
   for the same flexbox reason -- otherwise the fix above is undone one
   level up the tree. :has() scopes this to only the two columns that
   actually contain our keyed containers, same as everything above. */
div[data-testid="column"]:has(> div > .st-key-ia_main_col),
div[data-testid="column"]:has(> div > .st-key-ia_chat_col) {
    min-height: 0 !important;
}
</style>
"""


def _inject_custom_css():
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------- connection
# A single MySQL connection is cached for the session (st.cache_resource),
# and re-pinged/reconnected on each rerun since Streamlit can idle long
# enough for MySQL to drop the connection.


@st.cache_resource
def _open_connection():
    db_manager.init_db()
    return db_manager.get_connection()

@st.cache_resource(show_spinner="Setting up the intervention knowledge base...")
def _warm_up_rag() -> str:
    return ai_rag.warm_up()

def get_conn():
    conn = _open_connection()
    try:
        conn.ping(reconnect=True, attempts=3, delay=1)
    except Exception:
        _open_connection.clear()
        conn = _open_connection()
    return conn


# ------------------------------------------------------------------ helpers
def show_table(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        st.info("No data.")
    else:
        st.dataframe(df, width="stretch")


def class_filter(conn, key: str):
    classes = queries.get_available_classes(conn)
    if not classes:
        return None
    choice = st.selectbox("Class / grade", ["All classes"] + classes, key=key)
    return None if choice == "All classes" else choice


def subject_filter(conn, key: str):
    subjects = queries.get_available_subjects(conn)
    if not subjects:
        return None
    choice = st.selectbox("Subject", ["Overall (all subjects)"] + subjects, key=key)
    return None if choice == "Overall (all subjects)" else choice


def export_buttons(df: pd.DataFrame, fig, name_hint: str, key_prefix: str) -> None:
    """CSV/PNG download buttons, replacing the CLI's 'Export this report? (y/n)' prompt."""
    col1, col2 = st.columns(2)
    if df is not None and not df.empty:
        col1.download_button(
            "Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name=f"{name_hint}.csv",
            mime="text/csv",
            key=f"{key_prefix}_csv",
        )
    if fig is not None:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        col2.download_button(
            "Download PNG",
            buf.getvalue(),
            file_name=f"{name_hint}.png",
            mime="image/png",
            key=f"{key_prefix}_png",
        )


def require_data(conn) -> bool:
    if not db_manager.has_data(conn):
        st.warning("No data found yet. Go to **Load / Seed Data** first.")
        return False
    return True


def require_risk_artifacts(dataset: str) -> bool:
    """Sprint 3 pages read pre-trained model files and pre-computed CSVs
    from disk instead of MySQL -- this checks those exist, same role
    require_data() plays for the MySQL-backed Sprint 2 pages."""
    model_path = os.path.join(rs.MODELS_DIR, f"risk_model_{dataset}_v1.joblib")
    scores_path = os.path.join(rs.PROCESSED_DIR, f"{dataset}_risk_scores.csv")
    if not (os.path.exists(model_path) and os.path.exists(scores_path)):
        st.warning(
            f"No trained {dataset.upper()} risk model / risk scores found yet. "
            f"Run `notebooks/{'uci' if dataset == 'uci' else 'oulad'}_notebooks/"
            f"model_training_{dataset}.ipynb` first — it saves the files this page reads."
        )
        return False
    return True


# --------------------------------------------------------------------- pages
def page_seed_data(conn):
    st.header("Load / Seed Data from CSV")
    st.write(
        "Loads `data/students.csv`, `data/subjects.csv`, and `data/marks.csv` into MySQL. "
        "Existing rows are skipped automatically, so it's safe to re-run after adding new rows."
    )
    if st.button("Seed / Reload Data", type="primary"):
        try:
            summary = db_manager.seed_all(conn, DATA_DIR)
            st.success("Seeding complete.")
            st.json(summary)
        except (InvalidCSVError, FileNotFoundError) as e:
            st.error(f"Error loading data: {e}")


def page_class_average(conn):
    st.header("Class Average Report (per subject)")
    if not require_data(conn):
        return
    class_name = class_filter(conn, "avg_class")
    df = queries.get_class_average_by_subject(conn, class_name=class_name)
    label = class_name or "All Classes"

    show_table(df)
    if not df.empty:
        fig = charts.plot_class_average_bar(df, title=f"Class Average by Subject — {label}")
        st.pyplot(fig)
        export_buttons(df, fig, f"class_average_{label}", "avg")


def page_top_bottom(conn):
    st.header("Top/Bottom Performers")
    if not require_data(conn):
        return
    class_name = class_filter(conn, "tb_class")
    subject = subject_filter(conn, "tb_subject")
    n = st.number_input("How many top/bottom students to show?", min_value=1, max_value=50, value=5)

    top, bottom = queries.get_top_bottom_students(conn, n=int(n), subject=subject, class_name=class_name)
    if top.empty:
        st.info("No matching data found.")
        return

    label = class_name or "All Classes"
    st.subheader(f"Top {n} — {label}" + (f" ({subject})" if subject else ""))
    show_table(top)
    st.subheader(f"Bottom {n} — {label}" + (f" ({subject})" if subject else ""))
    show_table(bottom)

    title = f"Top/Bottom Performers — {label}" + (f" — {subject}" if subject else "")
    fig = charts.plot_ranked_bar(top, bottom, title=title)
    st.pyplot(fig)
    combined = pd.concat([top.assign(group="Top"), bottom.assign(group="Bottom")])
    export_buttons(combined, fig, f"top_bottom_{label}", "tb")


def page_weak_subjects(conn):
    st.header("Subject-wise Weak Area Detection")
    if not require_data(conn):
        return
    class_name = class_filter(conn, "weak_class")

    avg_df = queries.get_class_average_by_subject(conn, class_name=class_name)
    fail_df = queries.get_subject_failure_rates(conn, pass_mark=35, class_name=class_name)
    weak_df = analytics.identify_weak_subjects(avg_df, fail_df)
    label = class_name or "All Classes"

    show_table(weak_df)
    if not weak_df.empty:
        fig = charts.plot_weak_subjects(weak_df, title=f"Subject Averages — {label} (Weak Subjects Highlighted)")
        st.pyplot(fig)
        export_buttons(weak_df, fig, f"weak_subjects_{label}", "weak")


def page_grade_distribution(conn):
    st.header("Grade Distribution")
    if not require_data(conn):
        return
    class_name = class_filter(conn, "grade_class")
    subject = subject_filter(conn, "grade_subject")

    scores_df = queries.get_all_scores(conn, subject=subject, class_name=class_name)
    if scores_df.empty:
        st.info("No matching data found.")
        return

    label = class_name or "All Classes"
    band_counts = analytics.compute_grade_bands(scores_df["score"].values)
    band_df = band_counts.reset_index()
    band_df.columns = ["grade_band", "count"]
    show_table(band_df)

    title = f"Grade Distribution — {label}" + (f" — {subject}" if subject else "")
    fig = charts.plot_grade_histogram(scores_df["score"].values, title=title)
    st.pyplot(fig)
    export_buttons(band_df, fig, f"grade_distribution_{label}", "grade")


def page_student_report(conn):
    st.header("Individual Student Report")
    if not require_data(conn):
        return
    search_term = st.text_input("Enter student name or roll number")
    if not search_term:
        return

    matches = queries.find_student(conn, search_term)
    if matches.empty:
        st.info(f"No student found matching '{search_term}'.")
        return

    if len(matches) > 1:
        st.write("Multiple matches found — pick one:")
        options = {
            f"{row['name']} ({row['roll_number']}, Class {row['class_name']})": row
            for _, row in matches.iterrows()
        }
        pick = st.selectbox("Student", list(options.keys()), key="student_pick")
        student = options[pick]
    else:
        student = matches.iloc[0]

    student_id = int(student["student_id"])
    scores_df = queries.get_student_scores_over_time(conn, student_id)
    if scores_df.empty:
        st.info(f"{student['name']} has no recorded marks yet.")
        return

    # Z-score is computed against the student's own class/grade, not the whole school.
    class_scores = queries.get_all_scores(conn, class_name=student["class_name"])
    stats = analytics.class_stats(class_scores["score"].values)
    student_avg = scores_df["score"].mean()
    zscore = analytics.compute_zscore(student_avg, stats["mean"], stats["std"])

    per_test = scores_df.groupby("test_name")["score"].mean().reindex(scores_df["test_name"].unique())
    trend = analytics.classify_trend(per_test.values)

    st.subheader(f"{student['name']} ({student['roll_number']}, Class {student['class_name']})")
    show_table(scores_df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Student average", f"{student_avg:.2f}")
    c2.metric(f"Class {student['class_name']} average", f"{stats['mean']:.2f}")
    c3.metric("Z-score (vs own class)", zscore)
    st.write(f"**Trend:** {trend}")
    if zscore < -1:
        st.warning("Flag: Consistently underperforming relative to class.")

    trend_df = pd.DataFrame({"test_name": per_test.index, "score": per_test.values})
    fig = charts.plot_student_trend(trend_df, student["name"], stats["mean"])
    st.pyplot(fig)
    export_buttons(scores_df, fig, f"student_report_{student['roll_number']}", "student")


def page_pivot_table(conn):
    st.header("Pivot Table (Subject vs Student)")
    if not require_data(conn):
        return
    class_name = class_filter(conn, "pivot_class")

    scores_df = queries.get_all_scores(conn, class_name=class_name)
    pivot = analytics.build_pivot_table(scores_df)
    label = class_name or "All Classes"

    pivot_out = pivot.reset_index()
    show_table(pivot_out)
    export_buttons(pivot_out, None, f"pivot_table_{label}", "pivot")


def page_percentile_ranking(conn):
    st.header("Percentile Ranking")
    if not require_data(conn):
        return
    class_name = class_filter(conn, "pct_class")
    subject = subject_filter(conn, "pct_subject")

    avg_df = queries.get_student_averages(conn, subject=subject, class_name=class_name)
    if avg_df.empty:
        st.info("No matching data found.")
        return

    label = class_name or "All Classes"
    ranked = analytics.add_percentile_ranks(avg_df)
    show_table(ranked)

    title = f"Percentile Ranking — {label}" + (f" — {subject}" if subject else "")
    fig = charts.plot_percentile_ranks(ranked, title=title)
    st.pyplot(fig)
    export_buttons(ranked, fig, f"percentile_ranking_{label}", "pct")


def page_subject_strengths(conn):
    st.header("Subject Strengths (Best vs Weakest)")
    if not require_data(conn):
        return
    class_name = class_filter(conn, "strength_class")

    scores_df = queries.get_all_scores(conn, class_name=class_name)
    if scores_df.empty:
        st.info("No matching data found.")
        return

    label = class_name or "All Classes"
    pivot = analytics.build_pivot_table(scores_df)
    strengths = analytics.find_subject_strengths(pivot)
    show_table(strengths)

    fig = charts.plot_subject_strengths(strengths, title=f"Subject Strengths — {label}")
    st.pyplot(fig)
    export_buttons(strengths, fig, f"subject_strengths_{label}", "strength")


def page_improvement(conn):
    st.header("Improvement Leaderboard")
    if not require_data(conn):
        return
    class_name = class_filter(conn, "imp_class")
    subject = subject_filter(conn, "imp_subject")

    scores_df = queries.get_all_scores(conn, subject=subject, class_name=class_name)
    if scores_df.empty:
        st.info("No matching data found.")
        return

    label = class_name or "All Classes"
    improve_df = analytics.compute_improvement_by_student(scores_df)
    show_table(improve_df)

    title = f"Improvement Leaderboard — {label}" + (f" — {subject}" if subject else "")
    fig = charts.plot_improvement(improve_df, title=title)
    st.pyplot(fig)
    export_buttons(improve_df, fig, f"improvement_{label}", "imp")


def page_consistency(conn):
    st.header("Consistency Analysis")
    if not require_data(conn):
        return
    class_name = class_filter(conn, "cons_class")
    subject = subject_filter(conn, "cons_subject")

    scores_df = queries.get_all_scores(conn, subject=subject, class_name=class_name)
    if scores_df.empty:
        st.info("No matching data found.")
        return

    label = class_name or "All Classes"
    cons_df = analytics.compute_consistency_by_student(scores_df)
    show_table(cons_df)

    title = f"Consistency — {label}" + (f" — {subject}" if subject else "")
    fig = charts.plot_consistency_scatter(cons_df, title=title)
    st.pyplot(fig)
    export_buttons(cons_df, fig, f"consistency_{label}", "cons")


def page_outlier_detection(conn):
    st.header("Outlier Detection")
    if not require_data(conn):
        return
    class_name = class_filter(conn, "out_class")
    subject = subject_filter(conn, "out_subject")
    threshold = st.number_input("Outlier sensitivity (z-score threshold)", min_value=0.5, max_value=5.0, value=2.0, step=0.1)

    avg_df = queries.get_student_averages(conn, subject=subject, class_name=class_name)
    if avg_df.empty:
        st.info("No matching data found.")
        return

    label = class_name or "All Classes"
    full_df = analytics.add_zscore_column(avg_df)
    outliers = analytics.detect_outliers(avg_df, threshold=threshold)

    st.write(f"Threshold = {threshold}")
    if outliers.empty:
        st.info("No outliers found at this threshold.")
    else:
        show_table(outliers)

    title = f"Outlier Detection — {label}" + (f" — {subject}" if subject else "")
    fig = charts.plot_outliers_scatter(full_df, outliers, title=title)
    st.pyplot(fig)
    export_buttons(outliers, fig, f"outliers_{label}", "out")


def page_class_comparison(conn):
    st.header("Class Comparison (Grade vs Grade)")
    if not require_data(conn):
        return
    subject = subject_filter(conn, "cmp_subject")

    scores_df = queries.get_all_scores(conn, subject=subject)
    if scores_df.empty:
        st.info("No matching data found.")
        return

    overall = analytics.compare_classes(scores_df)
    by_subject = analytics.compare_classes_by_subject(scores_df)

    st.subheader("Overall Average")
    show_table(overall)
    st.subheader("By Subject")
    by_subject_out = by_subject.reset_index()
    show_table(by_subject_out)

    title = "Class vs Subject Comparison" + (f" — {subject}" if subject else "")
    fig = charts.plot_class_comparison_heatmap(by_subject, title=title)
    st.pyplot(fig)
    export_buttons(by_subject_out, fig, "class_comparison", "cmp")


def page_pass_rate(conn):
    st.header("Pass Rate Summary")
    if not require_data(conn):
        return
    class_name = class_filter(conn, "pass_class")

    group_by = "subject"
    if not class_name:
        choice = st.radio("Group pass rate by", ["Subject", "Class/Grade"], horizontal=True)
        group_by = "subject" if choice == "Subject" else "class_name"

    scores_df = queries.get_all_scores(conn, class_name=class_name)
    if scores_df.empty:
        st.info("No matching data found.")
        return

    label = class_name or "All Classes"
    pass_df = analytics.compute_pass_rate(scores_df, pass_mark=35, group_by=group_by)
    show_table(pass_df)

    fig = charts.plot_pass_rate_bar(pass_df, title=f"Pass Rate — {label}")
    st.pyplot(fig)
    export_buttons(pass_df, fig, f"pass_rate_{label}", "pass")


def page_subject_top_performers(conn):
    st.header("Subject Top Performers")
    if not require_data(conn):
        return
    class_name = class_filter(conn, "stp_class")
    top_n = st.number_input("How many top performers per subject?", min_value=1, max_value=20, value=1)

    scores_df = queries.get_all_scores(conn, class_name=class_name)
    if scores_df.empty:
        st.info("No matching data found.")
        return

    label = class_name or "All Classes"
    top_df = analytics.rank_subject_performers(scores_df, top_n=int(top_n))
    show_table(top_df)

    fig = charts.plot_subject_top_performers(top_df, title=f"Top Performers by Subject — {label}")
    st.pyplot(fig)
    export_buttons(top_df, fig, f"subject_top_performers_{label}", "stp")


# --------------------------------------------------------- Sprint 3: risk pages
# These pages don't use `conn` (MySQL) -- they read the pre-trained model
# files and pre-computed CSVs that model_training_uci.ipynb / model_training_
# oulad.ipynb save to models/ and data/processed/. Kept the same `(conn)`
# signature as every Sprint 2 page purely so main()'s dispatch loop doesn't
# need a special case; the parameter is simply unused here.

def _risk_level_badge(risk_level: str, risk_score: float) -> None:
    if risk_level == "High":
        st.error(f"**High risk** — risk score {risk_score:.1%}")
    elif risk_level == "Medium":
        st.warning(f"**Medium risk** — risk score {risk_score:.1%}")
    else:
        st.success(f"**Low risk** — risk score {risk_score:.1%}")


def page_risk_dashboard_uci(conn):
    st.header("Risk Dashboard — UCI Student Performance")
    if not require_risk_artifacts("uci"):
        return

    scores = rs.load_risk_scores("uci")
    summary = rs.load_model_summary("uci")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Students scored", f"{len(scores):,}")
    c2.metric("At-risk (predicted)", f"{(scores['predicted_label'] == 'at_risk').sum():,}")
    c3.metric("Model", rs.get_model_class_name("uci"))
    c4.metric("Test F1 (at-risk)", f"{summary.get('test_f1_at_risk', 0):.3f}")

    level_filter = st.multiselect("Filter by risk level", ["High", "Medium", "Low"],
                                   default=["High", "Medium", "Low"])
    filtered = scores[scores["risk_level"].isin(level_filter)].sort_values("risk_score", ascending=False)
    show_table(filtered)

    fig = rs.plot_risk_level_counts(filtered, title="Risk Level Distribution — UCI")
    st.pyplot(fig)
    export_buttons(filtered, fig, "uci_risk_scores", "uci_risk")


def page_risk_dashboard_oulad(conn):
    st.header("Risk Dashboard — OULAD")
    if not require_risk_artifacts("oulad"):
        return

    scores = rs.load_risk_scores("oulad")
    summary = rs.load_model_summary("oulad")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Students scored", f"{len(scores):,}")
    c2.metric("At-risk (predicted)", f"{(scores['predicted_label'] == 'at_risk').sum():,}")
    c3.metric("Model", rs.get_model_class_name("oulad"))
    c4.metric("Test F1 (at-risk)", f"{summary.get('test_f1_at_risk', 0):.3f}")

    level_filter = st.multiselect("Filter by risk level", ["High", "Medium", "Low"],
                                   default=["High", "Medium", "Low"], key="oulad_level_filter")
    filtered = scores[scores["risk_level"].isin(level_filter)].sort_values("risk_score", ascending=False)
    show_table(filtered.head(500))
    if len(filtered) > 500:
        st.caption(f"Showing top 500 of {len(filtered):,} matching rows — use the CSV download for the full set.")

    fig = rs.plot_risk_level_counts(filtered, title="Risk Level Distribution — OULAD")
    st.pyplot(fig)
    export_buttons(filtered, fig, "oulad_risk_scores", "oulad_risk")


def page_risk_model_performance_uci(conn):
    st.header("Risk Model Performance — UCI")
    if not require_risk_artifacts("uci"):
        return

    performance = risk_queries.get_model_performance("uci")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision (at-risk)", f"{performance['precision']:.3f}")
    c2.metric("Recall (at-risk)", f"{performance['recall']:.3f}")
    c3.metric("F1 (at-risk)", f"{performance['f1']:.3f}")
    c4.metric("ROC-AUC", f"{performance['roc_auc']:.3f}")
    st.caption(f"Model: {rs.get_model_class_name('uci')}  ·  threshold = {rs.RISK_THRESHOLD_UCI:.3f}")

    model = rs.load_model("uci")
    _, X_test, _, y_test = rs.get_test_split("uci")
    proba = model.predict_proba(X_test)[:, 1]
    preds = rs.predict_risk(model, X_test, rs.RISK_THRESHOLD_UCI)

    col1, col2 = st.columns(2)
    with col1:
        fig1 = rs.plot_precision_recall(y_test, proba, rs.RISK_THRESHOLD_UCI, title="Precision/Recall — UCI (test set)")
        st.pyplot(fig1)
    with col2:
        fig2 = rs.plot_risk_distribution(preds, rs.RISK_THRESHOLD_UCI, title="Risk Score Distribution — UCI (test set)")
        st.pyplot(fig2)


def page_risk_model_performance_oulad(conn):
    st.header("Risk Model Performance — OULAD")
    if not require_risk_artifacts("oulad"):
        return

    performance = risk_queries.get_model_performance("oulad")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision (at-risk)", f"{performance['precision']:.3f}")
    c2.metric("Recall (at-risk)", f"{performance['recall']:.3f}")
    c3.metric("F1 (at-risk)", f"{performance['f1']:.3f}")
    c4.metric("ROC-AUC", f"{performance['roc_auc']:.3f}")
    st.caption(f"Model: {rs.get_model_class_name('oulad')}  ·  threshold = {rs.RISK_THRESHOLD_OULAD:.3f}")

    model = rs.load_model("oulad")
    _, X_test, _, y_test = rs.get_test_split("oulad")
    proba = model.predict_proba(X_test)[:, 1]
    preds = rs.predict_risk(model, X_test, rs.RISK_THRESHOLD_OULAD)

    col1, col2 = st.columns(2)
    with col1:
        fig1 = rs.plot_precision_recall(y_test, proba, rs.RISK_THRESHOLD_OULAD, title="Precision/Recall — OULAD (test set)")
        st.pyplot(fig1)
    with col2:
        fig2 = rs.plot_risk_distribution(preds, rs.RISK_THRESHOLD_OULAD, title="Risk Score Distribution — OULAD (test set)")
        st.pyplot(fig2)

    st.subheader("Lead time & GRU comparison (precomputed, population-level)")
    st.caption(
        "These two need a full pass over the raw studentVle.csv (lead-time "
        "simulation) or a second trained model (GRU) — too heavy to (re)compute "
        "on a page load, so these are the saved images from risk_charts.ipynb "
        "and model_gru_oulad.ipynb. For a **live** GRU score and week-by-week "
        "risk trajectory on one real student, see the **Student Time-Series "
        "Risk Explorer (GRU)** page in the sidebar."
    )
    lead_time_path = os.path.join(REPORTS_DIR, "lead_time_oulad.png")
    gru_path = os.path.join(REPORTS_DIR, "baseline_vs_gru_oulad.png")
    col3, col4 = st.columns(2)
    if os.path.exists(lead_time_path):
        col3.image(lead_time_path, caption="Early-warning lead time (sampled withdrawn students)")
    if os.path.exists(gru_path):
        col4.image(gru_path, caption="Baseline vs. tuned/GRU comparison")

    with st.expander("⚡ Live sampled comparison (compute now, on a small real sample)"):
        st.caption(
            "The precomputed chart above is a full-population run saved from a notebook. This runs the same "
            "idea live, on a smaller sample (fast enough for a page load): samples real students across all "
            "four outcomes, scores each with both models from the same raw data, and reports precision/recall/F1 "
            "against actual Fail/Withdrawn labels for that sample. Expect noisier numbers than the frozen "
            "full-population chart — it's a smaller n."
        )
        if not require_gru_artifacts():
            pass
        else:
            n_per_outcome = st.slider("Students per outcome (× 4 outcomes)", min_value=5, max_value=30, value=15)
            if st.button("Run live comparison now"):
                with st.spinner(f"Scoring ~{n_per_outcome * 4} students with both models..."):
                    gru_model, gru_cfg = _get_gru_model()
                    xgb_model = rs.load_model("oulad")
                    comparison, metrics = rs.run_live_sampled_comparison(
                        gru_model, gru_cfg, xgb_model, sample_per_outcome=n_per_outcome
                    )
                fig = rs.plot_live_sampled_comparison(metrics, n_students=len(comparison))
                st.pyplot(fig)
                mcol1, mcol2 = st.columns(2)
                for col, (name, m) in zip([mcol1, mcol2], metrics.items()):
                    col.metric(f"{name} F1", m["f1"])
                    col.caption(f"precision={m['precision']}  ·  recall={m['recall']}")
                st.dataframe(
                    comparison[["id_student", "final_result", "gru_risk_score", "gru_predicted_label",
                                "xgb_risk_score", "xgb_predicted_label"]],
                    width="stretch",
                )


def page_predict_new_student_uci(conn):
    st.header("Predict Risk — New Student (UCI)")
    if not require_risk_artifacts("uci"):
        return
    st.caption("Enter a student's data to get a risk score before their final grade exists.")

    with st.form("predict_uci_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", 15, 22, 17)
            g1 = st.number_input("Period 1 grade (G1, 0-20)", 0, 20, 10)
            g2 = st.number_input("Period 2 grade (G2, 0-20)", 0, 20, 10)
            absences = st.number_input("Absences", 0, 93, 4)
            failures = st.number_input("Past class failures", 0, 3, 0)
            studytime = st.selectbox("Weekly study time", [1, 2, 3, 4],
                                      format_func=lambda v: {1: "<2h", 2: "2-5h", 3: "5-10h", 4: ">10h"}[v])
        with c2:
            medu = st.selectbox("Mother's education (0=none, 4=higher)", [0, 1, 2, 3, 4], index=2)
            fedu = st.selectbox("Father's education (0=none, 4=higher)", [0, 1, 2, 3, 4], index=2)
            traveltime = st.selectbox("Home-to-school travel time", [1, 2, 3, 4], index=0)
            famrel = st.slider("Family relationship quality (1-5)", 1, 5, 4)
            freetime = st.slider("Free time after school (1-5)", 1, 5, 3)
            goout = st.slider("Going out with friends (1-5)", 1, 5, 3)
        with c3:
            dalc = st.slider("Workday alcohol use (1-5)", 1, 5, 1)
            walc = st.slider("Weekend alcohol use (1-5)", 1, 5, 1)
            health = st.slider("Current health (1-5)", 1, 5, 4)
            subject = st.selectbox("Subject", ["Mathematics", "Portuguese"])
            school = st.selectbox("School", ["GP", "MS"])
            sex = st.selectbox("Sex", ["F", "M"])

        c4, c5 = st.columns(2)
        with c4:
            address = st.selectbox("Home address", ["Urban", "Rural"])
            famsize = st.selectbox("Family size", ["<=3", ">3"])
            schoolsup = st.checkbox("Extra educational school support")
            famsup = st.checkbox("Extra educational family support", value=True)
        with c5:
            higher = st.checkbox("Wants higher education", value=True)
            internet = st.checkbox("Internet access at home", value=True)
            romantic = st.checkbox("In a romantic relationship")

        submitted = st.form_submit_button("Predict risk", type="primary")

    if submitted:
        record = {
            "age": age, "medu": medu, "fedu": fedu, "traveltime": traveltime,
            "studytime": studytime, "failures": failures, "famrel": famrel,
            "freetime": freetime, "goout": goout, "dalc": dalc, "walc": walc,
            "health": health, "absences": absences, "g1": g1, "g2": g2,
            "school_MS": school == "MS", "sex_M": sex == "M",
            "address_U": address == "Urban", "famsize_LE3": famsize == "<=3",
            "schoolsup_yes": schoolsup, "famsup_yes": famsup, "higher_yes": higher,
            "internet_yes": internet, "romantic_yes": romantic,
            "subject_Portuguese": subject == "Portuguese",
        }
        X_new = rs.build_single_record(record, rs.UCI_FEATURE_COLUMNS)
        model = rs.load_model("uci")
        result = rs.predict_risk(model, X_new, rs.RISK_THRESHOLD_UCI).iloc[0]
        _risk_level_badge(result["risk_level"], result["risk_score"])


def page_predict_new_student_oulad(conn):
    st.header("Predict Risk — New Student (OULAD)")
    if not require_risk_artifacts("oulad"):
        return
    st.caption(
        "Engagement figures below would normally come from the LMS automatically "
        "mid-course — entered manually here for demonstration."
    )

    code_modules = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"]
    regions = [
        "East Anglian Region", "East Midlands Region", "Ireland", "London Region",
        "North Region", "North Western Region", "Scotland", "South East Region",
        "South Region", "South West Region", "Wales", "West Midlands Region", "Yorkshire Region",
    ]
    imd_bands = {"0-10%": 5, "10-20%": 15, "20-30%": 25, "30-40%": 35, "40-50%": 45,
                 "50-60%": 55, "60-70%": 65, "70-80%": 75, "80-90%": 85, "90-100%": 95}
    education_map = {"No Formal quals": 0, "Lower Than A Level": 1, "A Level or Equivalent": 2,
                      "HE Qualification": 3, "Post Graduate Qualification": 4}
    age_band_map = {"0-35": 0, "35-55": 1, "55<=": 2}

    with st.form("predict_oulad_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            num_weeks_active = st.number_input("Weeks active so far", 0, 40, 10)
            total_clicks_sum = st.number_input("Total clicks so far", 0, 20000, 150)
            total_clicks_max = st.number_input("Peak single-week clicks", 0, 5000, 40)
            days_active_sum = st.number_input("Total distinct active days", 0, 280, 20)
            pct_submitted = st.slider("Assessments submitted so far (%)", 0, 100, 60)
        with c2:
            engagement_trend = st.number_input(
                "Engagement trend (2nd-half minus 1st-half avg weekly clicks)",
                -200, 200, 0, help="Negative = declining engagement, positive = increasing.",
            )
            code_module = st.selectbox("Module", code_modules)
            region = st.selectbox("Region", regions)
            gender = st.selectbox("Gender", ["F", "M"])
            disability = st.selectbox("Disability", ["N", "Y"])
        with c3:
            age_band = st.selectbox("Age band", list(age_band_map.keys()))
            highest_education = st.selectbox("Highest education", list(education_map.keys()))
            imd_band_label = st.selectbox("IMD band (deprivation, 0-10%=most deprived)", list(imd_bands.keys()))
            num_of_prev_attempts = st.number_input("Previous attempts at this module", 0, 6, 0)
            studied_credits = st.number_input("Studied credits", 30, 655, 60)

        submitted = st.form_submit_button("Predict risk", type="primary")

    if submitted:
        record = {
            "total_clicks_sum": total_clicks_sum,
            "total_clicks_mean": total_clicks_sum / max(num_weeks_active, 1),
            "total_clicks_max": total_clicks_max,
            "days_active_sum": days_active_sum,
            "days_active_mean": days_active_sum / max(num_weeks_active, 1),
            "num_weeks_active": num_weeks_active,
            "engagement_trend": engagement_trend,
            "num_of_prev_attempts": num_of_prev_attempts,
            "studied_credits": studied_credits,
            "pct_assessments_submitted": pct_submitted / 100,
            "age_band_ordinal": age_band_map[age_band],
            "highest_education_ordinal": education_map[highest_education],
            "imd_band_missing": 0,
            "imd_band_ordinal": imd_bands[imd_band_label],
            "gender_M": gender == "M",
            "disability_Y": disability == "Y",
        }
        for m in code_modules[1:]:  # AAA is the drop_first baseline
            record[f"code_module_{m}"] = code_module == m
        for r in regions[1:]:  # East Anglian Region is the drop_first baseline
            record[f"region_{r}"] = region == r

        X_new = rs.build_single_record(record, rs.OULAD_FEATURE_COLUMNS)
        model = rs.load_model("oulad")
        result = rs.predict_risk(model, X_new, rs.RISK_THRESHOLD_OULAD).iloc[0]
        _risk_level_badge(result["risk_level"], result["risk_score"])


def require_gru_artifacts() -> bool:
    """Same role as require_risk_artifacts(), for the GRU checkpoint specifically."""
    pt_path = os.path.join(rs.MODELS_DIR, "risk_model_oulad_gru_v1.pt")
    txt_path = os.path.join(rs.MODELS_DIR, "risk_model_oulad_gru_v1.txt")
    if not (os.path.exists(pt_path) or os.path.exists(txt_path)):
        st.warning(
            "No trained OULAD GRU checkpoint found yet. Run "
            "`notebooks/oulad_notebooks/model_gru_oulad.ipynb` first — it "
            "saves `models/risk_model_oulad_gru_v1.pt`, which this page reads."
        )
        return False
    vle_path = os.path.join(rs.OULAD_RAW_DIR, "studentVle.csv")
    if not os.path.exists(vle_path):
        st.warning(f"Raw OULAD data not found at `{vle_path}` — this page needs it to build a student's weekly sequence.")
        return False
    return True


@st.cache_resource
def _get_gru_model():
    return rs.load_gru_model("oulad")


@st.cache_data(show_spinner=False)
def _get_student_directory(sample_per_outcome: int = 20):
    return rs.load_oulad_student_directory(sample_per_outcome=sample_per_outcome)


@st.cache_data(show_spinner=False)
def _get_student_weekly(id_student: int, max_weeks: int):
    return rs.build_student_weekly_engagement(id_student, max_weeks=max_weeks)


def page_student_gru_explorer(conn):
    st.header("Student Time-Series Risk Explorer (GRU)")
    st.caption(
        "Live inference, not a saved image: reloads the trained OULAD GRU checkpoint, "
        "streams `studentVle.csv` for one real student's weekly click/active-day history, "
        "and scores it — once for the full sequence, and once per week to replay how the "
        "model's read would have evolved as the course progressed. The XGBoost score for "
        "the same student (built from the same raw data) is shown alongside it."
    )
    if not require_risk_artifacts("oulad") or not require_gru_artifacts():
        return

    with st.spinner("Loading GRU checkpoint..."):
        gru_model, gru_cfg = _get_gru_model()
    st.caption(
        f"EngagementGRU · max_weeks={gru_cfg['max_weeks']} · features={gru_cfg['feature_cols']} "
        f"· threshold={rs.RISK_THRESHOLD_OULAD_GRU:.2f}"
    )

    pick_mode = st.radio(
        "Pick a student", ["Sample a real held-out student", "Enter an id_student directly"], horizontal=True
    )

    id_student = None
    if pick_mode == "Sample a real held-out student":
        with st.spinner("Sampling real students across all four outcomes..."):
            directory = _get_student_directory()
        directory = directory.copy()
        directory["label"] = directory.apply(
            lambda r: f"{r['id_student']}  ·  {r['code_module']}  ·  {r['region']}  ·  actual: {r['final_result']}",
            axis=1,
        )
        choice = st.selectbox("Student", directory["label"].tolist())
        id_student = int(choice.split(" ")[0])
    else:
        id_student = st.number_input("id_student", min_value=1, value=11391, step=1)

    if st.button("Analyze this student", type="primary"):
        try:
            with st.spinner(f"Scanning studentVle.csv for student {id_student}'s weekly engagement..."):
                weekly = _get_student_weekly(int(id_student), gru_cfg["max_weeks"])
        except Exception as e:
            st.error(f"Couldn't read engagement history: {e}")
            return

        if weekly["total_clicks"].sum() == 0:
            st.warning(f"Student {id_student} has zero recorded VLE activity — nothing to score.")
            return

        with st.spinner("Scoring with the GRU (full sequence + week-by-week trajectory)..."):
            gru_result = rs.score_student_gru(gru_model, gru_cfg, weekly)
            trajectory, first_high_risk_week = rs.compute_gru_risk_trajectory(gru_model, gru_cfg, weekly)
        event_week = rs.get_actual_event_week(int(id_student))

        xgb_result, final_result = None, None
        try:
            record, final_result = rs.build_oulad_record_from_raw(int(id_student), weekly)
            X_new = rs.build_single_record(record, rs.OULAD_FEATURE_COLUMNS)
            xgb_model = rs.load_model("oulad")
            xgb_result = rs.predict_risk(xgb_model, X_new, rs.RISK_THRESHOLD_OULAD).iloc[0]
        except ValueError as e:
            st.info(f"XGBoost comparison unavailable: {e}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Actual outcome", final_result or "—")
        c2.metric("Last active week", int(weekly.loc[weekly["total_clicks"] > 0, "week_number"].max())
                  if (weekly["total_clicks"] > 0).any() else "—")
        c3.metric("Actual withdrawal week", event_week if event_week is not None else "—")

        col_xgb, col_gru = st.columns(2)
        with col_xgb:
            st.subheader("XGBoost (tabular, full-course aggregate)")
            if xgb_result is not None:
                _risk_level_badge(xgb_result["risk_level"], xgb_result["risk_score"])
            else:
                st.info("Not available for this student.")
        with col_gru:
            st.subheader("GRU (weekly sequence)")
            _risk_level_badge(gru_result["risk_level"], gru_result["risk_score"])

        if xgb_result is not None and gru_result["predicted_label"] != xgb_result["predicted_label"]:
            st.info(
                f"Models disagree for this student: XGBoost says **{xgb_result['predicted_label']}**, "
                f"GRU says **{gru_result['predicted_label']}**. Worth a closer look at the weekly pattern below."
            )

        st.subheader("Weekly engagement & GRU risk trajectory")
        st.caption(
            "Bottom panel: each week's point re-scores the student using real data up to that week, with weeks "
            "after it zeroed out — i.e. 'how risky would this look if engagement stopped right here'. The y-axis "
            "is a **logit scale**, not linear — most of a trajectory's real movement happens in a narrow high "
            "band (0.85-1.0) that a plain 0-to-1 axis compresses into what looks like a flat line; logit scale "
            "stretches that band out so real week-to-week change is actually visible instead of hidden. Read the "
            "curve as a trend, not a calibrated real-time probability: early weeks run structurally high because "
            "a short burst of activity followed by nothing IS what real early withdrawals look like in the "
            "training data. The green marker (if shown) is the first week at or after week 7 where the score "
            "re-crosses the threshold — a demo-grade burn-in filter, not a validated flag. Two known failure "
            "modes: (1) it still fires for most fine students, since the zero-padding artifact hasn't fully faded "
            "by week 7; and (2) it shows nothing at all for students whose real engagement ends *before* week 7 — "
            "exactly the fastest, most urgent withdrawals — since the trajectory never reaches a week-7 point to "
            "check. Treat this marker as illustrative only; it is **not** the same rigorously-evaluated "
            "population-level lead time shown on the **Risk Model Performance — OULAD** page."
        )
        fig = rs.plot_student_engagement_and_risk(
            weekly, trajectory, rs.RISK_THRESHOLD_OULAD_GRU, first_high_risk_week, event_week,
            title=f"Student {id_student}" + (f" (actual: {final_result})" if final_result else ""),
        )
        st.pyplot(fig)

        merged = weekly.merge(trajectory, on="week_number", how="left")
        export_buttons(merged, fig, f"student_{id_student}_gru_timeseries", f"gru_ts_{id_student}")


def page_clustering_explorer(conn):
    st.header("Clustering Explorer (K-Means)")
    st.caption(
        "Runs K-Means live on the feature matrix — pick k yourself and see the "
        "elbow/silhouette curves, the resulting clusters in a PCA projection, and "
        "what each cluster actually looks like. This is the live version of what "
        "clustering_performance_personas.ipynb's find_optimal_k()/assign_personas() "
        "do offline; the **Performance Personas** page shows that notebook's saved, "
        "human-named result — this page lets you explore the clustering itself."
    )

    dataset_label = st.radio("Dataset", ["UCI", "OULAD"], horizontal=True, key="cluster_dataset")
    dataset = dataset_label.lower()
    if not require_risk_artifacts(dataset):
        return

    feature_df = rs.load_feature_matrix(dataset)
    view_cols = rs.UCI_CENTROID_VIEW_COLS if dataset == "uci" else rs.OULAD_CENTROID_VIEW_COLS

    with st.spinner("Fitting K-Means for k=2..7 (elbow + silhouette)..."):
        k_results = rs.find_optimal_k(feature_df)
    st.pyplot(rs.plot_elbow_silhouette(k_results))

    best_k_by_silhouette = max(k_results["silhouette"], key=k_results["silhouette"].get)
    st.caption(
        f"Silhouette score is maximized at k={best_k_by_silhouette} "
        f"(score={k_results['silhouette'][best_k_by_silhouette]:.3f}). That's the "
        "statistically tightest split, not necessarily the most useful one for an "
        "advisor — 2 clusters usually just re-partitions high-risk vs. low-risk, "
        "which the risk score already gives you. Try a higher k below for richer, "
        "more actionable groups."
    )

    k = st.slider("Number of clusters (k)", min_value=2, max_value=7, value=4)
    with st.spinner(f"Fitting K-Means with k={k}..."):
        clustered = rs.run_kmeans(feature_df, k)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.pyplot(rs.plot_cluster_scatter(clustered, title=f"{dataset_label} — K-Means clusters (k={k})"))
    with col2:
        st.subheader("Cluster centroids")
        st.caption("Mean of key features + at_risk rate per cluster — read this to name each one.")
        st.dataframe(rs.cluster_centroid_table(clustered, view_cols), width="stretch")

    export_buttons(clustered.drop(columns=["pc1", "pc2"]), None, f"{dataset}_clusters_k{k}", f"{dataset}_cluster_export")


def page_performance_personas(conn):
    st.header("Performance Personas")
    dataset_label = st.radio("Dataset", ["UCI", "OULAD"], horizontal=True)
    dataset = dataset_label.lower()
    if not require_risk_artifacts(dataset):
        return

    personas = rs.load_personas(dataset)
    counts = personas["persona_name"].value_counts()

    c1, c2 = st.columns(2)
    c1.metric("Students", f"{len(personas):,}")
    c2.metric("Personas found", personas["persona_name"].nunique())

    st.subheader("Persona × Risk Level")
    crosstab = pd.crosstab(personas["persona_name"], personas["risk_level"])
    show_table(crosstab.reset_index())

    fig = rs.plot_persona_risk_boxplot(personas, title=f"Risk Score by Persona — {dataset_label}")
    st.pyplot(fig)

    show_table(personas.sort_values("risk_score", ascending=False).head(200))
    export_buttons(personas, fig, f"{dataset}_personas", f"{dataset}_persona")


SPRINT2_PAGES = {
    "Load / Seed Data": page_seed_data,
    "Class Average Report": page_class_average,
    "Top/Bottom Performers": page_top_bottom,
    "Weak Area Detection": page_weak_subjects,
    "Grade Distribution": page_grade_distribution,
    "Individual Student Report": page_student_report,
    "Pivot Table": page_pivot_table,
    "Percentile Ranking": page_percentile_ranking,
    "Subject Strengths": page_subject_strengths,
    "Improvement Leaderboard": page_improvement,
    "Consistency Analysis": page_consistency,
    "Outlier Detection": page_outlier_detection,
    "Class Comparison": page_class_comparison,
    "Pass Rate Summary": page_pass_rate,
    "Subject Top Performers": page_subject_top_performers,
}

def _provider_status_caption():
    """Shared status line for both Sprint 4 pages: which provider is
    configured, and whether it'll actually be used or the app will fall
    back to the deterministic template. Matches require_risk_artifacts()'s
    role for Sprint 3 -- surface the real state, don't let the page pretend
    everything's ready when it isn't."""
    cfg = llm_client.get_llm_config()
    provider = cfg["provider"]
    if llm_client.is_configured(provider):
        st.caption(f"LLM provider: **{provider}** (configured) · model: `{cfg[provider + '_model']}`")
    else:
        st.caption(
            f"LLM provider: **{provider}** — no `{provider.upper()}_API_KEY` found in `.env`. "
            "Answers will use the deterministic, fully-grounded template instead of an LLM call "
            "(same underlying facts, no network/API cost)."
        )


def page_ai_assistant(conn):
    st.header("AI Assistant")
    st.caption(
        "Ask about a student, a cohort, or how the models work — one chat, no dataset or ID "
        "dropdown to fill in first. The assistant reads your message, decides what to look up, "
        "and answers using only real data it actually retrieved — it never invents a fact, and "
        "it tells you plainly when something can't be found instead of guessing."
    )
    _provider_status_caption()
    # Pays the RAG retriever's one-time setup cost here
    rag_backend = _warm_up_rag()
    if "hash-fallback" in rag_backend:
        st.caption(
            f"⚠ Intervention search is running on the offline hash-fallback embedding "
            f"(`{rag_backend}`), not real semantic search — sentence-transformers likely "
            f"couldn't be downloaded. Retrieval still works but is less precise; see "
            f"vectorstore.get_embedding_function()."
        )

    if not llm_client.is_configured():
        st.info(
            "No LLM provider configured — this page needs one (see `.env`'s `LLM_PROVIDER` / "
            "`OPENAI_API_KEY` / `GEMINI_API_KEY`) to understand free-form questions, since "
            "deciding which tool to call requires reading your message. The other Sprint 4 "
            "pages' single-purpose flows aren't available anymore now that this page replaces "
            "them — set a key to use this page."
        )

    with st.expander("Example questions"):
        st.markdown(
            "- \"Why is student 5 at risk?\" *(UCI/OULAD row-index)*\n"
            "- \"Check the risk score for id 101\"\n"
            "- \"How accurate is the OULAD model?\"\n"
            "- \"Which students need attention first?\" / \"Top 10 at risk\"\n"
            "- \"How is student 480959 doing over time?\" *(real OULAD student ID — shows a chart)*\n"
            "- \"Explain the graph\" / \"What does this chart show?\" *(follow-up — sends the actual chart image)*\n"
            "- \"When did their engagement start declining?\" *(follow-up, same student)*\n"
            "- \"What should I do to help them?\"\n"
            "- \"What is SHAP?\" *(general question, no lookup needed)*"
        )

    session_key = "ai_agent_chat"
    chart_state_key = "ai_agent_chat_last_chart"
    chart_image_key = "ai_agent_chat_last_chart_image_b64"

    for msg in genai_memory.get_session_history(st.session_state, session_key).messages:
        with st.chat_message("user" if msg.type == "human" else "assistant"):
            st.markdown(msg.content)

    question = st.chat_input("Ask about a student, a cohort, or how the models work…")

    if question:
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            result_holder = agent._StreamResult()
            cached_chart_image_b64 = st.session_state.get(chart_image_key)
            try:
                answer = st.write_stream(
                    agent.run_agent_stream_with_artifacts(
                        question, st.session_state, result_holder, session_key,
                        chart_image_b64=cached_chart_image_b64,
                    )
                )
            except Exception:
                # Defense-in-depth, same pattern as the rest of Sprint 4: fall
                # back to the non-streaming call rather than leaving the page
                # broken if something escapes the streaming path's own handling.
                full = agent.run_agent(question, st.session_state, session_key,
                                        chart_image_b64=cached_chart_image_b64)
                answer = full["answer"]
                result_holder.source = full["source"]
                result_holder.tool_calls_made = full["tool_calls_made"]
                result_holder.artifacts = full["artifacts"]
                result_holder.chart_image_attached = full["chart_image_attached"]
                st.markdown(answer)

            if result_holder.source == "unconfigured":
                st.caption("⚠️ No LLM configured")
            elif result_holder.chart_image_attached:
                st.caption(
                    f"🤖 LLM (streamed) · 📷 analyzed the actual chart image · "
                    f"tools used: {', '.join(result_holder.tool_calls_made) or 'none'}"
                )
            elif result_holder.tool_calls_made:
                st.caption(f"🤖 LLM (streamed) · tools used: {', '.join(result_holder.tool_calls_made)}")
            else:
                st.caption("🤖 LLM (streamed) · general question, no lookup needed")

            artifacts = result_holder.artifacts
            if artifacts.gru_weekly_df is not None:
                st.session_state[chart_state_key] = artifacts
                fig = rs.plot_student_engagement_and_risk(
                    artifacts.gru_weekly_df, artifacts.gru_trajectory_df, rs.RISK_THRESHOLD_OULAD_GRU,
                    artifacts.gru_first_high_risk_week, None,
                    title=f"Student {artifacts.gru_student_id}"
                    + (f" (actual: {artifacts.gru_actual_final_result})" if artifacts.gru_actual_final_result else ""),
                )
                st.pyplot(fig)

                # Cache the ACTUAL rendered chart as a real PNG, so a later
                # "explain the graph" question can send this exact image to
                # the LLM as real multimodal input -- not a description of
                # the data, the literal picture the person is looking at.
                png_buf = io.BytesIO()
                fig.savefig(png_buf, format="png", dpi=120, bbox_inches="tight")
                st.session_state[chart_image_key] = base64.b64encode(png_buf.getvalue()).decode("ascii")

    elif chart_state_key in st.session_state:
        # No new question this rerun -- still show the most recent chart so
        # it doesn't vanish just because some other widget triggered a rerun.
        artifacts = st.session_state[chart_state_key]
        with st.expander("Chart from your last question", expanded=True):
            fig = rs.plot_student_engagement_and_risk(
                artifacts.gru_weekly_df, artifacts.gru_trajectory_df, rs.RISK_THRESHOLD_OULAD_GRU,
                artifacts.gru_first_high_risk_week, None, title=f"Student {artifacts.gru_student_id}",
            )
            st.pyplot(fig)



SPRINT3_PAGES = {
    "Risk Dashboard — UCI": page_risk_dashboard_uci,
    "Risk Dashboard — OULAD": page_risk_dashboard_oulad,
    "Predict Risk — New Student (UCI)": page_predict_new_student_uci,
    "Predict Risk — New Student (OULAD)": page_predict_new_student_oulad,
    "Student Time-Series Risk Explorer (GRU)": page_student_gru_explorer,
    "Risk Model Performance — UCI": page_risk_model_performance_uci,
    "Risk Model Performance — OULAD": page_risk_model_performance_oulad,
    "Clustering Explorer (K-Means)": page_clustering_explorer,
    "Performance Personas": page_performance_personas,
}

# Kept for backward compatibility with anything importing PAGES directly.
PAGES = {**SPRINT2_PAGES, **SPRINT3_PAGES}

SPRINT4_PAGES = {
    "AI Assistant": page_ai_assistant,
}


# ============================================================
# Sprint 5 — Intervention Agent (LangGraph + CrewAI + FastAPI/PostgreSQL)
# ============================================================

def _decision_badge(decision: str) -> str:
    """Plain-text form -- kept as-is because expander titles can't render
    HTML, so this stays the fallback for those spots."""
    return {"nudge": "🟢 Nudge", "resource": "🟡 Resource", "escalate": "🔴 Escalate"}.get(decision, decision)


def _decision_pill_html(decision: str) -> str:
    """Styled pill form (.decision-pill CSS above) for spots that CAN
    render HTML, e.g. _render_decision's header -- same three colors as
    _decision_badge's emoji so both forms read as the same signal."""
    label = {"nudge": "Nudge", "resource": "Resource", "escalate": "Escalate"}.get(decision, decision)
    css_class = decision if decision in ("nudge", "resource", "escalate") else ""
    return f'<span class="decision-pill {css_class}">{label}</span>'


def _storage_backend_caption():
    """Surfaces which storage backend is actually active (real
    PostgreSQL vs. the SQLite/JSONL fallback) -- same "never pretend a
    fallback is the primary path" convention as _provider_status_caption()
    already uses for the LLM provider above."""
    try:
        backend_label = intervention_audit.get_storage_backend_label()
        st.caption(f"Audit storage: {backend_label}")
    except Exception:
        st.caption("Audit storage: logs/interventions.jsonl (database unavailable)")


def _show_and_cache_chart(label: str, fig):
    """Renders a matplotlib figure AND stashes it as a base64 PNG in
    st.session_state["ia_chart_gallery"], keyed by a human-readable
    label. This is the Sprint 5 equivalent of page_ai_assistant's single
    cached GRU chart (chart_image_key above) -- except several Sprint 5
    charts can be on screen across different pages in one session (a
    student's SHAP factors, a confidence gauge, the Teacher Dashboard's
    four diagrams), so this keeps a small gallery instead of one slot.
    The Intervention Agent Chat page reads this gallery and sends every
    chart in it to the multimodal model together when a question looks
    chart-related (ai_assistant.agent._build_message_history's
    chart_gallery parameter) -- the actual images the teacher is looking
    at, not a text description of what's in them."""
    st.pyplot(fig)
    png_buf = io.BytesIO()
    fig.savefig(png_buf, format="png", dpi=120, bbox_inches="tight")
    gallery = st.session_state.setdefault("ia_chart_gallery", {})
    gallery[label] = base64.b64encode(png_buf.getvalue()).decode("ascii")


def _render_agent_trace(decision: dict):
    """A visual step-by-step trace of what the graph actually did for
    this decision, in order -- 'classify' always runs; the node name
    after it is the routing outcome; a drafting step is shown only for
    nudge/resource (escalate never calls one, by design); a logging step
    is always last. This replaces a single 'Agent trace: a → b' caption
    with something that reads as a pipeline, not a chat reply."""
    node_labels = {
        "classify": ("🔎", "Classify"),
        "nudge_node": ("🟢", "Nudge"),
        "resource_node": ("🟡", "Resource"),
        "escalate_node": ("🔴", "Escalate"),
    }
    steps = [node_labels.get(n, ("•", n)) for n in decision["agent_trace"]]
    if decision["decision"] != "escalate":
        draft_icon = "🤖" if decision["source"] == "crew" else "📄"
        steps.append((draft_icon, f"Draft ({decision['source']})"))
    steps.append(("💾", "Logged"))

    cols = st.columns(len(steps) * 2 - 1)
    for i, (icon, label) in enumerate(steps):
        with cols[i * 2]:
            st.markdown(f"<div style='text-align:center'>{icon}<br><small>{label}</small></div>", unsafe_allow_html=True)
        if i * 2 + 1 < len(cols):
            with cols[i * 2 + 1]:
                st.markdown("<div style='text-align:center; padding-top:0.4rem'>→</div>", unsafe_allow_html=True)


def _render_resource_badge(resource_str: str):
    """Parses the '<text> (source: rem_XXX, <match_type>)' format
    resource_node encodes into resources_used, and renders the match
    type as a colored badge -- so an exact factor-tag match and an
    embedding-similarity re-rank don't look identical, per Section 5.2's
    distinction between the two retrieval paths."""
    if "(source: " in resource_str and resource_str.endswith(")"):
        text, meta = resource_str.rsplit("(source: ", 1)
        meta = meta[:-1] if meta.endswith(")") else meta  # strip exactly one trailing paren
        parts = [p.strip() for p in meta.split(",")]
        source_id = parts[0] if parts else "?"
        match_type = parts[1] if len(parts) > 1 else ""
    else:
        text, source_id, match_type = resource_str, "?", ""

    badge_color = "green" if "exact" in match_type else ("orange" if "re-ranked" in match_type else "gray")
    st.markdown(f"- {text.strip()}  \n  :{badge_color}[{match_type or 'n/a'}] · `{source_id}`")


def _render_decision(decision: dict):
    st.markdown(f"### {_decision_pill_html(decision['decision'])}", unsafe_allow_html=True)
    if decision.get("display_student_id"):
        st.caption(f"Student: **{decision['display_student_id']}**")
    _render_agent_trace(decision)

    gauge_col, rationale_col = st.columns([1, 1.4])
    with gauge_col:
        student_label = decision.get("display_student_id") or decision.get("student_id", "?")
        gauge_fig = intervention_charts.plot_confidence_gauge(decision["router_confidence"], intervention_graph.CONFIDENCE_FLOOR)
        _show_and_cache_chart(f"Router confidence — student {student_label} ({decision['decision']})", gauge_fig)
    with rationale_col:
        st.markdown("**Why this decision:**")
        st.write(decision["rationale"])

    if decision["decision"] == "escalate":
        st.error(
            f"🚨 **Human intervention needed.** No automated nudge or resource message was "
            f"sent to this student. Reason: {decision['escalation_reason']}."
        )
        if decision.get("mentor_brief"):
            st.markdown("**Briefing for the mentor:**")
            with st.container(border=True):
                st.markdown(decision["mentor_brief"])
        else:
            st.caption(
                "This is by design, not a missing feature — a disagreeing model pair or a "
                "borderline router-confidence score routes here instead of guessing which "
                "automated message would fit. Recommended next step: a human mentor reviews "
                "this student directly, using the risk data above."
            )
    else:
        st.markdown("**Drafted message:**")
        st.info(decision["drafted_message"])
        if decision["resources_used"]:
            st.markdown("**Resources used:**")
            for res in decision["resources_used"]:
                _render_resource_badge(res)


def _resolve_student_lookup(dataset: str, id_mode: str, raw_value: str) -> str:
    """Turns whatever the teacher typed into the 1-based row-index string
    every ai_assistant.tools function actually expects -- without
    modifying tools.py itself (Sprint 4, read-only). Two modes, because
    forcing "row-index only" was the actual source of the ID confusion
    reported: a teacher with a school roster naturally has the real
    student ID (e.g. OULAD's 628653, or UCI's "UCI-0091") on hand far
    more often than an arbitrary row position, and the two number
    formats look deceptively similar, so it's easy to type a real ID
    into a field that silently means something else.

    "Row-index": unchanged, passed straight through.
    "Real Student ID": reverse-looked-up against risk_queries directly
    -- for OULAD, against the risk table's own real id_student column
    (the exact same column _display_student_id() already reads to show
    "Confirmed: 628653" after a row-index lookup -- so typing 628653
    back in here now resolves to the same row); for UCI, against
    display_student_id (e.g. "UCI-0091", case/format-insensitive).
    """
    if id_mode == "Row-index":
        return raw_value

    risk_df = risk_queries.load_risk_scores(dataset)
    if dataset == "oulad":
        try:
            target = int(raw_value)
        except (TypeError, ValueError):
            raise StudentNotFoundError(f"'{raw_value}' isn't a valid OULAD id_student -- expected a number like 628653.")
        matches = risk_df.index[risk_df["id_student"] == target]
    else:
        normalized = raw_value.strip().upper()
        if not normalized.startswith("UCI-"):
            normalized = f"UCI-{normalized.zfill(4)}"
        matches = risk_df.index[risk_df["display_student_id"].str.upper() == normalized]

    if len(matches) == 0:
        raise StudentNotFoundError(f"No {dataset.upper()} student found with real ID '{raw_value}'.")
    return str(matches[0] + 1)  # back to 1-based, same convention _resolve_row_index() uses


def page_intervention_agent_student(conn):
    with _render_chat_toggle_and_columns():
        st.header("Intervention Agent — Single Student")
        st.caption(
            "Two steps: look up a student's real risk data first, then run the Sprint 5 LangGraph "
            "agent and see exactly why it routed the way it did — the confidence it routed on, the "
            "factors behind the drafted message, and which resources were an exact match vs. a "
            "re-ranked similarity guess."
        )
        _provider_status_caption()
        _storage_backend_caption()

        dataset = st.selectbox("Dataset", ["uci", "oulad"], key="ia_dataset")
        id_mode = st.radio(
            "Look up by", ["Row-index", "Real Student ID"], horizontal=True, key="ia_id_mode",
            help="Row-index: the position shown on the Risk Dashboard page (1, 2, 3...). "
                 "Real Student ID: the actual ID from your roster "
                 f"({'a number like 628653' if dataset == 'oulad' else 'e.g. UCI-0091'}).",
        )
        id_placeholder = "628653" if (dataset == "oulad" and id_mode == "Real Student ID") else \
                          "UCI-0091" if id_mode == "Real Student ID" else "0"
        raw_id = st.text_input(f"Student ID ({id_mode})", value="" if id_mode == "Real Student ID" else "0",
                                placeholder=id_placeholder, key="ia_student_id_raw")

        st.subheader("1. Look up student")
        if st.button("Look Up Student"):
            try:
                student_id = _resolve_student_lookup(dataset, id_mode, raw_id)
                st.session_state["ia_risk_explanation"] = ai_tools.get_risk_explanation(dataset, student_id)
                st.session_state["ia_lookup_key"] = f"{dataset}:{student_id}"
                st.session_state["ia_resolved_student_id"] = student_id
            except (StudentNotFoundError, DataNotFoundError) as e:
                st.session_state.pop("ia_risk_explanation", None)
                st.error(str(e))

        risk_explanation = st.session_state.get("ia_risk_explanation")
        student_id = st.session_state.get("ia_resolved_student_id")
        lookup_key = f"{dataset}:{student_id}" if student_id else None
        if risk_explanation and st.session_state.get("ia_lookup_key") == lookup_key:
            threshold = (risk_queries.RISK_THRESHOLD_UCI if dataset == "uci" else risk_queries.RISK_THRESHOLD_OULAD)
            if risk_explanation.get("display_student_id"):
                st.caption(f"✅ Confirmed: **{risk_explanation['display_student_id']}** — the same real-world-checkable ID the AI Assistant chat uses.")
            m1, m2, m3 = st.columns(3)
            m1.metric("Risk score", f"{risk_explanation['risk_score']:.3f}", delta=f"{risk_explanation['risk_score'] - threshold:+.3f} vs. threshold")
            m2.metric("Risk level", risk_explanation["risk_level"])
            m3.metric("Persona", risk_explanation.get("persona_name", "—"))
            student_label = risk_explanation.get("display_student_id") or risk_explanation.get("student_id", student_id)
            factors_fig = intervention_charts.plot_student_factors(risk_explanation)
            _show_and_cache_chart(f"SHAP factors — student {student_label}", factors_fig)

            st.subheader("2. Run Intervention Agent")
            check_disagreement = False
            oulad_id_student = None
            if dataset == "oulad":
                # For OULAD, display_student_id IS the real id_student
                # (see _display_student_id() in ai_assistant/tools.py --
                # it reads the row's own id_student column directly), so
                # the disagreement check can just reuse it instead of
                # asking the teacher to re-type it into a second field.
                # Previously this defaulted to a hardcoded placeholder
                # (480959) with NO relationship to whatever student was
                # actually just looked up -- silently running the
                # disagreement check against an unrelated student unless
                # the teacher noticed and overwrote it by hand.
                resolved_real_id = risk_explanation.get("display_student_id")
                same_student = st.checkbox(
                    f"Also run the GRU/XGBoost disagreement check for this same student"
                    f"{f' ({resolved_real_id})' if resolved_real_id else ''}",
                    value=bool(resolved_real_id), disabled=not resolved_real_id,
                )
                check_different = st.checkbox("...or check a different OULAD id_student instead")
                if same_student and not check_different and resolved_real_id:
                    check_disagreement = True
                    oulad_id_student = int(resolved_real_id)
                elif check_different:
                    check_disagreement = True
                    oulad_id_student = st.number_input("Real OULAD id_student to check", min_value=1, step=1)

            if st.button("Run Intervention Agent", type="primary"):
                models_agree = None
                if check_disagreement and oulad_id_student:
                    try:
                        gru_context, _, _ = ai_tools.get_gru_timeseries_context(int(oulad_id_student))
                        models_agree = gru_context["models_agree"]
                        if not models_agree:
                            st.warning(
                                f"GRU ({gru_context['gru_risk_level']}) and XGBoost "
                                f"({gru_context['xgboost_risk_level']}) disagree on this student."
                            )
                    except StudentNotFoundError as e:
                        st.warning(f"Disagreement check skipped: {e}")

                decision = intervention_graph.run_intervention(
                    dataset, student_id, risk_explanation,
                    oulad_id_student=int(oulad_id_student) if (check_disagreement and oulad_id_student) else None,
                    models_agree=models_agree,
                )
                logged = intervention_audit.log_decision(decision, run_id=f"single:{dataset}:{student_id}")
                st.divider()
                _render_decision(decision)
                st.caption(f"Logged to {logged['storage_backend']}.")
        else:
            st.info("Look up a student above to see their real risk data before running the agent.")


def page_intervention_agent_cohort(conn):
    with _render_chat_toggle_and_columns():
        st.header("Intervention Agent — Cohort Batch Run")
        st.caption(
            "Runs the Intervention Agent across a whole cohort slice at once (e.g. every High-risk "
            "OULAD student) and logs one audit entry per student — the coverage/consistency record "
            "the Sprint 5 plan calls out as currently missing."
        )
        _provider_status_caption()
        _storage_backend_caption()

        dataset = st.selectbox("Dataset", ["uci", "oulad"], key="ia_cohort_dataset")
        risk_level = st.selectbox("Risk level filter", ["High", "Medium", "Low", "(all)"], key="ia_cohort_risk_level")
        batch_size = st.slider("Number of students", min_value=1, max_value=25, value=5, key="ia_cohort_limit")

        page_signature = f"{dataset}:{risk_level}:{batch_size}"
        if st.session_state.get("ia_cohort_page_sig") != page_signature:
            st.session_state["ia_cohort_page_sig"] = page_signature
            st.session_state["ia_cohort_page"] = 0
        page = st.session_state["ia_cohort_page"]

        # Cheap call just to learn how many students exist in this
        # filtered slice -- get_cohort_summary's risk_level_counts /
        # total_students describe the WHOLE filtered population
        # regardless of `limit`, so limit=1 is enough to size the pool
        # without fetching a full page of IDs we don't need yet.
        pool_probe = ai_tools.get_cohort_summary(dataset, risk_level=None if risk_level == "(all)" else risk_level, limit=1)
        pool_size = (pool_probe["risk_level_counts"].get(risk_level, 0) if risk_level != "(all)"
                     else pool_probe["total_students"])
        next_disabled = (page + 1) * batch_size >= pool_size

        run_clicked = st.button("Run batch", type="primary")

        nav_prev, nav_label, nav_next = st.columns([1, 2, 1])
        with nav_prev:
            prev_clicked = st.button("◀ Previous 5", disabled=(page == 0))
        with nav_label:
            shown_to = min((page + 1) * batch_size, pool_size)
            st.caption(f"Page {page + 1} · students {page * batch_size + 1}–{shown_to} of {pool_size}")
        with nav_next:
            next_clicked = st.button("Next 5 ▶", disabled=next_disabled)

        if prev_clicked:
            st.session_state["ia_cohort_page"] = max(0, page - 1)
            st.rerun()
        if next_clicked:
            st.session_state["ia_cohort_page"] += 1
            st.rerun()
        page = st.session_state["ia_cohort_page"]

        if run_clicked or "ia_cohort_rows" in st.session_state:
            fetch_limit = (page + 1) * batch_size
            cohort = ai_tools.get_cohort_summary(
                dataset, risk_level=None if risk_level == "(all)" else risk_level, limit=fetch_limit
            )
            page_student_ids = cohort["top_student_ids"][page * batch_size: fetch_limit]

            if run_clicked:
                if not page_student_ids:
                    st.info("No students matched this filter.")
                    return

                rows, decisions_by_student = [], {}
                progress = st.progress(0.0)
                for i, sid in enumerate(page_student_ids):
                    try:
                        risk_explanation = ai_tools.get_risk_explanation(dataset, sid)
                        decision = intervention_graph.run_intervention(dataset, sid, risk_explanation)
                        intervention_audit.log_decision(decision, run_id=f"cohort:{dataset}:{risk_level}:page{page}")
                        decisions_by_student[sid] = decision
                        rows.append({
                            "student_id": sid,
                            "risk_score": risk_explanation["risk_score"],
                            "risk_level": risk_explanation["risk_level"],
                            "decision": decision["decision"],
                            "router_confidence": round(decision["router_confidence"], 3),
                            "source": decision["source"],
                        })
                    except (StudentNotFoundError, DataNotFoundError) as e:
                        rows.append({"student_id": sid, "risk_score": None, "risk_level": None,
                                      "decision": "error", "router_confidence": None, "source": str(e)})
                    progress.progress((i + 1) / len(page_student_ids))

                st.session_state["ia_cohort_rows"] = rows
                st.session_state["ia_cohort_decisions"] = decisions_by_student
                st.session_state["ia_cohort_run_id"] = f"cohort:{dataset}:{risk_level}:page{page}"

        if "ia_cohort_rows" in st.session_state:
            df = pd.DataFrame(st.session_state["ia_cohort_rows"])
            st.dataframe(df, use_container_width=True)
            st.caption(f"{len(df)} decisions logged (run_id={st.session_state['ia_cohort_run_id']}) — see the Teacher Dashboard page.")
            counts = df["decision"].value_counts()
            st.bar_chart(counts)

            st.subheader("Per-student detail")
            decisions_by_student = st.session_state["ia_cohort_decisions"]
            for sid, decision in decisions_by_student.items():
                label = decision.get("display_student_id") or f"Student {sid}"
                with st.expander(f"{label} — {_decision_badge(decision['decision'])}"):
                    _render_decision(decision)


def page_teacher_dashboard(conn):
    with _render_chat_toggle_and_columns():
        st.header("Teacher Dashboard — Intervention Agent")
        st.caption(
            "A read-only view of everything the Intervention Agent has done, across every run mode "
            "(single-student, cohort batch, and the FastAPI /agent endpoints) — for a teacher or "
            "admin to see coverage and consistency at a glance, not just the outcome of the one "
            "student they last looked up."
        )
        _storage_backend_caption()

        col1, col2, col3 = st.columns(3)
        with col1:
            dataset_filter = st.selectbox("Dataset", ["(all)", "uci", "oulad"], key="dash_dataset")
        with col2:
            decision_filter = st.selectbox("Decision", ["(all)", "nudge", "resource", "escalate"], key="dash_decision")
        with col3:
            max_entries = st.slider("Max raw entries to load (before dedup)", 10, 1000, 200, key="dash_max")

        dedup = st.checkbox(
            "Show one row per student (most recent decision only)", value=True, key="dash_dedup",
            help="Off shows every logged run, including repeats from re-running the same student or "
                 "cohort batch multiple times. On (default) keeps only each student's latest decision, "
                 "per dataset, so totals and charts reflect current status rather than run history.",
        )

        entries = intervention_audit.read_audit_log(limit=max_entries)
        if not entries:
            st.info(
                "No decisions logged yet. Run the agent from the Single Student or Cohort Batch Run "
                "pages (or POST to the FastAPI /agent/run endpoint) to populate this dashboard."
            )
            return

        df = pd.DataFrame(entries)  # already most-recent-first from read_audit_log()
        if dataset_filter != "(all)":
            df = df[df["dataset"] == dataset_filter]
        if decision_filter != "(all)":
            df = df[df["decision"] == decision_filter]

        if df.empty:
            st.info("No decisions match the current filters.")
            return

        raw_count = len(df)
        if dedup:
            df = df.drop_duplicates(subset=["dataset", "student_id"], keep="first")  # keep=first == most recent, since df is already ordered that way
            if raw_count > len(df):
                st.caption(f"Deduplicated {raw_count} logged runs down to {len(df)} unique students (most recent decision kept for each).")

        total = len(df)
        escalated = int((df["decision"] == "escalate").sum())
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        metric_col1.metric("Unique students" if dedup else "Total decisions", total)
        metric_col2.metric("Nudges", int((df["decision"] == "nudge").sum()))
        metric_col3.metric("Resources sent", int((df["decision"] == "resource").sum()))
        metric_col4.metric("Escalated to human", escalated, delta=f"{escalated / total * 100:.0f}% of total" if total else None)

        st.subheader("Diagrams")
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            _show_and_cache_chart("Teacher Dashboard — Decisions by Branch", intervention_charts.plot_decision_distribution(df))
        with chart_col2:
            _show_and_cache_chart("Teacher Dashboard — Decisions by Risk Level", intervention_charts.plot_risk_level_breakdown(df))

        _show_and_cache_chart("Teacher Dashboard — Decisions Over Time", intervention_charts.plot_decisions_over_time(df))

        if df["dataset"].nunique() > 1:
            _show_and_cache_chart("Teacher Dashboard — Escalation Rate by Dataset", intervention_charts.plot_escalation_rate_by_dataset(df))

        st.subheader("Recent decisions")
        display_cols = [c for c in ["logged_at", "run_id", "dataset", "student_id", "display_student_id", "risk_level",
                                     "decision", "router_confidence", "source", "escalation_reason"] if c in df.columns]
        st.dataframe(df[display_cols], use_container_width=True)

        # CSV export -- "shown" = exactly the table above; "full" = every
        # column (rationale, drafted_message, resources, trace...) for the
        # same filtered rows. utf-8-sig so Excel opens it cleanly.
        def _csv_bytes(frame: pd.DataFrame) -> bytes:
            out = frame.copy()
            for c in out.columns:  # lists (resources_used, agent_trace) -> JSON text
                if out[c].map(lambda v: isinstance(v, (list, dict))).any():
                    out[c] = out[c].map(lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v)
            return out.to_csv(index=False).encode("utf-8-sig")

        dl_col1, dl_col2 = st.columns(2)
        dl_col1.download_button(
            "⬇️ Download table as CSV", data=_csv_bytes(df[display_cols]),
            file_name="intervention_decisions.csv", mime="text/csv", key="dash_dl_shown",
        )
        dl_col2.download_button(
            "⬇️ Download full detail as CSV", data=_csv_bytes(df),
            file_name="intervention_decisions_full.csv", mime="text/csv", key="dash_dl_full",
        )

        with st.expander("Full decision detail"):
            entries_filtered = df.to_dict("records")
            selected = st.selectbox(
                "Pick an entry", options=range(len(entries_filtered)),
                format_func=lambda i: f"{entries_filtered[i].get('logged_at', '?')} — student "
                                       f"{entries_filtered[i].get('student_id', '?')} — {entries_filtered[i].get('decision', '?')}",
            )
            _render_decision(entries_filtered[selected])


def _current_page_student_context() -> str | None:
    """One-line description of whatever student is currently loaded on
    screen -- Single Student's last lookup, or if that's empty, the last
    cohort batch run's headline. Passed as page_context to the agent
    (ai_assistant/agent.py) so "why is this student at risk" resolves
    without the agent asking which student you mean, the same way
    ia_chart_gallery already lets "explain this graph" resolve to
    whatever chart is on screen. Returns None when nothing's loaded yet
    -- the agent then behaves exactly as it did before this existed."""
    r = st.session_state.get("ia_risk_explanation")
    sid = st.session_state.get("ia_resolved_student_id")
    if r and sid:
        label = r.get("display_student_id") or sid
        return (f"Currently displayed on this page: {st.session_state.get('ia_dataset', '?')} student "
                f"{label} (row-index {sid}), risk_score={r.get('risk_score')}, "
                f"risk_level={r.get('risk_level')}, persona={r.get('persona_name', '—')}.")

    rows = st.session_state.get("ia_cohort_rows")
    run_id = st.session_state.get("ia_cohort_run_id")
    if rows:
        return (f"Currently displayed on this page: a cohort batch run ({run_id}) covering "
                f"{len(rows)} students -- student_ids {[r.get('student_id') for r in rows]}.")

    return None


def _render_intervention_chat_panel(show_examples: bool = True):
    """The actual chat UI -- message history + input + response rendering.
    Factored out of page_intervention_agent_chat() so the SAME chat, same
    session (session_key="ia_chat"), same chart gallery, can run either as
    its own full page OR embedded in an expander at the bottom of the
    Single Student / Cohort Batch Run / Teacher Dashboard pages -- so a
    teacher can ask about what they're looking at without leaving the
    page (Sprint 5's "chat in parallel with the rest of the UI" mode,
    not only its own separate destination)."""
    session_key = "ia_chat"
    teacher_id = st.session_state.get("teacher_id", "").strip()
    if teacher_id:
        genai_memory.load_history_into_session(st.session_state, teacher_id, session_key)

    if show_examples:
        with st.expander("Example messages"):
            st.markdown(
                "- \"Run an intervention for student 12\" *(uci/oulad row-index)*\n"
                "- \"Why is student 5 at risk?\"\n"
                "- \"What should I do for student 8?\"\n"
                "- \"Explain this graph\" / \"What does the escalation rate chart mean?\" "
                "*(after viewing a chart on Single Student, Cohort Batch Run, or Teacher Dashboard)*\n"
                "- \"Escalate student 20 to a mentor\"\n"
            )

    gallery = st.session_state.get("ia_chart_gallery", {})
    page_context = _current_page_student_context()
    if page_context:
        st.caption(f"🧑‍🎓 {page_context}")
    if gallery:
        st.caption(f"📊 Charts available to reference: {', '.join(gallery.keys())}")
    else:
        st.caption("📊 No charts viewed yet this session — visit another Sprint 5 page first if you want to ask about one.")

    for msg in genai_memory.get_session_history(st.session_state, session_key).messages:
        with st.chat_message("user" if msg.type == "human" else "assistant"):
            st.markdown(msg.content)

    question = st.chat_input("Ask about a student, a chart, or tell it to run an intervention…", key=f"ia_chat_input_{session_key}")
    if question:
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            result_holder = agent._StreamResult()
            current_gallery = st.session_state.get("ia_chart_gallery")
            current_page_context = _current_page_student_context()
            try:
                answer = st.write_stream(
                    agent.run_agent_stream_with_artifacts(
                        question, st.session_state, result_holder, session_key,
                        chart_gallery=current_gallery, page_context=current_page_context,
                    )
                )
            except Exception:
                # Same defense-in-depth fallback as page_ai_assistant.
                full = agent.run_agent(question, st.session_state, session_key,
                                        chart_gallery=current_gallery, page_context=current_page_context)
                answer = full["answer"]
                result_holder.source = full["source"]
                result_holder.tool_calls_made = full["tool_calls_made"]
                result_holder.chart_image_attached = full["chart_image_attached"]
                st.markdown(answer)

            if teacher_id:
                genai_memory.save_turn_to_db(teacher_id, session_key, question, answer)

            if result_holder.source == "unconfigured":
                st.caption("⚠️ No LLM configured")
            elif result_holder.chart_image_attached:
                st.caption(
                    f"🤖 LLM (streamed) · 📷 analyzed the visible chart(s) · "
                    f"tools used: {', '.join(result_holder.tool_calls_made) or 'none'}"
                )
            elif result_holder.tool_calls_made:
                st.caption(f"🤖 LLM (streamed) · tools used: {', '.join(result_holder.tool_calls_made)}")
            else:
                st.caption("🤖 LLM (streamed) · general question, no lookup needed")


def _render_side_chat_panel():
    """Chat panel content for the right-hand column when open -- see
    _render_chat_toggle_and_columns() below, which decides whether it's
    shown at all and owns the 1/3-width column it renders into."""
    st.markdown("#### 💬 Ask the Intervention Agent")
    st.caption("About this page, a chart you've viewed, or tell it to run an intervention.")
    _render_intervention_chat_panel(show_examples=False)


def _render_chat_toggle_and_columns():
    """Shared layout for all three Sprint 5 pages (Single Student, Cohort
    Batch Run, Teacher Dashboard) -- one robot-icon button that toggles
    the chat panel on and off, and hands back a container to render the
    page's own content into. Use as:

        def page_x(conn):
            with _render_chat_toggle_and_columns():
                st.header(...)
                ...rest of the page, unchanged...

    Closed by default: the page gets the FULL width, not 1/3 given away
    to an empty panel. Clicking 🤖 allocates the right 1/3 of the screen
    to "Ask the Intervention Agent" (st.columns([2, 1])) and swaps the
    icon to ✕ to close it again -- same toggle, same button, same spot,
    on all three pages.

    Both branches return a container keyed "ia_main_col", so the
    scrollable-column CSS below applies identically whether chat is open
    or closed -- the main content always gets its own independent
    scrollbar, separate from the full browser-page scrollbar and (when
    open) separate from the chat panel's own scrollbar. Three distinct
    scroll regions, as asked for.
    """
    st.session_state.setdefault("ia_chat_open", False)

    header_col, toggle_col = st.columns([5, 1])  # header_col intentionally empty -- the page's own
    # st.header() below renders full-width beneath this row regardless; this row exists only to
    # place the toggle button top-right, matching where the panel itself opens.
    with toggle_col:
        if st.session_state["ia_chat_open"]:
            clicked = st.button("✕ Close chat", key="ia_chat_toggle", use_container_width=True)
        else:
            clicked = st.button("🤖 Ask Agent", key="ia_chat_toggle", use_container_width=True, type="primary")
        if clicked:
            st.session_state["ia_chat_open"] = not st.session_state["ia_chat_open"]
            st.rerun()

    if st.session_state["ia_chat_open"]:
        main_col, chat_col = st.columns([2, 1])
        with chat_col.container(key="ia_chat_col"):
            _render_side_chat_panel()
    else:
        main_col = st.container()

    return main_col.container(key="ia_main_col")


def page_intervention_agent_chat(conn):
    st.header("Intervention Agent — Chat")
    st.caption(
        "Type what you want in plain language — the assistant decides which tool to call, "
        "including actually running and logging a real intervention when you explicitly ask "
        "for one. It can also explain any chart you've viewed on the other Sprint 5 pages in "
        "this session, in plain language — just ask about \"the graph\" or \"this chart\". This "
        "same chat is also available at the bottom of the Single Student, Cohort Batch Run, and "
        "Teacher Dashboard pages, so you don't have to come here to use it."
    )
    _provider_status_caption()
    _storage_backend_caption()
    _render_intervention_chat_panel(show_examples=True)


SPRINT5_PAGES = {
    "Intervention Agent — Chat": page_intervention_agent_chat,
    "Intervention Agent — Single Student": page_intervention_agent_student,
    "Intervention Agent — Cohort Batch Run": page_intervention_agent_cohort,
    "Teacher Dashboard": page_teacher_dashboard,
}


def main():
    _inject_custom_css()

    # Same one-time setup page_ai_assistant() (Sprint 4) already pays for
    # itself -- moved here so it runs at the app's actual initial load,
    # regardless of which module a teacher opens first. Previously this
    # was only ever called from page_ai_assistant(), so a teacher going
    # straight to any Sprint 5 page skipped it entirely: the sentence-
    # transformer model load + Chroma collection build then happened for
    # the first time deep inside resource_node's first real call, with
    # no spinner and no visible cause for the delay. @st.cache_resource
    # means this still only actually runs once per app process no matter
    # how many pages call it -- calling it here doesn't duplicate work
    # for page_ai_assistant, it just guarantees it always happens early.
    rag_backend = _warm_up_rag()
    if "hash-fallback" in rag_backend:
        st.sidebar.caption(
            f"⚠ Running on the offline hash-fallback embedding, not real semantic search "
            f"— sentence-transformers likely couldn't be downloaded."
        )

    st.sidebar.markdown("### 📊 Student Performance Analytics")
    st.sidebar.caption("Sprint 2 → 5 · analytics, risk, explanation, intervention")
    st.session_state["teacher_id"] = st.sidebar.text_input(
        "Teacher ID", value=st.session_state.get("teacher_id", ""),
        placeholder="e.g. your name or staff ID",
        help="Not a login -- no password. Typing the same ID again on a later visit "
             "reloads that name's previous Sprint 5 chat history from the database. "
             "Leave blank and chat still works, it just won't be remembered next time.",
    )
    st.sidebar.divider()
    module = st.sidebar.selectbox(
        "Module",
        [
            "Sprint 2 — Student Performance Analytics",
            "Sprint 3 — Risk Prediction",
            "Sprint 4 — Risk Explanation (AI Assistant)",
            "Sprint 5 — Intervention Agent",
        ],
    )
    if module.startswith("Sprint 2"):
        pages = SPRINT2_PAGES
    elif module.startswith("Sprint 3"):
        pages = SPRINT3_PAGES
    elif module.startswith("Sprint 4"):
        pages = SPRINT4_PAGES
    else:
        pages = SPRINT5_PAGES
    choice = st.sidebar.radio("Report", list(pages.keys()))

    try:
        conn = get_conn()
    except Exception as e:
        if pages is not SPRINT2_PAGES:
            # Sprint 3 and Sprint 4 pages read from disk (models/,
            # data/processed/) or call an LLM API -- neither needs the
            # MySQL connection Sprint 2 requires, so don't block them
            # behind one that isn't there.
            conn = None
        else:
            st.error(
                "Could not connect to MySQL. Check your `.env` (DB_HOST/DB_PORT/DB_USER/"
                f"DB_PASSWORD/DB_NAME) and that MySQL is running.\n\nDetails: {e}"
            )
            return

    try:
        pages[choice](conn)
    except (DataNotFoundError, StudentNotFoundError) as e:
        st.error(str(e))
    except Exception as e:  # last-resort guard so the app never hard-crashes
        st.error(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()