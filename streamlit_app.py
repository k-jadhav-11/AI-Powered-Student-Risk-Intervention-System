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
import os

import numpy as np
import pandas as pd
import streamlit as st

from src.analytics import db_manager, queries, analytics, charts
from src.analytics.exceptions import InvalidCSVError, DataNotFoundError, StudentNotFoundError
from src import risk as rs
from ai_assistant import memory as genai_memory, llm_client, agent, rag as ai_rag

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")

st.set_page_config(page_title="Student Performance Analytics", layout="wide")

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

    summary = rs.load_model_summary("uci")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision (at-risk)", f"{summary['test_precision_at_risk']:.3f}")
    c2.metric("Recall (at-risk)", f"{summary['test_recall_at_risk']:.3f}")
    c3.metric("F1 (at-risk)", f"{summary['test_f1_at_risk']:.3f}")
    c4.metric("ROC-AUC", f"{summary['test_roc_auc']:.3f}")
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

    summary = rs.load_model_summary("oulad")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision (at-risk)", f"{summary['test_precision_at_risk']:.3f}")
    c2.metric("Recall (at-risk)", f"{summary['test_recall_at_risk']:.3f}")
    c3.metric("F1 (at-risk)", f"{summary['test_f1_at_risk']:.3f}")
    c4.metric("ROC-AUC", f"{summary['test_roc_auc']:.3f}")
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


def main():
    st.sidebar.title("Student Performance Analytics")
    module = st.sidebar.selectbox(
        "Module",
        [
            "Sprint 2 — Student Performance Analytics",
            "Sprint 3 — Risk Prediction",
            "Sprint 4 — Risk Explanation (AI Assistant)",
        ],
    )
    if module.startswith("Sprint 2"):
        pages = SPRINT2_PAGES
    elif module.startswith("Sprint 3"):
        pages = SPRINT3_PAGES
    else:
        pages = SPRINT4_PAGES
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