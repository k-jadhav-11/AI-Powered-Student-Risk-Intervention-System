/* ============================================================
   sprint2Config.js — declarative list of every Sprint 2 report,
   consumed by reportEngine.js. One entry per endpoint in
   backend/routers/sprint2.py, with filters/labels/defaults/ranges
   matching streamlit_app.py's own pages exactly (class_filter(),
   subject_filter(), and each page_* function's st.number_input /
   st.radio ranges), plus a `chart` spec so the generic result
   renderer can draw the same chart Streamlit shows via st.pyplot(),
   and a `csv` flag so every report gets the same "Download CSV"
   button as Streamlit's export_buttons().
   ============================================================ */

const SPRINT2_REPORTS = [
  {
    id: "load-data", title: "Load / Seed Data",
    desc: "Loads data/students.csv, data/subjects.csv, and data/marks.csv into MySQL. Existing rows are skipped automatically, so it's safe to re-run. POST /sprint2/load-data.",
    method: "POST", path: () => "/sprint2/load-data", custom: "load-data",
    fields: [{ name: "data_dir", label: "Data directory (optional — defaults to the project's data/ folder)", type: "text", optional: true }],
  },
  {
    id: "class-average", title: "Class Average Report (per subject)",
    desc: "Average score per subject. GET /sprint2/class-average",
    method: "GET", path: () => "/sprint2/class-average",
    fields: [{ name: "class_name", label: "Class / grade", type: "select-class", optional: true, allLabel: "All classes" }],
    csv: true,
    chart: { type: "bar", labelKey: "subject", valueKey: "average_score", title: (v, f) => `Class Average by Subject — ${f.class_name || "All Classes"}` },
  },
  {
    id: "top-bottom", title: "Top/Bottom Performers",
    desc: "Highest and lowest average scores. GET /sprint2/top-bottom",
    method: "GET", path: () => "/sprint2/top-bottom",
    fields: [
      { name: "class_name", label: "Class / grade", type: "select-class", optional: true, allLabel: "All classes" },
      { name: "subject", label: "Subject", type: "select-subject", optional: true, allLabel: "Overall (all subjects)" },
      { name: "n", label: "How many top/bottom students to show?", type: "number", default: 5, min: 1, max: 50 },
    ],
    csv: true,
    chart: { type: "top-bottom", labelKey: "name", valueKey: "average_score" },
  },
  {
    id: "weak-subjects", title: "Subject-wise Weak Area Detection",
    desc: "Subjects with low averages or high failure rates (merges class-average + failure-rate, then flags is_weak). GET /sprint2/weak-subjects",
    method: "GET", path: () => "/sprint2/weak-subjects",
    fields: [
      { name: "class_name", label: "Class / grade", type: "select-class", optional: true, allLabel: "All classes" },
      { name: "avg_threshold", label: "Average-score threshold", type: "number", default: 55 },
      { name: "fail_threshold", label: "Failure-rate threshold (%)", type: "number", default: 25 },
      { name: "pass_mark", label: "Pass mark", type: "number", default: 35 },
    ],
    csv: true,
    chart: { type: "bar", labelKey: "subject", valueKey: "average_score", highlightKey: "is_weak", title: (v, f) => `Subject Averages — ${f.class_name || "All Classes"} (Weak Subjects Highlighted)` },
  },
  {
    id: "grade-distribution", title: "Grade Distribution",
    desc: "Count of scores in each A–F band. GET /sprint2/grade-distribution",
    method: "GET", path: () => "/sprint2/grade-distribution",
    fields: [
      { name: "class_name", label: "Class / grade", type: "select-class", optional: true, allLabel: "All classes" },
      { name: "subject", label: "Subject", type: "select-subject", optional: true, allLabel: "Overall (all subjects)" },
    ],
    csv: true,
    chart: { type: "bar", labelKey: "grade_band", valueKey: "count", title: (v, f) => `Grade Distribution — ${f.class_name || "All Classes"}${f.subject ? " — " + f.subject : ""}` },
  },
  {
    id: "student-report", title: "Individual Student Report",
    desc: "Full marks, average, class average, z-score, trend. GET /sprint2/student-report",
    method: "GET", path: () => "/sprint2/student-report",
    fields: [
      { name: "search", label: "Enter student name or roll number", type: "text", placeholder: "e.g. Rahul or 101" },
      { name: "class_name", label: "Class / grade", type: "select-class", optional: true, allLabel: "All classes" },
    ],
    csv: true,
  },
  {
    id: "pivot-table", title: "Pivot Table (Subject vs Student)",
    desc: "Average score matrix, one row per student. GET /sprint2/pivot-table",
    method: "GET", path: () => "/sprint2/pivot-table",
    fields: [{ name: "class_name", label: "Class / grade", type: "select-class", optional: true, allLabel: "All classes" }],
    csv: true,
  },
  {
    id: "percentile-ranking", title: "Percentile Ranking",
    desc: "Each student's percentile within the group. GET /sprint2/percentile-ranking",
    method: "GET", path: () => "/sprint2/percentile-ranking",
    fields: [
      { name: "class_name", label: "Class / grade", type: "select-class", optional: true, allLabel: "All classes" },
      { name: "subject", label: "Subject", type: "select-subject", optional: true, allLabel: "Overall (all subjects)" },
    ],
    csv: true,
    chart: { type: "bar", labelKey: "name", valueKey: "percentile_rank", title: (v, f) => `Percentile Ranking — ${f.class_name || "All Classes"}${f.subject ? " — " + f.subject : ""}` },
  },
  {
    id: "subject-strengths", title: "Subject Strengths (Best vs Weakest)",
    desc: "Each student's strongest and weakest subject. GET /sprint2/subject-strengths",
    method: "GET", path: () => "/sprint2/subject-strengths",
    fields: [{ name: "class_name", label: "Class / grade", type: "select-class", optional: true, allLabel: "All classes" }],
    csv: true,
    chart: { type: "bar", labelKey: "student_name", valueKey: "strongest_score", title: (v, f) => `Subject Strengths (strongest score) — ${f.class_name || "All Classes"}` },
  },
  {
    id: "improvement-leaderboard", title: "Improvement Leaderboard",
    desc: "Score change from first to latest test. GET /sprint2/improvement-leaderboard",
    method: "GET", path: () => "/sprint2/improvement-leaderboard",
    fields: [
      { name: "class_name", label: "Class / grade", type: "select-class", optional: true, allLabel: "All classes" },
      { name: "subject", label: "Subject", type: "select-subject", optional: true, allLabel: "Overall (all subjects)" },
    ],
    csv: true,
    chart: { type: "bar", labelKey: "student_name", valueKey: "improvement", signed: true, title: (v, f) => `Improvement Leaderboard — ${f.class_name || "All Classes"}${f.subject ? " — " + f.subject : ""}` },
  },
  {
    id: "consistency", title: "Consistency Analysis",
    desc: "Standard deviation of each student's scores — lower is more consistent. GET /sprint2/consistency",
    method: "GET", path: () => "/sprint2/consistency",
    fields: [
      { name: "class_name", label: "Class / grade", type: "select-class", optional: true, allLabel: "All classes" },
      { name: "subject", label: "Subject", type: "select-subject", optional: true, allLabel: "Overall (all subjects)" },
    ],
    csv: true,
    chart: { type: "scatter", xKey: "average_score", yKey: "consistency_std", xLabel: "Average score", yLabel: "Consistency (std dev)", title: (v, f) => `Consistency — ${f.class_name || "All Classes"}${f.subject ? " — " + f.subject : ""}` },
  },
  {
    id: "outliers", title: "Outlier Detection",
    desc: "Students whose average deviates significantly from the group (z-score). GET /sprint2/outliers",
    method: "GET", path: () => "/sprint2/outliers",
    fields: [
      { name: "class_name", label: "Class / grade", type: "select-class", optional: true, allLabel: "All classes" },
      { name: "subject", label: "Subject", type: "select-subject", optional: true, allLabel: "Overall (all subjects)" },
      { name: "threshold", label: "Outlier sensitivity (z-score threshold)", type: "number", default: 2.0, step: 0.1, min: 0.5, max: 5.0 },
    ],
    csv: true,
    chart: { type: "scatter", xKey: "average_score", yKey: "zscore", xLabel: "Average score", yLabel: "Z-score", highlightAll: true, title: (v, f) => `Outlier Detection — ${f.class_name || "All Classes"}${f.subject ? " — " + f.subject : ""}` },
  },
  {
    id: "class-comparison", title: "Class Comparison (Grade vs Grade)",
    desc: "Average performance across classes/grades, overall and per subject. GET /sprint2/class-comparison",
    method: "GET", path: () => "/sprint2/class-comparison",
    fields: [
      { name: "subject", label: "Subject", type: "select-subject", optional: true, allLabel: "Overall (all subjects)" },
      { name: "by", label: "Compare by", type: "static-select", options: [{ value: "both", label: "Both (overall + by subject)" }, { value: "overall", label: "Overall only" }, { value: "subject", label: "By subject only" }], default: "both" },
    ],
    csv: true,
  },
  {
    id: "pass-rate", title: "Pass Rate Summary",
    desc: "Pass rate grouped by subject or by class/grade. GET /sprint2/pass-rate",
    method: "GET", path: () => "/sprint2/pass-rate",
    fields: [
      { name: "class_name", label: "Class / grade", type: "select-class", optional: true, allLabel: "All classes" },
      { name: "group_by", label: "Group pass rate by", type: "static-select", options: [{ value: "subject", label: "Subject" }, { value: "class_name", label: "Class/Grade" }], default: "subject" },
      { name: "pass_mark", label: "Pass mark", type: "number", default: 35 },
    ],
    csv: true,
    chart: { type: "bar", labelKey: null, valueKey: "pass_rate_pct", title: (v, f) => `Pass Rate — ${f.class_name || "All Classes"}` },
  },
  {
    id: "subject-top-performers", title: "Subject Top Performers",
    desc: "Top student(s) in each subject. GET /sprint2/subject-top-performers",
    method: "GET", path: () => "/sprint2/subject-top-performers",
    fields: [
      { name: "class_name", label: "Class / grade", type: "select-class", optional: true, allLabel: "All classes" },
      { name: "top_n", label: "How many top performers per subject?", type: "number", default: 1, min: 1, max: 20 },
    ],
    csv: true,
    chart: { type: "bar", labelKey: "__subject_student", valueKey: "score", title: (v, f) => `Top Performers by Subject — ${f.class_name || "All Classes"}` },
  },
];
