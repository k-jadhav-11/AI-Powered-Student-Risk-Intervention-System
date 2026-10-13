/* ============================================================
   sprint3Config.js — declarative list of every Sprint 3 report,
   consumed by reportEngine.js. Most are dataset-scoped (uci/oulad,
   picked once at the top of the page); GRU ones are OULAD-only.
   ============================================================ */

// Real trained feature columns (src/risk/risk_queries.py), used to build
// the dynamic "Predict Risk — New Student" form per dataset.
const SPRINT3_FEATURES = {
  uci: [
    "age", "medu", "fedu", "traveltime", "studytime", "failures", "famrel",
    "freetime", "goout", "dalc", "walc", "health", "absences", "g1", "g2",
    "school_MS", "sex_M", "address_U", "famsize_LE3", "schoolsup_yes",
    "famsup_yes", "higher_yes", "internet_yes", "romantic_yes", "subject_Portuguese",
  ],
  oulad: [
    "total_clicks_sum", "total_clicks_mean", "total_clicks_max", "days_active_sum",
    "days_active_mean", "num_weeks_active", "engagement_trend", "num_of_prev_attempts",
    "studied_credits", "pct_assessments_submitted", "age_band_ordinal",
    "highest_education_ordinal", "imd_band_missing", "imd_band_ordinal",
    "code_module_BBB", "code_module_CCC", "code_module_DDD", "code_module_EEE",
    "code_module_FFF", "code_module_GGG",
    "region_East Midlands Region", "region_Ireland", "region_London Region",
    "region_North Region", "region_North Western Region", "region_Scotland",
    "region_South East Region", "region_South Region", "region_South West Region",
    "region_Wales", "region_West Midlands Region", "region_Yorkshire Region",
    "gender_M", "disability_Y",
  ],
};

const SPRINT3_REPORTS = [
  {
    id: "predict", title: "Predict Risk — New Student",
    desc: "Score a brand-new student's features through the real trained model. POST /sprint3/{dataset}/predict",
    datasetScoped: true, method: "POST", custom: "predict",
    path: (v, ds) => `/sprint3/${ds}/predict`,
    fields: [],
  },
  {
    id: "risk-scores", title: "Risk Dashboard",
    desc: "Every student's risk score and level. GET /sprint3/{dataset}/risk-scores",
    datasetScoped: true, method: "GET",
    path: (v, ds) => `/sprint3/${ds}/risk-scores`,
    fields: [
      { name: "risk_level", label: "Filter by risk level", type: "static-multiselect", options: ["High", "Medium", "Low"], default: ["High", "Medium", "Low"] },
      { name: "limit", label: "Rows per page (max 500)", type: "number", default: 500, min: 1, max: 500 },
      { name: "offset", label: "Offset", type: "number", default: 0, min: 0 },
    ],
  },
  {
    id: "model-performance", title: "Risk Model Performance — metrics",
    desc: "Accuracy / precision / recall / F1 / ROC-AUC / confusion matrix on the held-out split at the configured risk threshold. GET /sprint3/{dataset}/model-performance",
    datasetScoped: true, method: "GET",
    path: (v, ds) => `/sprint3/${ds}/model-performance`,
    fields: [],
  },
  {
    id: "student-risk", title: "Student Risk Lookup",
    desc: "One student's risk score and level. GET /sprint3/{dataset}/student/{id}/risk",
    datasetScoped: true, method: "GET",
    path: (v, ds) => `/sprint3/${ds}/student/${encodeURIComponent(v.student_id)}/risk`,
    fields: [{ name: "student_id", label: "Student ID (row index)", type: "text", default: "1", pathParam: true }],
  },
  {
    id: "student-shap", title: "Student SHAP Factors",
    desc: "The real factors behind one student's score. GET /sprint3/{dataset}/student/{id}/shap",
    datasetScoped: true, method: "GET",
    path: (v, ds) => `/sprint3/${ds}/student/${encodeURIComponent(v.student_id)}/shap`,
    fields: [
      { name: "student_id", label: "Student ID (row index)", type: "text", default: "1", pathParam: true },
      { name: "top_n", label: "Top N factors", type: "number", default: 5, min: 1, max: 25 },
    ],
  },
  {
    id: "cohort-summary", title: "Cohort Summary",
    desc: "Risk-level breakdown across the dataset. GET /sprint3/{dataset}/cohort-summary",
    datasetScoped: true, method: "GET",
    path: (v, ds) => `/sprint3/${ds}/cohort-summary`,
    fields: [
      { name: "risk_level", label: "Risk level", type: "static-select", options: ["", "Low", "Medium", "High"], optional: true },
      { name: "limit", label: "Limit", type: "number", default: 10 },
    ],
  },
  {
    id: "personas", title: "Performance Personas",
    desc: "K-Means persona assigned to each student. GET /sprint3/{dataset}/personas",
    datasetScoped: true, method: "GET",
    path: (v, ds) => `/sprint3/${ds}/personas`,
    fields: [
      { name: "limit", label: "Rows to show", type: "number", default: 200, min: 1, max: 1000 },
      { name: "offset", label: "Offset", type: "number", default: 0, min: 0 },
    ],
  },
  {
    id: "clustering-optimal-k", title: "Clustering Explorer — optimal k",
    desc: "Inertia / silhouette score per candidate k. GET /sprint3/{dataset}/clustering/optimal-k",
    datasetScoped: true, method: "GET",
    path: (v, ds) => `/sprint3/${ds}/clustering/optimal-k`,
    fields: [],
  },
  {
    id: "clustering", title: "Clustering Explorer — run K-Means",
    desc: "Cluster centroids + per-student cluster assignment. GET /sprint3/{dataset}/clustering",
    datasetScoped: true, method: "GET",
    path: (v, ds) => `/sprint3/${ds}/clustering`,
    fields: [{ name: "k", label: "Number of clusters (k)", type: "number", default: 4, min: 2, max: 7 }],
  },
  {
    id: "gru-student-directory", title: "GRU — student directory",
    desc: "Sample of real OULAD students for the time-series explorer. GET /sprint3/oulad/gru/student-directory",
    datasetScoped: false, ouladOnly: true, method: "GET",
    path: () => `/sprint3/oulad/gru/student-directory`,
    fields: [{ name: "sample_per_outcome", label: "Students per outcome (5–30)", type: "number", default: 15, min: 5, max: 30 }],
  },
  {
    id: "gru-timeseries", title: "Student Time-Series Risk Explorer (GRU)",
    desc: "Weekly engagement + GRU risk trajectory for one real OULAD student. GET /sprint3/oulad/gru/{id_student}/timeseries",
    datasetScoped: false, ouladOnly: true, method: "GET",
    path: (v) => `/sprint3/oulad/gru/${encodeURIComponent(v.id_student)}/timeseries`,
    fields: [
      { name: "student_choice", label: "Choose a sampled real student", type: "select-gru-student", clientOnly: true },
      { name: "id_student", label: "Or enter a real OULAD id_student", type: "text", placeholder: "e.g. 11391", pathParam: true },
    ],
  },
];
