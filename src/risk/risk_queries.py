# src/risk/risk_queries.py
"""
Sprint 3 -- model loading, prediction, and raw data-file access.

This is the "read frozen artifacts from disk, never retrain here" layer:
model files (models/), pre-computed CSVs (data/processed/). Design choice,
spelled out up front (same reasoning as the original risk_service.py this
was split out of): unlike src/analytics/db_manager.py + queries.py (which
read/write MySQL live), this module reads whatever's currently on disk --
retrain a model, re-run its notebook, and the next call here picks it up
automatically, no code change needed.

ai_assistant/tools.py and src/explainability.py import ONLY from this module --
not from risk_charts.py, clustering.py, model_gru.py, or evaluation.py --
so Sprint 4's grounding layer has the smallest possible dependency surface
onto Sprint 3.
"""
import os
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
UCI_RAW_DIR = os.path.join(RAW_DIR, "uci_data")
OULAD_RAW_DIR = os.path.join(RAW_DIR, "oulad_data")

RANDOM_STATE = 42  # must match every training notebook's train_test_split call,
                    # or get_test_split() below won't reconstruct the same rows

# Feature columns, in the exact order each model was trained on. Hardcoded
# (rather than re-derived from the CSV every time) so a single-record
# prediction from a Streamlit form can never silently drift from training
# order -- the #1 real-world inference bug, per inference_uci.ipynb /
# inference_oulad.ipynb's own notes.
UCI_FEATURE_COLUMNS = [
    "age", "medu", "fedu", "traveltime", "studytime", "failures", "famrel",
    "freetime", "goout", "dalc", "walc", "health", "absences", "g1", "g2",
    "school_MS", "sex_M", "address_U", "famsize_LE3", "schoolsup_yes",
    "famsup_yes", "higher_yes", "internet_yes", "romantic_yes", "subject_Portuguese",
]

OULAD_FEATURE_COLUMNS = [
    "total_clicks_sum", "total_clicks_mean", "total_clicks_max", "days_active_sum",
    "days_active_mean", "num_weeks_active", "engagement_trend", "num_of_prev_attempts",
    "studied_credits", "pct_assessments_submitted", "age_band_ordinal",
    "highest_education_ordinal", "imd_band_missing", "imd_band_ordinal",
    "code_module_BBB", "code_module_CCC", "code_module_DDD", "code_module_EEE",
    "code_module_FFF", "code_module_GGG", "region_East Midlands Region", "region_Ireland",
    "region_London Region", "region_North Region", "region_North Western Region",
    "region_Scotland", "region_South East Region", "region_South Region",
    "region_South West Region", "region_Wales", "region_West Midlands Region",
    "region_Yorkshire Region", "gender_M", "disability_Y",
]

# F1-optimal thresholds, derived from each model's precision-recall curve on
# its held-out test split (same method as model_training_*.ipynb Section 7b).
# NOT recomputed live on every page load -- deliberately frozen the same way
# the models themselves are, so the app's behavior doesn't shift on a
# Streamlit rerun. Re-derive these (see get_test_split() + sklearn's
# precision_recall_curve) after retraining either model, since a new model's
# optimal cutoff won't generally match the old one.
RISK_THRESHOLD_UCI = 0.8558
RISK_THRESHOLD_OULAD = 0.4473

# The GRU never got a threshold-tuning pass in model_gru_oulad.ipynb (no
# Section 7b-equivalent for it), so 0.5 is the honest default -- same value
# inference_oulad_gru.ipynb uses. Re-derive this the same way as the two
# XGBoost thresholds above if you tune it later.
RISK_THRESHOLD_OULAD_GRU = 0.5


# --------------------------------------------------------------------------
# Model + data loading
# --------------------------------------------------------------------------
def get_model_class_name(dataset: str) -> str:
    """The single source of truth for 'what model is this' -- reads it off
    the loaded model object itself (type(model).__name__) rather than the
    training summary JSON. The JSON's schema isn't consistent across
    datasets: UCI's summary records model_class directly, but OULAD's only
    records tuning_method ("GridSearchCV") -- falling back to that field
    showed the search strategy, not the model, on the OULAD dashboard."""
    model = load_model(dataset)
    return type(model).__name__


_MODEL_CACHE: dict = {}


def load_model(dataset: str):
    """... (unchanged docstring, plus a note that it's now cached) ..."""
    if dataset in _MODEL_CACHE:
        return _MODEL_CACHE[dataset]
    path = os.path.join(MODELS_DIR, f"risk_model_{dataset}_v1.joblib")
    model = joblib.load(path)
    _MODEL_CACHE[dataset] = model
    return model


def load_model_summary(dataset: str) -> dict:
    path = os.path.join(MODELS_DIR, f"risk_model_{dataset}_v1_summary.json")
    with open(path) as f:
        return json.load(f)


def load_risk_scores(dataset: str) -> pd.DataFrame:
    """Pre-computed risk_score / predicted_label / risk_level per student,
    from model_training_{dataset}.ipynb Section 9. This is what the risk
    dashboard pages display -- not recomputed here."""
    path = os.path.join(PROCESSED_DIR, f"{dataset}_risk_scores.csv")
    return pd.read_csv(path)


def load_personas(dataset: str) -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, f"{dataset}_personas.csv")
    return pd.read_csv(path)


def load_feature_matrix(dataset: str) -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, f"{dataset}_feature_matrix.csv")
    return pd.read_csv(path)


def get_test_split(dataset: str):
    """Reconstructs the exact held-out test split each training notebook
    used (same random_state/test_size/stratify) -- needed for the live
    precision-recall chart, since that can't be precomputed as a static image
    without losing the interactive threshold marker."""
    df = load_feature_matrix(dataset)
    X = df.drop(columns=["at_risk"])
    y = df["at_risk"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------
def predict_risk(model, X: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Same shape as every training/inference notebook's predict_risk()."""
    proba = model.predict_proba(X)[:, 1]

    def bucket(p):
        if p < 0.3:
            return "Low"
        elif p < 0.6:
            return "Medium"
        return "High"

    return pd.DataFrame(
        {
            "risk_score": proba.round(4),
            "predicted_label": np.where(proba >= threshold, "at_risk", "not_at_risk"),
            "risk_level": [bucket(p) for p in proba],
        },
        index=X.index,
    )


def build_single_record(input_dict: dict, feature_columns: list) -> pd.DataFrame:
    """Builds a one-row DataFrame in the exact training column order from a
    {column: value} dict -- the single most common inference bug is a
    mismatched/missing column, so this is the one place that enforcement
    happens, shared by both the UCI and OULAD prediction form pages."""
    missing = set(feature_columns) - set(input_dict.keys())
    if missing:
        raise ValueError(f"Missing required fields: {sorted(missing)}")
    return pd.DataFrame([input_dict])[feature_columns]

