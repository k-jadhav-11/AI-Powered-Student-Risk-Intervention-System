# src/risk/model_gru.py
"""
Sprint 3 -- OULAD GRU sequence model: live inference + per-student
time-series analysis (originally notebooks/oulad_notebooks/model_gru_oulad.ipynb
/ inference_oulad_gru.ipynb).

Everything above this module either loads a frozen artifact (joblib model,
precomputed CSV, in risk_queries.py) or fits something fast enough to run
on a page load (K-Means, in clustering.py). The GRU used to fall into
neither bucket for the Streamlit app -- a full-population GRU pass + a raw
studentVle.csv scan were judged too heavy for a page load, so
risk_charts.ipynb produced static PNGs instead.

What *is* cheap enough to run live is scoring ONE real student:
  - reload the trained EngagementGRU checkpoint (same round-trip
    inference_oulad_gru.ipynb verifies: .pt/.txt -> state_dict -> eval())
  - stream studentVle.csv filtered to that single student's rows
  - run the GRU once for an end-of-course score, and once per week for a
    week-by-week "risk trajectory" -- replaying what the model would have
    said in real time as engagement data accumulated
This is what powers the Student Time-Series Risk Explorer page.
"""
import os
import re
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .risk_queries import MODELS_DIR, OULAD_RAW_DIR, RISK_THRESHOLD_OULAD_GRU

# --------------------------------------------------------------------------
# GRU sequence model -- live inference + per-student time-series analysis
# --------------------------------------------------------------------------
# Everything above this point either loads a frozen artifact (joblib model,
# precomputed CSV) or fits something fast enough to run on a page load
# (K-Means). The GRU used to fall into neither bucket for the Streamlit app
# -- risk_charts.ipynb / model_gru_oulad.ipynb produced static PNGs
# (baseline_vs_gru_oulad.png, lead_time_oulad.png) because a full-population
# GRU pass + a raw studentVle.csv scan were judged too heavy for a page load.
#
# What *is* cheap enough to run live is scoring ONE real student:
#   - reload the trained EngagementGRU checkpoint (same round-trip
#     inference_oulad_gru.ipynb verifies: .pt/.txt -> state_dict -> eval())
#   - stream studentVle.csv filtered to that single student's rows
#   - run the GRU once for an end-of-course score, and once per week for a
#     week-by-week "risk trajectory" -- replaying what the model would have
#     said in real time as engagement data accumulated
# This is what powers the new Student Time-Series Risk Explorer page.
class EngagementGRU(nn.Module):
    """Identical to model_gru_oulad.ipynb / inference_oulad_gru.ipynb's
    definition -- this class must exist wherever the checkpoint is loaded,
    since torch.save() only stores the weights (state_dict), never the
    architecture code itself."""

    def __init__(self, input_size=2, hidden_size=32, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out, h_n = self.gru(x)
        final_h = h_n[-1]
        prob = self.head(final_h)
        return prob.squeeze(-1)


_GRU_MODEL_CACHE: dict = {}


def load_gru_model(dataset: str = "oulad"):
    if dataset in _GRU_MODEL_CACHE:
        return _GRU_MODEL_CACHE[dataset]

    pt_path = os.path.join(MODELS_DIR, f"risk_model_{dataset}_gru_v1.pt")
    txt_path = os.path.join(MODELS_DIR, f"risk_model_{dataset}_gru_v1.txt")
    path = pt_path if os.path.exists(pt_path) else txt_path
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    model = EngagementGRU(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    cfg = {
        "max_weeks": checkpoint["max_weeks"],
        "feature_cols": checkpoint["feature_cols"],
        "log1p_clicks": checkpoint["log1p_clicks"],
    }
    _GRU_MODEL_CACHE[dataset] = (model, cfg)
    return model, cfg


def load_gru_model_summary(dataset: str = "oulad") -> dict:
    path = os.path.join(MODELS_DIR, f"risk_model_{dataset}_gru_v1_summary.json")
    with open(path) as f:
        return json.load(f)


_READ_OULAD_CACHE: dict = {}


def _read_oulad(name: str, **kwargs) -> pd.DataFrame:
    if name in _READ_OULAD_CACHE and not kwargs:
        return _READ_OULAD_CACHE[name]
    df = pd.read_csv(os.path.join(OULAD_RAW_DIR, name), na_values="?", **kwargs)
    if not kwargs:
        _READ_OULAD_CACHE[name] = df
    return df


_ACTIVE_STUDENT_IDS_CACHE = None
_VLE_CACHE: pd.DataFrame | None = None


def _load_vle_cached() -> pd.DataFrame:
    global _VLE_CACHE
    if _VLE_CACHE is not None:
        return _VLE_CACHE
    vle_path = os.path.join(OULAD_RAW_DIR, "studentVle.csv")
    _VLE_CACHE = pd.read_csv(vle_path, na_values="?", usecols=["id_student", "date", "sum_click"])
    return _VLE_CACHE


def get_active_student_ids() -> set:
    global _ACTIVE_STUDENT_IDS_CACHE
    if _ACTIVE_STUDENT_IDS_CACHE is not None:
        return _ACTIVE_STUDENT_IDS_CACHE
    _ACTIVE_STUDENT_IDS_CACHE = set(_load_vle_cached()["id_student"].unique().tolist())
    return _ACTIVE_STUDENT_IDS_CACHE


def load_oulad_student_directory(sample_per_outcome: int = 20, seed: int = 42,
                                  require_activity: bool = True) -> pd.DataFrame:
    """A light sample of real OULAD student IDs spanning all 4 final_result
    outcomes, for populating a 'pick a real held-out student' selector --
    mirrors inference_oulad_gru.ipynb Section 4's sampling approach rather
    than loading all ~32K students into a single dropdown.

    require_activity: when True (default), only samples students who have
    at least one row in studentVle.csv, so every entry in the selector has
    something real to show on the time-series page.

    Returns columns: id_student, code_module, region, final_result.
    """
    student_info = _read_oulad("studentInfo.csv")
    outcome_rank = {"Distinction": 0, "Pass": 1, "Fail": 2, "Withdrawn": 3}
    info_dedup = (
        student_info.assign(_rank=student_info["final_result"].map(outcome_rank))
        .sort_values("_rank", ascending=False)
        .drop_duplicates(subset="id_student", keep="first")
    )
    if require_activity:
        active_ids = get_active_student_ids()
        info_dedup = info_dedup[info_dedup["id_student"].isin(active_ids)]

    rng_state = np.random.default_rng(seed)
    chunks = []
    for result in ["Distinction", "Pass", "Fail", "Withdrawn"]:
        pool = info_dedup[info_dedup["final_result"] == result]
        n = min(sample_per_outcome, len(pool))
        idx = rng_state.choice(pool.index.values, size=n, replace=False)
        chunks.append(pool.loc[idx])
    directory = pd.concat(chunks)[["id_student", "code_module", "region", "final_result"]]
    return directory.sort_values("id_student").reset_index(drop=True)


def build_student_weekly_engagement(id_student: int, max_weeks: int = 40) -> pd.DataFrame:
    """Streams studentVle.csv chunk-by-chunk, filtering down to a single
    student's rows, and returns a week-indexed DataFrame with columns
    [week_number, total_clicks, days_active] covering weeks 0..max_weeks-1
    (zero-filled for inactive weeks). Same aggregation as
    feature_engineering_oulad.ipynb / model_gru_oulad.ipynb, just scoped to
    one student instead of the whole ~10.65M-row population -- cheap enough
    to run on a Streamlit page load."""
    vle_all = _load_vle_cached()
    chunk = vle_all[vle_all["id_student"] == id_student]

    full_weeks = pd.DataFrame({"week_number": range(max_weeks)})
    if chunk.empty:
        weekly = full_weeks.copy()
        weekly["total_clicks"] = 0
        weekly["days_active"] = 0
        return weekly

    vle = chunk.reset_index(drop=True)
    vle["week_number"] = (vle["date"].clip(lower=0) // 7).astype(int)
    weekly = (
        vle.groupby("week_number")
        .agg(total_clicks=("sum_click", "sum"), days_active=("date", "nunique"))
        .reset_index()
    )
    weekly = full_weeks.merge(weekly, on="week_number", how="left").fillna(0)
    weekly["total_clicks"] = weekly["total_clicks"].astype(int)
    weekly["days_active"] = weekly["days_active"].astype(int)
    return weekly[weekly["week_number"] < max_weeks].reset_index(drop=True)


def _weekly_df_to_gru_tensor(weekly_df: pd.DataFrame, cfg: dict) -> torch.Tensor:
    seq = (
        weekly_df.set_index("week_number")
        .reindex(range(cfg["max_weeks"]), fill_value=0)[cfg["feature_cols"]]
        .values.astype(np.float32)
    )
    X = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)  # add batch dim
    if cfg["log1p_clicks"]:
        X[:, :, 0] = torch.log1p(X[:, :, 0])
    return X


def score_student_gru(model, cfg: dict, weekly_df: pd.DataFrame,
                       threshold: float = RISK_THRESHOLD_OULAD_GRU) -> dict:
    """Single end-of-sequence GRU risk score for one student's full weekly
    engagement history -- same shape as inference_oulad_gru.ipynb Section 5's
    single-record inference."""
    X = _weekly_df_to_gru_tensor(weekly_df, cfg)
    with torch.no_grad():
        proba = float(model(X).item())

    if proba < 0.3:
        level = "Low"
    elif proba < 0.6:
        level = "Medium"
    else:
        level = "High"

    return {
        "risk_score": round(proba, 4),
        "predicted_label": "at_risk" if proba >= threshold else "not_at_risk",
        "risk_level": level,
    }


def compute_gru_risk_trajectory(model, cfg: dict, weekly_df: pd.DataFrame,
                                 threshold: float = RISK_THRESHOLD_OULAD_GRU,
                                 min_weeks_for_signal: int = 7):
    """Scores the student week-by-week using only the engagement data
    available up through each week -- replaying what the GRU would have
    said in real time as the course progressed. This is what turns a single
    end-of-sequence score into an early-warning / lead-time signal.

    How "up through week w" is fed to the model, and why: each week's row is
    real data for weeks 0..w, then zero-filled for w+1..max_weeks-1 -- i.e.
    "assume no further engagement from here on", rather than a shorter,
    unpadded sequence. This keeps every call in-distribution (the GRU was
    only ever trained on full max_weeks-length, zero-padded sequences, so a
    genuinely short input is out-of-distribution noise, not a real signal).
    The trade-off: read each week's score as "how at-risk would this
    student look if their engagement stopped right here", not as a
    calibrated real-time probability -- it runs structurally high in the
    first couple of weeks purely because one active week followed by
    nothing but zeros IS what a real early-withdrawal looks like in the
    training data, even for a student who goes on to keep engaging.
    `min_weeks_for_signal` gives the trajectory a short burn-in period
    before trusting a crossing for first_high_risk_week, for that reason
    (the full trajectory is still returned so the chart shows all of it,
    including the noisy early weeks).

    Returns:
        (trajectory_df, first_high_risk_week)
        trajectory_df: columns [week_number, risk_score, at_risk], one row
            per week from 0 up to the student's last week with any recorded
            VLE activity.
        first_high_risk_week: int or None -- the earliest week_number (at or
            after min_weeks_for_signal) where risk_score >= threshold
            (mirrors the plan's first_high_risk_week vs. event_week
            lead-time calculation).
    """
    full_seq = (
        weekly_df.set_index("week_number")
        .reindex(range(cfg["max_weeks"]), fill_value=0)[cfg["feature_cols"]]
        .values.astype(np.float32)
    )

    active_weeks = weekly_df.loc[weekly_df["total_clicks"] > 0, "week_number"]
    last_active_week = int(active_weeks.max()) if not active_weeks.empty else 0

    rows = []
    with torch.no_grad():
        for w in range(last_active_week + 1):
            truncated = np.zeros_like(full_seq)
            truncated[: w + 1] = full_seq[: w + 1]
            X = torch.tensor(truncated, dtype=torch.float32).unsqueeze(0)
            if cfg["log1p_clicks"]:
                X[:, :, 0] = torch.log1p(X[:, :, 0])
            proba = float(model(X).item())
            rows.append({"week_number": w, "risk_score": round(proba, 4)})

    trajectory = pd.DataFrame(rows)
    trajectory["at_risk"] = trajectory["risk_score"] >= threshold
    flagged = trajectory.loc[
        trajectory["at_risk"] & (trajectory["week_number"] >= min_weeks_for_signal), "week_number"
    ]
    first_high_risk_week = int(flagged.iloc[0]) if not flagged.empty else None
    return trajectory, first_high_risk_week


def get_actual_event_week(id_student: int):
    """Looks up the actual withdrawal week from studentRegistration.csv
    (date_unregistration, in days since course start; '?' if the student
    never unregistered). Returns None if the student didn't withdraw or
    isn't present -- used to compute a real lead_time = event_week -
    first_high_risk_week for a Withdrawn student, same formula as the
    Sprint 3 plan's Lead-Time Evaluation step."""
    reg = _read_oulad("studentRegistration.csv")
    match = reg[reg["id_student"] == id_student].dropna(subset=["date_unregistration"])
    if match.empty:
        return None
    date_unreg = match["date_unregistration"].astype(float).iloc[0]
    return int(max(date_unreg, 0) // 7)


# Ordinal encodings, identical to feature_engineering_oulad.ipynb Section on
# ordinal mappings -- kept here so a real student's raw studentInfo.csv row
# can be encoded into the exact same OULAD_FEATURE_COLUMNS shape the XGBoost
# model was trained on, for an apples-to-apples GRU-vs-XGBoost comparison.
_AGE_BAND_MAP = {"0-35": 0, "35-55": 1, "55<=": 2}
_EDUCATION_MAP = {
    "No Formal quals": 0, "Lower Than A Level": 1, "A Level or Equivalent": 2,
    "HE Qualification": 3, "Post Graduate Qualification": 4,
}
_OULAD_CODE_MODULES = ["BBB", "CCC", "DDD", "EEE", "FFF", "GGG"]  # AAA = drop_first baseline
_OULAD_REGIONS = [
    "East Midlands Region", "Ireland", "London Region", "North Region",
    "North Western Region", "Scotland", "South East Region", "South Region",
    "South West Region", "Wales", "West Midlands Region", "Yorkshire Region",
]  # East Anglian Region = drop_first baseline


def _imd_band_to_numeric(band):
    """'0-10%' -> 5, '10-20' -> 15, etc. -- midpoint of each band, robust to
    the inconsistent '%' suffix in the raw data. Same helper as
    feature_engineering_oulad.ipynb."""
    if pd.isna(band):
        return None
    match = re.match(r"(\d+)-(\d+)", str(band))
    if not match:
        return None
    low, high = int(match.group(1)), int(match.group(2))
    return (low + high) / 2


def build_oulad_record_from_raw(id_student: int, weekly_df: pd.DataFrame):
    """Builds an OULAD_FEATURE_COLUMNS-shaped {column: value} dict for a
    real student directly from raw studentInfo.csv / assessments.csv /
    studentAssessment.csv -- the same encoding
    feature_engineering_oulad.ipynb used to build the training matrix. This
    lets predict_risk() score the SAME real student the GRU just scored,
    for a fair XGBoost-vs-GRU side-by-side on the Student Time-Series Risk
    Explorer page.

    Returns:
        (record_dict, final_result) -- final_result is the ground-truth
        outcome for that student, purely for display (never fed to a model).
    """
    student_info = _read_oulad("studentInfo.csv")
    outcome_rank = {"Distinction": 0, "Pass": 1, "Fail": 2, "Withdrawn": 3}
    info_dedup = (
        student_info.assign(_rank=student_info["final_result"].map(outcome_rank))
        .sort_values("_rank", ascending=False)
        .drop_duplicates(subset="id_student", keep="first")
        .set_index("id_student")
    )
    if id_student not in info_dedup.index:
        raise ValueError(f"id_student {id_student} not found in studentInfo.csv")
    info = info_dedup.loc[id_student]

    active = weekly_df[weekly_df["total_clicks"] > 0]
    n_active = len(active)
    if n_active >= 2:
        ordered = active.sort_values("week_number")
        mid = n_active // 2
        trend = float(ordered["total_clicks"].iloc[mid:].mean() - ordered["total_clicks"].iloc[:mid].mean())
    else:
        trend = 0.0

    assessments = _read_oulad("assessments.csv")
    student_assessment = _read_oulad("studentAssessment.csv")
    non_exam = assessments[assessments["assessment_type"] != "Exam"]
    module_assessments = set(non_exam.loc[non_exam["code_module"] == info["code_module"], "id_assessment"])
    submitted = set(student_assessment.loc[student_assessment["id_student"] == id_student, "id_assessment"])
    pct_submitted = len(submitted & module_assessments) / len(module_assessments) if module_assessments else 0.0

    imd_val = _imd_band_to_numeric(info.get("imd_band"))

    record = {
        "total_clicks_sum": float(active["total_clicks"].sum()),
        "total_clicks_mean": float(active["total_clicks"].mean()) if n_active else 0.0,
        "total_clicks_max": float(active["total_clicks"].max()) if n_active else 0.0,
        "days_active_sum": float(active["days_active"].sum()),
        "days_active_mean": float(active["days_active"].mean()) if n_active else 0.0,
        "num_weeks_active": n_active,
        "engagement_trend": trend,
        "num_of_prev_attempts": float(info["num_of_prev_attempts"]),
        "studied_credits": float(info["studied_credits"]),
        "pct_assessments_submitted": pct_submitted,
        "age_band_ordinal": _AGE_BAND_MAP.get(info.get("age_band"), 1),
        "highest_education_ordinal": _EDUCATION_MAP.get(info.get("highest_education"), 2),
        "imd_band_missing": int(imd_val is None),
        # 55 = midpoint of the 50-60% band, a reasonable fallback matching
        # feature_engineering_oulad.ipynb's median-imputation intent when the
        # exact training-set median isn't available at inference time.
        "imd_band_ordinal": imd_val if imd_val is not None else 55.0,
        "gender_M": info.get("gender") == "M",
        "disability_Y": info.get("disability") == "Y",
    }
    for m in _OULAD_CODE_MODULES:
        record[f"code_module_{m}"] = info.get("code_module") == m
    for r in _OULAD_REGIONS:
        record[f"region_{r}"] = info.get("region") == r

    return record, info.get("final_result")


def build_weekly_engagement_batch(student_ids, max_weeks: int = 40) -> dict:
    """Single chunked pass over studentVle.csv, filtering to a *set* of
    students at once and grouping by (id_student, week_number) -- built for
    scoring many students together (e.g. a live sampled population
    comparison), where looping build_student_weekly_engagement() one id at a
    time would mean rescanning the full ~10.65M-row file once per student.

    Returns: {id_student: weekly_df} dict, one weekly_df per input id
    (zero-filled for any id with no VLE rows at all).
    """
    student_ids = {int(s) for s in student_ids}
    vle_all = _load_vle_cached()
    vle = vle_all[vle_all["id_student"].isin(student_ids)]

    full_weeks = pd.DataFrame({"week_number": range(max_weeks)})

    def _zero_filled():
        weekly = full_weeks.copy()
        weekly["total_clicks"] = 0
        weekly["days_active"] = 0
        return weekly

    if vle.empty:
        return {sid: _zero_filled() for sid in student_ids}

    vle = vle.reset_index(drop=True)
    vle["week_number"] = (vle["date"].clip(lower=0) // 7).astype(int)
    grouped = (
        vle.groupby(["id_student", "week_number"])
        .agg(total_clicks=("sum_click", "sum"), days_active=("date", "nunique"))
        .reset_index()
    )

    result = {}
    for sid in student_ids:
        sub = grouped.loc[grouped["id_student"] == sid, ["week_number", "total_clicks", "days_active"]]
        if sub.empty:
            result[sid] = _zero_filled()
            continue
        weekly = full_weeks.merge(sub, on="week_number", how="left").fillna(0)
        weekly["total_clicks"] = weekly["total_clicks"].astype(int)
        weekly["days_active"] = weekly["days_active"].astype(int)
        result[sid] = weekly[weekly["week_number"] < max_weeks].reset_index(drop=True)
    return result


def score_students_gru_batch(model, cfg: dict, weekly_by_student: dict,
                              threshold: float = RISK_THRESHOLD_OULAD_GRU) -> pd.DataFrame:
    """Scores many students in one forward pass -- stacks each student's
    full max_weeks sequence into a single batch tensor instead of looping
    score_student_gru() one call at a time. Same math as the single-student
    path, just batched for a live population-level comparison."""
    ids = list(weekly_by_student.keys())
    seqs = []
    for sid in ids:
        seq = (
            weekly_by_student[sid].set_index("week_number")
            .reindex(range(cfg["max_weeks"]), fill_value=0)[cfg["feature_cols"]]
            .values.astype(np.float32)
        )
        seqs.append(seq)
    X = torch.tensor(np.stack(seqs), dtype=torch.float32)
    if cfg["log1p_clicks"]:
        X[:, :, 0] = torch.log1p(X[:, :, 0])
    with torch.no_grad():
        proba = model(X).numpy()

    return pd.DataFrame({
        "id_student": ids,
        "risk_score": proba.round(4),
        "predicted_label": np.where(proba >= threshold, "at_risk", "not_at_risk"),
    })

