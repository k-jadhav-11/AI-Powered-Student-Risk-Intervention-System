# backend/schemas_sprint3.py
"""
Sprint 3 REST layer -- request schemas + a shared JSON-sanitization helper.

This module does NOT reimplement any model loading, feature engineering or
scoring logic -- it only shapes requests going in (the two dataset-specific
/predict bodies, built from the REAL trained feature-column order in
src/risk/risk_queries.py's UCI_FEATURE_COLUMNS / OULAD_FEATURE_COLUMNS) and
sanitizes whatever pandas/numpy/torch output comes back out, so every
sprint3 router endpoint can share one `sanitize()` call instead of each
hand-rolling its own NaN/int64/Timestamp handling.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


# ============================================================
# JSON sanitization -- shared by every sprint3 endpoint
# ============================================================

def sanitize(obj):
    """Recursively converts a value tree that may contain pandas/numpy
    objects into plain JSON-safe Python (NaN/NaT -> None, numpy scalar
    types -> native int/float/bool, DataFrame -> list[dict], Series ->
    dict, Timestamp -> isoformat string). Use this on every value that
    ultimately came from a pandas DataFrame/Series or a numpy/torch
    computation before returning it from a sprint3 endpoint.
    """
    if obj is None:
        return None

    if isinstance(obj, pd.DataFrame):
        # Replace NaN/NaT with None first so to_dict() doesn't hand back
        # float('nan'), which is not valid JSON.
        clean = obj.astype(object).where(pd.notnull(obj), None)
        return [sanitize(row) for row in clean.to_dict(orient="records")]

    if isinstance(obj, pd.Series):
        clean = obj.where(pd.notnull(obj), None)
        return {sanitize(k): sanitize(v) for k, v in clean.to_dict().items()}

    if isinstance(obj, dict):
        return {sanitize(k): sanitize(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [sanitize(v) for v in obj]

    if isinstance(obj, np.ndarray):
        return [sanitize(v) for v in obj.tolist()]

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if math.isnan(v) else v

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()

    if isinstance(obj, float) and math.isnan(obj):
        return None

    if obj is pd.NaT:
        return None

    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass

    return obj


# ============================================================
# Predict -- UCI
# ============================================================
# Field order/names copied verbatim from risk_queries.UCI_FEATURE_COLUMNS.
# The one-hot/binary columns (school_MS ... subject_Portuguese) are the
# *_yes / *_M / *_U / *_LE3 / *_Portuguese dummies the training notebooks
# produced via pd.get_dummies(..., drop_first=True) -- always 0 or 1.

class UCIPredictRequest(BaseModel):
    age: float
    medu: float
    fedu: float
    traveltime: float
    studytime: float
    failures: float
    famrel: float
    freetime: float
    goout: float
    dalc: float
    walc: float
    health: float
    absences: float
    g1: float
    g2: float
    school_MS: int = Field(ge=0, le=1, description="1 if school == 'MS', else 0")
    sex_M: int = Field(ge=0, le=1, description="1 if sex == 'M', else 0")
    address_U: int = Field(ge=0, le=1, description="1 if address == 'U' (urban), else 0")
    famsize_LE3: int = Field(ge=0, le=1, description="1 if famsize <= 3, else 0")
    schoolsup_yes: int = Field(ge=0, le=1)
    famsup_yes: int = Field(ge=0, le=1)
    higher_yes: int = Field(ge=0, le=1)
    internet_yes: int = Field(ge=0, le=1)
    romantic_yes: int = Field(ge=0, le=1)
    subject_Portuguese: int = Field(ge=0, le=1, description="1 if subject == Portuguese, else 0 (Math)")

    def to_feature_dict(self) -> dict:
        return self.model_dump()


# ============================================================
# Predict -- OULAD
# ============================================================
# Field order/names copied verbatim from risk_queries.OULAD_FEATURE_COLUMNS.
# The region_* dummy columns contain spaces in their real column name
# (e.g. "region_East Midlands Region"), which can't be a Python
# identifier, so each uses a Python-safe attribute name with `alias` set
# to the real training-time column name. to_feature_dict() dumps
# by_alias=True so the dict handed to build_single_record() has exactly
# the real column names risk_queries.OULAD_FEATURE_COLUMNS expects.

class OuladPredictRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    total_clicks_sum: float
    total_clicks_mean: float
    total_clicks_max: float
    days_active_sum: float
    days_active_mean: float
    num_weeks_active: float
    engagement_trend: float
    num_of_prev_attempts: float
    studied_credits: float
    pct_assessments_submitted: float
    age_band_ordinal: float
    highest_education_ordinal: float
    imd_band_missing: int = Field(ge=0, le=1)
    imd_band_ordinal: float
    code_module_BBB: int = Field(ge=0, le=1)
    code_module_CCC: int = Field(ge=0, le=1)
    code_module_DDD: int = Field(ge=0, le=1)
    code_module_EEE: int = Field(ge=0, le=1)
    code_module_FFF: int = Field(ge=0, le=1)
    code_module_GGG: int = Field(ge=0, le=1)
    region_east_midlands: int = Field(ge=0, le=1, alias="region_East Midlands Region")
    region_ireland: int = Field(ge=0, le=1, alias="region_Ireland")
    region_london: int = Field(ge=0, le=1, alias="region_London Region")
    region_north: int = Field(ge=0, le=1, alias="region_North Region")
    region_north_western: int = Field(ge=0, le=1, alias="region_North Western Region")
    region_scotland: int = Field(ge=0, le=1, alias="region_Scotland")
    region_south_east: int = Field(ge=0, le=1, alias="region_South East Region")
    region_south: int = Field(ge=0, le=1, alias="region_South Region")
    region_south_west: int = Field(ge=0, le=1, alias="region_South West Region")
    region_wales: int = Field(ge=0, le=1, alias="region_Wales")
    region_west_midlands: int = Field(ge=0, le=1, alias="region_West Midlands Region")
    region_yorkshire: int = Field(ge=0, le=1, alias="region_Yorkshire Region")
    gender_M: int = Field(ge=0, le=1)
    disability_Y: int = Field(ge=0, le=1)

    def to_feature_dict(self) -> dict:
        return self.model_dump(by_alias=True)


# ============================================================
# Predict response
# ============================================================

class PredictResponse(BaseModel):
    dataset: str
    risk_score: float
    predicted_label: str
    risk_level: str
    threshold_used: float
