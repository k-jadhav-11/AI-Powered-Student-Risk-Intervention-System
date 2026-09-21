"""
Complete, static metadata for every real feature column in
risk_queries.UCI_FEATURE_COLUMNS and OULAD_FEATURE_COLUMNS.

This file defines:
- The category of each feature.
- A plain-English description of each feature.
- Which features are binary 0/1 columns.
- Helper functions to retrieve metadata and build SHAP explanations.

A third CONTEXTUAL category is used for demographic and background
features that are neither academic nor engagement-related.
"""

from typing import Literal


# Allowed categories for each feature
Category = Literal["academic", "engagement", "contextual"]


# ============================================================
# UCI Feature Metadata
# ============================================================

# Category assigned to each UCI feature
UCI_FEATURE_CATEGORY: dict[str, Category] = {
    "age": "contextual",
    "medu": "contextual",  # mother's education level -- background, not the student's own academic record
    "fedu": "contextual",  # father's education level
    "traveltime": "contextual",
    "studytime": "academic",
    "failures": "academic",
    "famrel": "contextual",
    "freetime": "engagement",  # how the student spends time outside class -- behavioral
    "goout": "engagement",
    "dalc": "engagement",  # workday alcohol use -- behavioral pattern, not academic record
    "walc": "engagement",  # weekend alcohol use
    "health": "contextual",
    "absences": "engagement",  # attendance is behavioral, not an academic grade
    "g1": "academic",
    "g2": "academic",
    "school_MS": "contextual",  # which school -- background
    "sex_M": "contextual",
    "address_U": "contextual",  # urban vs rural home address
    "famsize_LE3": "contextual",
    "schoolsup_yes": "academic",  # extra educational school support
    "famsup_yes": "contextual",  # family educational support -- a household characteristic, not the student's own behavior
    "higher_yes": "academic",  # aspires to higher education
    "internet_yes": "contextual",  # home internet access
    "romantic_yes": "contextual",
    "subject_Portuguese": "academic",  # which course this record is for
}


# Plain-English descriptions for UCI features
UCI_FEATURE_DESCRIPTION: dict[str, str] = {
    "age": "the student's age",
    "medu": "the mother's education level",
    "fedu": "the father's education level",
    "traveltime": "home-to-school travel time",
    "studytime": "weekly study time",
    "failures": "previous class failures",
    "famrel": "quality of family relationships",
    "freetime": "free time after school",
    "goout": "how often the student goes out with friends",
    "dalc": "workday alcohol consumption",
    "walc": "weekend alcohol consumption",
    "health": "self-reported health status",
    "absences": "recorded school absences",
    "g1": "the first-period grade",
    "g2": "the second-period grade",
    "school_MS": "attending the MS school (vs. GP)",
    "sex_M": "male (vs. female)",
    "address_U": "living in an urban area (vs. rural)",
    "famsize_LE3": "from a family of 3 or fewer",
    "schoolsup_yes": "receiving extra educational school support",
    "famsup_yes": "receiving family educational support",
    "higher_yes": "aspiring to pursue higher education",
    "internet_yes": "able to access the internet at home",
    "romantic_yes": "in a romantic relationship",
    "subject_Portuguese": "enrolled in the Portuguese course (rather than math)",
}


# ============================================================
# OULAD Feature Metadata
# ============================================================

# Category assigned to each OULAD feature
OULAD_FEATURE_CATEGORY: dict[str, Category] = {
    "total_clicks_sum": "engagement",
    "total_clicks_mean": "engagement",
    "total_clicks_max": "engagement",
    "days_active_sum": "engagement",
    "days_active_mean": "engagement",
    "num_weeks_active": "engagement",
    "engagement_trend": "engagement",
    "num_of_prev_attempts": "academic",
    "studied_credits": "academic",
    "pct_assessments_submitted": "academic",
    "age_band_ordinal": "contextual",
    "highest_education_ordinal": "academic",  # the student's prior education level going into this course
    "imd_band_missing": "contextual",
    "imd_band_ordinal": "contextual",  # socioeconomic deprivation index of home area
    "code_module_BBB": "academic",  # which course module -- academic/course selection
    "code_module_CCC": "academic",
    "code_module_DDD": "academic",
    "code_module_EEE": "academic",
    "code_module_FFF": "academic",
    "code_module_GGG": "academic",
    "region_East Midlands Region": "contextual",
    "region_Ireland": "contextual",
    "region_London Region": "contextual",
    "region_North Region": "contextual",
    "region_North Western Region": "contextual",
    "region_Scotland": "contextual",
    "region_South East Region": "contextual",
    "region_South Region": "contextual",
    "region_South West Region": "contextual",
    "region_Wales": "contextual",
    "region_West Midlands Region": "contextual",
    "region_Yorkshire Region": "contextual",
    "gender_M": "contextual",
    "disability_Y": "contextual",
}


# Plain-English descriptions for OULAD features
OULAD_FEATURE_DESCRIPTION: dict[str, str] = {
    "total_clicks_sum": "total VLE clicks across the whole course",
    "total_clicks_mean": "average weekly VLE clicks",
    "total_clicks_max": "the single busiest week's VLE clicks",
    "days_active_sum": "total days active on the course platform",
    "days_active_mean": "average active days per week",
    "num_weeks_active": "number of weeks with any recorded activity",
    "engagement_trend": "the weekly engagement trend score (negative means declining, positive means rising)",
    "num_of_prev_attempts": "number of previous attempts at this course",
    "studied_credits": "total credits currently being studied",
    "pct_assessments_submitted": "percentage of assessments actually submitted",
    "age_band_ordinal": "the student's age band",
    "highest_education_ordinal": "the student's highest prior education level",
    "imd_band_missing": "missing a recorded home-area deprivation band",
    "imd_band_ordinal": "the home-area deprivation index",
    "code_module_BBB": "enrolled in course module BBB",
    "code_module_CCC": "enrolled in course module CCC",
    "code_module_DDD": "enrolled in course module DDD",
    "code_module_EEE": "enrolled in course module EEE",
    "code_module_FFF": "enrolled in course module FFF",
    "code_module_GGG": "enrolled in course module GGG",
    "region_East Midlands Region": "living in the East Midlands region",
    "region_Ireland": "living in Ireland",
    "region_London Region": "living in the London region",
    "region_North Region": "living in the North region",
    "region_North Western Region": "living in the North Western region",
    "region_Scotland": "living in Scotland",
    "region_South East Region": "living in the South East region",
    "region_South Region": "living in the South region",
    "region_South West Region": "living in the South West region",
    "region_Wales": "living in Wales",
    "region_West Midlands Region": "living in the West Midlands region",
    "region_Yorkshire Region": "living in the Yorkshire region",
    "gender_M": "male (vs. female)",
    "disability_Y": "recorded as having a declared disability",
}


# ============================================================
# Feature Lookup Functions
# ============================================================

def get_category(dataset: str, feature: str) -> Category:
    """
    Return the category assigned to a feature for the selected dataset.
    Unmapped features default to the contextual category.
    """

    # Select the metadata table based on the dataset
    table = UCI_FEATURE_CATEGORY if dataset == "uci" else OULAD_FEATURE_CATEGORY

    # Return the feature category or use contextual as the default
    return table.get(feature, "contextual")


def get_description(dataset: str, feature: str) -> str:
    """
    Return the plain-English description of a feature for the dataset.
    Uses the feature name as a readable fallback when no description exists.
    """

    # Select the description table based on the dataset
    table = UCI_FEATURE_DESCRIPTION if dataset == "uci" else OULAD_FEATURE_DESCRIPTION

    # Return the description or convert the feature name into readable text
    return table.get(feature, feature.replace("_", " "))


# ============================================================
# Binary Feature Definitions
# ============================================================

# UCI binary 0/1 features
# These are described using "is/is not" sentences
UCI_BINARY_FEATURES = {
    "school_MS", "sex_M", "address_U", "famsize_LE3", "schoolsup_yes", "famsup_yes",
    "higher_yes", "internet_yes", "romantic_yes", "subject_Portuguese",
}


# OULAD binary 0/1 features
# These are also described using "is/is not" sentences
OULAD_BINARY_FEATURES = {
    "imd_band_missing", "gender_M", "disability_Y",
    "code_module_BBB", "code_module_CCC", "code_module_DDD", "code_module_EEE",
    "code_module_FFF", "code_module_GGG",
    "region_East Midlands Region", "region_Ireland", "region_London Region", "region_North Region",
    "region_North Western Region", "region_Scotland", "region_South East Region", "region_South Region",
    "region_South West Region", "region_Wales", "region_West Midlands Region", "region_Yorkshire Region",
}


# ============================================================
# Binary Feature Check
# ============================================================

def is_binary_feature(dataset: str, feature: str) -> bool:
    """
    Check whether the given feature is a binary 0/1 feature.
    Returns True when the feature belongs to the dataset's binary feature set.
    """

    # Select the binary feature table based on the dataset
    table = UCI_BINARY_FEATURES if dataset == "uci" else OULAD_BINARY_FEATURES

    # Check whether the feature exists in the selected table
    return feature in table


# ============================================================
# SHAP Factor Explanation
# ============================================================

def build_factor_explanation(
    dataset: str,
    feature: str,
    value: float,
    direction: str
) -> str:
    """
    Convert one SHAP factor into a plain-English explanation for a teacher.
    Uses the real feature value and direction without inventing additional information.
    """

    # Get the human-readable feature description
    description = get_description(dataset, feature)

    # Convert the SHAP direction into a readable phrase
    direction_phrase = (
        "increases the predicted risk"
        if direction == "increases_risk"
        else "decreases the predicted risk"
    )

    # Binary features are described as "is" or "is not"
    if is_binary_feature(dataset, feature):
        state = "is" if value >= 0.5 else "is not"

        return (
            f"The student {state} {description}, "
            f"which {direction_phrase}."
        )

    # Non-binary features are described using their actual value
    return (
        f"The student has {value:g} for {description}, "
        f"which {direction_phrase}."
    )