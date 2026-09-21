# tests/test_ai_assistant/test_feature_metadata.py
"""Completeness checks for ai_assistant/feature_metadata.py -- every real
feature column that risk_queries actually uses must have both a category
and a description, with no typos/extras in either direction. No LLM/network
needed."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai_assistant import feature_metadata as fm
from src.risk import risk_queries


def test_every_uci_column_has_a_category():
    missing = set(risk_queries.UCI_FEATURE_COLUMNS) - set(fm.UCI_FEATURE_CATEGORY.keys())
    assert not missing, f"UCI columns missing a category: {missing}"


def test_every_uci_column_has_a_description():
    missing = set(risk_queries.UCI_FEATURE_COLUMNS) - set(fm.UCI_FEATURE_DESCRIPTION.keys())
    assert not missing, f"UCI columns missing a description: {missing}"


def test_every_oulad_column_has_a_category():
    missing = set(risk_queries.OULAD_FEATURE_COLUMNS) - set(fm.OULAD_FEATURE_CATEGORY.keys())
    assert not missing, f"OULAD columns missing a category: {missing}"


def test_every_oulad_column_has_a_description():
    missing = set(risk_queries.OULAD_FEATURE_COLUMNS) - set(fm.OULAD_FEATURE_DESCRIPTION.keys())
    assert not missing, f"OULAD columns missing a description: {missing}"


def test_no_stale_entries_beyond_real_columns():
    """Catches typos the other direction: an entry here that doesn't match
    any real column name would silently never be used."""
    uci_extra = set(fm.UCI_FEATURE_CATEGORY.keys()) - set(risk_queries.UCI_FEATURE_COLUMNS)
    oulad_extra = set(fm.OULAD_FEATURE_CATEGORY.keys()) - set(risk_queries.OULAD_FEATURE_COLUMNS)
    assert not uci_extra, f"Stale UCI entries not in real columns: {uci_extra}"
    assert not oulad_extra, f"Stale OULAD entries not in real columns: {oulad_extra}"


def test_categories_are_valid_values():
    valid = {"academic", "engagement", "contextual"}
    for table in (fm.UCI_FEATURE_CATEGORY, fm.OULAD_FEATURE_CATEGORY):
        for feature, category in table.items():
            assert category in valid, f"{feature} has invalid category {category!r}"


def test_build_factor_explanation_is_grounded():
    text = fm.build_factor_explanation("uci", "failures", 2.0, "increases_risk")
    assert "2" in text
    assert "failures" in text.lower() or "previous class failures" in text
    assert "increases" in text


def test_binary_features_are_valid_real_columns():
    """Every entry in UCI_BINARY_FEATURES / OULAD_BINARY_FEATURES must be a
    real column, or it would silently never trigger the is/is-not phrasing
    it exists for."""
    uci_bad = fm.UCI_BINARY_FEATURES - set(risk_queries.UCI_FEATURE_COLUMNS)
    oulad_bad = fm.OULAD_BINARY_FEATURES - set(risk_queries.OULAD_FEATURE_COLUMNS)
    assert not uci_bad, f"UCI_BINARY_FEATURES has non-existent columns: {uci_bad}"
    assert not oulad_bad, f"OULAD_BINARY_FEATURES has non-existent columns: {oulad_bad}"


def test_binary_feature_explanation_uses_is_is_not_phrasing():
    text_yes = fm.build_factor_explanation("uci", "subject_Portuguese", 1.0, "increases_risk")
    text_no = fm.build_factor_explanation("uci", "subject_Portuguese", 0.0, "increases_risk")
    assert "The student is " in text_yes
    assert "The student is not " in text_no
    assert "has 0" not in text_no and "has 1" not in text_yes


def test_continuous_feature_explanation_uses_value_phrasing():
    text = fm.build_factor_explanation("uci", "absences", 6.0, "decreases_risk")
    assert "6" in text
    assert "The student has" in text


def test_no_binary_feature_composes_awkwardly():
    """Regression test for a real bug: several binary descriptions started
    with 'being'/'having', which doubled up with the template's own 'is'
    ('is being male', 'is having internet access'). Every binary feature's
    composed sentence, for both value=0 and value=1, must avoid that."""
    for dataset, binary_set in [("uci", fm.UCI_BINARY_FEATURES), ("oulad", fm.OULAD_BINARY_FEATURES)]:
        for feature in binary_set:
            for value in (0.0, 1.0):
                text = fm.build_factor_explanation(dataset, feature, value, "increases_risk")
                assert "is being" not in text, f"{dataset}/{feature}: {text!r}"
                assert "is having" not in text, f"{dataset}/{feature}: {text!r}"
