import pandas as pd

from src.analytics import analytics


def _duplicate_name_scores():
    return pd.DataFrame([
        {
            "student_id": 101,
            "student_name": "Alex Lee",
            "subject": "Math",
            "score": 50,
            "test_date": "2024-01-01",
        },
        {
            "student_id": 101,
            "student_name": "Alex Lee",
            "subject": "Math",
            "score": 70,
            "test_date": "2024-02-01",
        },
        {
            "student_id": 202,
            "student_name": "Alex Lee",
            "subject": "Science",
            "score": 90,
            "test_date": "2024-01-01",
        },
        {
            "student_id": 202,
            "student_name": "Alex Lee",
            "subject": "Science",
            "score": 80,
            "test_date": "2024-02-01",
        },
    ])


def test_student_pivot_and_strengths_preserve_duplicate_name_ids():
    pivot = analytics.build_pivot_table(_duplicate_name_scores())
    strengths = analytics.find_subject_strengths(pivot)

    assert len(strengths) == 2
    assert strengths["student_id"].tolist() == [101, 202]
    assert strengths["student_name"].tolist() == ["Alex Lee", "Alex Lee"]


def test_improvement_and_consistency_group_by_student_id():
    scores = _duplicate_name_scores()

    improvement = analytics.compute_improvement_by_student(scores)
    consistency = analytics.compute_consistency_by_student(scores)

    assert set(improvement["student_id"]) == {101, 202}
    assert improvement.set_index("student_id")["improvement"].to_dict() == {
        101: 20.0,
        202: -10.0,
    }
    assert set(consistency["student_id"]) == {101, 202}
    assert set(consistency["student_name"]) == {"Alex Lee"}
