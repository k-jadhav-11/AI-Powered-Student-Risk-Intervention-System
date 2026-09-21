"""
Raw SQL query functions against MySQL. Each returns a pandas DataFrame so
downstream analytics/chart code has a consistent interface regardless of query shape.

Most functions accept an optional `class_name` (e.g. "5th", "10th") to scope
the report to a single grade instead of the whole school. Leave it as None
for a school-wide (all-grades) report.

Note: mysql-connector-python's cursor requires params as a tuple (or dict),
not a list -- every params list below is converted with tuple(...) before
being handed to pd.read_sql_query.
"""
import pandas as pd

# 1. Get available classes
def get_available_classes(conn) -> list:
    """List distinct class/grade names present in the students table, sorted naturally by grade number."""
    df = pd.read_sql_query(
        "SELECT DISTINCT class_name FROM students WHERE class_name IS NOT NULL", conn
    )
    if df.empty:
        return []

    def grade_num(name):
        digits = "".join(ch for ch in str(name) if ch.isdigit())
        return int(digits) if digits else 999

    return sorted(df["class_name"].tolist(), key=grade_num)

# 2. Get available subjects
def get_available_subjects(conn) -> list:
    """List distinct subject names present in the subjects table, alphabetically sorted."""
    df = pd.read_sql_query("SELECT DISTINCT subject_name FROM subjects", conn)
    if df.empty:
        return []
    return sorted(df["subject_name"].tolist())


# 3. Get class average by subject, optionally scoped to a single grade.
def get_class_average_by_subject(conn, class_name: str = None) -> pd.DataFrame:
    """Average score per subject, optionally scoped to a single grade (class_name)."""
    query = """
        SELECT sub.subject_name AS subject,
               ROUND(AVG(m.score), 2) AS average_score,
               COUNT(*) AS num_records
        FROM marks m
        JOIN subjects sub ON m.subject_id = sub.subject_id
        JOIN students st ON m.student_id = st.student_id
    """
    params = []
    if class_name:
        query += " WHERE st.class_name = %s"
        params.append(class_name)
    query += " GROUP BY sub.subject_name ORDER BY average_score DESC"
    return pd.read_sql_query(query, conn, params=tuple(params))


# 4. Get student averages, optionally scoped to a subject and/or a single grade.
def get_student_averages(conn, subject: str = None, class_name: str = None) -> pd.DataFrame:
    """Average score per student, optionally filtered to a subject and/or a single grade."""
    query = """
        SELECT st.student_id, st.name, st.roll_number, st.class_name,
               ROUND(AVG(m.score), 2) AS average_score
        FROM marks m
        JOIN students st ON m.student_id = st.student_id
        JOIN subjects sub ON m.subject_id = sub.subject_id
    """
    conditions = []
    params = []
    if subject:
        conditions.append("sub.subject_name = %s")
        params.append(subject)
    if class_name:
        conditions.append("st.class_name = %s")
        params.append(class_name)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " GROUP BY st.student_id ORDER BY average_score DESC"
    return pd.read_sql_query(query, conn, params=tuple(params))


# 5. Get top/bottom N students by average score, optionally scoped to a subject and/or a single grade.
def get_top_bottom_students(
    conn, n: int = 5, subject: str = None, class_name: str = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (top_n, bottom_n) DataFrames ranked by average score, optionally scoped to one grade."""
    df = get_student_averages(conn, subject, class_name)
    if df.empty:
        return df, df
    top = df.head(n)
    bottom = df.tail(n).sort_values("average_score")
    return top, bottom


# 6. Get subject failure rates, optionally scoped to a single grade.
def get_subject_failure_rates(conn, pass_mark: float = 35, class_name: str = None) -> pd.DataFrame:
    """Percentage of records below pass_mark, grouped by subject, optionally scoped to one grade."""
    query = """
        SELECT sub.subject_name AS subject,
               ROUND(100.0 * SUM(CASE WHEN m.score < %s THEN 1 ELSE 0 END) / COUNT(*), 2)
                   AS failure_rate_pct
        FROM marks m
        JOIN subjects sub ON m.subject_id = sub.subject_id
        JOIN students st ON m.student_id = st.student_id
    """
    params = [pass_mark]
    if class_name:
        query += " WHERE st.class_name = %s"
        params.append(class_name)
    query += " GROUP BY sub.subject_name ORDER BY failure_rate_pct DESC"
    return pd.read_sql_query(query, conn, params=tuple(params))


# 7. Get all individual score records, optionally filtered by subject and/or a single grade.
def get_all_scores(conn, subject: str = None, class_name: str = None) -> pd.DataFrame:
    """Every individual score record, optionally filtered by subject and/or a single grade."""
    query = """
        SELECT st.name AS student_name, st.roll_number, st.class_name,
               sub.subject_name AS subject, m.test_name, m.score, m.max_score, m.test_date
        FROM marks m
        JOIN students st ON m.student_id = st.student_id
        JOIN subjects sub ON m.subject_id = sub.subject_id
    """
    conditions = []
    params = []
    if subject:
        conditions.append("sub.subject_name = %s")
        params.append(subject)
    if class_name:
        conditions.append("st.class_name = %s")
        params.append(class_name)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    return pd.read_sql_query(query, conn, params=tuple(params))


# 8. Find a student by exact roll_number or partial name match, optionally scoped to one grade.
def find_student(conn, search_term: str, class_name: str = None) -> pd.DataFrame:
    """Find students by exact roll_number or partial name match, optionally scoped to one grade."""
    query = """
        SELECT student_id, name, roll_number, class_name
        FROM students
        WHERE (roll_number = %s OR name LIKE %s)
    """
    params = [search_term, f"%{search_term}%"]
    if class_name:
        query += " AND class_name = %s"
        params.append(class_name)
    return pd.read_sql_query(query, conn, params=tuple(params))


# 9. Get all of one student's scores across tests/subjects, ordered by date.
def get_student_scores_over_time(conn, student_id: int) -> pd.DataFrame:
    """All of one student's scores across tests/subjects, ordered by date."""
    query = """
        SELECT sub.subject_name AS subject, m.test_name, m.score, m.max_score, m.test_date
        FROM marks m
        JOIN subjects sub ON m.subject_id = sub.subject_id
        WHERE m.student_id = %s
        ORDER BY m.test_date, sub.subject_name
    """
    return pd.read_sql_query(query, conn, params=(student_id,))
