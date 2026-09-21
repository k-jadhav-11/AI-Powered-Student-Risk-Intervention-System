# src/db_manager.py
"""
Handles MySQL connection setup, schema creation, and CSV-based seeding.
"""
import os
import pandas as pd
import mysql.connector
from mysql.connector import errors as mysql_errors

from src.analytics.exceptions import InvalidCSVError
from src.analytics.db_config import get_db_config

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "db", "schema.sql")


def get_connection(config: dict = None) -> mysql.connector.MySQLConnection:
    """Open a MySQL connection using .env-provided settings (or an explicit config dict)."""
    config = config or get_db_config()
    return mysql.connector.connect(**config)


def init_db(config: dict = None) -> None:
    """Create the database's tables/indexes if they don't already exist (idempotent)."""
    conn = get_connection(config)
    cur = conn.cursor()

    with open(SCHEMA_PATH, "r") as f:
        schema_sql = f.read()

    # MySQL's connector doesn't support executing a whole multi-statement
    
    statements = [s.strip() for s in schema_sql.split(";") if s.strip()]
    for stmt in statements:
        try:
            cur.execute(stmt)
        except mysql_errors.ProgrammingError as e:
            # 1061 = Duplicate key name -> index already exists; safe to ignore on re-run.
            if getattr(e, "errno", None) == 1061:
                continue
            raise

    conn.commit()
    cur.close()
    conn.close()


# Helper function to validate required columns in a CSV DataFrame.
def _require_columns(df: pd.DataFrame, required: list, source: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise InvalidCSVError(f"{source}: missing required column(s): {missing}")


def load_students_csv(conn, path: str) -> dict:
    """Insert students from CSV, skipping duplicates on roll_number."""
    df = pd.read_csv(path)
    _require_columns(df, ["name", "roll_number"], path)

    inserted, skipped = 0, 0
    cur = conn.cursor()
    for _, row in df.iterrows():
        class_name = row.get("class_name")
        class_name = None if pd.isna(class_name) else class_name
        try:
            cur.execute(
                "INSERT INTO students (name, roll_number, class_name) VALUES (%s, %s, %s)",
                (row["name"], row["roll_number"], class_name),
            )
            inserted += 1
        except mysql_errors.IntegrityError:
            skipped += 1
    conn.commit()
    cur.close()
    return {"inserted": inserted, "skipped": skipped}


def load_subjects_csv(conn, path: str) -> dict:
    """Insert subjects from CSV, skipping duplicates on subject_name."""
    df = pd.read_csv(path)
    _require_columns(df, ["subject_name"], path)

    inserted, skipped = 0, 0
    cur = conn.cursor()
    for _, row in df.iterrows():
        try:
            cur.execute(
                "INSERT INTO subjects (subject_name) VALUES (%s)",
                (row["subject_name"],),
            )
            inserted += 1
        except mysql_errors.IntegrityError:
            skipped += 1
    conn.commit()
    cur.close()
    return {"inserted": inserted, "skipped": skipped}


def load_marks_csv(conn, path: str) -> dict:
    """
    Insert marks from CSV, resolving roll_number/subject_name to their FK ids.
    Unknown students/subjects and duplicate (student, subject, test_name) rows are skipped.
    """
    df = pd.read_csv(path)
    _require_columns(
        df, ["roll_number", "subject_name", "test_name", "score"], path
    )

    lookup_cur = conn.cursor()
    lookup_cur.execute("SELECT roll_number, student_id FROM students")
    student_map = dict(lookup_cur.fetchall())
    lookup_cur.execute("SELECT subject_name, subject_id FROM subjects")
    subject_map = dict(lookup_cur.fetchall())
    lookup_cur.close()

    inserted, skipped, unknown = 0, 0, 0
    cur = conn.cursor()
    for _, row in df.iterrows():
        student_id = student_map.get(row["roll_number"])
        subject_id = subject_map.get(row["subject_name"])
        if student_id is None or subject_id is None:
            unknown += 1
            continue

        max_score = row.get("max_score", 100)
        if pd.isna(max_score):
            max_score = 100

        test_date = row.get("test_date")
        test_date = None if pd.isna(test_date) else test_date

        try:
            cur.execute(
                """INSERT INTO marks
                   (student_id, subject_id, test_name, score, max_score, test_date)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (
                    student_id,
                    subject_id,
                    row["test_name"],
                    float(row["score"]),
                    float(max_score),
                    test_date,
                ),
            )
            inserted += 1
        except mysql_errors.IntegrityError:
            skipped += 1
    conn.commit()
    cur.close()
    return {"inserted": inserted, "skipped": skipped, "unknown_refs": unknown}


# Helper function to seed the database with all CSV data.
def seed_all(conn, data_dir: str) -> dict:
    """Run all three loaders in the correct order (students/subjects before marks)."""
    summary = {}
    summary["students"] = load_students_csv(conn, os.path.join(data_dir, "students.csv"))
    summary["subjects"] = load_subjects_csv(conn, os.path.join(data_dir, "subjects.csv"))
    summary["marks"] = load_marks_csv(conn, os.path.join(data_dir, "marks.csv"))

    # print("Database seeding complete. Summary:", summary)

    return summary


   
# Helper function to check if the database has any marks data.
def has_data(conn) -> bool:
    """Quick check used by the CLI to warn if the DB is empty."""
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM marks")
    count = cur.fetchone()[0]
    cur.close()
    return count > 0


# if __name__ == "__main__":
#     conn = get_connection()
#     seed_all(conn, "data")
#     conn.close()