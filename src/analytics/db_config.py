# src/db_config.py
"""
Loads MySQL connection settings from environment variables (via a .env file).
Keeping credentials out of source code and out of version control.
"""
import os
from dotenv import load_dotenv

load_dotenv()  # reads .env in the project root if present; no-op if missing


def get_db_config() -> dict:
    return {
        "host": os.environ.get("DB_HOST", "localhost"),
        "port": int(os.environ.get("DB_PORT", 3306)),
        "user": os.environ.get("DB_USER", "root"),
        "password": os.environ.get("DB_PASSWORD", ""),
        "database": os.environ.get("DB_NAME", "student_analytics"),
    }
