# src/exceptions.py
"""Custom exceptions for clearer error handling across the CLI."""


class DataNotFoundError(Exception):
    """Raised when a query expects data that doesn't exist (e.g. empty DB)."""
    pass


class StudentNotFoundError(Exception):
    """Raised when a student lookup (by name or roll number) fails."""
    pass


class InvalidCSVError(Exception):
    """Raised when a seed CSV is missing required columns or has bad data."""
    pass
