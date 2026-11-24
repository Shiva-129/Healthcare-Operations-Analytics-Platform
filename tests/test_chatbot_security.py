"""
test_chatbot_security.py
-------------------------
Unit tests for AI chatbot security, read-only SQL validation,
multi-statement blocking, and offline analytical fallbacks.
"""

import pytest

from src.rag.chatbot import (
    validate_sql_query,
    execute_safe_query,
    ask_hospital_ai,
)


def test_validate_sql_query_valid():
    """Verifies that legitimate SELECT and CTE queries are permitted."""
    valid_queries = [
        "SELECT * FROM patients",
        "SELECT COUNT(*) AS total FROM patients WHERE age > 60",
        "SELECT department, AVG(billing_amount) FROM patients GROUP BY department",
        "WITH avg_b AS (SELECT AVG(billing_amount) AS a FROM patients) SELECT * FROM patients CROSS JOIN avg_b",
    ]
    for q in valid_queries:
        is_valid, msg = validate_sql_query(q)
        assert is_valid is True, f"Legitimate query was rejected: {q} ({msg})"


def test_validate_sql_query_blocks_destructive_keywords():
    """Verifies that all destructive SQL keywords are blocked."""
    destructive_queries = [
        "DROP TABLE patients",
        "DELETE FROM patients WHERE id = 1",
        "UPDATE patients SET billing_amount = 0",
        "INSERT INTO patients VALUES (1, 'Hacked')",
        "ALTER TABLE patients DROP COLUMN name",
        "TRUNCATE TABLE patients",
        "CREATE TABLE backdoors (id INT)",
        "PRAGMA table_info(patients)",
    ]
    for q in destructive_queries:
        is_valid, msg = validate_sql_query(q)
        assert is_valid is False, f"Destructive query was not blocked: {q}"
        assert "Security violation" in msg


def test_validate_sql_query_blocks_multistatement_injection():
    """Verifies that multi-statement SQL execution attempts are blocked."""
    injections = [
        "SELECT * FROM patients; DROP TABLE patients;",
        "SELECT * FROM patients; DELETE FROM patients;",
        "SELECT 1; SELECT 2",
    ]
    for q in injections:
        is_valid, msg = validate_sql_query(q)
        assert is_valid is False, f"Multi-statement injection was not blocked: {q}"
        assert "Multi-statement" in msg


def test_ask_hospital_ai_empty_question():
    """Verifies that empty question returns a helpful prompt."""
    res = ask_hospital_ai("")
    assert "Please enter a question" in res


def test_ask_hospital_ai_fallback_total_patients():
    """Verifies that analytical fallback accurately queries total patients."""
    res = ask_hospital_ai("What is the total patient count?")
    assert "Database Record" in res
    assert "patients recorded" in res
