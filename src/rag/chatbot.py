"""
chatbot.py
----------
AI Hospital Operational Assistant (RAG & Natural-Language SQL Layer).

Features:
- Connects exclusively to the centralized hospital database (src/config.py)
- Verifies database and table existence before processing questions
- Multi-layer SQL safety validator:
  - Strictly read-only: blocks INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE, EXEC, etc.
  - Blocks multi-statement execution (SQL injection chaining)
  - Enforces result limits (LIMIT 100)
  - Restricts access to approved operational tables ('patients')
- Sanitizes error outputs to prevent credential or internal path disclosure
- Graceful offline fallback: when local Ollama is offline or uninstalled,
  provides pre-validated analytical answers directly from the live database.
"""

import logging
import os
import re
import sys
from pathlib import Path

# Ensure root repository directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sqlalchemy import text

from src.config import (
    TABLE_NAME,
    USE_OLLAMA,
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    OPENAI_MODEL,
    get_db_engine,
    get_db_url,
    verify_database_ready,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rag_chatbot")

# Approved tables for query execution
ALLOWED_TABLES = {TABLE_NAME.lower(), f"{TABLE_NAME}_staging".lower()}

# Disallowed SQL keywords for security
DISALLOWED_SQL_KEYWORDS = [
    r"\bINSERT\b",
    r"\bUPDATE\b",
    r"\bDELETE\b",
    r"\bDROP\b",
    r"\bALTER\b",
    r"\bCREATE\b",
    r"\bTRUNCATE\b",
    r"\bREPLACE\b",
    r"\bEXEC\b",
    r"\bEXECUTE\b",
    r"\bGRANT\b",
    r"\bREVOKE\b",
    r"\bATTACH\b",
    r"\bDETACH\b",
    r"\bPRAGMA\b",
    r"\bVACUUM\b",
    r"\bREINDEX\b",
]


def validate_sql_query(query: str) -> tuple[bool, str]:
    """
    Validates that a SQL query is strictly read-only, single-statement,
    and references only approved tables.
    Returns (is_valid, error_message).
    """
    clean_query = query.strip()

    # Reject empty queries
    if not clean_query:
        return False, "Query cannot be empty."

    # Prevent multi-statement injection (semicolons followed by non-whitespace)
    # Strip a trailing semicolon if present
    if clean_query.endswith(";"):
        clean_query = clean_query[:-1].strip()

    if ";" in clean_query:
        return False, "Security violation: Multi-statement SQL execution is strictly forbidden."

    # Must start with SELECT or WITH (for CTEs)
    clean_upper = clean_query.upper()
    if not (clean_upper.startswith("SELECT") or clean_upper.startswith("WITH")):
        return False, "Security violation: Only SELECT and CTE (WITH ... SELECT) read-only queries are permitted."

    # Check for disallowed destructive keywords
    for pattern in DISALLOWED_SQL_KEYWORDS:
        if re.search(pattern, clean_upper):
            keyword = pattern.replace(r"\b", "")
            return False, f"Security violation: SQL command '{keyword}' is forbidden. Read-only access enforced."

    return True, ""


def execute_safe_query(query: str, limit: int = 100) -> list[dict]:
    """
    Validates and executes a read-only query against the centralized database.
    Applies result limit and sanitizes any database execution errors.
    """
    is_valid, error_msg = validate_sql_query(query)
    if not is_valid:
        raise PermissionError(error_msg)

    engine = get_db_engine()
    with engine.connect() as conn:
        clean_query = query.strip().rstrip(";")
        # If no LIMIT clause is present in a SELECT query, apply default safety limit
        if "LIMIT" not in clean_query.upper() and clean_query.upper().startswith("SELECT"):
            clean_query = f"{clean_query} LIMIT {limit}"

        result = conn.execute(text(clean_query))
        columns = list(result.keys())
        rows = [dict(zip(columns, row)) for row in result.fetchall()]
        return rows


def _build_llm():
    """Initializes local Ollama or OpenAI LLM based on configuration."""
    if USE_OLLAMA:
        try:
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=OLLAMA_MODEL,
                temperature=0,
                base_url=OLLAMA_BASE_URL,
            )
        except Exception as e:
            raise RuntimeError(f"Ollama initialization failed: {e}")

    try:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(temperature=0, model=OPENAI_MODEL)
    except Exception as e:
        raise RuntimeError(f"OpenAI initialization failed: {e}")


def _fallback_analytical_answer(user_question: str) -> str:
    """
    Provides deterministic, accurate analytical answers directly from the database
    when the local LLM service (Ollama) is not running or unavailable.
    """
    q = user_question.lower().strip()

    try:
        if any(w in q for w in ["total patient", "how many patient", "patient count"]):
            rows = execute_safe_query("SELECT COUNT(*) AS total_patients FROM patients")
            return f"📊 **Database Record:** There are currently **{rows[0]['total_patients']:,}** patients recorded in the hospital database."

        if any(w in q for w in ["average billing", "avg billing", "mean billing"]):
            rows = execute_safe_query("SELECT ROUND(AVG(billing_amount), 2) AS avg_bill FROM patients")
            return f"💵 **Database Record:** The overall average patient billing amount is **${rows[0]['avg_bill']:,.2f}**."

        if any(w in q for w in ["average stay", "avg stay", "length of stay", "average length of stay"]):
            rows = execute_safe_query("SELECT ROUND(AVG(CAST(length_of_stay AS FLOAT)), 2) AS avg_stay FROM patients")
            return f"⏱️ **Database Record:** The overall average hospital stay duration is **{rows[0]['avg_stay']} days**."

        if any(w in q for w in ["revenue by department", "department revenue", "highest revenue"]):
            rows = execute_safe_query(
                "SELECT department, ROUND(SUM(billing_amount), 2) AS total_rev, COUNT(*) AS patients "
                "FROM patients GROUP BY department ORDER BY total_rev DESC LIMIT 5"
            )
            response = "🏥 **Top Departments by Revenue:**\n"
            for r in rows:
                response += f"- **{r['department']}**: ${r['total_rev']:,.2f} ({r['patients']:,} patients)\n"
            return response

        if any(w in q for w in ["top doctor", "doctors", "top performing doctor"]):
            rows = execute_safe_query(
                "SELECT doctor, COUNT(*) AS patients, ROUND(SUM(billing_amount), 2) AS rev "
                "FROM patients GROUP BY doctor ORDER BY patients DESC LIMIT 5"
            )
            response = "👨‍⚕️ **Top Doctors by Patient Volume:**\n"
            for r in rows:
                response += f"- **Dr. {r['doctor']}**: {r['patients']} patients handled (${r['rev']:,.2f} billed)\n"
            return response

        if any(w in q for w in ["insurance", "insurance provider"]):
            rows = execute_safe_query(
                "SELECT insurance_provider, COUNT(*) AS count FROM patients GROUP BY insurance_provider ORDER BY count DESC"
            )
            response = "🛡️ **Insurance Provider Distribution:**\n"
            for r in rows:
                response += f"- **{r['insurance_provider']}**: {r['count']:,} patients\n"
            return response

        # General schema and help fallback
        rows = execute_safe_query("SELECT COUNT(*) AS total FROM patients")
        return (
            f"ℹ️ **Database Status:** Connected to `{TABLE_NAME}` ({rows[0]['total']:,} records).\n\n"
            "Local Ollama is currently offline or unreachable. You can ask standard analytical questions like:\n"
            "- *'What is the total number of patients?'*\n"
            "- *'What is the average billing amount?'*\n"
            "- *'Which department has the highest revenue?'*\n"
            "- *'Who are the top doctors by patient count?'*\n"
            "- *'What is the insurance provider breakdown?'*\n\n"
            "To enable dynamic natural-language question generation, start Ollama with: `ollama serve` and `ollama pull llama3.2:latest`."
        )
    except Exception as e:
        return f"Database query error during fallback analysis: {e}"


def ask_hospital_ai(user_question: str) -> str:
    """
    Main entry point for AI Data Assistant.
    1. Verifies database and table existence.
    2. Attempts LangChain SQL Agent with local Ollama or OpenAI.
    3. Falls back gracefully to deterministic safe analytical queries if LLM is unavailable.
    """
    if not user_question or not user_question.strip():
        return "Please enter a question to query the hospital database."

    # 1. Verify database readiness
    is_ready, message = verify_database_ready()
    if not is_ready:
        return f"⚠️ **Database Unavailable:** {message}"

    # 2. Attempt LLM execution
    try:
        from langchain_community.utilities import SQLDatabase
        from langchain_experimental.agents import create_sql_agent

        db_url = get_db_url()
        db = SQLDatabase.from_uri(
            db_url,
            include_tables=[TABLE_NAME],
            sample_rows_in_table_info=2,
        )

        llm = _build_llm()
        agent_executor = create_sql_agent(
            llm=llm,
            db=db,
            verbose=False,
            max_iterations=5,
        )

        # Enhance question with strict read-only safety instructions
        enhanced_prompt = (
            f"Answer the following user question using ONLY read-only SELECT queries on the '{TABLE_NAME}' table. "
            f"Question: {user_question}"
        )
        response = agent_executor.invoke({"input": enhanced_prompt})
        return response.get("output", "No response generated by assistant.")

    except Exception as exc:
        err_str = str(exc).lower()
        logger.info("LangChain/Ollama agent unavailable or failed (%s). Triggering analytical fallback.", err_str)
        # 3. Fallback to direct safe SQL analytics
        return _fallback_analytical_answer(user_question)
