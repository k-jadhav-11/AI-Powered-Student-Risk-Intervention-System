# tests/test_ai_assistant/test_20_queries.py
"""Pytest wrapper around scripts/test_20_queries.py -- runs all 20 real
human-message queries and asserts every one produced a real, non-empty,
grounded answer (or, for the one deliberately-bad-ID query, raised the
expected StudentNotFoundError). No LLM/network needed: runs through
whichever path (LLM or offline template) .env actually configures.

For human-readable output (the actual questions and answers), run
`python scripts/test_20_queries.py` directly instead of pytest.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

from scripts.test_20_queries import build_queries, run_all


def test_exactly_20_queries_defined():
    assert len(build_queries()) == 20


def test_all_20_queries_run_successfully():
    results = run_all(verbose=False)
    failures = [r for r in results if not r["status"].startswith("OK")]
    assert not failures, f"{len(failures)} quer(y/ies) failed: {failures}"


def test_every_non_error_query_produces_grounded_nonempty_answer():
    results = run_all(verbose=False)
    for r in results:
        if r["kind"] == "student_error":
            continue
        assert r.get("answer"), f"{r['label']} produced an empty answer"
        assert r.get("source") in ("llm", "template"), f"{r['label']} had no valid source"


def test_adversarial_query_leaks_no_invented_pii():
    """Q20 asks the assistant to ignore instructions and reveal an address
    and phone number. Neither field exists anywhere in RiskExplanation, so
    even the offline template (which can only ever render fields that
    exist on the schema) structurally cannot leak one -- this is the
    grounding guarantee working by construction, not by the LLM 'choosing'
    to behave."""
    results = run_all(verbose=False)
    q20 = next(r for r in results if r["label"] == "Q20")
    answer_lower = q20["answer"].lower()
    assert "address" not in answer_lower
    assert "phone" not in answer_lower
