# tests/test_ai_assistant/test_memory.py
"""ai_assistant/memory.py -- session-scoped chat history. Tested against a plain
dict standing in for st.session_state, so these run with no Streamlit
runtime needed."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai_assistant import memory


def test_memory_roundtrip():
    fake_session_state = {}
    memory.add_turn(fake_session_state, "question one", "answer one")
    memory.add_turn(fake_session_state, "question two", "answer two")
    rendered = memory.format_history_for_prompt(fake_session_state)
    assert "question one" in rendered
    assert "answer two" in rendered


def test_memory_clear_removes_history():
    fake_session_state = {}
    memory.add_turn(fake_session_state, "q", "a")
    memory.clear(fake_session_state)
    assert memory.format_history_for_prompt(fake_session_state) == "(no prior conversation)"


def test_memory_scoped_per_key_not_shared():
    """Different students/cohorts must not bleed into each other's chat
    history -- streamlit_app.py scopes the key per dataset+student_id for
    exactly this reason."""
    fake_session_state = {}
    memory.add_turn(fake_session_state, "about student A", "answer A", key="chat_a")
    memory.add_turn(fake_session_state, "about student B", "answer B", key="chat_b")
    assert "student A" in memory.format_history_for_prompt(fake_session_state, key="chat_a")
    assert "student A" not in memory.format_history_for_prompt(fake_session_state, key="chat_b")


def test_memory_respects_max_turns():
    fake_session_state = {}
    for i in range(10):
        memory.add_turn(fake_session_state, f"question {i}", f"answer {i}")
    rendered = memory.format_history_for_prompt(fake_session_state, max_turns=2)
    assert "question 9" in rendered  # most recent kept
    assert "question 0" not in rendered  # oldest dropped
