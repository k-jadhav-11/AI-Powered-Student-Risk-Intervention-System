# ai_assistant/memory.py
"""
Manage in-memory chat history for the current Streamlit session.
Optionally persist and restore chat history using the database layer.
"""


from langchain_core.chat_history import InMemoryChatMessageHistory


# ============================================================
# Session History
# ============================================================

def get_session_history(
    session_state: dict,
    key: str = "genai_chat_history"
) -> InMemoryChatMessageHistory:
    """
    Get the chat history stored in the current Streamlit session.
    Creates a new InMemoryChatMessageHistory when no history exists.
    """

    # Create chat history on the first request
    if key not in session_state:
        session_state[key] = InMemoryChatMessageHistory()

    # Return the existing or newly created history
    return session_state[key]


# ============================================================
# Add Conversation Turn
# ============================================================

def add_turn(
    session_state: dict,
    question: str,
    answer: str,
    key: str = "genai_chat_history"
) -> None:
    """
    Add one user question and AI answer to the session history.
    This keeps both sides of the conversation available for follow-up questions.
    """

    # Get the current session's conversation history
    history = get_session_history(session_state, key)

    # Store the user's question
    history.add_user_message(question)

    # Store the assistant's response
    history.add_ai_message(answer)


# ============================================================
# Format History for Prompt
# ============================================================

def format_history_for_prompt(
    session_state: dict,
    key: str = "genai_chat_history",
    max_turns: int = 5
) -> str:
    """
    Convert the latest conversation turns into plain text for an LLM prompt.
    Limits the history to max_turns to keep the prompt and token usage manageable.
    """

    # Get the current conversation history
    history = get_session_history(session_state, key)

    # Keep only the requested number of recent conversation turns
    messages = history.messages[-(max_turns * 2):]

    # Return a default message when there is no previous conversation
    if not messages:
        return "(no prior conversation)"

    # Store formatted conversation lines
    lines = []

    # Convert each message into Admin or Assistant format
    for m in messages:
        role = "Admin" if m.type == "human" else "Assistant"
        lines.append(f"{role}: {m.content}")

    # Combine all messages into one prompt-ready string
    return "\n".join(lines)


# ============================================================
# Clear Session History
# ============================================================

def clear(
    session_state: dict,
    key: str = "genai_chat_history"
) -> None:
    """
    Remove the stored conversation history from the current session.
    A new history will be created automatically when get_session_history() is called again.
    """

    # Remove the conversation history from session state
    session_state.pop(key, None)


# ============================================================
# Normalize Message Content to Text
# ============================================================

def _as_text(content) -> str:
    """
    Return message content as a plain string.
    Some providers (e.g. Gemini) return a list of content blocks instead of
    a str; a list cannot be stored in ChatMessage.content, so flatten it to
    its text using the same helper the streaming path already relies on.
    """

    # Already plain text -- nothing to do
    if isinstance(content, str):
        return content

    # Imported lazily to avoid any import cycle with llm_client
    from ai_assistant import llm_client
    return llm_client._extract_chunk_text(content)


# ============================================================
# Save Conversation Turn to Database
# ============================================================

def save_turn_to_db(
    teacher_id: str,
    session_key: str,
    question: str,
    answer: str,
) -> None:
    """
    Persist one chat turn to the database for session-based history.
    Uses best-effort persistence so database failures do not interrupt chat.
    The identifier is not an authenticated teacher identity.
    """

    # Skip persistence entirely when no identifier was supplied
    if not teacher_id:
        return

    # Guard against list-of-blocks content (see _as_text)
    question = _as_text(question)
    answer = _as_text(answer)

    try:
        from backend.database import get_session
        from backend.models_db import ChatMessage

        # Open a database session and store both sides of the turn
        with get_session() as session:
            session.add(ChatMessage(teacher_id=teacher_id, session_key=session_key, role="human", content=question))
            session.add(ChatMessage(teacher_id=teacher_id, session_key=session_key, role="ai", content=answer))
            session.commit()
    except Exception as e:
        # Log the failure but let the conversation continue uninterrupted
        import logging
        logging.getLogger(__name__).error(
            "Failed to persist chat turn for teacher_id=%r: %s", teacher_id, e, exc_info=True
        )


# ============================================================
# Load Conversation History from Database
# ============================================================

def load_history_into_session(
    session_state: dict,
    teacher_id: str,
    session_key: str,
    limit: int = 40,
) -> None:
    """
    Restore saved chat turns into the current session history.
    Loads each teacher/session history only once per Streamlit session.
    """

    # Skip when no identifier was supplied, or history was already loaded once
    loaded_flag = f"_loaded_{session_key}_{teacher_id}"
    if not teacher_id or session_state.get(loaded_flag):
        return
    session_state[loaded_flag] = True

    try:
        from backend.database import get_session
        from backend.models_db import ChatMessage

        # Read this teacher's saved turns for this chat, oldest first
        with get_session() as session:
            rows = (
                session.query(ChatMessage)
                .filter(ChatMessage.teacher_id == teacher_id, ChatMessage.session_key == session_key)
                .order_by(ChatMessage.created_at.asc())
                .limit(limit)
                .all()
            )
    except Exception as e:
        # Log the failure and leave session history empty rather than raising
        import logging
        logging.getLogger(__name__).error(
            "Failed to load chat history for teacher_id=%r: %s", teacher_id, e, exc_info=True
        )
        return

    # Nothing saved yet for this teacher -- leave history empty
    if not rows:
        return

    # Replay each saved row back into the in-memory history
    history = get_session_history(session_state, session_key)
    for row in rows:
        if row.role == "human":
            history.add_user_message(row.content)
        else:
            history.add_ai_message(row.content)