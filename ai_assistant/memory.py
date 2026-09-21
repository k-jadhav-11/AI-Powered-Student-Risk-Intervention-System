# ai_assistant/memory.py

"""
Sprint 4 -- chat memory (Section 5.2 of the plan).

Scoped to a Streamlit session (st.session_state), not persisted to a
database -- persistent, multi-session chat history is a later-sprint
concern once a persistence layer is actually in the stack, consistent with
how Sprint 3 itself avoided adding infrastructure (like MLflow) that the
sprint didn't strictly need.

API note: the original plan named `ConversationBufferMemory`
(`langchain.memory`), which no longer exists in current LangChain -- 
`langchain.memory` and the standalone Memory classes were removed in favor
of composing `Runnable`s with `langchain_core.chat_history`. This module
uses `InMemoryChatMessageHistory`, which is that class's direct successor
for a simple "keep every turn" buffer -- same behavior, current API.
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