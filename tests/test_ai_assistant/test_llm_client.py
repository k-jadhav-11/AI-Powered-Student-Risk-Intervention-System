# tests/test_ai_assistant/test_llm_client.py
"""Token budgeting, provider config detection, and error handling for
ai_assistant/llm_client.py. No real API key or network access needed -- these
test the guardrails around the LLM call, not the LLM itself."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import os
import pytest

from ai_assistant import llm_client, prompts, tools


def test_token_budget_within_limit_for_normal_query():
    result = tools.get_risk_explanation("uci", "0")
    user_prompt = prompts.build_explanation_user_prompt("Why is this student at risk?", result)
    budget = llm_client.check_token_budget(prompts.RISK_EXPLANATION_SYSTEM_PROMPT, user_prompt)
    assert budget["within_budget"] is True
    assert budget["total_tokens"] > 0


def test_token_budget_flags_oversized_prompt():
    huge_prompt = "word " * 20000  # comfortably over TOKEN_BUDGET_PER_QUERY
    budget = llm_client.check_token_budget(prompts.RISK_EXPLANATION_SYSTEM_PROMPT, huge_prompt)
    assert budget["within_budget"] is False


def test_count_tokens_never_raises_even_offline():
    """count_tokens must degrade to an approximation rather than crash if
    tiktoken's vocab file can't be downloaded (e.g. no network, corporate
    proxy) -- this is the exact failure mode hit while building this sprint."""
    assert llm_client.count_tokens("a short piece of text") > 0


def test_is_configured_false_without_env_keys(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert llm_client.is_configured("openai") is False
    assert llm_client.is_configured("gemini") is False


def test_complete_raises_llm_error_when_unconfigured(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(llm_client.LLMError):
        llm_client.complete("system", "user", provider="openai")


def test_unknown_provider_raises_llm_error():
    with pytest.raises(llm_client.LLMError):
        llm_client.complete("system", "user", provider="not_a_real_provider")


def test_get_chat_model_raises_when_unconfigured(monkeypatch):
    """get_chat_model() is the LangChain-native path streaming and
    LangSmith both depend on -- must fail the same clean way complete()
    does, not with a raw SDK exception."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(llm_client.LLMError):
        llm_client.get_chat_model("openai")


def test_stream_complete_raises_before_yielding_any_chunk(monkeypatch):
    """A generator only runs its body once iterated -- this confirms the
    LLMError actually surfaces on the first iteration (before any chunk
    reaches the caller), not silently swallowed or deferred further."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    gen = llm_client.stream_complete("system", "user", provider="openai")
    chunks_received = []
    with pytest.raises(llm_client.LLMError):
        for chunk in gen:
            chunks_received.append(chunk)
    assert chunks_received == []


def test_configure_langsmith_returns_false_without_key(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    assert llm_client.configure_langsmith() is False


def test_configure_langsmith_sets_langchain_env_vars_when_key_present(monkeypatch):
    """LangChain's own tracing looks for LANGCHAIN_TRACING_V2 /
    LANGCHAIN_API_KEY / LANGCHAIN_PROJECT -- this project's .env uses
    LANGSMITH_* names for consistency with the other provider keys, so
    configure_langsmith() has to actually translate one to the other, not
    just check presence."""
    monkeypatch.setenv("LANGSMITH_API_KEY", "fake-key-for-test")
    monkeypatch.setenv("LANGSMITH_PROJECT", "test-project")
    try:
        assert llm_client.configure_langsmith() is True
        assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
        assert os.environ["LANGCHAIN_API_KEY"] == "fake-key-for-test"
        assert os.environ["LANGCHAIN_PROJECT"] == "test-project"
    finally:
        for var in ["LANGCHAIN_TRACING_V2", "LANGCHAIN_API_KEY", "LANGCHAIN_PROJECT"]:
            os.environ.pop(var, None)


# --- Regression coverage for a real bug hit with gemini-3.5-flash-lite:
# newer Gemini models can return chunk.content as a list of content-block
# dicts instead of a plain string (observed alongside a "thought signature"
# extras field), and some langchain-google-genai versions crash trying to
# reconstruct the final merged AIMessage from those chunks once streaming
# finishes. See stream_complete()'s docstring for the full story.

def test_extract_chunk_text_handles_plain_string():
    assert llm_client._extract_chunk_text("hello world") == "hello world"


def test_extract_chunk_text_handles_gemini_content_block_shape():
    """The exact shape observed live: a list of {"type": "text", "text":
    ..., "index": ...} dicts."""
    content = [{"type": "text", "text": "Based on", "index": 0}]
    assert llm_client._extract_chunk_text(content) == "Based on"


def test_extract_chunk_text_handles_multiple_blocks():
    content = [
        {"type": "text", "text": "part one ", "index": 0},
        {"type": "text", "text": "part two", "index": 1},
    ]
    assert llm_client._extract_chunk_text(content) == "part one part two"


def test_extract_chunk_text_handles_unexpected_extra_nesting():
    """Defensive: even a doubly-nested list (list-of-lists) shouldn't crash
    -- it should recurse and extract whatever text is findable."""
    content = [[{"type": "text", "text": "double nested", "index": 0}]]
    assert llm_client._extract_chunk_text(content) == "double nested"


def test_extract_chunk_text_handles_none_and_empty():
    assert llm_client._extract_chunk_text(None) == ""
    assert llm_client._extract_chunk_text([]) == ""
    assert llm_client._extract_chunk_text("") == ""


def test_stream_complete_survives_crash_after_partial_success(monkeypatch):
    """Reproduces the exact real failure: several good text chunks stream
    successfully, then the underlying generator raises (simulating
    LangChain's post-stream AIMessage-reconstruction bug). The caller must
    still receive every chunk that was already produced, with no exception
    propagating -- a successful answer must never be thrown away over a
    finalization-time failure."""
    class _FakeChunk:
        def __init__(self, content):
            self.content = content

    def _fake_stream(messages):
        yield _FakeChunk([{"type": "text", "text": "Based on", "index": 0}])
        yield _FakeChunk([{"type": "text", "text": " the student model", "index": 0}])
        raise ValueError("35 validation errors for AIMessage content.str (simulated)")

    fake_model = type("FakeModel", (), {"stream": staticmethod(_fake_stream)})()
    monkeypatch.setattr(llm_client, "get_chat_model", lambda provider=None: fake_model)

    chunks = list(llm_client.stream_complete("system", "user", provider="openai"))
    assert chunks == ["Based on", " the student model"]
    assert "".join(chunks) == "Based on the student model"
