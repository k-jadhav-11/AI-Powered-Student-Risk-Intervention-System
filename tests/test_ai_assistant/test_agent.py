# tests/test_ai_assistant/test_agent.py
"""Tests for ai_assistant/agent.py -- the real tool-calling loop. No live
LLM access from this environment, so the chat model itself is a scripted
fake (FakeChatModel below) that mimics the real LangChain contract
(bind_tools -> invoke returns an AIMessage with .tool_calls, or .stream
yields chunks) closely enough to exercise the loop's real control flow:
multi-round tool calling, error-as-fact propagation, the artifacts side
channel, and the round-limit cap.

This tests the AGENT'S LOGIC, not whether a real LLM picks good tool calls
-- that requires a live provider and is out of reach in this sandbox.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from langchain_core.messages import AIMessage, AIMessageChunk

from ai_assistant import agent, llm_client
from src.risk import model_gru


class FakeChatModel:
    """Scripted fake matching the LangChain chat-model contract used by
    agent.py: bind_tools() returns self (tools aren't actually needed by
    this fake -- args are pre-scripted), invoke() returns the next
    AIMessage in `responses`, and stream() yields real AIMessageChunks
    (tool-call chunks for a tool-call response, text chunks for a final
    answer) so chunk accumulation via `+` reconstructs .tool_calls exactly
    the way it would from a real streamed response."""

    def __init__(self, responses: list[AIMessage]):
        self._responses = list(responses)
        self._call_count = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        response = self._responses[self._call_count]
        self._call_count += 1
        return response

    def stream(self, messages):
        response = self._responses[self._call_count]
        self._call_count += 1

        if response.tool_calls:
            for i, call in enumerate(response.tool_calls):
                yield AIMessageChunk(content="", tool_call_chunks=[{
                    "name": call["name"], "args": __import__("json").dumps(call["args"]),
                    "id": call["id"], "index": i,
                }])
        else:
            mid = len(response.content) // 2
            yield AIMessageChunk(content=response.content[:mid])
            yield AIMessageChunk(content=response.content[mid:])


def _tool_call_message(name, args, call_id="call_1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


def _final_message(text):
    return AIMessage(content=text, tool_calls=[])


def test_run_agent_calls_the_right_tool_from_the_message(monkeypatch):
    """The core regression test for the original bug: the tool call's
    arguments must come from what the fake model decided (standing in for
    what a real LLM would extract from the user's message), not from any
    UI-side default."""
    monkeypatch.setattr(llm_client, "is_configured", lambda provider=None: True)
    fake = FakeChatModel([
        _tool_call_message("get_student_risk", {"dataset": "uci", "student_id": "5"}),
        _final_message("Student 5 has a real risk score."),
    ])
    monkeypatch.setattr(llm_client, "get_chat_model", lambda provider=None: fake)

    result = agent.run_agent("check risk for student 5", {})
    assert result["tool_calls_made"] == ["get_student_risk"]
    assert result["answer"] == "Student 5 has a real risk score."
    assert result["source"] == "llm"


def test_run_agent_multi_round_tool_calling(monkeypatch):
    """Several questions in your real transcript needed more than one tool
    (risk + shap + interventions) -- verify the loop actually supports
    multiple sequential tool calls before a final answer."""
    monkeypatch.setattr(llm_client, "is_configured", lambda provider=None: True)
    fake = FakeChatModel([
        _tool_call_message("get_student_risk", {"dataset": "uci", "student_id": "0"}, "call_1"),
        _tool_call_message("get_student_shap", {"dataset": "uci", "student_id": "0"}, "call_2"),
        _final_message("Student 0 is High risk because of failures and grades."),
    ])
    monkeypatch.setattr(llm_client, "get_chat_model", lambda provider=None: fake)

    result = agent.run_agent("why is student 0 at risk", {})
    assert result["tool_calls_made"] == ["get_student_risk", "get_student_shap"]
    assert "High risk" in result["answer"]


def test_run_agent_reports_tool_error_as_a_fact(monkeypatch):
    """Regression test for the false 'the requested ID was not in the
    dataset' explanation from your real transcript: the tool's actual
    error message must reach the model as a real tool result, so any
    'not found' the model reports traces back to something real, not a
    guess about what might have gone wrong."""
    monkeypatch.setattr(llm_client, "is_configured", lambda provider=None: True)
    captured_tool_results = []

    class _CapturingFake(FakeChatModel):
        def invoke(self, messages):
            # Capture whatever ToolMessage content was appended before this call.
            for m in messages:
                if type(m).__name__ == "ToolMessage":
                    captured_tool_results.append(m.content)
            return super().invoke(messages)

    fake = _CapturingFake([
        _tool_call_message("get_student_risk", {"dataset": "uci", "student_id": "999999999"}),
        _final_message("I couldn't find that student."),
    ])
    monkeypatch.setattr(llm_client, "get_chat_model", lambda provider=None: fake)

    result = agent.run_agent("check risk for student 999999999", {})
    assert any("not found" in r for r in captured_tool_results)
    assert result["answer"] == "I couldn't find that student."


def test_run_agent_stops_at_max_rounds(monkeypatch):
    """A model that never stops calling tools must not loop forever."""
    monkeypatch.setattr(llm_client, "is_configured", lambda provider=None: True)
    # Every response is a tool call, never a final answer.
    responses = [_tool_call_message("get_model_information", {"dataset": "uci"}, f"call_{i}")
                 for i in range(agent.MAX_TOOL_ROUNDS + 2)]
    fake = FakeChatModel(responses)
    monkeypatch.setattr(llm_client, "get_chat_model", lambda provider=None: fake)

    result = agent.run_agent("infinite loop test", {})
    assert len(result["tool_calls_made"]) == agent.MAX_TOOL_ROUNDS
    assert "rephrase" in result["answer"].lower() or "specific" in result["answer"].lower()


def test_run_agent_unconfigured_gives_honest_message(monkeypatch):
    monkeypatch.setattr(llm_client, "is_configured", lambda provider=None: False)
    result = agent.run_agent("why is student 5 at risk", {})
    assert result["source"] == "unconfigured"
    assert "configured" in result["answer"].lower()


def test_get_gru_context_tool_populates_artifacts(monkeypatch):
    """The GRU chart tool must stash real DataFrames on the artifacts
    channel, not just return the dict -- this is what lets the UI render
    the chart after the agent finishes."""
    monkeypatch.setattr(llm_client, "is_configured", lambda provider=None: True)
    directory = model_gru.load_oulad_student_directory(sample_per_outcome=1)
    sid = int(directory.iloc[0]["id_student"])

    fake = FakeChatModel([
        _tool_call_message("get_gru_context", {"id_student": sid}),
        _final_message(f"Here is student {sid}'s time series."),
    ])
    monkeypatch.setattr(llm_client, "get_chat_model", lambda provider=None: fake)

    result = agent.run_agent(f"how is student {sid} doing over time", {})
    assert result["artifacts"].gru_weekly_df is not None
    assert result["artifacts"].gru_trajectory_df is not None
    assert result["artifacts"].gru_student_id == str(sid)


def test_build_tools_returns_all_eight_tools():
    artifacts = agent.AgentArtifacts()
    tools = agent.build_tools(artifacts)
    names = {t.name for t in tools}
    assert names == {
        "get_student_risk", "get_student_shap", "get_model_information", "search_interventions",
        "get_cohort_summary", "get_student_engagement", "get_student_gru_explanation", "get_gru_context",
    }


def test_agent_artifacts_fresh_per_run_no_leakage(monkeypatch):
    """Different runs must not share artifacts -- each run_agent() call
    builds a fresh AgentArtifacts, closed over by that run's tools only."""
    monkeypatch.setattr(llm_client, "is_configured", lambda provider=None: True)
    directory = model_gru.load_oulad_student_directory(sample_per_outcome=1)
    sid = int(directory.iloc[0]["id_student"])

    fake1 = FakeChatModel([_tool_call_message("get_gru_context", {"id_student": sid}), _final_message("ok")])
    monkeypatch.setattr(llm_client, "get_chat_model", lambda provider=None: fake1)
    result1 = agent.run_agent("q1", {})
    assert result1["artifacts"].gru_weekly_df is not None

    fake2 = FakeChatModel([_final_message("a general answer, no tool needed")])
    monkeypatch.setattr(llm_client, "get_chat_model", lambda provider=None: fake2)
    result2 = agent.run_agent("what is SHAP", {})
    assert result2["artifacts"].gru_weekly_df is None  # fresh instance, no leakage from result1


def test_streaming_agent_yields_full_answer(monkeypatch):
    monkeypatch.setattr(llm_client, "is_configured", lambda provider=None: True)
    fake = FakeChatModel([
        _tool_call_message("get_student_risk", {"dataset": "uci", "student_id": "0"}),
        _final_message("Student 0 is High risk."),
    ])
    monkeypatch.setattr(llm_client, "get_chat_model", lambda provider=None: fake)

    result = agent._StreamResult()
    chunks = list(agent.run_agent_stream_with_artifacts("test question", {}, result))
    assert "".join(chunks) == "Student 0 is High risk."
    assert result.tool_calls_made == ["get_student_risk"]
