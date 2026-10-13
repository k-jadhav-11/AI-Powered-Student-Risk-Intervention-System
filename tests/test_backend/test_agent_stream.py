import asyncio
import base64
import json
import pandas as pd

from backend.routers import agent as agent_router
from ai_assistant import agent as explain_agent
from backend.schemas import ChatRequest


def test_chat_stream_keeps_agent_context_across_sse_chunks(monkeypatch):
    forwarded = {}
    monkeypatch.setattr(
        agent_router.genai_memory,
        "load_history_into_session",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        agent_router.genai_memory,
        "save_turn_to_db",
        lambda *_args: None,
    )

    def context_sensitive_stream(_question, _session_state, result, **kwargs):
        forwarded.update(kwargs)
        result.artifacts.gru_weekly_df = pd.DataFrame([
            {"week_number": 0, "total_clicks": 12, "days_active": 2},
            {"week_number": 1, "total_clicks": 18, "days_active": 3},
        ])
        result.artifacts.gru_trajectory_df = pd.DataFrame([
            {"week_number": 0, "risk_score": 0.72},
            {"week_number": 1, "risk_score": 0.31},
        ])
        result.artifacts.gru_student_id = "6516"
        result.artifacts.gru_first_high_risk_week = 0
        token = explain_agent._current_artifacts.set(result.artifacts)
        try:
            yield "Hello "
            yield "there."
        finally:
            explain_agent._current_artifacts.reset(token)

    monkeypatch.setattr(
        explain_agent,
        "run_agent_stream_with_artifacts",
        context_sensitive_stream,
    )

    response = agent_router.chat_stream(
        ChatRequest(
            question="Explain this chart",
            page_context="Dashboard shows 31 nudges.",
            chart_gallery={"Decision breakdown": "data:image/jpeg;base64,dGVzdA=="},
        )
    )

    async def collect_body():
        chunks = [chunk async for chunk in response.body_iterator]
        return "".join(chunk.decode() if isinstance(chunk, bytes) else chunk for chunk in chunks)

    body = asyncio.run(collect_body())
    assert "event: start" in body
    assert 'event: token\ndata: {"text": "Hello "}' in body
    assert 'event: token\ndata: {"text": "there."}' in body
    assert "event: complete" in body
    completion = json.loads(body.split("event: complete\ndata: ", 1)[1].split("\n\n", 1)[0])
    assert completion["charts"]["gru_timeseries"]["student_id"] == "6516"
    assert completion["charts"]["gru_timeseries"]["weekly_engagement"][1]["total_clicks"] == 18
    assert completion["charts"]["gru_timeseries"]["risk_trajectory"][0]["risk_score"] == 0.72
    assert base64.b64decode(completion["charts"]["gru_timeseries"]["image_png_base64"]).startswith(
        b"\x89PNG\r\n\x1a\n"
    )
    assert forwarded["page_context"] == "Dashboard shows 31 nudges."
    assert forwarded["chart_gallery"]["Decision breakdown"].startswith("data:image/jpeg;base64,")


def test_chart_question_includes_chart_images_and_number_references():
    assert explain_agent._looks_like_chart_question("Why is 31 high?")
    assert explain_agent._looks_like_chart_question("Can you explain this?")
    assert not explain_agent._looks_like_chart_question("Say hello")

    messages, image_attached = explain_agent._build_message_history(
        {},
        "test",
        "Why is 31 high?",
        "system prompt",
        chart_gallery={"Decision breakdown": "data:image/jpeg;base64,dGVzdA=="},
        page_context="Dashboard shows 31 nudges.",
    )

    assert image_attached
    assert "Dashboard shows 31 nudges." in messages[-1].content[0]["text"]
    assert messages[-1].content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
