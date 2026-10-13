# ai_assistant/agent.py

import contextvars
import base64
from io import BytesIO
import re
from typing import Iterator
import pandas as pd
import matplotlib
from langsmith import traceable, trace

matplotlib.use("Agg")

from src.analytics.exceptions import DataNotFoundError, StudentNotFoundError
from src.risk import model_gru, risk_charts, risk_queries
from . import llm_client, memory
from . import tools as ai_tools
from .prompts import AGENT_SYSTEM_PROMPT, CHART_IMAGE_ANALYSIS_ADDENDUM

MAX_TOOL_ROUNDS = 6


class AgentArtifacts:
    """
    Store GRU chart data and student information for the current run.
    Keeps non-JSON data separate from the LLM tool responses.
    """

    def __init__(self):
        self.gru_weekly_df = None
        self.gru_trajectory_df = None
        self.gru_student_id: str | None = None
        self.gru_first_high_risk_week: int | None = None
        self.gru_actual_final_result: str | None = None
        self.gru_event_week: int | None = None

    def chart_payload(self) -> dict:
        if self.gru_weekly_df is None or self.gru_trajectory_df is None:
            return {}

        weekly = self.gru_weekly_df[
            ["week_number", "total_clicks", "days_active"]
        ]
        trajectory = self.gru_trajectory_df[
            ["week_number", "risk_score"]
        ]
        figure = risk_charts.plot_student_engagement_and_risk(
            self.gru_weekly_df,
            self.gru_trajectory_df,
            risk_queries.RISK_THRESHOLD_OULAD_GRU,
            self.gru_first_high_risk_week,
            self.gru_event_week,
            title=f"Student {self.gru_student_id}"
            + (
                f" (actual: {self.gru_actual_final_result})"
                if self.gru_actual_final_result
                else ""
            ),
        )
        image = BytesIO()
        try:
            figure.savefig(image, format="png", dpi=144, bbox_inches="tight")
        finally:
            risk_charts.plt.close(figure)

        return {
            "gru_timeseries": {
                "student_id": self.gru_student_id,
                "image_png_base64": base64.b64encode(image.getvalue()).decode("ascii"),
                "weekly_engagement": [
                    {
                        "week_number": int(row.week_number),
                        "total_clicks": int(row.total_clicks),
                        "days_active": int(row.days_active),
                    }
                    for row in weekly.itertuples(index=False)
                ],
                "risk_trajectory": [
                    {
                        "week_number": int(row.week_number),
                        "risk_score": float(row.risk_score),
                    }
                    for row in trajectory.itertuples(index=False)
                ],
                "first_high_risk_week": self.gru_first_high_risk_week,
            }
        }


_current_artifacts: contextvars.ContextVar[AgentArtifacts] = contextvars.ContextVar("current_artifacts")


def build_tools() -> list:
    """
    Create the tools available to the AI agent for one run.
    Each tool safely calls the corresponding analytics function.
    """
    from langchain_core.tools import tool

    @tool
    def get_student_risk(dataset: str, student_id: str) -> dict:
        """
        Get an existing student's risk score and risk level.
        Uses the dataset row ID provided by the user.
        """
        try:
            return ai_tools.get_student_risk(dataset, student_id)
        except (StudentNotFoundError, ValueError) as e:
            return {"error": str(e)}

    @tool
    def get_student_shap(dataset: str, student_id: str, top_n: int = 5) -> dict:
        """
        Get the main factors affecting a student's risk score.
        Uses SHAP values to explain the model's prediction.
        """
        try:
            return ai_tools.get_student_shap(dataset, student_id, top_n=top_n)
        except (StudentNotFoundError, ValueError) as e:
            return {"error": str(e)}

    @tool
    def get_model_information(dataset: str) -> dict:
        """
        Get the trained model name and its evaluation metrics.
        Returns metrics such as precision, recall, F1, and ROC-AUC.
        """
        try:
            return ai_tools.get_model_information(dataset)
        except ValueError as e:
            return {"error": str(e)}

    @tool
    def search_interventions(risk_factors: list[str], top_k: int = 2) -> dict:
        """
        Find suitable interventions for the student's risk factors.
        Uses the curated intervention knowledge base.
        """
        return ai_tools.search_interventions(risk_factors, top_k=top_k)

    @tool
    def get_cohort_summary(dataset: str, risk_level: str | None = None, limit: int = 10) -> dict:
        """
        Get risk statistics and high-risk students for a dataset.
        Can filter students by Low, Medium, or High risk level.
        """
        try:
            return ai_tools.get_cohort_summary(dataset, risk_level=risk_level, limit=limit)
        except Exception as e:
            return {"error": str(e)}

    @tool
    def get_student_engagement(id_student: int) -> dict:
        """
        Get weekly engagement information for an OULAD student.
        Shows clicks, active weeks, trends, and engagement decline.
        """
        try:
            result = ai_tools.get_student_engagement(id_student)
            chart_rows = result.pop("_weekly_chart_data")
            weekly = pd.DataFrame(chart_rows)
            gru_model_obj, gru_cfg = model_gru.load_gru_model("oulad")
            trajectory, first_high_risk_week = model_gru.compute_gru_risk_trajectory(
                gru_model_obj, gru_cfg, weekly
            )
            artifacts = _current_artifacts.get()
            artifacts.gru_weekly_df = weekly
            artifacts.gru_trajectory_df = trajectory
            artifacts.gru_student_id = str(id_student)
            artifacts.gru_first_high_risk_week = first_high_risk_week
            artifacts.gru_event_week = model_gru.get_actual_event_week(id_student)
            return result
        except StudentNotFoundError as e:
            return {"error": str(e)}

    @tool
    def get_student_gru_explanation(id_student: int) -> dict:
        """
        Explain an OULAD student's GRU risk prediction.
        Shows important weeks and engagement features affecting the score.
        """
        try:
            return ai_tools.get_student_gru_explanation(id_student)
        except StudentNotFoundError as e:
            return {"error": str(e)}

    @tool
    def get_gru_context(id_student: int) -> dict:
        """
        Get the complete GRU and XGBoost risk information for a student.
        Also prepares weekly engagement and risk trajectory data for the chart.
        """
        try:
            context, weekly, trajectory = ai_tools.get_gru_timeseries_context(id_student)
            artifacts = _current_artifacts.get()
            artifacts.gru_weekly_df = weekly
            artifacts.gru_trajectory_df = trajectory
            artifacts.gru_student_id = str(id_student)
            artifacts.gru_first_high_risk_week = context["first_high_risk_week"]
            artifacts.gru_actual_final_result = context["actual_final_result"]
            artifacts.gru_event_week = model_gru.get_actual_event_week(id_student)
            return context
        except StudentNotFoundError as e:
            return {"error": str(e)}

    @tool
    def run_intervention_agent(
        dataset: str,
        student_id: str,
    ) -> dict:
        """
        Run an intervention for a student's current risk.
        Routes the student to a nudge, resource, or mentor escalation and logs the decision.
        
        """
        # Import intervention modules only when the tool is called.
        from orchestration import audit as intervention_audit
        from orchestration import graph as intervention_graph

        try:
            risk_explanation = ai_tools.get_risk_explanation(
                dataset,
                student_id,
            )
        except (
            StudentNotFoundError,
            DataNotFoundError,
            ValueError,
        ) as e:
            return {"error": str(e)}

        decision = intervention_graph.run_intervention(
            dataset,
            student_id,
            risk_explanation,
        )

        intervention_audit.log_decision(
            decision,
            run_id=f"chat:{dataset}:{student_id}",
        )

        return decision

    return [
        get_student_risk, get_student_shap, get_model_information, search_interventions,
        get_cohort_summary, get_student_engagement, get_student_gru_explanation, get_gru_context,
        run_intervention_agent,
    ]


_TOOLS_CACHE: list | None = None
_MODEL_CACHE: dict = {}


def _get_cached_model_and_tools(provider: str | None):
    global _TOOLS_CACHE

    if _TOOLS_CACHE is None:
        _TOOLS_CACHE = build_tools()

    key = provider or llm_client.get_llm_config()["provider"]

    if key not in _MODEL_CACHE:
        model = llm_client.get_chat_model(provider)
        model_with_tools = model.bind_tools(_TOOLS_CACHE)
        _MODEL_CACHE[key] = (model, model_with_tools)

    model, model_with_tools = _MODEL_CACHE[key]
    return model, model_with_tools, _TOOLS_CACHE


_CHART_REFERENCE_WORDS = (
    "graph", "chart", "diagram", "plot", "image", "picture", "trajectory",
    "statistic", "number", "numbers", "value", "count", "percentage", "percent", "rate",
    "average", "metric", "bar", "trend", "this", "that", "these", "those",
    "shown", "above", "how many", "total", "unique", "risk level", "risk-level",
    "escalation", "students",
)


def _looks_like_chart_question(question: str) -> bool:
    """Check whether the question explicitly refers to a chart or visual."""
    q = question.lower()
    return (
        any(re.search(rf"\b{re.escape(word)}\b", q) for word in _CHART_REFERENCE_WORDS)
        or re.search(r"\b\d+(?:\.\d+)?\s*(?:%|percent)?\b", q) is not None
    )


def _build_message_history(session_state: dict, session_key: str, question: str, system_prompt: str,
                            chart_image_b64: str | None = None,
                            chart_gallery: dict[str, str] | None = None,
                            page_context: str | None = None) -> tuple[list, bool]:
    """Build agent messages from the system prompt, chat history, and new question.
    Attach relevant chart images only when the question explicitly refers to a visual.
    Include the current page context so references such as "this student" remain grounded."""

    from langchain_core.messages import HumanMessage, SystemMessage

    history = memory.get_session_history(session_state, session_key)

    images = chart_gallery if chart_gallery else ({"chart": chart_image_b64} if chart_image_b64 else {})
    attach_image = bool(images) and _looks_like_chart_question(question)
    effective_system_prompt = system_prompt + CHART_IMAGE_ANALYSIS_ADDENDUM if attach_image else system_prompt

    messages = [SystemMessage(content=effective_system_prompt)]
    messages.extend(history.messages)

    question_text = f"({page_context})\n\n{question}" if page_context else question

    if attach_image:
        labels = ", ".join(images.keys())
        content = [{"type": "text", "text": f"{question_text}\n\n(Charts currently visible: {labels}.)"}]
        for b64 in images.values():
            if b64.startswith("data:image/"):
                image_url = b64
            else:
                image_url = f"data:image/png;base64,{b64}"
            content.append({"type": "image_url", "image_url": {"url": image_url}})
        messages.append(HumanMessage(content=content))
    else:
        messages.append(HumanMessage(content=question_text))
    return messages, attach_image


@traceable(name="Agent LLM Call", run_type="llm")
def _call_agent_llm(model_with_tools, messages: list):
    """
    Invoke the LLM with tool bindings and trace the call.
    Returns the model's response (which may include tool calls or final answer).
    """
    return model_with_tools.invoke(messages)

# Run the agent implementation (non-streaming) and return its result.
def _run_agent_impl(question: str, session_state: dict, session_key: str,
                    provider: str | None = None, chart_image_b64: str | None = None,
                    chart_gallery: dict[str, str] | None = None,
                    page_context: str | None = None) -> dict:
    """Run the non-streaming agent implementation."""
    from langchain_core.messages import ToolMessage

    artifacts = AgentArtifacts()
    artifacts_token = _current_artifacts.set(artifacts)
    tool_calls_made = []

    if not llm_client.is_configured(provider):
        _current_artifacts.reset(artifacts_token)
        answer = (
            "I need an LLM provider configured (see .env's LLM_PROVIDER / OPENAI_API_KEY / "
            "GEMINI_API_KEY) to understand free-form questions like this one. Once configured, "
            "I can look up any student or cohort you ask about directly from your message."
        )
        return {"answer": answer, "source": "unconfigured", "artifacts": artifacts,
                "tool_calls_made": tool_calls_made, "chart_image_attached": False}
    
    # Trace the agent setup and tool/model retrieval, including building the message history.
    with trace(name="Agent Setup", run_type="chain") as setup_run:
        model, model_with_tools, tools_list = _get_cached_model_and_tools(provider)
        tools_by_name = {t.name: t for t in tools_list}
        messages, image_attached = _build_message_history(session_state, session_key, question, AGENT_SYSTEM_PROMPT,
                                                            chart_image_b64=chart_image_b64,
                                                            chart_gallery=chart_gallery,
                                                            page_context=page_context)
        setup_run.end(outputs={"tool_count": len(tools_list), "image_attached": image_attached})

    try:
        for _ in range(MAX_TOOL_ROUNDS):
            response = _call_agent_llm(model_with_tools, messages)
            messages.append(response)

            if not response.tool_calls:
               # Normalize provider-specific content to plain text for API and database compatibility.
                answer = llm_client._extract_chunk_text(response.content)
                memory.add_turn(session_state, question, answer, session_key)
                return {"answer": answer, "source": "llm", "artifacts": artifacts,
                        "tool_calls_made": tool_calls_made, "chart_image_attached": image_attached}

            for call in response.tool_calls:
                tool_calls_made.append(call["name"])
                tool_fn = tools_by_name.get(call["name"])
                if tool_fn is None:
                    result = {"error": f"Unknown tool {call['name']!r}"}
                else:
                    result = tool_fn.invoke(call["args"])
                messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

        answer = (
            "I wasn't able to settle on a final answer within this conversation's tool-call limit. "
            "Could you rephrase or ask a more specific question?"
        )
        return {"answer": answer, "source": "llm", "artifacts": artifacts,
                "tool_calls_made": tool_calls_made, "chart_image_attached": image_attached}
    finally:
        _current_artifacts.reset(artifacts_token)

# Run the agent and return its result, tracing the entire process.
def run_agent(question: str, session_state: dict, session_key: str = "ai_agent_chat",
              provider: str | None = None, chart_image_b64: str | None = None,
              chart_gallery: dict[str, str] | None = None,
              page_context: str | None = None) -> dict:
    """Run the AI agent and return its result."""
    with trace(
        name="AI Student Risk Assistant",
        run_type="chain",
        inputs={"question": question}
    ) as trace_run:
        result = _run_agent_impl(
            question,
            session_state,
            session_key,
            provider,
            chart_image_b64=chart_image_b64,
            chart_gallery=chart_gallery,
            page_context=page_context,
        )
        trace_run.end(outputs={"answer": result["answer"], "tools_used": result["tool_calls_made"]})
        return result


class _StreamResult:
    """
    Store streaming results, tool calls, and generated artifacts.
    These values are updated while the agent is running.
    """

    def __init__(self):
        self.artifacts = AgentArtifacts()
        self.tool_calls_made: list[str] = []
        self.source = "llm"
        self.chart_image_attached = False


def _run_agent_stream_impl(question: str, session_state: dict, result: _StreamResult,
                           session_key: str, provider: str | None = None,
                           chart_image_b64: str | None = None,
                           chart_gallery: dict[str, str] | None = None,
                           page_context: str | None = None) -> Iterator[str]:
    """Run the agent and stream generated text."""
    from langchain_core.messages import ToolMessage

    if not llm_client.is_configured(provider):
        result.source = "unconfigured"
        yield (
            "I need an LLM provider configured (see .env's LLM_PROVIDER / OPENAI_API_KEY / "
            "GEMINI_API_KEY) to understand free-form questions like this one."
        )
        return

    artifacts_token = _current_artifacts.set(result.artifacts)

    with trace(name="Agent Setup", run_type="chain") as setup_run:
        model, model_with_tools, tools_list = _get_cached_model_and_tools(provider)
        tools_by_name = {t.name: t for t in tools_list}
        messages, image_attached = _build_message_history(session_state, session_key, question, AGENT_SYSTEM_PROMPT,
                                                            chart_image_b64=chart_image_b64,
                                                            chart_gallery=chart_gallery,
                                                            page_context=page_context)
        result.chart_image_attached = image_attached
        setup_run.end(outputs={"tool_count": len(tools_list), "image_attached": image_attached})

    try:
        for _ in range(MAX_TOOL_ROUNDS):
            accumulated = None
            full_answer = []
            for chunk in model_with_tools.stream(messages):
                accumulated = chunk if accumulated is None else accumulated + chunk
                text = llm_client._extract_chunk_text(chunk.content)
                if text:
                    full_answer.append(text)
                    yield text

            messages.append(accumulated)

            if not accumulated.tool_calls:
                memory.add_turn(session_state, question, "".join(full_answer), session_key)
                return

            for call in accumulated.tool_calls:
                result.tool_calls_made.append(call["name"])
                tool_fn = tools_by_name.get(call["name"])
                if tool_fn is None:
                    tool_result = {"error": f"Unknown tool {call['name']!r}"}
                else:
                    tool_result = tool_fn.invoke(call["args"])
                messages.append(ToolMessage(content=str(tool_result), tool_call_id=call["id"]))

        yield (
            "I wasn't able to settle on a final answer within this conversation's tool-call limit. "
            "Could you rephrase or ask a more specific question?"
        )
    finally:
        _current_artifacts.reset(artifacts_token)


def run_agent_stream_with_artifacts(question: str, session_state: dict, result: _StreamResult,
                                     session_key: str = "ai_agent_chat", provider: str | None = None,
                                     chart_image_b64: str | None = None,
                                     chart_gallery: dict[str, str] | None = None,
                                     page_context: str | None = None) -> Iterator[str]:
    """Run the agent and stream its response."""
    with trace(
        name="AI Student Risk Assistant",
        run_type="chain",
        inputs={"question": question}
    ) as trace_run:
        full_answer = []
        try:
            for chunk in _run_agent_stream_impl(
                question,
                session_state,
                result,
                session_key,
                provider,
                chart_image_b64=chart_image_b64,
                chart_gallery=chart_gallery,
                page_context=page_context,
            ):
                full_answer.append(chunk)
                yield chunk
            trace_run.end(outputs={"answer": "".join(full_answer), "tools_used": result.tool_calls_made})
        except Exception as e:
            trace_run.end(outputs={"error": str(e)})
            raise