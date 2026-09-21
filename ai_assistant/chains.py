
from typing import Iterator
from langsmith import traceable
from . import llm_client, prompts

# ============================================================
# Risk Explanation Chain
# ============================================================

@traceable(name="explain_risk_chain", run_type="chain")
def explain_risk_chain(question: str, tool_result: dict) -> dict:
    """
    Generate a plain-language explanation of a student's risk prediction.
    Uses the LLM when available and a template as a fallback.
    """

    # Get the system instructions and user prompt
    system_prompt = prompts.RISK_EXPLANATION_SYSTEM_PROMPT
    user_prompt = prompts.build_explanation_user_prompt(question, tool_result)

    # Try to generate the explanation using the configured LLM
    if llm_client.is_configured():
        try:
            answer = llm_client.complete(system_prompt, user_prompt)

            return {
                "answer": answer,
                "source": "llm",
                "provider": llm_client.get_llm_config()["provider"]
            }

        # Use the template fallback if the LLM call fails
        except llm_client.LLMError:
            pass

    # Return a deterministic explanation without using the LLM
    return {
        "answer": _template_explanation(tool_result),
        "source": "template",
        "provider": None
    }


# ============================================================
# Risk Explanation Streaming Chain
# ============================================================

@traceable(name="explain_risk_chain_stream", run_type="chain")
def explain_risk_chain_stream(
    question: str,
    tool_result: dict,
    status: dict | None = None
) -> Iterator[str]:
    """
    Stream the student's risk explanation as it is generated.
    Falls back to a template if the LLM is unavailable or fails.
    """

    # Create a status dictionary if one was not provided
    status = status if status is not None else {}

    # Stream the explanation using the configured LLM
    if llm_client.is_configured():
        try:
            yield from llm_client.stream_complete(
                prompts.RISK_EXPLANATION_SYSTEM_PROMPT,
                prompts.build_explanation_user_prompt(question, tool_result),
            )

            # Record that the response came from the LLM
            status["source"] = "llm"
            return

        # Use the template if streaming fails
        except llm_client.LLMError:
            pass

    # Record and return the template-based response
    status["source"] = "template"
    yield _template_explanation(tool_result)


# ============================================================
# Cohort Explanation Chain
# ============================================================

@traceable(name="explain_cohort_chain", run_type="chain")
def explain_cohort_chain(question: str, tool_result: dict) -> dict:
    """
    Generate a plain-language explanation of cohort-level risk information.
    Uses the LLM when available and a template as a fallback.
    """

    # Get the cohort-specific system instructions and user prompt
    system_prompt = prompts.COHORT_SYSTEM_PROMPT
    user_prompt = prompts.build_explanation_user_prompt(question, tool_result)

    # Try to generate the explanation using the configured LLM
    if llm_client.is_configured():
        try:
            answer = llm_client.complete(system_prompt, user_prompt)

            return {
                "answer": answer,
                "source": "llm",
                "provider": llm_client.get_llm_config()["provider"]
            }

        # Use the template if the LLM call fails
        except llm_client.LLMError:
            pass

    # Return the deterministic cohort summary
    return {
        "answer": _template_cohort_summary(tool_result),
        "source": "template",
        "provider": None
    }


# ============================================================
# Cohort Explanation Streaming Chain
# ============================================================

@traceable(name="explain_cohort_chain_stream", run_type="chain")
def explain_cohort_chain_stream(
    question: str,
    tool_result: dict,
    status: dict | None = None
) -> Iterator[str]:
    """
    Stream the cohort risk explanation to the chat interface.
    Falls back to a template if the LLM call fails.
    """

    # Create a status dictionary if needed
    status = status if status is not None else {}

    # Stream the response using the configured LLM
    if llm_client.is_configured():
        try:
            yield from llm_client.stream_complete(
                prompts.COHORT_SYSTEM_PROMPT,
                prompts.build_explanation_user_prompt(question, tool_result),
            )

            # Record the response source
            status["source"] = "llm"
            return

        # Fall back to the template response
        except llm_client.LLMError:
            pass

    # Return the deterministic cohort summary
    status["source"] = "template"
    yield _template_cohort_summary(tool_result)


# ============================================================
# GRU Time-Series Explanation Chain
# ============================================================

@traceable(name="explain_gru_timeseries_chain", run_type="chain")
def explain_gru_timeseries_chain(question: str, tool_result: dict) -> dict:
    """
    Generate a plain-language explanation of GRU time-series results.
    Uses the LLM when available and a template as a fallback.
    """

    # Get the GRU-specific system instructions and user prompt
    system_prompt = prompts.GRU_TIMESERIES_SYSTEM_PROMPT
    user_prompt = prompts.build_explanation_user_prompt(question, tool_result)

    # Try to generate the explanation using the configured LLM
    if llm_client.is_configured():
        try:
            answer = llm_client.complete(system_prompt, user_prompt)

            return {
                "answer": answer,
                "source": "llm",
                "provider": llm_client.get_llm_config()["provider"]
            }

        # Use the template if the LLM call fails
        except llm_client.LLMError:
            pass

    # Return the deterministic GRU explanation
    return {
        "answer": _template_gru_timeseries(tool_result),
        "source": "template",
        "provider": None
    }


# ============================================================
# GRU Time-Series Streaming Chain
# ============================================================

@traceable(name="explain_gru_timeseries_chain_stream", run_type="chain")
def explain_gru_timeseries_chain_stream(
    question: str,
    tool_result: dict,
    status: dict | None = None
) -> Iterator[str]:
    """
    Stream the GRU time-series explanation to the chat interface.
    Falls back to a template if the LLM call fails.
    """

    # Create a status dictionary if needed
    status = status if status is not None else {}

    # Stream the GRU explanation using the configured LLM
    if llm_client.is_configured():
        try:
            yield from llm_client.stream_complete(
                prompts.GRU_TIMESERIES_SYSTEM_PROMPT,
                prompts.build_explanation_user_prompt(question, tool_result),
            )

            # Record the response source
            status["source"] = "llm"
            return

        # Fall back to the template response
        except llm_client.LLMError:
            pass

    # Return the deterministic GRU explanation
    status["source"] = "template"
    yield _template_gru_timeseries(tool_result)


# ============================================================
# Template Fallback Functions
# ============================================================

def _template_explanation(r: dict) -> str:
    """
    Create a deterministic explanation from the student's risk results.
    Uses model evidence, risk factors, persona information, and recommendations.
    """

    # Format individual risk factors as bullet points
    def _factor_lines(factors: list[dict]) -> str:
        return "\n".join(
            f"  - {f['explanation']} (contribution: {f['contribution']:+.3f})"
            for f in factors
        )

    # Get different categories of risk factors
    academic = r.get("academic_factors") or []
    engagement = r.get("engagement_factors") or []
    contextual = r.get("contextual_factors") or []

    # Build the factor sections
    sections = []

    if academic:
        sections.append(
            "Academic factors:\n" + _factor_lines(academic)
        )

    if engagement:
        sections.append(
            "Engagement/behavioral factors:\n" + _factor_lines(engagement)
        )

    if contextual:
        sections.append(
            "Background/contextual factors:\n" + _factor_lines(contextual)
        )

    # Combine all factor sections
    factors_block = (
        "\n\n".join(sections)
        if sections
        else "  (no factors returned)"
    )

    # Get model performance metrics
    m = r["model_metrics"]

    metrics_line = (
        f"precision={m['precision']}, recall={m['recall']}, F1={m['f1']}"
        + (
            f", ROC-AUC={m['roc_auc']}"
            if m.get("roc_auc") is not None
            else ""
        )
    )

    # Add persona information when available
    persona_line = (
        f"\n\n**Persona context:** this student's profile matches the "
        f"**{r['persona_name']}** persona "
        f"({r['persona_student_count']} students), whose average risk score is "
        f"{r['persona_avg_risk_score']:.3f} -- "
        + (
            "close to typical for that group."
            if abs(
                r['risk_score'] - r['persona_avg_risk_score']
            ) < 0.15
            else "notably different from this student's own score."
        )
    ) if r.get("persona_name") else ""

    # Add recommended actions when available
    remedial_line = ""

    if r.get("mitigation_strategies"):
        bullets = "\n".join(
            f"  - {s} (source: {src})"
            for s, src in zip(
                r["mitigation_strategies"],
                r.get("evidence_sources")
                or [""] * len(r["mitigation_strategies"])
            )
        )

        remedial_line = (
            f"\n\n**Recommended actions** "
            f"(from the intervention knowledge base):\n{bullets}"
        )

    # Build the final explanation
    return (
        f"## Model Evidence\n"
        f"Student {r['student_id']} ({r['dataset'].upper()}) is flagged "
        f"**{r['risk_level']} risk** "
        f"(score {r['risk_score']:.3f}, predicted label: "
        f"{r['predicted_label']}), from {r['model_name']} "
        f"({metrics_line} on held-out test data).\n\n"
        + (f"**ID:** {r['display_student_id']}\n\n" if r.get("display_student_id") else "")   # <-- new
        + f"## Interpretation\n"

        f"{factors_block}\n\n"

        f"These factors describe what the model weighted in this prediction, "
        f"not a proven cause of the student's situation."
        f"{persona_line}"
        f"{remedial_line}"
    )


# ============================================================
# Cohort Summary Template
# ============================================================

def _template_cohort_summary(c: dict) -> str:
    """
    Create a deterministic summary of cohort-level risk information.
    Shows student counts, risk levels, and the highest-risk students.
    """

    # Format the number of students in each risk level
    counts_line = ", ".join(
        f"{level}: {n}"
        for level, n in c["risk_level_counts"].items()
    )

    # Show the applied risk filter when available
    filter_line = (
        f" filtered to {c['filter_applied']} risk"
        if c["filter_applied"]
        else ""
    )

    # Format the highest-risk student IDs and scores
    top_line = ", ".join(
        f"#{sid} ({score:.3f})"
        for sid, score in zip(
            c["top_student_ids"],
            c["top_risk_scores"]
        )
    )

    # Return the final cohort summary
    return (
        f"{c['dataset'].upper()} cohort: "
        f"{c['total_students']} students total ({counts_line}).\n\n"
        f"Top{filter_line} students by risk score: {top_line}"
    )


# ============================================================
# GRU Time-Series Template
# ============================================================

def _template_gru_timeseries(t: dict) -> str:
    """
    Create a deterministic summary of GRU and XGBoost predictions.
    Includes model agreement, risk scores, active week, and actual outcome.
    """

    # Check whether both models produced the same risk assessment
    agree_line = (
        "The two models agree"
        if t["models_agree"]
        else "The two models **disagree**"
    )

    # Add early-warning information when available
    lead_line = (
        f" An illustrative early-warning flag first appeared around week "
        f"{t['first_high_risk_week']} "
        "(not a validated lead-time estimate -- see the Student "
        "Time-Series Risk Explorer page for that caveat)."
        if t.get("first_high_risk_week") is not None
        else ""
    )

    # Add the actual recorded outcome when available
    actual_line = (
        f" Actual outcome on record: **{t['actual_final_result']}**."
        if t.get("actual_final_result")
        else ""
    )

    # Return the final GRU time-series summary
    return (
        f"Real OULAD student {t['student_id']}: "
        f"GRU (sequence model) reads "
        f"**{t['gru_risk_level']} risk** "
        f"(score {t['gru_risk_score']:.3f}); "
        f"XGBoost (tabular model) reads "
        f"**{t['xgboost_risk_level']} risk** "
        f"(score {t['xgboost_risk_score']:.3f}). "
        f"{agree_line} on this student. "
        f"Last recorded active week: {t['last_active_week']}."
        f"{lead_line}{actual_line}"
    )