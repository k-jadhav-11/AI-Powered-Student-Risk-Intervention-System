# orchestration/crew.py
"""
CrewAI drafting layer for intervention messages.

Pipeline for nudge / resource branches:
    Risk Analyst -> Intervention Writer -> Message Reviewer
followed by a deterministic guard. Falls back to a deterministic, grounded
template when CrewAI is unavailable, fails, or the guard rejects the draft.

Student-facing text only ever sees "student-safe facts": actionable academic
and engagement factors in plain language. Raw numbers, SHAP wording, persona
labels and background/demographic factors (family, region, gender, disability,
alcohol use...) are never given to the writer or used in the template --
those belong in the mentor briefing (orchestration/briefing.py) instead.
"""
from __future__ import annotations

import logging
import re
import zlib

from langsmith import traceable

from ai_assistant import llm_client

logger = logging.getLogger(__name__)

RISK_ANALYST_GOAL = (
    "Summarize, in plain language, which areas of this student's course work may need "
    "attention and which are going well, using only the facts provided. Never invent, "
    "estimate, or add unsupported information."
)
RISK_ANALYST_BACKSTORY = (
    "You read a short list of pre-approved, student-safe facts. You report only what "
    "that list establishes and never claim that a factor caused the student's outcome."
)

WRITER_GOAL = (
    "Turn the analyst's summary into a concise, supportive student-facing message. "
    "For a nudge, encourage one practical next step; for a resource intervention, "
    "connect the student with the supplied resources."
)
WRITER_BACKSTORY = (
    "You write clear, respectful, encouraging messages for students. Use only the "
    "supplied summary and resources. Never invent resources, statistics, causes, or "
    "student details."
)

REVIEWER_GOAL = (
    "Check the drafted student message against the source facts and the supplied "
    "resources. Remove or fix anything unsupported, alarming, blaming, or that exposes "
    "internal scoring, and return only the final message text."
)
REVIEWER_BACKSTORY = (
    "You are a careful quality reviewer for student-facing communication. You compare "
    "a draft with the source facts line by line, you never add new facts, and you never "
    "add resources that were not supplied."
)

# ============================================================
# Student-safe facts
# ============================================================

# Only actionable academic/engagement features get a student-facing phrase.
# Anything not listed here (demographics, family, region, subject/module,
# alcohol use, ...) is never shown to the student.
_STUDENT_PHRASES = {
    # UCI
    "g1": "your first-period grade",
    "g2": "your second-period grade",
    "failures": "how earlier classes have gone",
    "studytime": "weekly study time",
    "absences": "missed classes",
    # OULAD
    "pct_assessments_submitted": "keeping up with assessment submissions",
    "total_clicks_sum": "time spent on the course platform",
    "total_clicks_mean": "time spent on the course platform",
    "total_clicks_max": "time spent on the course platform",
    "days_active_sum": "how regularly you log in to the course",
    "days_active_mean": "how regularly you log in to the course",
    "num_weeks_active": "staying active week to week",
    "engagement_trend": "your recent activity on the course",
    "num_of_prev_attempts": "earlier attempts at this course",
    "studied_credits": "your overall workload",
}

# Positive framing when a factor is protecting the student (e.g. few absences
# -> "your attendance", not "missed classes"). Falls back to the concern phrase.
_STRENGTH_PHRASES = {
    "failures": "your track record in earlier classes",
    "studytime": "your study habits",
    "absences": "your attendance",
    "pct_assessments_submitted": "your assessment submissions",
    "total_clicks_sum": "your time on the course platform",
    "total_clicks_mean": "your time on the course platform",
    "total_clicks_max": "your time on the course platform",
    "days_active_sum": "how regularly you log in",
    "days_active_mean": "how regularly you log in",
    "engagement_trend": "your recent activity on the course",
    "num_of_prev_attempts": "your track record with this course",
}


def _student_safe_facts(r: dict, max_concerns: int = 3, max_strengths: int = 2) -> dict:
    """Plain-language areas needing attention vs. going well, strongest first."""
    factors = []
    for bucket in ("academic_factors", "engagement_factors"):  # contextual is excluded on purpose
        factors += list(r.get(bucket) or [])
    factors.sort(key=lambda f: abs(f["contribution"]), reverse=True)

    concerns, strengths = [], []
    for f in factors:
        phrase = _STUDENT_PHRASES.get(f["feature"])
        if not phrase:
            continue
        is_concern = f["direction"] == "increases_risk"
        if not is_concern:
            phrase = _STRENGTH_PHRASES.get(f["feature"], phrase)
        target = concerns if is_concern else strengths
        if phrase not in target:
            target.append(phrase)
    return {"concerns": concerns[:max_concerns], "strengths": strengths[:max_strengths]}

# A concrete, concern-specific next step, written as a topic so it reads
# naturally in every frame ("...to talk through {action}" / "...chat about {action}").
_CONCERN_ACTIONS = {
    "your first-period grade": "which parts of your early work could be the quickest wins",
    "your second-period grade": "which parts of your recent work could be the quickest wins",
    "how earlier classes have gone": "which topics from earlier classes are still tricky",
    "weekly study time": "a small, realistic weekly study routine that fits your schedule",
    "missed classes": "what has been getting in the way of attending",
    "keeping up with assessment submissions": "a plan for the next few deadlines so none sneak up on you",
    "time spent on the course platform": "a short, fixed time each week to log in and look around",
    "how regularly you log in to the course": "a short, fixed time each week to log in",
    "staying active week to week": "a simple weekly rhythm for the course",
    "your recent activity on the course": "a gentle plan for getting back into the course material",
    "earlier attempts at this course": "what could make this attempt feel different",
    "your overall workload": "how to balance your workload so it feels more manageable",
}
_GENERIC_ACTION = "what would help most right now"


def _next_step_for(facts: dict) -> str:
    """The concrete next step matching the strongest concern."""
    concerns = facts["concerns"]
    return _CONCERN_ACTIONS.get(concerns[0], _GENERIC_ACTION) if concerns else _GENERIC_ACTION


def _joined(phrases: list[str]) -> str:
    if len(phrases) <= 1:
        return "".join(phrases)
    return ", ".join(phrases[:-1]) + " and " + phrases[-1]


def _safe_facts_text(facts: dict) -> str:
    def block(items):
        return "\n".join(f"- {p}" for p in items) if items else "- (none identified)"
    return (
        f"Areas that may need attention:\n{block(facts['concerns'])}\n\n"
        f"Positive signs:\n{block(facts['strengths'])}"
    )


def _student_advice(resource: str) -> str:
    """Knowledge-base entries read 'Diagnostic label: advice'. The label is
    written for staff (and can contain internal names like 'g1/g2'), so a
    student only gets the advice part."""
    _, sep, advice = resource.partition(": ")
    text = advice.strip() if sep and advice.strip() else resource.strip()
    return text[:1].upper() + text[1:]


# ============================================================
# Agents and tasks
# ============================================================

def _crewai_llm():
    """Build the LLM object CrewAI itself expects, from the same .env settings
    as ai_assistant.llm_client.

    llm_client.get_chat_model() returns a LangChain model, which is right for
    the rest of the app but is rejected by current CrewAI versions (Agent(llm=...)
    only accepts a model-name string, a crewai.LLM, or a BaseLLM). CrewAI routes
    through its own provider layer, so the model name carries a provider prefix.
    """
    from crewai import LLM

    cfg = llm_client.get_llm_config()
    provider = cfg["provider"]
    if provider == "openai":
        if not cfg["openai_api_key"]:
            raise llm_client.LLMError("OPENAI_API_KEY is not set in .env")
        return LLM(model=f"openai/{cfg['openai_model']}", api_key=cfg["openai_api_key"],
                   temperature=0.3, timeout=45)
    if provider == "gemini":
        if not cfg["gemini_api_key"]:
            raise llm_client.LLMError("GEMINI_API_KEY is not set in .env")
        return LLM(model=f"gemini/{cfg['gemini_model']}", api_key=cfg["gemini_api_key"],
                   temperature=0.3, timeout=45)
    raise llm_client.LLMError(f"Unknown provider: {provider!r} (expected 'openai' or 'gemini')")


def _build_agents():
    """Build the Risk Analyst, Intervention Writer and Message Reviewer agents."""
    from crewai import Agent

    crew_llm = _crewai_llm()

    def make(role, goal, backstory):
        return Agent(role=role, goal=goal, backstory=backstory, llm=crew_llm, verbose=False)

    return (
        make("Risk Analyst", RISK_ANALYST_GOAL, RISK_ANALYST_BACKSTORY),
        make("Intervention Writer", WRITER_GOAL, WRITER_BACKSTORY),
        make("Message Reviewer", REVIEWER_GOAL, REVIEWER_BACKSTORY),
    )


def _build_tasks(risk_analyst, intervention_writer, message_reviewer,
                 r: dict, branch: str, resources: list[str] | None):
    """Build the analysis, writing and review tasks."""
    from crewai import Task

    facts = _student_safe_facts(r)
    facts_text = _safe_facts_text(facts)

    analyst_task = Task(
        description=(
            "A student is getting a supportive check-in. These are the ONLY facts you may use:\n\n"
            f"{facts_text}\n\n"
            "Write a 2-4 sentence plain-English summary of the areas that may need attention "
            "and, if any are listed, the positive signs. Do not add numbers, scores, causes, "
            "labels, or any detail that is not in the list above."
        ),
        expected_output="A 2-4 sentence plain-English summary grounded only in the listed facts.",
        agent=risk_analyst,
    )

    next_step = _next_step_for(facts)
    if branch == "nudge":
        writer_description = (
            "Using the analyst's summary, write a short, warm, low-pressure message directly "
            "to the student, 3-5 sentences. Offer ONE practical next step: a short chat about "
            f"{next_step}. Use your own natural wording and avoid stock openings. Do not "
            "mention scores, models, SHAP, predictions, risk levels or personas, and do not "
            "invent facts."
        )
    else:
        resources_block = "\n".join(f"- {res}" for res in (resources or []))
        writer_description = (
            "Using the analyst's summary and the supplied resources, write a short message "
            "directly to the student, 4-6 sentences.\n\n"
            f"Available resources:\n{resources_block}\n\n"
            "Explain why the resources may help, based only on the summary. Mention only the "
            "resources above and do not invent or recommend any others. Do not mention scores, "
            "models, SHAP, predictions, risk levels or personas. Keep the tone supportive."
        )
    writer_task = Task(
        description=writer_description,
        expected_output="A short, clear, supportive, ready-to-send message.",
        agent=intervention_writer,
        context=[analyst_task],
    )

    if branch == "nudge":
        facts_text += f"\n\nSUGGESTED NEXT STEP (allowed to be offered):\n- a short chat about {next_step}"

    allowed_resources = (
        "\n".join(f"- {res}" for res in (resources or []))
        or "(none -- this message must not mention any resource)"
    )
    reviewer_task = Task(
        description=(
            "Review the Intervention Writer's draft against these source facts.\n\n"
            f"SOURCE FACTS:\n{facts_text}\n\n"
            f"ALLOWED RESOURCES:\n{allowed_resources}\n\n"
            "Rules:\n"
            "1. Every statement about the student's situation must be supported by the source facts.\n"
            "2. Mention only the allowed resources -- no others.\n"
            "3. Never mention a risk score, SHAP, a model, a prediction, a persona, or label the "
            "student as 'at risk' / 'high risk'.\n"
            "4. Do not blame the student or claim a factor caused their results.\n"
            "5. Keep the tone warm and supportive, and keep the original length limits.\n\n"
            "If the draft already follows every rule, return it unchanged. Otherwise fix ONLY the "
            "offending parts. Return ONLY the final message text -- no notes, no preamble."
        ),
        expected_output="The final, ready-to-send student message and nothing else.",
        agent=message_reviewer,
        context=[analyst_task, writer_task],
    )
    return [analyst_task, writer_task, reviewer_task]


# ============================================================
# Deterministic message guard
# ============================================================

_FORBIDDEN_PATTERNS = [
    (r"\bshap\b", "mentions SHAP"),
    (r"\brisk score\b", "mentions the risk score"),
    (r"\b(?:at|high|medium|low)[- ]risk\b", "labels the student by risk level"),
    (r"\b(?:model|algorithm|prediction|predicted)\b", "mentions the model/prediction"),
    (r"\bpersona\b", "mentions the persona"),
    (r"\b[a-z]+_[a-z_]+\b", "leaks an internal label (snake_case)"),
    (r"\d+\.\d{2,}", "contains a score-like decimal number"),
    (r"[\[\]{}]|<[^>]+>", "contains template/markup leakage"),
]
_MAX_SENTENCES = {"nudge": 6, "resource": 8}  # prompts ask for 3-5 / 4-6; small slack allowed


def _validate_message(message: str, branch: str) -> list[str]:
    """Cheap, deterministic checks on the final message. Returns a list of
    problems (empty = passes). This is the hard guarantee; the Reviewer agent
    is the soft one."""
    if not message.strip():
        return ["empty output"]
    lowered = message.lower()
    problems = [label for pattern, label in _FORBIDDEN_PATTERNS if re.search(pattern, lowered)]
    if len(re.findall(r"[.!?]+(?:\s|$)", message)) > _MAX_SENTENCES[branch]:
        problems.append("too long for this branch")
    return problems


# ============================================================
# Draft intervention
# ============================================================

@traceable(name="CrewAI Drafting", run_type="chain")
def draft_intervention(risk_explanation: dict, branch: str, resources: list[str] | None) -> dict:
    """
    Draft an intervention using the CrewAI agents (analyst -> writer -> reviewer).
    Falls back to the deterministic template when CrewAI is unavailable, fails, or
    the guard rejects the message. The reason for any fallback is returned in
    "fallback_reason" and logged instead of being swallowed silently.
    """
    if branch not in ("nudge", "resource"):
        raise ValueError(f"draft_intervention() only handles 'nudge' or 'resource', got {branch!r}")

    resources = [_student_advice(res) for res in (resources or [])] or None

    fallback_reason = "LLM provider not configured"
    if llm_client.is_configured():
        try:
            from crewai import Crew

            analyst, writer, reviewer = _build_agents()
            tasks = _build_tasks(analyst, writer, reviewer, risk_explanation, branch, resources)
            result = Crew(agents=[analyst, writer, reviewer], tasks=tasks, verbose=False).kickoff()
            message = str(result).strip()
            problems = _validate_message(message, branch)
            if not problems:
                return {"message": message, "source": "crew", "fallback_reason": None}
            fallback_reason = "draft rejected by guard: " + "; ".join(problems)
        except Exception as e:
            fallback_reason = f"CrewAI error: {type(e).__name__}: {e}"

    logger.warning("Using template draft for %s branch -- %s", branch, fallback_reason)
    return {
        "message": _template_draft(risk_explanation, branch, resources),
        "source": "template",
        "fallback_reason": fallback_reason,
    }


# ============================================================
# Deterministic template fallback (stable per-student variation)
# ============================================================


_NUDGE_OPENERS = [
    "Hi there -- I wanted to check in.",
    "Hi -- hope your week is going okay.",
    "Hello -- just reaching out for a quick check-in.",
    "Hi there -- I've been thinking about how the course is going and wanted to say hello.",
]
_CONCERN_SENTENCES = [
    "{Focus} might be worth a closer look, and I'd love to help make it feel more manageable.",
    "It looks like {focus} could be where things have felt toughest lately.",
    "One area that might benefit from a little attention is {focus}.",
    "A few things around {focus} seem to be taking extra effort right now.",
]
_POSITIVE_SENTENCES = [
    "There are also good signs around {s}.",
    "I can also see some encouraging signs with {s}.",
    "On the bright side, there are positive signs in {s}.",
]
_ACTION_SENTENCES = [
    "Would you have 10 minutes this week to talk through {action}?",
    "If it helps, we could set aside a few minutes to talk through {action}.",
    "How about a short chat this week about {action}?",
]
_CLOSERS = [
    "No pressure -- I just want to make sure you have what you need.",
    "Totally fine if now isn't a good time; the offer stands.",
    "There's no rush at all, and I'm happy to work around your schedule.",
]
_CHECKIN_MESSAGES = [
    "Hi there -- just a quick check-in. {steady}{watch}, I'm happy to help. "
    "Would you like a short chat this week? No pressure at all.",
    "Hello -- a quick note to say hi. {steady}{watch}, just let me know and we can sort "
    "something out. A short chat is always an option.",
    "Hi -- checking in briefly. {steady}{watch}, I'm here to help. "
    "Feel free to reply whenever suits you.",
]
_RESOURCE_OPENERS = [
    "Hi there -- based on how the course is going{focus_clause}, here are a couple of things "
    "that have helped other students in a similar spot:",
    "Hello -- looking at how the course is going{focus_clause}, these are two things that "
    "tend to help:",
    "Hi -- I thought these might be useful{focus_clause}. They've helped other students "
    "in a similar spot:",
]
_RESOURCE_CLOSERS = [
    "Take a look and let me know which one sounds most useful -- happy to help you get started.",
    "Have a look when you get a moment, and tell me if you'd like a hand with either.",
    "Pick whichever feels easiest to try first -- I'm glad to help you get going.",
]


def _variation_seed(r: dict) -> int:
    """Stable across runs and processes (unlike Python's salted hash())."""
    key = f"{r.get('dataset', '')}:{r.get('display_student_id') or r.get('student_id', '')}"
    return zlib.crc32(key.encode("utf-8"))


def _pick(options: list[str], seed: int, slot: int) -> str:
    return options[(seed >> (slot * 4)) % len(options)]


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def _template_draft(r: dict, branch: str, resources: list[str] | None) -> str:
    """Grounded message built only from student-safe facts: no raw numbers, no
    'predicted risk' wording, no persona label, no background factors."""
    facts = _student_safe_facts(r)
    concerns, strengths = facts["concerns"], facts["strengths"]
    seed = _variation_seed(r)

    if branch == "nudge":
        if r.get("risk_level") == "Low" or not concerns:
            steady = (
                "Things look fairly steady at the moment, and I wanted to make sure you have "
                "everything you need. "
                if r.get("risk_level") == "Low"
                else "I wanted to see how the course is going and make sure you have "
                     "everything you need. "
            )
            watch = (
                f"If anything around {_joined(concerns[:2])} ever feels tricky" if concerns
                else "If anything about the course ever feels tricky"
            )
            return _pick(_CHECKIN_MESSAGES, seed, 0).format(steady=steady, watch=watch)

        focus = _joined(concerns[:2])
        parts = [
            _pick(_NUDGE_OPENERS, seed, 0),
            _pick(_CONCERN_SENTENCES, seed, 1).format(focus=focus, Focus=_cap(focus)),
        ]
        if strengths:
            parts.append(_pick(_POSITIVE_SENTENCES, seed, 2).format(s=_joined(strengths)))
        parts.append(_pick(_ACTION_SENTENCES, seed, 3).format(action=_next_step_for(facts)))
        parts.append(_pick(_CLOSERS, seed, 4))
        return " ".join(parts)

    # resource branch
    bullets = "\n".join(f"- {res}" for res in (resources or []))
    focus_clause = f" (particularly {_joined(concerns[:2])})" if concerns else ""
    return (
        _pick(_RESOURCE_OPENERS, seed, 0).format(focus_clause=focus_clause)
        + f"\n\n{bullets}\n\n"
        + _pick(_RESOURCE_CLOSERS, seed, 1)
    )