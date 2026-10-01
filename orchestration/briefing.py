# orchestration/briefing.py
"""
Mentor briefing for escalated students.

The escalate branch deliberately sends NO message to the student, but the
human who picks the case up still needs everything the agent knows. This
module turns the structured RiskExplanation into a readable markdown brief:
why it was escalated, the persona, every factor pushing risk up or down,
conversation starters, and the knowledge-base support options.

Fully deterministic (no LLM): every line comes from the data, so the brief
can never invent a fact, and it works even when no LLM key is configured.
"""
from __future__ import annotations

from ai_assistant import feature_metadata

_CATEGORY_TITLES = {
    "academic": "Academic",
    "engagement": "Engagement",
    "contextual": "Background / context",
}
_BUCKETS = (
    ("academic", "academic_factors"),
    ("engagement", "engagement_factors"),
    ("contextual", "contextual_factors"),
)


def _fmt_value(dataset: str, feature: str, value) -> str:
    """Readable value: yes/no for binary features, trimmed number otherwise."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if feature_metadata.is_binary_feature(dataset, feature):
        return "yes" if number >= 0.5 else "no"
    return f"{number:g}"


def _factor_line(dataset: str, factor: dict) -> str:
    desc = feature_metadata.get_description(dataset, factor["feature"])
    desc = desc[:1].upper() + desc[1:]
    value = _fmt_value(dataset, factor["feature"], factor.get("value"))
    return f"- **{desc}**: {value} _(impact {factor['contribution']:+.2f})_"


def _factors_by_direction(r: dict, direction: str) -> dict[str, list[dict]]:
    """{category: [factors]} for one direction, strongest first."""
    grouped: dict[str, list[dict]] = {}
    for category, key in _BUCKETS:
        items = [f for f in (r.get(key) or []) if f.get("direction") == direction]
        items.sort(key=lambda f: abs(f["contribution"]), reverse=True)
        if items:
            grouped[category] = items
    return grouped


def _reason_guidance(reason: str | None) -> str:
    text = (reason or "").lower()
    if "disagree" in text:
        return (
            "The time-series (GRU) model and the snapshot (XGBoost) model give different "
            "readings for this student. Check the weekly engagement trajectory before "
            "deciding how urgent this is."
        )
    if "borderline" in text:
        return (
            "The score sits close to the model's decision boundary, so this is a judgement "
            "call rather than a clear-cut flag. Verify against recent grades and attendance."
        )
    return (
        "Risk-increasing factors appear in several categories at once (academic, "
        "engagement and background), which is why an automated message was not sent."
    )


def _conversation_starters(up: dict[str, list[dict]]) -> list[str]:
    starters = []
    if "academic" in up:
        starters.append(
            "Ask how the coursework feels right now, and which assessments or topics are hardest."
        )
    if "engagement" in up:
        starters.append(
            "Ask what has made it hard to stay engaged (time, access, workload, motivation) "
            "and listen before suggesting fixes."
        )
    if "contextual" in up:
        starters.append(
            "Background factors are correlations in the data, not causes -- don't raise them "
            "directly with the student; use them only to shape how supportive the check-in is."
        )
    starters.append("Agree one concrete next step and a follow-up date.")
    return starters


def build_mentor_brief(
    r: dict,
    escalation_reason: str | None,
    router_confidence: float | None = None,
) -> str:
    """Build the markdown brief shown to the human reviewer."""
    dataset = r["dataset"]
    student = r.get("display_student_id") or r.get("student_id", "?")
    up = _factors_by_direction(r, "increases_risk")
    down = _factors_by_direction(r, "decreases_risk")

    lines = [
        "**Case summary**",
        f"Student **{student}** ({dataset.upper()}) is rated **{r['risk_level']}** risk "
        f"(score {r['risk_score']:.3f})"
        + (f", router confidence {router_confidence:.2f}." if router_confidence is not None else "."),
        f"**Why this was escalated:** {escalation_reason or 'not recorded'}.",
        _reason_guidance(escalation_reason),
        "_No automated message was sent to the student._",
    ]

    persona = r.get("persona_name")
    if persona:
        avg = r.get("persona_avg_risk_score")
        count = r.get("persona_student_count")
        persona_line = f"**Persona:** {persona}"
        if count:
            persona_line += f" -- {count} students share this profile"
        if avg is not None:
            relation = "above" if r["risk_score"] > avg else "below"
            persona_line += (
                f"; their average risk score is {avg:.3f}, so this student is {relation} it"
            )
        lines += ["", persona_line + "."]

    lines += ["", "**What is pushing risk up**"]
    if up:
        for category, factors in up.items():
            lines.append(f"_{_CATEGORY_TITLES[category]}_")
            lines += [_factor_line(dataset, f) for f in factors]
    else:
        lines.append("- No risk-increasing factors were returned.")

    if down:
        lines += ["", "**What is protecting this student**"]
        strongest = sorted(
            (f for factors in down.values() for f in factors),
            key=lambda f: abs(f["contribution"]),
            reverse=True,
        )[:3]
        lines += [_factor_line(dataset, f) for f in strongest]

    lines += ["", "**Suggested conversation starters**"]
    lines += [f"- {s}" for s in _conversation_starters(up)]

    strategies = [s for s in (r.get("mitigation_strategies") or []) if s][:3]
    if strategies:
        lines += ["", "**Support options from the knowledge base**"]
        lines += [f"- {s}" for s in strategies]

    return "\n".join(lines)
