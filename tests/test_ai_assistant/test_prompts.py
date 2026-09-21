# tests/test_ai_assistant/test_prompts.py
"""ai_assistant/prompts.py -- the grounding rules the LLM is instructed to follow,
and the user-prompt builder that wraps a question with the tool's grounded
output. No LLM call here; this just checks the prompt text itself."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai_assistant import prompts, tools


def test_system_prompt_forbids_inventing_facts():
    """The core grounding rule must actually be present in the system
    prompt text, not just asserted in a docstring somewhere."""
    assert "never invent" in prompts.RISK_EXPLANATION_SYSTEM_PROMPT.lower()
    assert "never invent" in prompts.COHORT_SYSTEM_PROMPT.lower()


def test_build_explanation_user_prompt_includes_question_and_tool_result():
    result = tools.get_risk_explanation("uci", "0")
    prompt = prompts.build_explanation_user_prompt("Why is this student at risk?", result)
    assert "Why is this student at risk?" in prompt
    assert str(result["risk_score"]) in prompt


def test_example_tool_result_matches_schema_shape():
    """The few-shot example in the prompt module must have the same keys a
    real tool result has, or the example would teach the LLM a shape that
    doesn't match reality."""
    real_result = tools.get_risk_explanation("uci", "0")
    assert set(prompts.EXAMPLE_TOOL_RESULT.keys()) == set(real_result.keys())
