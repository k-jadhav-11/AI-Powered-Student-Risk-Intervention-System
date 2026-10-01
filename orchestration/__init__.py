# orchestration/__init__.py
"""
Sprint 5 -- the Intervention Agent's orchestration layer (LangGraph +
CrewAI), sitting entirely above the unchanged Sprint 3/4 artifacts
(ai_assistant/, src/risk/, src/explainability.py, src/gru_explainability.py).

graph.py    -- the 3-branch conditional LangGraph (nudge / resource / escalate)
crew.py     -- the two-agent CrewAI drafting layer (Risk Analyst + Intervention Writer)
audit.py    -- one row per decision, to PostgreSQL (falls back to
               logs/interventions.jsonl if the database is unreachable)
dashboard_charts.py -- matplotlib charts for the Streamlit Teacher Dashboard

Nothing in this package modifies models/, data/processed/, or
ai_assistant/tools.py -- Section 4's "read-only from this sprint's
perspective" boundary.
"""
