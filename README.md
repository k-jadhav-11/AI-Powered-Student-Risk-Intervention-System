# Sprint 5 — Intervention Agent

Part of the **At-Risk Student Prediction & Intervention Platform** capstone project. Sprint 5 adds a decision-and-action layer on top of Sprint 4's explanation layer, so a flagged student doesn't just get explained — they get routed, a message gets drafted, and the decision gets logged, automatically and consistently, whether triggered by a teacher, a cohort batch run, a chat message, or an external system over REST.

## What Sprint 5 delivers

- A LangGraph state machine (`classify_severity`) that routes every student to exactly one of three branches — **nudge**, **resource**, or **escalate** — using real router confidence (distance from the dataset's actual decision threshold) and a compounding-risk check across academic, engagement, and contextual factors together
- The branch decision itself is fully deterministic — no LLM is ever called to decide nudge vs. resource vs. escalate, only to draft the message afterward
- A two-agent CrewAI crew (Risk Analyst + Intervention Writer) drafts the nudge/resource message, reusing Sprint 4's own LLM provider config — with the same deterministic **template fallback** discipline Sprint 4 established, so a decision is always produced even with zero LLM key configured
- `escalate_node` never drafts a message at all and never calls an LLM — a compounding-risk case is handed to a human mentor, not an automated message pretending to be one
- Magnitude-weighted re-ranking of retrieved remedial resources, built on Sprint 4's existing vector store, so two students sharing the same top factor *names* don't always receive identical resources
- A persistent audit trail (PostgreSQL, with an automatic SQLite fallback) — one row per decision, from every entry point, queryable by dataset, decision, or student
- A FastAPI service exposing the identical pipeline over REST, including a stateful chat endpoint that remembers a conversation by `session_id` without the caller resending history
- Four Streamlit pages (Chat, Single Student, Cohort Batch Run, Teacher Dashboard), each with a collapsible, context-aware chat sub-agent that already knows which student or chart is on screen

## Package layout

```
orchestration/
├── graph.py              # LangGraph StateGraph: classify_severity + nudge/resource/escalate nodes
├── crew.py                # CrewAI two-agent drafting (Risk Analyst + Intervention Writer) + template fallback
├── audit.py                # Decision log: SQLAlchemy over Postgres/SQLite, one row per run
├── dashboard_charts.py      # Matplotlib figures for the Teacher Dashboard
└── supervisor.py            # Single-root-trace wrapper nesting Sprint 4's agent as a sub-agent of this system

ai_assistant/
└── rerank.py              # NEW this sprint — magnitude-weighted re-ranking over Sprint 4's vectorstore.py

backend/
├── main.py                 # FastAPI app
├── database.py              # DATABASE_URL -> Postgres; falls back to SQLite automatically if unset
├── models_db.py              # SQLAlchemy models: InterventionLog, ChatMessage
├── schemas.py                 # Pydantic request/response contracts
└── routers/
    └── agent.py                # /agent/run  /batch  /audit-log  /chat  /chat/history
```

`streamlit_app.py` registers the four pages above under `SPRINT5_PAGES` ("Intervention Agent"), alongside the Sprint 2–4 modules already in the sidebar. `tests/test_orchestration/` has one test file per module above, plus `test_rerank.py` covering the re-ranking logic and a precision@k / MRR evaluation harness (`scripts/eval_retrieval.py`) against a hand-labeled retrieval set.

## How it's grounded (no hallucination by design, extended to routing)

Every `classify_severity` decision reports the real `risk_score`, the real dataset decision threshold, and the `router_confidence` that resulted from comparing them — never a qualitative verdict without the number behind it. `crew.py`'s drafted message is always built from `resources_used` that `graph.py` actually retrieved; the crew is never allowed to reference a resource it wasn't handed, same discipline Sprint 4's `chains.py` established for explanations. Nothing in Sprint 3's models or Sprint 4's tools was modified to build any of this — `orchestration/` and `backend/` only read their output.

## Setup

```bash
pip install -r requirements.txt
```

In `.env`, in addition to Sprint 4's `LLM_PROVIDER` / `OPENAI_API_KEY` / `GEMINI_API_KEY`:

```
DATABASE_URL=postgresql+psycopg2://user:pass@host:5432/dbname   # optional
```

Omit it and the audit trail and chat history both run on an automatic SQLite fallback — nothing hard-fails on a missing connection string.

## Running

```bash
streamlit run streamlit_app.py
```

Select **Sprint 5 — Intervention Agent** from the sidebar module dropdown.

```bash
uvicorn backend.main:app --reload --port 8000
```

Interactive API docs at `http://127.0.0.1:8000/docs`.

## Testing

```bash
pytest tests/test_orchestration/
```

Covers all three routing branches, the compounding-risk trigger, the model-disagreement escalation path, the CrewAI template fallback, audit log read/write on both storage backends, and the re-ranking logic in `test_rerank.py` — including a `precision@k` / MRR check against `data/eval/retrieval_eval.json` run as a real assertion, not just observed by eye.


