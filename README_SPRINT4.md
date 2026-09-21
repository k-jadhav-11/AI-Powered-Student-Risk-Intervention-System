# Sprint 4 — Risk Explanation (AI Assistant)

Part of the **At-Risk Student Prediction & Intervention Platform** capstone project. Sprint 4 adds a natural-language explanation layer on top of Sprint 3's risk models (XGBoost/CatBoost + GRU), so a teacher or admin can ask plain-English questions about a student's or cohort's risk and get a grounded, factual answer — with the underlying SHAP factors and model output always in the loop, never guessed at by the LLM.

## What Sprint 4 delivers

- Function-calling agent (LangChain + OpenAI/Gemini) that answers questions about individual students, cohorts, and GRU/XGBoost engagement trajectories
- Every answer is grounded in real tool output (risk scores, SHAP factors, persona context) — the LLM explains and phrases, it does not invent numbers
- A deterministic **template fallback** for every explanation path, so the app still returns a complete, correct answer with zero LLM key configured
- Retrieval-augmented remedial-content suggestions (hybrid exact tag-match + similarity search)
- Per-session conversation memory
- A Streamlit "AI Assistant" page wired into the existing Sprint 2/3 app

## Package layout

```
ai_assistant/
├── schemas.py       # Pydantic contracts: RiskExplanation, GRUTimeSeriesContext, CohortQuestion
├── llm_client.py     # Provider config (OpenAI / Gemini), is_configured() guard
├── prompts.py        # System prompts for the agent / explanation chains
├── tools.py          # get_risk_explanation, get_cohort_summary, get_gru_timeseries_context, ...
├── chains.py          # LangChain explanation chains + deterministic template fallback
├── memory.py          # Per-session chat history
├── rag.py             # Hybrid retrieval: exact factor-tag match + similarity search
└── vectorstore.py     # STARTER_REMEDIAL_CONTENT knowledge base + embedding pipeline
```

`streamlit_app.py` registers these under `SPRINT4_PAGES` ("AI Assistant"), alongside the Sprint 2 and Sprint 3 modules already in the sidebar. `tests/test_ai_assistant/` has one test file per module above.

## How it's grounded (no hallucination by design)

Every user-facing explanation traces back to a `RiskExplanation` object that Sprint 3's actual model output populated — `tools.py` is the only place that touches the trained models / SHAP values / persona clusters. The LLM (`chains.py`) is only ever asked to phrase what's already in that object; it never computes a risk score, a factor, or a persona itself. If no `OPENAI_API_KEY` / `GEMINI_API_KEY` is configured, `chains.py`'s template path renders the same `RiskExplanation` fields directly into a fixed sentence structure — same facts, no LLM involved, so the feature never silently degrades to "unavailable."

## Retrieval (RAG)

`rag.py` retrieves remedial-content suggestions relevant to a student's actual top risk factors:

1. **Exact factor-tag match** against `vectorstore.STARTER_REMEDIAL_CONTENT` — fast, precise when a factor's tag lines up with a remedy.
2. **Similarity search** (Chroma) fills in the remainder when exact matches run short.

`vectorstore.py`'s embedding function tries real `sentence-transformers` first and falls back to a small deterministic hash embedding when the model can't be downloaded (no network, first run before caching) — clearly labeled in the return value so nothing pretends an offline hash vector is a semantic embedding.

## Setup

```bash
pip install -r requirements.txt
```

In `.env`:

```
LLM_PROVIDER=openai        # or "gemini"
OPENAI_API_KEY=...         # if using OpenAI
GEMINI_API_KEY=...         # if using Gemini
```

Both are optional — omit them and the app runs entirely on the template fallback path described above.

## Running

```bash
streamlit run streamlit_app.py
```

Select **Sprint 4 — Risk Explanation (AI Assistant)** from the sidebar module dropdown, pick a dataset (`uci` or `oulad`) and a student ID, and ask a question.

## Testing

```bash
pytest tests/test_ai_assistant/
```

Tests run against real Sprint 3 model output where possible (no mocked risk scores), and cover the no-LLM-key template path explicitly, so the fallback is a tested guarantee, not an assumption.

## What's next (Sprint 5)

Sprint 5 builds a LangGraph intervention agent (nudge / resource / escalate) on top of this explanation layer, with CrewAI-drafted messages and an expanded, re-ranked version of the RAG library already started here.
