# AI-Powered Student Risk Intervention System

A student analytics application that combines academic reporting, student-risk analysis, an AI assistant, and explainable intervention workflows. It provides three user-facing interfaces backed by shared Python services:

- **Streamlit analytics application** for interactive reports, model exploration, an AI assistant, and intervention workflows.
- **HTML web console** served through nginx for browser-based reports and chat.
- **FastAPI** for programmatic access to analytics, risk insights, interventions, and chat.

The project is organized around **Sprint 2 through Sprint 5**. Each sprint adds a different layer: descriptive academic analytics, predictive risk analysis, conversational exploration, and actionable interventions with an audit trail.

> **Data and responsible use:** Model scores are decision-support signals, not diagnoses or guarantees about a student's outcome. Review explanations and source data before acting. Follow applicable privacy, consent, and data-retention requirements when using real student records.

## Contents

- [System overview](#system-overview)
- [Sprint-by-sprint guide](#sprint-by-sprint-guide)
- [Data and model artifacts](#data-and-model-artifacts)
- [Run locally](#run-locally)
- [Run with Docker Compose](#run-with-docker-compose)
- [Configuration](#configuration)
- [FastAPI reference](#fastapi-reference)
- [Tests](#tests)
- [Project layout](#project-layout)
- [Troubleshooting](#troubleshooting)

## System overview

```text
Browser
  ├── Streamlit :8502 ─────────┐
  ├── HTML console :8080 ─ nginx┼── FastAPI :8000
  └── API clients :8000 ───────┘       ├── MySQL: Sprint 2 marks and class data
                                       ├── PostgreSQL: intervention and chat audit
                                       │   (SQLite fallback outside Compose)
                                       ├── saved model and processed risk artifacts
                                       └── optional OpenAI / Gemini provider
```

The interfaces share application logic, but their feature surfaces are not identical. Streamlit is the most complete interactive analytics interface, including its report filters and model-exploration pages. The HTML console is a separate frontend and may not expose every Streamlit filter or visualization.

| Component | Responsibility |
| --- | --- |
| Streamlit (`streamlit_app.py`) | Sprint 2–5 interactive pages and charts |
| FastAPI (`backend/`) | JSON API for reports, risk analysis, intervention runs, audit data, and chat |
| HTML frontend (`frontend/`) | Browser console served by nginx |
| MySQL | Sprint 2 student, subject, and marks tables |
| PostgreSQL | Sprint 5 intervention audit and API chat history in Docker Compose |
| SQLite | Local fallback for intervention persistence if `DATABASE_URL` is not configured |
| `data/processed/`, `models/` | Saved feature tables, risk scores, personas, and model artifacts used by risk pages |

## Sprint-by-sprint guide

### Sprint 2 — Academic performance analytics

Sprint 2 provides descriptive reporting over student marks, subjects, and classes. The reports use the MySQL tables seeded from CSV files supplied under `data/`.

**Streamlit pages**

1. **Load / Seed Data** — initializes the MySQL schema and loads the student, subject, and marks CSVs from `data/`.
2. **Class Average Report** — class-level average performance, with report filters and visualizations.
3. **Top/Bottom Performers** — ranks students by performance.
4. **Weak Area Detection** — highlights subjects or areas with lower performance.
5. **Grade Distribution** — summarizes grade outcomes.
6. **Individual Student Report** — performance details for a selected student.
7. **Pivot Table** — compares results across student/class/subject dimensions.
8. **Percentile Ranking** — places performance in cohort context.
9. **Subject Strengths** — identifies relative subject strengths.
10. **Improvement Leaderboard** — ranks improvement across available marks.
11. **Consistency Analysis** — examines consistency across subjects or assessments.
12. **Outlier Detection** — flags unusual performance values for review.
13. **Class Comparison** — compares class-level outcomes.
14. **Pass Rate Summary** — summarizes passing outcomes.
15. **Subject Top Performers** — identifies leading performers by subject.

Where implemented, Streamlit reports provide interactive controls and chart/table downloads. The HTML reporting interface also provides CSV exports for supported reports.

The REST API is grouped under `/sprint2` and includes health, class and subject lookups, report endpoints corresponding to the analytics pages, and `POST /sprint2/load-data` for data loading. See the generated API schema for exact request and response fields.

### Sprint 3 — Risk prediction and student exploration

Sprint 3 adds model-based risk dashboards for **UCI Student Performance** and **OULAD**. The dashboards primarily read saved scores and model artifacts; they do not retrain the risk models for each page request.

**Streamlit pages**

- UCI and OULAD Risk Dashboards.
- New-student prediction forms for each dataset.
- Student Time-Series Risk Explorer for OULAD's GRU engagement model.
- Model Performance views for each dataset.
- K-Means Clustering Explorer.
- Performance Personas.

Risk dashboards support cohort filters and student-level exploration, including model explanations where the required artifacts are available. The GRU explorer computes a single OULAD student's engagement trajectory from `studentVle.csv`; it is not a full-cohort GRU recomputation.

**Identifier conventions matter:** the UCI dashboard and tabular risk artifacts use a 1-based row index for lookup/display. OULAD engagement analysis uses the dataset's real `id_student`. The intervention API also supports both row-index lookup and real-student-ID lookup; check its `real_student_id` flag and response `lookup_by` field before joining results to source records.

The REST API is grouped under `/sprint3`. It exposes health and model metadata, saved risk scores, prediction, student risk and SHAP explanations, cohort summaries, GRU time-series/engagement data, personas, clustering, and model-performance results. Dataset-specific paths include `/{dataset}/risk-scores`, `/{dataset}/model-info`, `/{dataset}/predict`, `/{dataset}/student/{student_id}/risk`, `/{dataset}/student/{student_id}/shap`, `/{dataset}/cohort-summary`, `/{dataset}/personas`, `/{dataset}/clustering`, and `/{dataset}/model-performance`; OULAD GRU paths are under `/oulad/gru/`.

### Sprint 4 — AI Assistant

Sprint 4 adds the **AI Assistant** Streamlit page. The assistant can answer natural-language questions using application tools that retrieve student risk, SHAP/explanation details, cohort summaries, model information, OULAD engagement, and GRU time-series context. Conversation state is retained in the active Streamlit session.

The assistant is intended to explain retrieved facts rather than invent student data. It can display a weekly engagement chart, and image/chart follow-up questions require a configured multimodal model provider. Free-form model interpretation requires an API key for the configured provider; without one, the assistant reports that it is not configured.

Provider selection uses `LLM_PROVIDER` (`openai` or `gemini`). The default provider is OpenAI, with `gpt-4o-mini` as the default OpenAI model. The default Gemini model is `gemini-3.5-flash-lite`. Optional LangSmith tracing can be enabled with `LANGSMITH_API_KEY`.

### Sprint 5 — Intervention Agent and audit workflow

Sprint 5 turns risk insights into reviewable actions through deterministic routing, optional message drafting, and persisted decisions.

**Streamlit pages**

- **Intervention Agent — Chat** — contextual assistant interaction, including visible student or chart context where available.
- **Intervention Agent — Student** — look up a student and run one intervention.
- **Intervention Agent — Cohort Batch Run** — select a dataset, risk-level filter, and page size (1–25 students), navigate cohort pages, and run a batch.
- **Teacher Dashboard — Intervention Agent** — review logged runs; filter by dataset and decision, set the number of raw entries to load, and choose whether to show only each student's latest decision.

**Decision behavior**

- A LangGraph router selects **nudge**, **resource**, or **escalate** from risk score and dataset threshold, confidence, and compounding factors.
- An optional OULAD GRU/XGBoost disagreement check can inform single-student runs when the required GRU inputs are present.
- CrewAI/LLM-backed drafting can generate nudge or resource text. Templates provide a fallback when a drafting provider is unavailable.
- Escalation produces a mentor briefing rather than a drafted student message.
- Resource recommendations are re-ranked using the magnitude of the identified risk factors.
- Each decision is written to the audit store. The application reports whether the storage backend is PostgreSQL or the SQLite fallback.
- Teacher chat history is retained by Teacher ID; API chat supports session IDs and streaming responses.

Batch results are paged and logged per student. A batch can include per-student errors without discarding successful decisions.

## Data and model artifacts

### Sprint 2 CSVs and MySQL

The expected seed files are:

```text
data/
├── students.csv
├── subjects.csv
└── marks.csv
```

Supply `students.csv`, `subjects.csv`, and `marks.csv` under `data/`, then start MySQL, open **Load / Seed Data**, and run the seed action before opening Sprint 2 reports. The repository ignores local data files because student records may be sensitive, so these CSVs are not included in a Git checkout. Seeding initializes the tables and imports the supplied data. In Compose, the MySQL database persists in its named volume.

### Sprint 3 processed files and model files

The local `data/processed/` directory is expected to contain the saved UCI and OULAD risk scores, feature matrices, persona tables, and student-ID lookup CSVs. These files are ignored by Git because they may contain student-level information. The current expected names include:

```text
data/processed/
├── uci_risk_scores.csv
├── uci_feature_matrix.csv
├── uci_personas.csv
├── oulad_risk_scores.csv
├── oulad_feature_matrix.csv
├── oulad_personas.csv
└── oulad_student_ids.csv
```

Trained model artifacts are loaded from `models/`. Pages that need a missing score table, model, or metadata file cannot produce that model view until the artifact is supplied. The OULAD GRU checkpoint is named `risk_model_oulad_gru_v1.pt` or `risk_model_oulad_gru_v1.txt`.

The per-student OULAD GRU explorer additionally needs the raw OULAD files, including `studentVle.csv`, under `data/raw/oulad_data/`. Raw datasets are excluded from the Docker build context; Compose mounts the local `data/raw/` directory read-only into the app containers. Processed data files are also ignored by Git; when building an image locally, place required processed CSVs under `data/processed/` before running `docker compose up --build`, since the Dockerfile copies application files into the image. Provide source files locally before using features that read them. UCI/OULAD raw data required for any other locally computed feature must likewise be present in the location expected by its loader.

### Sprint 5 persistence

Compose starts PostgreSQL for intervention auditing and stateful API chat. Outside Compose, when no `DATABASE_URL` is set, the backend can use a local SQLite file. Compose named volumes keep PostgreSQL data, the SQLite fallback file, logs/reports, and the Hugging Face cache across container recreation. Removing a named volume deletes its persisted data; do not use volume-removal commands unless you intend to reset that data.

## Run locally

Use **Python 3.12**, the runtime used for the current Docker configuration. These commands are for PowerShell from the repository root.

1. Create and activate a virtual environment, then install dependencies:

   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

2. Configure environment variables as needed (see [Configuration](#configuration)). Do not commit credentials.
3. Start MySQL yourself for Sprint 2. For Sprint 5, you can also start PostgreSQL or use the SQLite fallback when `DATABASE_URL` is unset. Alternatively, use Compose for the full dependency stack.
4. Start the API and Streamlit in separate terminals:

   ```powershell
   uvicorn backend.main:app --host 0.0.0.0 --port 8000
   ```

   ```powershell
   streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
   ```

5. Open `http://localhost:8501` for Streamlit and `http://localhost:8000/docs` for the API schema.
6. Seed MySQL from the Streamlit **Load / Seed Data** page before using Sprint 2 reports.

An optional command-line entry point is available with `python main.py`.

## Run with Docker Compose

Docker Desktop (or another working Docker Engine) and the Compose plugin are required. Compose reads a root `.env` file for application variables and references it as a service environment file. If you do not already have one, create an empty local file without overwriting an existing file:

```powershell
if (-not (Test-Path .env)) { New-Item -ItemType File .env }
```

Add local provider keys or database overrides to this file only when needed; it is ignored by Git. The Compose defaults are for local development, not production. From the repository root:

```powershell
docker compose config
docker compose up -d --build
docker compose ps
```

The API and Streamlit services use the application image; nginx and the database services use their own images. It is expected to see multiple containers and multiple images for this full stack.

**Local URLs**

| Service | URL | Notes |
| --- | --- | --- |
| Streamlit | `http://localhost:8502` | Container listens on 8501 |
| HTML console | `http://localhost:8080` | nginx frontend |
| FastAPI | `http://localhost:8000` | JSON API |
| FastAPI Swagger docs | `http://localhost:8000/docs` | Interactive API reference |
| FastAPI ReDoc | `http://localhost:8000/redoc` | Alternative API reference |

Useful lifecycle and health commands:

```powershell
# Start/recreate the app services without rebuilding the image
docker compose up -d --no-build api streamlit

# Follow service logs
docker compose logs -f api streamlit

# Check service health and published ports
docker compose ps

# Stop containers but keep named volumes/data
docker compose down
```

To rebuild after changing application code or dependencies, run `docker compose up -d --build`. `docker compose down` does **not** remove database volumes. Only use `docker compose down -v` when intentionally deleting the persisted databases and other named-volume data.

Docker Compose uses the `student-risk-intervention:local` application image for both API and Streamlit. Images with different tags, such as an old baseline tag, can refer to earlier builds and remain on disk until removed. Before removing an image, check whether stopped or running containers still reference it:

```powershell
docker image ls
docker ps -a
docker inspect <container-name> --format '{{.Name}} uses {{.Image}}'
```

The image contains substantial ML dependencies, so its size is expected to be larger than a minimal web application image. The two-stage Dockerfile reduces build/runtime baggage; actual image size depends on the dependency versions and platform.

## Configuration

Configuration is read from environment variables. For local development, set them in the shell or a local `.env` file. Never add real API keys or database passwords to source control.

| Variable | Purpose |
| --- | --- |
| `LLM_PROVIDER` | Select `openai` or `gemini`; defaults to `openai` |
| `OPENAI_API_KEY` | Enables OpenAI-backed assistant/drafting features |
| `OPENAI_MODEL` | Optional OpenAI model override; defaults to `gpt-4o-mini` |
| `GEMINI_API_KEY` | Enables Gemini-backed assistant/drafting features |
| `GEMINI_MODEL` | Optional Gemini model override; defaults to `gemini-3.5-flash-lite` |
| `LANGSMITH_API_KEY` | Optional LangSmith tracing key |
| `LANGSMITH_PROJECT` | Optional LangSmith project name |
| `LANGSMITH_ENDPOINT` | Optional LangSmith endpoint |
| `DATABASE_URL` | PostgreSQL connection URL for the backend; without it, the backend can fall back to SQLite |
| `SQLITE_DB_PATH` | Optional path override for the SQLite fallback database |
| `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME` | Sprint 2 MySQL connection settings |
| `MYSQL_PASSWORD`, `MYSQL_ROOT_PASSWORD` | Compose MySQL application/root passwords; local-development defaults are provided in `compose.yaml` |
| `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD` | Compose PostgreSQL database and user settings; local-development defaults are provided in `compose.yaml` |

In Docker Compose, the database service settings and internal service hostnames are supplied to the containers by `compose.yaml`. Set any LLM keys in your local Compose environment if you want provider-backed features. Keep `.env` private and do not paste secrets into logs or documentation.

## FastAPI reference

The full, versioned request/response contract is available at `/docs` and `/openapi.json`.

| Prefix | Main capabilities |
| --- | --- |
| `/sprint2` | Health check, class/subject lookups, academic report queries, and CSV data loading |
| `/sprint3` | Health, saved risk scores and model information, predictions, student risk/SHAP details, cohort summaries, GRU engagement/time-series, personas, clustering, and model performance |
| `/agent` | Health, single and batch interventions, audit-log queries, chat, streaming chat, and chat history |

The Sprint 5 intervention API includes:

- `GET /agent/health`
- `POST /agent/run`
- `POST /agent/batch`
- `GET /agent/audit-log`
- `POST /agent/chat`
- `POST /agent/chat/stream` (server-sent events)
- `GET /agent/chat/history`

For `/agent/run`, send `dataset` (`uci` or `oulad`) and `student_id`. `real_student_id` defaults to `false`, meaning the identifier is treated as a 1-based row index; set it to `true` to use the actual student ID. The optional `check_disagreement` enables the supported model-disagreement check. Batch requests accept `dataset`, optional `risk_level` (`Low`, `Medium`, or `High`), `limit` (1–50), and `offset`.

## Tests

Run the test suite from the repository root with the project environment activated:

```powershell
pytest tests
```

Test areas include analytics, risk analysis, AI assistant, and backend behavior. Some tests or features that call external LLM providers require valid provider configuration; keep unit tests and provider credentials separate, and do not commit secrets.

## Project layout

```text
.
├── ai_assistant/       # Sprint 4 assistant tools, prompts, LLM clients
├── backend/            # FastAPI app, schemas, database setup, routers
├── data/
│   ├── processed/      # Model scores, features, personas, lookup tables
│   └── raw/            # Local source datasets; mounted into Docker
├── db/                 # Database scripts and related assets
├── frontend/           # HTML/CSS/JS console served through nginx
├── models/             # Saved model artifacts
├── orchestration/      # Sprint 5 decision graph and intervention logic
├── reports/            # Generated report artifacts
├── scripts/            # Data/model utility scripts
├── src/
│   ├── analytics/      # Sprint 2 analytics and MySQL access
│   └── risk/           # Sprint 3 model inference and risk queries
├── tests/              # Automated test suite
├── compose.yaml        # Full Docker Compose stack
├── Dockerfile          # Two-stage application image
├── requirements.txt    # Python dependencies
├── streamlit_app.py    # Streamlit pages for Sprints 2–5
└── main.py             # Optional CLI entry point
```

## Troubleshooting

### A report says there is no data

- Confirm the required service is running with `docker compose ps` or check the local database connection.
- For Sprint 2, supply the three CSVs under `data/`, use **Load / Seed Data**, and confirm the MySQL tables contain the imported rows.
- For Sprint 3, verify that the corresponding CSVs under `data/processed/` and required model artifacts under `models/` exist.
- For the live OULAD GRU explorer, verify the GRU checkpoint and raw `data/raw/oulad_data/studentVle.csv` are available.

### Assistant or intervention text generation is unavailable

Check that `LLM_PROVIDER` is supported and the matching key (`OPENAI_API_KEY` or `GEMINI_API_KEY`) is available to the process/container. The application can provide template-backed intervention drafts in supported paths, but provider-dependent free-form assistant answers require a configured model.

### A port is already in use

Check the host ports listed above and stop or reconfigure the conflicting process before starting Compose. `docker compose ps` shows which project services published each port.

### Docker refuses to remove an image

An existing container (including a stopped one) may still reference the image. Use `docker ps -a` and inspect the container's image ID. Remove only the specific obsolete container after confirming it is not needed, then remove the old image tag. Do not remove active containers, project volumes, or data as a general cleanup step.
