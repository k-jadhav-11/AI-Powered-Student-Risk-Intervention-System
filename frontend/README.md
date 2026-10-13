# Intervention Console — standalone HTML/CSS/JS frontend

A plain HTML/CSS/JavaScript UI for the At-Risk Student Prediction & Intervention
Platform, covering **all four sprints (2–5)**. No build step, no framework, no
npm install — open `index.html` or serve the folder, point it at your FastAPI
backend, and go.

## What's live now

Sprints 2 and 3 didn't have a REST API in this codebase before — only Sprint 5
did. This console now ships **new backend routers** (`backend/routers/sprint2.py`,
`backend/routers/sprint3.py` — see `../backend_patch/`) that expose every report
your Streamlit apps show, calling the exact same underlying functions
(`src/analytics/*`, `src/risk/*`, `ai_assistant/tools.py`). Nothing was faked —
every endpoint wraps a real function from your repo; you can diff the router
files against your `src/` modules yourself.

| Page | Sprint | Backend | Status |
|---|---|---|---|
| Pipeline Overview | 2–5 | — | explains the whole pipeline, no calls |
| Class & Student Reports | 2 | `backend/routers/sprint2.py` (18 endpoints) | **live** |
| Risk & Engagement Models | 3 | `backend/routers/sprint3.py` (12 endpoint groups) | **live** |
| Ask the Assistant | 4 | `POST /agent/chat/stream` | **live** — streamed SHAP/GRU explanation + RAG via the function-calling agent |
| Run — Single Student | 5 | `POST /agent/run` | live |
| Run — Cohort Batch | 5 | `POST /agent/batch` | live |
| Audit Log | 5 | `GET /agent/audit-log` | live |
| Teacher Dashboard | 5 | `GET /agent/audit-log` (aggregated client-side) | live |
| Connection | — | `GET /agent/health` | live |

**You must apply `../backend_patch/PATCHES.md` to your backend first** — it
copies the two new router files in, registers them in `backend/main.py`, adds
the CORS middleware this console needs, and wires `page_context` through to
the chat agent. Skipping this is the #1 cause of "Request failed (0)" errors.

## Sprint 2 — Class & Student Reports

All 15 of your Streamlit "Student Performance Analytics" pages, as one
generic report browser: pick a report from the left sub-nav, fill in the
(optional) filters, **Run**. Includes class average, top/bottom performers,
weak-area detection, grade distribution, individual student report, pivot
table, percentile ranking, subject strengths, improvement leaderboard,
consistency analysis, outlier detection, class comparison, pass-rate summary,
subject top performers, and a **Load / Seed Data** action that calls
`POST /sprint2/load-data` to seed MySQL from your CSVs (safe to re-run).

Every filter matches `streamlit_app.py`'s own pages exactly — same labels
("Class / grade", "Subject"), same defaults ("All classes",
"Overall (all subjects)"), same numeric ranges (e.g. outlier z-score
threshold 0.5–5.0, top/bottom N 1–50). Every report also gets:
- **A chart** below the table — the client-side equivalent of Streamlit's
  `st.pyplot(fig)` for that page (bar chart, scatter plot, line/trend chart,
  or a class×subject heatmap, matching what that specific Streamlit page
  draws).
- **A "Download CSV" button** — the same data as the table, downloaded
  client-side (Streamlit's `export_buttons()` equivalent; there's no PNG
  download since the chart here is already an on-page `<canvas>`, not a
  server-rendered image).
- The **Individual Student Report** also shows the three metric tiles
  (student average / class average / z-score), the trend label, and the
  "consistently underperforming" flag, same as Streamlit.
- The **Class Comparison** report renders the "by subject" result as a
  colored heatmap table, same idea as Streamlit's `plot_class_comparison_heatmap`.

## Sprint 3 — Risk & Engagement Models

All 12 of your Streamlit risk-prediction pages, with a **uci / oulad dataset
toggle** at the top (OULAD-only pages — the GRU time-series ones — hide
themselves when "uci" is selected). Includes the risk dashboard, model
info/performance, student risk lookup, SHAP factors, cohort summary,
personas, K-Means clustering explorer, GRU student directory and
timeseries, and a **Predict Risk — New Student** form that renders every
real feature column your trained model expects (25 for UCI, 34 for OULAD)
with a "Fill from student" button that pulls a real row from
`/sprint3/{dataset}/risk-scores` to pre-populate it.

## Ask the Assistant, everywhere

A floating chat button (bottom-right, every page) — the HTML-UI equivalent of
each Streamlit page's collapsible chat sub-agent. It automatically tells the
assistant what's on screen (which report, which dataset, which student) via
`ChatRequest.page_context`, the same way each Streamlit page already knows
its own context. It shares one conversation (`session_id`, kept in
`sessionStorage`) with the dedicated **Ask the Assistant** page, so switching
between the floating panel and the full page continues the same thread
instead of starting two separate ones.

## Files

```
frontend/
├── index.html
├── css/
│   └── styles.css          # design tokens, layout, components, report sub-nav, assistant panel
├── js/
│   ├── api.js               # fetch wrapper — one function per real backend endpoint
│   ├── toast.js               # shared toast module
│   ├── charts.js                # tiny dependency-free canvas bar chart
│   ├── pages.js                   # HTML templates for the Sprint 5 + static pages
│   ├── sprint2Config.js             # declarative list of all 15 Sprint 2 reports
│   ├── sprint3Config.js               # declarative list of all 12 Sprint 3 reports + feature columns
│   ├── reportEngine.js                  # generic renderer that turns a report config into a working page
│   ├── assistant.js                       # the floating, context-aware chat widget
│   └── app.js                               # routing + event wiring, ties everything together
└── README.md
```

## Running it

1. Apply `../backend_patch/PATCHES.md` to your backend (copy in the 2 new
   router files, register them, add CORS, add `page_context`).

2. Start your backend (from the project root):

   ```bash
   uvicorn backend.main:app --reload
   ```

   Restart it fully (don't rely on `--reload` picking up the new routers/
   middleware) after applying the patch.

3. Serve this folder:

   ```bash
   cd frontend
   python3 -m http.server 8080
   ```

   Then open `http://127.0.0.1:8080`.

4. Click **Connection** in the sidebar and confirm the API base URL matches
   where uvicorn is running (defaults to `http://127.0.0.1:8000`, stored in
   `localStorage` once you save it).

5. Sprint 2 needs MySQL seeded — **Class & Student Reports → Load / Seed
   Data**. Sprint 3 needs your trained models already in `models/` (same as
   Streamlit requires).

6. Try **Run — Single Student** with dataset `uci`, student ID `1`, as a
   smoke test.

## If you see "Request failed (0)"

This means the browser couldn't reach the backend at all — almost always one
of:

- The backend isn't running, or was restarted with `--reload` but the CORS
  middleware didn't hot-reload (restart it fully).
- CORS was never added — see `../backend_patch/PATCHES.md`.
- The API base URL in **Connection** doesn't match where uvicorn is actually
  listening.

## Notes on specific pages

- **Teacher Dashboard** has no dedicated backend endpoint — it re-reads
  `GET /agent/audit-log` (last 200 records) and aggregates counts, escalation
  rate, and average router confidence in the browser. The filtered decisions
  can be browsed in the scrollable recent-decisions and high-risk tables,
  selected in **Pick an entry** for full details, and downloaded as a display
  table or full-detail CSV.
- **Teacher ID** is optional and stored in this browser's local storage. It is
  sent with chat requests to associate persisted assistant history with the
  same teacher ID; it is an identifier only, not an authenticated account.
- **Audit Log**'s "one row per student" checkbox dedups client-side by
  `(dataset, student_id)`, keeping the first (most recent) record.
- **Chat** (and the floating assistant) keep `session_id` in `sessionStorage`
  so a page refresh resumes the same conversation; "New" clears it.
- Chat requests include the active page's visible data and filters. The
  assistant also captures up to four visible chart canvases and attaches them
  to graph/statistic questions so the model can explain the chart itself,
  rather than relying only on a generic page label.
- Sprint 2/3 report results render generically from whatever shape the
  backend returns (rows / top+bottom / pivot table / nested object) — there's
  no per-endpoint template to go out of sync with the API.
