/* ============================================================
   pages.js — static HTML for each page + small render helpers
   shared across pages (decision card, badges, tables).
   app.js wires up events and calls Api.
   ============================================================ */

const Pages = (() => {

  function escapeHtml(s) {
    if (s === null || s === undefined) return "";
    return String(s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function decisionBadge(decision) {
    const d = (decision || "").toLowerCase();
    const cls = d.includes("escalate") ? "badge-escalate" : d.includes("resource") ? "badge-resource" : "badge-nudge";
    return `<span class="badge ${cls}">${escapeHtml(decision || "unknown")}</span>`;
  }

  function riskBadge(level) {
    const l = (level || "").toLowerCase();
    const cls = l === "high" ? "badge-high" : l === "medium" ? "badge-medium" : "badge-low";
    return `<span class="badge ${cls}">${escapeHtml(level || "—")}</span>`;
  }

  function mentorBriefHtml(brief) {
    return escapeHtml(brief || "")
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/_(.+?)_/g, "<em>$1</em>")
      .replace(/\r?\n/g, "<br>");
  }

  function formatAssistantText(text) {
    return escapeHtml(text || "")
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/\r?\n/g, "<br>");
  }

  /* ---------------- Decision result card (used by Run + Batch) ---------------- */
  function decisionCard(decision) {
    const d = (decision.decision || "").toLowerCase().includes("escalate") ? "escalate"
      : (decision.decision || "").toLowerCase().includes("resource") ? "resource" : "nudge";

    const trace = (decision.agent_trace || [])
      .map(step => `<span class="trace-step">${escapeHtml(step)}</span>`)
      .join('<span class="trace-arrow">&rarr;</span>');

    const confidencePct = Math.round((decision.router_confidence ?? 0) * 100);

    const messageBlock = decision.drafted_message
      ? `<div class="card-title" style="margin-top:18px;">Drafted message</div>
         <div class="message-box">${escapeHtml(decision.drafted_message)}</div>`
      : decision.decision === "escalate"
      ? `<div class="card-title" style="margin-top:18px;">Escalation reason</div>
         <div class="message-box">${escapeHtml(decision.escalation_reason || "This case was routed to a human mentor.")}</div>
         <p class="card-sub" style="margin-top:8px;">No drafted message for an escalation, by design — this is routed to a human mentor instead of a templated message.</p>
         ${decision.mentor_brief
            ? `<div class="card-title" style="margin-top:18px;">Briefing for the mentor</div>
               <div class="message-box mentor-brief">${mentorBriefHtml(decision.mentor_brief)}</div>`
            : `<div class="card-sub" style="margin-top:8px;">No mentor briefing was returned for this escalation.</div>`}`
      : "";

    const resourcesBlock = (decision.resources_used && decision.resources_used.length)
      ? `<div class="card-title" style="margin-top:18px;">Resources used</div>
         <ul class="resource-list">${decision.resources_used.map(r => `<li>${escapeHtml(r)}</li>`).join("")}</ul>`
      : "";

    return `
      <div class="decision-card">
        <div class="decision-head ${d}">
          <div class="decision-head-left">
            ${decisionBadge(decision.decision)}
            ${riskBadge(decision.risk_level)}
          </div>
          <div class="decision-head-id">
            dataset: <strong>${escapeHtml(decision.dataset)}</strong> &nbsp;·&nbsp;
            student_id: <strong>${escapeHtml(decision.student_id)}</strong>
            ${decision.display_student_id && decision.display_student_id !== decision.student_id
              ? `&nbsp;·&nbsp; display ID: <strong>${escapeHtml(decision.display_student_id)}</strong>`
              : ""}
            ${decision.row_index != null ? ` (row ${escapeHtml(decision.row_index)})` : ""}
            &nbsp;·&nbsp; lookup: <code class="inline">${escapeHtml(decision.lookup_by)}</code>
          </div>
        </div>
        <div class="decision-body">
          <div class="card-title">Router confidence</div>
          <div class="confidence-wrap">
            <div class="confidence-track"><div class="confidence-fill" style="width:${confidencePct}%"></div></div>
            <div class="confidence-value">${confidencePct}%</div>
          </div>

          <div class="card-title">Agent trace</div>
          <div class="trace-row">${trace || '<span class="card-sub">no trace recorded</span>'}</div>

          <div class="card-title">Rationale</div>
          <p style="font-size:13.5px; line-height:1.6; color:var(--ink); margin:0 0 4px;">${escapeHtml(decision.rationale)}</p>

          ${messageBlock}
          ${resourcesBlock}

          <div class="kv-grid">
            <div class="kv-item"><div class="kv-label">Source</div><div class="kv-value">${escapeHtml(decision.source)}</div></div>
            <div class="kv-item"><div class="kv-label">Decision</div><div class="kv-value">${escapeHtml(decision.decision)}</div></div>
            <div class="kv-item"><div class="kv-label">Risk level</div><div class="kv-value">${escapeHtml(decision.risk_level)}</div></div>
          </div>
        </div>
      </div>`;
  }

  /* ============================================================
     OVERVIEW
     ============================================================ */
  function overview() {
    return `
      <div class="page" id="page-overview">
        <div class="section-head">
          <h1>Pipeline Overview</h1>
          <p>One console across all four sprints — every report and model page your Streamlit app has, now backed by REST. The tags on each row say honestly which backend router each sprint calls.</p>
        </div>

        <div class="grid grid-2" style="margin-bottom:18px;">
          <div class="card">
            <div class="card-title">What this console talks to</div>
            <div class="card-sub">backend/main.py + routers: agent.py, sprint2.py, sprint3.py</div>
            <div class="sprint-timeline">
              <div class="sprint-row">
                <div class="sprint-num live">2</div>
                <div class="sprint-copy">
                  <h4>Marks Analytics <span class="pill-tag pill-live">live</span></h4>
                  <p>All 14 class/student reports (class average, top/bottom, weak areas, pivot table, percentile, outliers, pass rate, and more), MySQL-backed, via <strong>backend/routers/sprint2.py</strong> calling your real <code class="inline">src/analytics/queries.py</code> + <code class="inline">analytics.py</code>.</p>
                </div>
              </div>
              <div class="sprint-row">
                <div class="sprint-num live">3</div>
                <div class="sprint-copy">
                  <h4>Risk Prediction <span class="pill-tag pill-live">live</span></h4>
                  <p>Risk dashboard, new-student prediction, SHAP factors, GRU time-series, clustering/personas, and model performance — via <strong>backend/routers/sprint3.py</strong> calling your real trained XGBoost/CatBoost/GRU models and <code class="inline">ai_assistant/tools.py</code>.</p>
                </div>
              </div>
              <div class="sprint-row">
                <div class="sprint-num live">4</div>
                <div class="sprint-copy">
                  <h4>Risk Explanation (AI Assistant) <span class="pill-tag pill-live">live</span></h4>
                  <p>SHAP/Integrated-Gradients explanations, persona context, and RAG-matched remedial content — reachable through <strong>Ask the Assistant</strong> and the floating assistant panel on every page, both calling the same function-calling agent.</p>
                </div>
              </div>
              <div class="sprint-row">
                <div class="sprint-num live">5</div>
                <div class="sprint-copy">
                  <h4>Intervention Agent <span class="pill-tag pill-live">live</span></h4>
                  <p>LangGraph router + CrewAI drafting + audit log, exposed over REST. This is what <strong>Run</strong>, <strong>Batch</strong>, and <strong>Audit Log</strong> call directly.</p>
                </div>
              </div>
            </div>
          </div>

          <div class="card">
            <div class="card-title">Quick start</div>
            <div class="card-sub">If a page shows a connection error, start here</div>
            <ol style="font-size:13px; color:var(--ink-soft); line-height:1.9; padding-left:18px; margin:0;">
              <li>The backend mounts the Sprint 2 &amp; 3 routers and enables CORS in <code class="inline">backend/main.py</code>.</li>
              <li>Run the backend: <code class="inline">uvicorn backend.main:app --reload</code></li>
              <li>Open <strong>Connection</strong> in the sidebar and confirm the API base URL matches where it's running (default <code class="inline">http://127.0.0.1:8000</code>).</li>
              <li>Sprint 2 needs MySQL seeded (<strong>Class &amp; Student Reports → Load / Seed Data</strong>). Sprint 3 needs your trained models in <code class="inline">models/</code> — same as Streamlit already requires.</li>
              <li>Use <strong>Run — Single Student</strong> with dataset <code class="inline">uci</code> and student id <code class="inline">1</code> as a smoke test.</li>
            </ol>
            <div class="card-title" style="margin-top:20px;">Decision branches, at a glance</div>
            <div class="row" style="margin-top:10px;">
              <span class="badge badge-nudge">nudge</span>
              <span class="badge badge-resource">resource</span>
              <span class="badge badge-escalate">escalate</span>
            </div>
            <p class="card-sub" style="margin-top:10px;">Every run of the Intervention Agent ends in exactly one of these three — never a partial or ambiguous outcome.</p>
          </div>
        </div>

        <div class="card" style="margin-top:18px;">
          <div class="card-title">Ask the Assistant, everywhere</div>
          <p class="card-sub" style="margin-bottom:0;">The &#129302; button in the bottom-right corner is on every page, not just the dedicated Chat page. It automatically tells the assistant what you're currently looking at (which report, which dataset, which student) as context — the same way each Streamlit page's collapsible chat sub-agent already knows its own page. It shares one conversation/session with the dedicated <strong>Ask the Assistant</strong> page, so switching between them continues the same thread.</p>
        </div>
      </div>`;
  }

  /* ============================================================
     RUN — single student
     ============================================================ */
  function run() {
    return `
      <div class="page" id="page-run">
        <div class="section-head">
          <h1>Run — Single Student</h1>
          <p>Calls <code class="inline">POST /agent/run</code>. Runs the LangGraph router for one student and returns the full decision, including the CrewAI-drafted message when the branch is nudge or resource.</p>
        </div>

        <div class="grid" style="grid-template-columns: 340px 1fr; align-items:start;">
          <div class="card">
            <div class="card-title">Request</div>
            <div class="field">
              <label class="label">Dataset</label>
              <select id="runDataset">
                <option value="uci">uci</option>
                <option value="oulad">oulad</option>
              </select>
            </div>
            <div class="field">
              <label class="label">Student ID</label>
              <input type="text" id="runStudentId" placeholder="e.g. 1" value="1">
            </div>
            <div class="checkbox-row">
              <input type="checkbox" id="runRealId">
              <label for="runRealId">This is a real student ID (not a row index)</label>
            </div>
            <div class="checkbox-row">
              <input type="checkbox" id="runDisagreement">
              <label for="runDisagreement">Check GRU/XGBoost disagreement (OULAD only)</label>
            </div>
            <button class="btn btn-primary" id="runSubmit" style="width:100%; justify-content:center; margin-top:6px;">
              Run Intervention
            </button>
          </div>

          <div id="runResult">
            <div class="card empty-state">Run a student to see the routed decision here.</div>
          </div>
        </div>
      </div>`;
  }

  /* ============================================================
     BATCH — cohort
     ============================================================ */
  function batch() {
    return `
      <div class="page" id="page-batch">
        <div class="section-head">
          <h1>Run — Cohort Batch</h1>
          <p>Calls <code class="inline">POST /agent/batch</code>. Filter the cohort by dataset and risk, choose a page size, and run or rerun the previous/next slice.</p>
        </div>

        <div class="card" style="margin-bottom:18px;">
          <div class="toolbar">
            <div class="field">
              <label class="label">Dataset</label>
              <select id="batchDataset">
                <option value="uci">uci</option>
                <option value="oulad">oulad</option>
              </select>
            </div>
            <div class="field">
              <label class="label">Risk level</label>
              <select id="batchRiskLevel">
                <option value="">Any</option>
                <option value="Low">Low</option>
                <option value="Medium">Medium</option>
                <option value="High">High</option>
              </select>
            </div>
            <div class="field" style="max-width:150px;">
              <label class="label">Students per batch</label>
              <select id="batchLimit">
                <option value="5" selected>5 students</option>
                <option value="10">10 students</option>
                <option value="25">25 students</option>
              </select>
            </div>
            <div class="field" style="max-width:140px;">
              <label class="label">Starting at</label>
              <input type="number" id="batchOffset" value="0" min="0" step="1">
            </div>
            <button class="btn btn-ghost" id="batchPrev" disabled>◀ Previous batch</button>
            <button class="btn btn-ghost" id="batchNext" disabled>Next batch ▶</button>
            <div class="toolbar-spacer"></div>
            <button class="btn btn-primary" id="batchSubmit">Run current batch</button>
          </div>
          <div id="batchPageNote" class="card-sub" style="margin:12px 0 0;">Choose filters and run a batch to load students.</div>
        </div>

        <div id="batchSummary"></div>
        <div id="batchResult" style="margin-top:18px;">
          <div class="card empty-state">Run a batch to see per-student decisions and any errors.</div>
        </div>
      </div>`;
  }

  /* ============================================================
     AUDIT LOG
     ============================================================ */
  function audit() {
    return `
      <div class="page" id="page-audit">
        <div class="section-head">
          <h1>Audit Log</h1>
          <p>Calls <code class="inline">GET /agent/audit-log</code>. Every decision the Intervention Agent has ever made — from this console, from the Streamlit app, or from the chat assistant — is readable here.</p>
        </div>

        <div class="card" style="margin-bottom:18px;">
          <div class="toolbar">
            <div class="field">
              <label class="label">Dataset</label>
              <select id="auditDataset">
                <option value="">Any</option>
                <option value="uci">uci</option>
                <option value="oulad">oulad</option>
              </select>
            </div>
            <div class="field">
              <label class="label">Decision</label>
              <select id="auditDecision">
                <option value="">Any</option>
                <option value="nudge">nudge</option>
                <option value="resource">resource</option>
                <option value="escalate">escalate</option>
              </select>
            </div>
            <div class="field" style="max-width:120px;">
              <label class="label">Limit</label>
              <input type="number" id="auditLimit" value="50" min="1" max="500">
            </div>
            <div class="checkbox-row" style="margin-bottom:0; align-self:center;">
              <input type="checkbox" id="auditDedup" checked>
              <label for="auditDedup">One row per student (latest only)</label>
            </div>
            <div class="toolbar-spacer"></div>
            <button class="btn btn-ghost btn-sm" id="auditExport">Export CSV</button>
            <button class="btn btn-primary" id="auditRefresh">Refresh</button>
          </div>
        </div>

        <div id="auditStorageNote" class="card-sub" style="margin-bottom:10px;"></div>
        <div id="auditTable">
          <div class="card empty-state">Loading…</div>
        </div>
      </div>`;
  }

  /* ============================================================
     TEACHER DASHBOARD (derived client-side from the audit log --
     there is no separate /dashboard endpoint, so this reuses
     GET /agent/audit-log and summarizes it in the browser)
     ============================================================ */
  function dashboard() {
    return `
      <div class="page" id="page-dashboard">
        <div class="section-head">
          <h1>Teacher Dashboard</h1>
          <p>A summary view built in the browser from <code class="inline">GET /agent/audit-log</code> — there's no separate dashboard endpoint in the backend, so this page re-aggregates the same audit trail instead of inventing new data.</p>
        </div>

        <div class="card" style="margin-bottom:18px;">
          <div class="toolbar">
            <div class="field">
              <label class="label">Dataset</label>
              <select id="dashboardDataset">
                <option value="">(all)</option>
                <option value="uci">uci</option>
                <option value="oulad">oulad</option>
              </select>
            </div>
            <div class="field">
              <label class="label">Decision</label>
              <select id="dashboardDecision">
                <option value="">(all)</option>
                <option value="nudge">nudge</option>
                <option value="resource">resource</option>
                <option value="escalate">escalate</option>
              </select>
            </div>
            <div class="field" style="max-width:180px;">
              <label class="label">Max raw entries</label>
              <input type="number" id="dashboardLimit" value="200" min="10" max="1000" step="10">
            </div>
            <div class="checkbox-row" style="margin-bottom:0; align-self:center;">
              <input type="checkbox" id="dashboardDedup" checked>
              <label for="dashboardDedup">One row per student (latest only)</label>
            </div>
            <div class="toolbar-spacer"></div>
            <button class="btn btn-primary" id="dashboardRefresh">Refresh</button>
          </div>
        </div>

        <div class="grid grid-4" id="dashboardStats" style="margin-bottom:18px;">
          <div class="card stat-tile"><div class="stat-num">—</div><div class="stat-label">Total logged decisions</div></div>
          <div class="card stat-tile"><div class="stat-num">—</div><div class="stat-label">Unique students</div></div>
          <div class="card stat-tile"><div class="stat-num">—</div><div class="stat-label">Escalation rate</div></div>
          <div class="card stat-tile"><div class="stat-num">—</div><div class="stat-label">Avg. router confidence</div></div>
        </div>

        <div id="dashboardSummaryNote" class="card-sub" style="margin-bottom:14px;"></div>
        <div class="grid grid-2" style="margin-bottom:18px;">
          <div class="card">
            <div class="card-title">Decision breakdown</div>
            <div class="card-sub">Nudges, resources, and human escalations</div>
            <canvas id="dashboardChart" style="width:100%; height:180px;"></canvas>
          </div>
          <div class="card">
            <div class="card-title">Decisions by risk level</div>
            <div class="card-sub">Current filtered decisions</div>
            <canvas id="dashboardRiskChart" style="width:100%; height:180px;"></canvas>
          </div>
        </div>
        <div class="grid grid-2">
          <div class="card">
            <div class="card-title">Decisions over time</div>
            <div class="card-sub">Daily decision counts from the selected records</div>
            <canvas id="dashboardTimelineChart" style="width:100%; height:220px;"></canvas>
          </div>
          <div class="card">
            <div class="card-title">Escalation rate by dataset</div>
            <div class="card-sub">Share of decisions routed to a human mentor</div>
            <canvas id="dashboardDatasetEscalationChart" style="width:100%; height:220px;"></canvas>
          </div>
          <div class="card dashboard-wide-card">
            <div class="card-title">Students flagged High risk, most recent first</div>
            <div id="dashboardHighRisk" class="card-sub">Loading…</div>
          </div>
        </div>

        <div class="card" style="margin-top:18px;">
          <div class="card-title">Recent decisions</div>
          <div class="card-sub">All rows matching the selected filters and raw-entry limit. Scroll within the table to browse.</div>
          <div id="dashboardRecentDecisions" class="dashboard-table-scroll">Loading…</div>
          <div class="toolbar" style="margin:12px 0 0;">
            <button class="btn btn-ghost btn-sm" id="dashboardDownloadTable">Download table as CSV</button>
            <button class="btn btn-ghost btn-sm" id="dashboardDownloadFull">Download full detail as CSV</button>
          </div>
        </div>

        <div class="card" style="margin-top:18px;">
          <div class="card-title">Full decision detail</div>
          <div class="field" style="margin-top:12px;">
            <label class="label" for="dashboardEntrySelect">Pick an entry</label>
            <select id="dashboardEntrySelect" disabled><option>Loading…</option></select>
          </div>
          <div id="dashboardEntryDetail"></div>
        </div>
      </div>`;
  }

  /* ============================================================
     CHAT
     ============================================================ */
  function chat() {
    return `
      <div class="page" id="page-chat">
        <div class="section-head">
          <h1>Ask the Assistant</h1>
          <p>Streams from <code class="inline">POST /agent/chat/stream</code> — the Sprint 4 function-calling agent, now also able to trigger Sprint 5's Intervention Agent directly from conversation (e.g. "run an intervention for uci student 3").</p>
        </div>

        <div class="card">
          <div class="chat-shell">
            <div class="chat-log" id="chatLog">
              <div class="msg msg-ai">
                Ask me about a student's risk, their SHAP factors, a cohort summary, or ask me to run an intervention for them directly.
              </div>
            </div>
            <div class="chat-input-row">
              <input type="text" id="chatInput" placeholder="e.g. Why is uci student 3 at risk?">
              <button class="btn btn-primary" id="chatSend">Send</button>
              <button class="btn btn-ghost" id="chatNewSession" title="Start a fresh session">New chat</button>
            </div>
          </div>
        </div>
      </div>`;
  }

  /* ============================================================
     SETTINGS / CONNECTION
     ============================================================ */
  function settings() {
    return `
      <div class="page" id="page-settings">
        <div class="section-head">
          <h1>Connection</h1>
          <p>Point this console at wherever your FastAPI backend is running.</p>
        </div>

        <div class="card" style="max-width:520px;">
          <div class="field">
            <label class="label">API base URL</label>
            <input type="text" id="apiBaseInput" placeholder="http://127.0.0.1:8000">
          </div>
          <button class="btn btn-primary" id="apiBaseSave">Save &amp; test connection</button>
          <div id="settingsResult" style="margin-top:16px;"></div>

          <div class="card-title" style="margin-top:26px;">CORS reminder</div>
          <p class="card-sub">If this console is opened as a static file (or from a different port) than the API, add this to <code class="inline">backend/main.py</code> once:</p>
          <pre style="background:var(--bg-card-soft); border:1px solid var(--border); border-radius:8px; padding:12px; font-size:12px; overflow-x:auto;">from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # or restrict to this console's origin
    allow_methods=["*"],
    allow_headers=["*"],
)</pre>
        </div>
      </div>`;
  }

  return {
    escapeHtml, decisionBadge, riskBadge, mentorBriefHtml, formatAssistantText, decisionCard,
    overview, run, batch, audit, dashboard, chat, settings,
  };
})();
