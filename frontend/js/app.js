/* ============================================================
   app.js — routing, page wiring, and all event handling.
   ============================================================ */

(function () {
  const pageRoot = document.getElementById("pageRoot");
  const pageTitle = document.getElementById("pageTitle");
  const apiChip = document.getElementById("apiChip");
  const showToast = Toast.show; // local alias, kept so existing calls below don't need renaming
  const TEACHER_ID_KEY = "ic_teacher_id";

  function getTeacherId() {
    return localStorage.getItem(TEACHER_ID_KEY)?.trim() || null;
  }

  const TITLES = {
    overview: "Pipeline Overview",
    sprint2: "Sprint 2 — Class & Student Reports",
    sprint3: "Sprint 3 — Risk & Engagement Models",
    run: "Run — Single Student",
    batch: "Run — Cohort Batch",
    audit: "Audit Log",
    dashboard: "Teacher Dashboard",
    chat: "Ask the Assistant",
    settings: "Connection",
  };

  const RENDERERS = {
    overview: Pages.overview,
    sprint2: () => ReportEngine.renderShell("sprint2", SPRINT2_REPORTS, {
      title: "Sprint 2 — Class & Student Reports",
      subtitle: "Every report your Streamlit \"Student Performance Analytics\" pages show, calling the same MySQL-backed queries.py / analytics.py functions through backend/routers/sprint2.py.",
    }),
    sprint3: () => ReportEngine.renderShell("sprint3", SPRINT3_REPORTS, {
      title: "Sprint 3 — Risk & Engagement Models",
      subtitle: "Risk prediction, engagement, clustering, and model-performance reports using the trained XGBoost/CatBoost/GRU models and ai_assistant/tools.py helpers through backend/routers/sprint3.py.",
      datasetToggle: true,
    }),
    run: Pages.run,
    batch: Pages.batch,
    audit: Pages.audit,
    dashboard: Pages.dashboard,
    chat: Pages.chat,
    settings: Pages.settings,
  };

  const AFTER_RENDER = {
    sprint2: () => ReportEngine.wire("sprint2", SPRINT2_REPORTS, {
      title: "Sprint 2 — Class & Student Reports",
    }),
    sprint3: () => ReportEngine.wire("sprint3", SPRINT3_REPORTS, {
      title: "Sprint 3 — Risk & Engagement Models",
      datasetToggle: true,
    }),
    run: wireRunPage,
    batch: wireBatchPage,
    audit: wireAuditPage,
    dashboard: wireDashboardPage,
    chat: wireChatPage,
    settings: wireSettingsPage,
  };

  const PAGE_CONTEXT = {
    overview: "Viewing the Pipeline Overview page.",
    run: "Viewing the Sprint 5 single-student intervention page.",
    batch: "Viewing the Sprint 5 cohort batch page.",
    audit: "Viewing the Sprint 5 intervention audit log.",
    dashboard: "Viewing the Sprint 5 Teacher Dashboard.",
    chat: "Viewing the dedicated Ask the Assistant page.",
    settings: "Viewing the Connection settings page.",
  };

  let currentPage = null;

  function navigate(page) {
    if (!RENDERERS[page]) page = "overview";
    currentPage = page;

    pageRoot.innerHTML = RENDERERS[page]();
    pageRoot.querySelector(".page").classList.add("active");
    pageTitle.textContent = TITLES[page];

    document.querySelectorAll(".nav-item").forEach(btn => {
      btn.classList.toggle("active", btn.dataset.page === page);
    });

    if (AFTER_RENDER[page]) AFTER_RENDER[page]();
    if (PAGE_CONTEXT[page] && window.Assistant) Assistant.setPageContext(PAGE_CONTEXT[page]);
    window.scrollTo(0, 0);
  }

  document.getElementById("nav").addEventListener("click", (e) => {
    const btn = e.target.closest(".nav-item");
    if (btn) navigate(btn.dataset.page);
  });

  /* ============================================================
     Health pill (top of sidebar) + API chip (top bar)
     ============================================================ */
  async function refreshHealth() {
    const dot = document.getElementById("healthDot");
    const text = document.getElementById("healthText");
    apiChip.textContent = Api.getBase();
    dot.className = "dot pending";
    text.textContent = "checking backend…";
    try {
      const h = await Api.agentHealth();
      dot.className = "dot " + (h.status === "ok" ? "ok" : "pending");
      text.textContent = `${h.status} · ${h.storage_backend}`;
    } catch (err) {
      dot.className = "dot err";
      text.textContent = "backend unreachable";
    }
  }

  /* ============================================================
     RUN page
     ============================================================ */
  function wireRunPage() {
    document.getElementById("runSubmit").addEventListener("click", async () => {
      const btn = document.getElementById("runSubmit");
      const resultBox = document.getElementById("runResult");
      const payload = {
        dataset: document.getElementById("runDataset").value,
        student_id: document.getElementById("runStudentId").value.trim(),
        real_student_id: document.getElementById("runRealId").checked,
        check_disagreement: document.getElementById("runDisagreement").checked,
      };
      if (!payload.student_id) { showToast("Enter a student ID first.", "err"); return; }

      btn.disabled = true;
      btn.innerHTML = '<span class="spinner"></span> Running…';
      try {
        const decision = await Api.runIntervention(payload);
        resultBox.innerHTML = Pages.decisionCard(decision);
        showToast("Intervention decision returned.", "ok");
        if (window.Assistant) Assistant.setPageContext(
          `Viewing the intervention result for ${decision.dataset} student ${decision.student_id}: ${decision.risk_level} risk, decision=${decision.decision}.`
        );
      } catch (err) {
        resultBox.innerHTML = `<div class="card" style="border-color:var(--escalate);">
          <div class="card-title" style="color:var(--escalate);">Request failed (${err.status ?? "network"})</div>
          <p style="font-size:13px; color:var(--ink-soft);">${Pages.escapeHtml(err.message)}</p>
        </div>`;
        showToast(err.message, "err");
      } finally {
        btn.disabled = false;
        btn.textContent = "Run Intervention";
      }
    });
  }

  /* ============================================================
     BATCH page
     ============================================================ */
  function wireBatchPage() {
    const offsetInput = document.getElementById("batchOffset");
    const limitInput = document.getElementById("batchLimit");
    const previousButton = document.getElementById("batchPrev");
    const nextButton = document.getElementById("batchNext");
    const pageNote = document.getElementById("batchPageNote");
    let totalStudents = 0;

    async function runBatch(offset = Number(offsetInput.value)) {
      const btn = document.getElementById("batchSubmit");
      const summaryBox = document.getElementById("batchSummary");
      const resultBox = document.getElementById("batchResult");
      const limit = Number(limitInput.value);
      offset = Number.isInteger(offset) && offset >= 0 ? offset : 0;
      offsetInput.value = String(offset);
      const payload = {
        dataset: document.getElementById("batchDataset").value,
        risk_level: document.getElementById("batchRiskLevel").value || null,
        limit,
        offset,
      };

      btn.disabled = true;
      previousButton.disabled = true;
      nextButton.disabled = true;
      btn.innerHTML = '<span class="spinner"></span> Running batch…';
      try {
        const res = await Api.runBatch(payload);
        totalStudents = res.total_students ?? offset + res.results.length + res.errors.length;
        offsetInput.value = String(res.offset ?? offset);
        renderBatchSummary(summaryBox, res);
        renderBatchTable(resultBox, res);
        const shown = res.results.length + res.errors.length;
        const rangeStart = shown ? offset + 1 : offset;
        const rangeEnd = offset + shown;
        pageNote.textContent = totalStudents === 0
          ? `No students match the selected dataset and risk-level filter.`
          : shown
          ? `Showing students ${rangeStart}–${rangeEnd} of ${totalStudents} matching ${payload.risk_level || "all-risk"} ${payload.dataset} students.`
          : `No students at offset ${offset}; the filtered cohort contains ${totalStudents}. Set a lower offset or use Previous batch.`;
        previousButton.disabled = offset <= 0;
        nextButton.disabled = offset + limit >= totalStudents;
        showToast(`Batch complete: ${res.results.length} succeeded, ${res.errors.length} failed.`, "ok");
      } catch (err) {
        resultBox.innerHTML = `<div class="card" style="border-color:var(--escalate);">
          <div class="card-title" style="color:var(--escalate);">Request failed (${err.status ?? "network"})</div>
          <p style="font-size:13px; color:var(--ink-soft);">${Pages.escapeHtml(err.message)}</p>
        </div>`;
        pageNote.textContent = "Batch request failed; check the error and retry.";
        showToast(err.message, "err");
      } finally {
        btn.disabled = false;
        btn.textContent = "Run current batch";
        previousButton.disabled = Number(offsetInput.value) <= 0;
        nextButton.disabled = Number(offsetInput.value) + Number(limitInput.value) >= totalStudents;
      }
    }

    document.getElementById("batchSubmit").addEventListener("click", () => runBatch());
    previousButton.addEventListener("click", () => {
      const limit = Number(limitInput.value);
      runBatch(Math.max(0, Number(offsetInput.value) - limit));
    });
    nextButton.addEventListener("click", () => {
      runBatch(Number(offsetInput.value) + Number(limitInput.value));
    });
    ["batchDataset", "batchRiskLevel", "batchLimit"].forEach(id => {
      document.getElementById(id).addEventListener("change", () => {
        const summaryBox = document.getElementById("batchSummary");
        const resultBox = document.getElementById("batchResult");
        offsetInput.value = "0";
        totalStudents = 0;
        previousButton.disabled = true;
        nextButton.disabled = true;
        summaryBox.innerHTML = "";
        resultBox.innerHTML = '<div class="card card-sub">Run a batch to view students matching the selected filters.</div>';
        pageNote.textContent = "Filters changed. Run a batch to load the matching students.";
      });
    });
    offsetInput.addEventListener("change", () => {
      const value = Number(offsetInput.value);
      if (!Number.isInteger(value) || value < 0) offsetInput.value = "0";
    });
  }

  function renderBatchSummary(box, res) {
    const counts = { nudge: 0, resource: 0, escalate: 0 };
    res.results.forEach(r => {
      const decision = normalizeDecision(r.decision);
      if (counts[decision] !== undefined) counts[decision]++;
    });
    box.innerHTML = `
      <div class="grid grid-4">
        <div class="card stat-tile"><div class="stat-num">${res.results.length}</div><div class="stat-label">Succeeded</div></div>
        <div class="card stat-tile"><div class="stat-num">${res.errors.length}</div><div class="stat-label">Errors</div></div>
        <div class="card stat-tile"><div class="stat-num">${counts.escalate}</div><div class="stat-label">Escalated</div></div>
        <div class="card stat-tile"><div class="stat-num" style="font-size:13px; font-family:var(--mono); font-weight:600;">${Pages.escapeHtml(res.run_id)}</div><div class="stat-label">Run ID</div></div>
      </div>`;
  }

  function renderBatchTable(box, res) {
    const rows = res.results.map(r => `
      <tr>
        <td class="cell-mono">${Pages.escapeHtml(r.display_student_id || r.student_id)}</td>
        <td>${Pages.riskBadge(r.risk_level)}</td>
        <td>${Pages.decisionBadge(r.decision)}</td>
        <td>${Math.round((r.router_confidence ?? 0) * 100)}%</td>
        <td style="max-width:360px;">${Pages.escapeHtml((r.rationale || "").slice(0, 140))}${(r.rationale||"").length > 140 ? "…" : ""}
          ${r.mentor_brief ? `<details style="margin-top:8px;"><summary>Mentor briefing</summary><div class="message-box mentor-brief">${Pages.mentorBriefHtml(r.mentor_brief)}</div></details>` : ""}
        </td>
      </tr>`).join("");

    const errorRows = res.errors.map(e => `
      <tr>
        <td class="cell-mono">${Pages.escapeHtml(e.student_id)}</td>
        <td colspan="4" style="color:var(--escalate);">${Pages.escapeHtml(e.error)}</td>
      </tr>`).join("");

    box.innerHTML = `
      <div class="table-wrap">
        <table>
          <thead><tr><th>Student</th><th>Risk</th><th>Decision</th><th>Confidence</th><th>Rationale</th></tr></thead>
          <tbody>${rows || ""}${errorRows}</tbody>
        </table>
      </div>`;
  }

  /* ============================================================
     AUDIT LOG page
     ============================================================ */
  function dedupe(entries) {
    const seen = new Set();
    const out = [];
    for (const e of entries) {
      const key = `${e.dataset}:${e.student_id}`;
      if (seen.has(key)) continue;
      seen.add(key);
      out.push(e);
    }
    return out;
  }

  async function loadAuditLog() {
    const tableBox = document.getElementById("auditTable");
    const noteBox = document.getElementById("auditStorageNote");
    tableBox.innerHTML = `<div class="card empty-state">Loading…</div>`;
    try {
      const page = await Api.getAuditLog({
        limit: parseInt(document.getElementById("auditLimit").value, 10) || 50,
        dataset: document.getElementById("auditDataset").value || null,
        decision: document.getElementById("auditDecision").value || null,
      });
      noteBox.textContent = `Storage backend: ${page.storage_backend} · ${page.total_returned} record(s) returned`;

      let entries = page.entries;
      if (document.getElementById("auditDedup").checked) entries = dedupe(entries);

      window._lastAuditEntries = entries; // for CSV export

      if (!entries.length) {
        tableBox.innerHTML = `<div class="card empty-state">No matching audit records.</div>`;
        return;
      }

      const rows = entries.map(e => `
        <tr>
          <td>${Pages.escapeHtml(e.logged_at || "")}</td>
          <td class="cell-mono">${Pages.escapeHtml(e.dataset)}</td>
          <td class="cell-mono">${Pages.escapeHtml(e.student_id)}</td>
          <td>${Pages.riskBadge(e.risk_level)}</td>
          <td>${Pages.decisionBadge(e.decision)}</td>
          <td>${Math.round((e.router_confidence ?? 0) * 100)}%</td>
          <td><code class="inline">${Pages.escapeHtml(e.lookup_by)}</code></td>
        </tr>`).join("");

      tableBox.innerHTML = `
        <div class="table-wrap">
          <table>
            <thead><tr><th>Logged at</th><th>Dataset</th><th>Student</th><th>Risk</th><th>Decision</th><th>Confidence</th><th>Lookup</th></tr></thead>
            <tbody>${rows}</tbody>
          </table>
        </div>`;
    } catch (err) {
      tableBox.innerHTML = `<div class="card" style="border-color:var(--escalate);">
        <div class="card-title" style="color:var(--escalate);">Could not load audit log (${err.status ?? "network"})</div>
        <p style="font-size:13px; color:var(--ink-soft);">${Pages.escapeHtml(err.message)}</p>
      </div>`;
    }
  }

  function exportAuditCsv() {
    const entries = window._lastAuditEntries || [];
    if (!entries.length) { showToast("Nothing to export yet.", "err"); return; }
    const cols = ["logged_at", "dataset", "student_id", "risk_level", "decision", "router_confidence", "lookup_by", "run_id"];
    const lines = [cols.join(",")];
    entries.forEach(e => {
      lines.push(cols.map(c => `"${String(e[c] ?? "").replace(/"/g, '""')}"`).join(","));
    });
    const blob = new Blob([lines.join("\n")], { type: "text/csv" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "intervention_audit_log.csv";
    a.click();
    URL.revokeObjectURL(a.href);
  }

  function wireAuditPage() {
    document.getElementById("auditRefresh").addEventListener("click", loadAuditLog);
    document.getElementById("auditDedup").addEventListener("change", loadAuditLog);
    document.getElementById("auditExport").addEventListener("click", exportAuditCsv);
    loadAuditLog();
  }

  /* ============================================================
     TEACHER DASHBOARD page (derived from the audit log)
     ============================================================ */
  function normalizeDecision(decision) {
    return String(decision || "").toLowerCase().replace(/_node$/, "");
  }

  async function wireDashboardPage() {
    const statsBox = document.getElementById("dashboardStats");
    const highRiskBox = document.getElementById("dashboardHighRisk");
    const recentBox = document.getElementById("dashboardRecentDecisions");
    const entrySelect = document.getElementById("dashboardEntrySelect");
    const entryDetail = document.getElementById("dashboardEntryDetail");
    const summaryNote = document.getElementById("dashboardSummaryNote");
    let filteredEntries = [];
    const chartIds = [
      "dashboardChart",
      "dashboardRiskChart",
      "dashboardTimelineChart",
      "dashboardDatasetEscalationChart",
    ];

    async function loadDashboard() {
      summaryNote.textContent = "Loading intervention records…";
      try {
        const page = await Api.getAuditLog({
          limit: Math.min(1000, Math.max(10, Number(document.getElementById("dashboardLimit").value) || 200)),
          dataset: document.getElementById("dashboardDataset").value || null,
          decision: document.getElementById("dashboardDecision").value || null,
        });
        const rawEntries = page.entries || [];
        const entries = document.getElementById("dashboardDedup").checked
          ? dedupe(rawEntries)
          : rawEntries;
        filteredEntries = entries;
        const total = entries.length;
        const uniqueStudents = dedupe(entries).length;
        const decisionCounts = { nudge: 0, resource: 0, escalate: 0 };
        const riskCounts = { High: 0, Medium: 0, Low: 0 };
        const dailyCounts = {};
        const datasetCounts = {};
        entries.forEach(entry => {
          const decision = normalizeDecision(entry.decision);
          if (decisionCounts[decision] !== undefined) decisionCounts[decision]++;
          const risk = String(entry.risk_level || "").toLowerCase();
          if (risk === "high") riskCounts.High++;
          else if (risk === "medium") riskCounts.Medium++;
          else if (risk === "low") riskCounts.Low++;
          const dataset = String(entry.dataset || "").toLowerCase();
          if (dataset) {
            datasetCounts[dataset] ||= { total: 0, escalations: 0 };
            datasetCounts[dataset].total++;
            if (decision === "escalate") datasetCounts[dataset].escalations++;
          }
          if (entry.logged_at) {
            const timestamp = Date.parse(entry.logged_at);
            if (!Number.isNaN(timestamp)) {
              const day = new Date(timestamp).toISOString().slice(0, 10);
              dailyCounts[day] ||= { nudge: 0, resource: 0, escalate: 0 };
              if (dailyCounts[day][decision] !== undefined) dailyCounts[day][decision]++;
            }
          }
        });
        const escalations = decisionCounts.escalate;
        const escalationRate = total ? Math.round((escalations / total) * 100) : 0;
        const avgConfidence = total
          ? Math.round(100 * entries.reduce((sum, entry) => sum + (entry.router_confidence ?? 0), 0) / total)
          : 0;

        statsBox.innerHTML = `
          <div class="card stat-tile"><div class="stat-num">${total}</div><div class="stat-label">${document.getElementById("dashboardDedup").checked ? "Unique students" : "Total decisions"}</div></div>
          <div class="card stat-tile"><div class="stat-num">${decisionCounts.nudge}</div><div class="stat-label">Nudges</div></div>
          <div class="card stat-tile"><div class="stat-num">${decisionCounts.resource}</div><div class="stat-label">Resources sent</div></div>
          <div class="card stat-tile"><div class="stat-num">${escalations} <span class="card-sub">(${escalationRate}%)</span></div><div class="stat-label">Escalated to human</div></div>
          <div class="card stat-tile"><div class="stat-num">${avgConfidence}%</div><div class="stat-label">Avg. router confidence</div></div>`;
        summaryNote.textContent = rawEntries.length > entries.length
          ? `Loaded ${rawEntries.length} filtered records; showing ${entries.length} after keeping each student's latest decision. Storage: ${page.storage_backend}.`
          : `${rawEntries.length} filtered record(s) loaded. Storage: ${page.storage_backend}.`;

        if (total) {
          Charts.barBreakdown(document.getElementById("dashboardChart"), [
            { label: "Nudge", value: decisionCounts.nudge, key: "nudge" },
            { label: "Resource", value: decisionCounts.resource, key: "resource" },
            { label: "Escalate", value: decisionCounts.escalate, key: "escalate" },
          ]);
          Charts.barChart(document.getElementById("dashboardRiskChart"), [
            { label: "High", value: riskCounts.High, color: "#e0583b" },
            { label: "Medium", value: riskCounts.Medium, color: "#d39b27" },
            { label: "Low", value: riskCounts.Low, color: "#0ea5a0" },
          ]);

          const days = Object.keys(dailyCounts).sort();
          if (days.length) {
            const colors = { nudge: "#0ea5a0", resource: "#4f5bd5", escalate: "#e0583b" };
            Charts.multiLineChart(
              document.getElementById("dashboardTimelineChart"),
              Object.keys(colors).map(decision => ({
                name: decision,
                color: colors[decision],
                points: days.map(day => ({ x: day, y: dailyCounts[day][decision] })),
              }))
            );
          } else {
            const canvas = document.getElementById("dashboardTimelineChart");
            canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
          }

          const escalationByDataset = Object.entries(datasetCounts).map(([dataset, counts]) => ({
            label: dataset.toUpperCase(),
            value: Math.round(100 * counts.escalations / counts.total),
            color: "#e0583b",
          }));
          Charts.barChart(
            document.getElementById("dashboardDatasetEscalationChart"),
            escalationByDataset,
            { valueSuffix: "%" }
          );
        } else {
          chartIds.forEach(id => {
            const canvas = document.getElementById(id);
            canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
          });
          summaryNote.textContent = "No decisions match the current filters.";
        }

        const highRisk = entries.filter(entry => (entry.risk_level || "").toLowerCase() === "high");
        highRiskBox.innerHTML = highRisk.length
          ? `<div class="dashboard-table-scroll" style="max-height:320px; border:none;"><table>
              <thead><tr><th>Student</th><th>Dataset</th><th>Decision</th></tr></thead>
              <tbody>${highRisk.map(entry => `<tr><td class="cell-mono">${Pages.escapeHtml(entry.display_student_id || entry.student_id)}</td><td class="cell-mono">${Pages.escapeHtml(entry.dataset)}</td><td>${Pages.decisionBadge(entry.decision)}</td></tr>`).join("")}</tbody>
            </table></div>`
          : `<div class="empty-state">No High-risk students in the current filtered records.</div>`;

        const displayColumns = ["logged_at", "run_id", "dataset", "student_id", "display_student_id", "risk_level", "decision", "router_confidence", "source", "escalation_reason"]
          .filter(key => entries.some(entry => Object.prototype.hasOwnProperty.call(entry, key)));
        recentBox.innerHTML = entries.length
          ? `<table>
              <thead><tr>${displayColumns.map(key => `<th>${Pages.escapeHtml(key.replaceAll("_", " "))}</th>`).join("")}</tr></thead>
              <tbody>${entries.map(entry => `<tr>${displayColumns.map(key => `<td>${Pages.escapeHtml(formatDashboardCell(entry[key]))}</td>`).join("")}</tr>`).join("")}</tbody>
            </table>`
          : `<div class="empty-state" style="padding:16px;">No decisions match the current filters.</div>`;
        entrySelect.innerHTML = entries.map((entry, index) => {
          const label = `${entry.logged_at || "Unknown time"} — student ${entry.display_student_id || entry.student_id || "?"} — ${entry.decision || "?"}`;
          return `<option value="${index}">${Pages.escapeHtml(label)}</option>`;
        }).join("");
        entrySelect.disabled = entries.length === 0;
        renderDashboardEntryDetail();
      } catch (err) {
        statsBox.innerHTML = `<div class="card" style="border-color:var(--escalate); grid-column: 1 / -1;">
          <div class="card-title" style="color:var(--escalate);">Could not load dashboard data (${err.status ?? "network"})</div>
          <p style="font-size:13px; color:var(--ink-soft);">${Pages.escapeHtml(err.message)}</p>
        </div>`;
        summaryNote.textContent = "";
        highRiskBox.innerHTML = "";
        recentBox.innerHTML = "";
        entrySelect.innerHTML = '<option value="">No entries available</option>';
        entrySelect.disabled = true;
        entryDetail.innerHTML = "";
      }
    }

    function renderDashboardEntryDetail() {
      const entry = filteredEntries[Number(entrySelect.value)];
      entryDetail.innerHTML = entry
        ? Pages.decisionCard({ ...entry, lookup_by: entry.lookup_by || "audit log" })
        : '<div class="empty-state">Select a decision to view its full details.</div>';
    }

    function downloadCsv(rows, columns, filename) {
      if (!rows.length) {
        showToast("There are no matching records to download.", "err");
        return;
      }
      const csvCell = value => {
        let text = formatDashboardCell(value);
        if (/^[=+\-@]/.test(text)) text = `'${text}`;
        return `"${text.replace(/"/g, '""')}"`;
      };
      const csv = "\uFEFF" + [
        columns.map(csvCell).join(","),
        ...rows.map(row => columns.map(column => csvCell(row[column])).join(",")),
      ].join("\r\n");
      const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
      const link = document.createElement("a");
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    }

    document.getElementById("dashboardRefresh").addEventListener("click", loadDashboard);
    ["dashboardDataset", "dashboardDecision", "dashboardLimit", "dashboardDedup"].forEach(id => {
      document.getElementById(id).addEventListener("change", loadDashboard);
    });
    entrySelect.addEventListener("change", renderDashboardEntryDetail);
    document.getElementById("dashboardDownloadTable").addEventListener("click", () => {
      const columns = ["logged_at", "run_id", "dataset", "student_id", "display_student_id", "risk_level", "decision", "router_confidence", "source", "escalation_reason"]
        .filter(key => filteredEntries.some(entry => Object.prototype.hasOwnProperty.call(entry, key)));
      downloadCsv(filteredEntries, columns, "intervention_decisions.csv");
    });
    document.getElementById("dashboardDownloadFull").addEventListener("click", () => {
      const columns = [...new Set(filteredEntries.flatMap(entry => Object.keys(entry)))];
      downloadCsv(filteredEntries, columns, "intervention_decisions_full.csv");
    });
    loadDashboard();
  }

  function formatDashboardCell(value) {
    if (value === null || value === undefined) return "";
    return typeof value === "object" ? JSON.stringify(value) : String(value);
  }

  /* ============================================================
     CHAT page
     ============================================================ */
  let chatSessionId = null;

  function appendMessage(role, content, meta = "") {
    const log = document.getElementById("chatLog");
    const div = document.createElement("div");
    div.className = "msg " + (role === "human" ? "msg-human" : "msg-ai");
    div.innerHTML = (role === "human" ? Pages.escapeHtml(content).replace(/\n/g, "<br>") : Pages.formatAssistantText(content))
      + (meta ? `<div class="msg-meta">${meta}</div>` : "");
    log.appendChild(div);
    log.scrollTop = log.scrollHeight;
  }

  function createChatStreamMessage() {
    const log = document.getElementById("chatLog");
    const message = document.createElement("div");
    message.className = "msg msg-ai";
    const content = document.createElement("span");
    content.textContent = "Thinking…";
    const meta = document.createElement("div");
    meta.className = "msg-meta";
    message.append(content, meta);
    log.appendChild(message);

    return {
      append(text) {
        content.dataset.answer = (content.dataset.answer || "") + text;
        content.innerHTML = Pages.formatAssistantText(content.dataset.answer);
        log.scrollTop = log.scrollHeight;
      },
      complete(result) {
        meta.textContent = `source: ${result.source}`;
        if (result.tool_calls_made && result.tool_calls_made.length) {
          const tools = document.createElement("div");
          tools.className = "tool-pill-row";
          result.tool_calls_made.forEach(name => {
            const pill = document.createElement("span");
            pill.className = "tool-pill";
            pill.textContent = name;
            tools.appendChild(pill);
          });
          message.appendChild(tools);
        }
        if (result.charts?.gru_timeseries) {
          Charts.appendGruTimeseries(message, result.charts.gru_timeseries);
        }
        log.scrollTop = log.scrollHeight;
      },
      fail(error) {
        content.textContent = `Something went wrong: ${error.message}`;
        meta.textContent = "";
      },
    };
  }

  function wireChatPage() {
    // Shares its session_id with the floating Assistant widget (same
    // sessionStorage key), so a conversation started here continues
    // seamlessly from any other page's floating panel, and vice versa --
    // one conversation, not two independent chat histories.
    chatSessionId = (window.Assistant && Assistant.getSessionId()) || sessionStorage.getItem("ic_chat_session") || null;

    const send = async () => {
      const input = document.getElementById("chatInput");
      const btn = document.getElementById("chatSend");
      const question = input.value.trim();
      if (!question) return;

      appendMessage("human", question);
      input.value = "";
      btn.disabled = true;
      const streamingMessage = createChatStreamMessage();

      try {
        const res = await Api.chatStream(
          {
            question,
            session_id: chatSessionId,
            teacher_id: getTeacherId(),
            page_context: `${PAGE_CONTEXT.chat}\n\n${document.querySelector("#page-chat")?.innerText.replace(/\s+/g, " ").slice(0, 10000) || ""}`,
            chart_gallery: Assistant.captureVisibleCharts(),
          },
          { onStart: data => {
            chatSessionId = data.session_id;
            sessionStorage.setItem("ic_chat_session", chatSessionId);
            if (window.Assistant) Assistant.setSessionId(chatSessionId);
          }, onToken: streamingMessage.append }
        );
        chatSessionId = res.session_id;
        sessionStorage.setItem("ic_chat_session", chatSessionId);
        if (window.Assistant) Assistant.setSessionId(chatSessionId);
        streamingMessage.complete(res);
      } catch (err) {
        streamingMessage.fail(err);
        showToast(err.message, "err");
      } finally {
        btn.disabled = false;
      }
    };

    document.getElementById("chatSend").addEventListener("click", send);
    document.getElementById("chatInput").addEventListener("keydown", (e) => { if (e.key === "Enter") send(); });
    document.getElementById("chatNewSession").addEventListener("click", () => {
      chatSessionId = null;
      sessionStorage.removeItem("ic_chat_session");
      document.getElementById("chatLog").innerHTML = `<div class="msg msg-ai">New session started. Ask me anything about a student, a cohort, or an intervention.</div>`;
      showToast("Started a new chat session.");
    });
  }

  /* ============================================================
     SETTINGS page
     ============================================================ */
  function wireSettingsPage() {
    document.getElementById("apiBaseInput").value = Api.getBase();
    document.getElementById("apiBaseSave").addEventListener("click", async () => {
      const val = document.getElementById("apiBaseInput").value.trim();
      if (!val) return;
      Api.setBase(val);
      const resultBox = document.getElementById("settingsResult");
      resultBox.innerHTML = `<p class="card-sub">Testing connection…</p>`;
      try {
        const h = await Api.agentHealth();
        resultBox.innerHTML = `<p style="color:var(--ok); font-size:13px;">Connected — status: ${Pages.escapeHtml(h.status)}, storage: ${Pages.escapeHtml(h.storage_backend)}</p>`;
        showToast("Connected to backend.", "ok");
      } catch (err) {
        resultBox.innerHTML = `<p style="color:var(--escalate); font-size:13px;">${Pages.escapeHtml(err.message)}</p>`;
        showToast("Could not connect.", "err");
      }
      refreshHealth();
    });
  }

  /* ============================================================
     Boot
     ============================================================ */
  const teacherIdInput = document.getElementById("teacherIdInput");
  teacherIdInput.value = getTeacherId() || "";
  teacherIdInput.addEventListener("input", () => {
    const teacherId = teacherIdInput.value.trim();
    if (teacherId) localStorage.setItem(TEACHER_ID_KEY, teacherId);
    else localStorage.removeItem(TEACHER_ID_KEY);
  });
  Assistant.mount();
  navigate("overview");
  refreshHealth();
  setInterval(refreshHealth, 30000);
})();
