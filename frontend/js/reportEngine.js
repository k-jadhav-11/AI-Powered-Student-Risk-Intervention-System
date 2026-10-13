/* ============================================================
   reportEngine.js — one generic, data-driven page renderer shared
   by Sprint 2 (marks analytics) and Sprint 3 (risk prediction).

   Given a list of report configs (sprint2Config.js / sprint3Config.js),
   it renders: a sub-nav of reports, a dynamic form built from each
   report's `fields`, a Run button, and a generic result renderer that
   handles every shape the real backend actually returns (rows / top+bottom
   / columns+rows pivot / nested objects / scalars) without per-endpoint
   custom code -- except the two genuinely special cases (Sprint 2's
   "Load / Seed Data" and Sprint 3's "Predict Risk") which get their own
   small renderers via the `custom` field.
   ============================================================ */

const ReportEngine = (() => {

  const sprint3Renderer = (() => {
    function appendSprint3Chart(resultEl, sprintId, key, title, height = 260) {
      const section = document.createElement("section");
      const chartId = `${sprintId}_chart_${key}`;
      section.innerHTML = `<div class="card-title" style="margin-top:18px;">${Pages.escapeHtml(title)}</div>
        <canvas id="${chartId}" style="width:100%; height:${height}px; display:block;"></canvas>`;
      resultEl.appendChild(section);
      return document.getElementById(chartId);
    }

    function appendCsvButton(resultEl, filename, rows) {
      if (!Array.isArray(rows) || !rows.length) return;
      const button = document.createElement("button");
      button.className = "btn btn-ghost btn-sm";
      button.style.marginTop = "14px";
      button.textContent = "Download CSV";
      button.addEventListener("click", () => downloadCsv(filename, rows));
      resultEl.appendChild(button);
    }

    function renderSprint3Extras(resultEl, sprintId, report, data) {
      if (report.id === "risk-scores" && data.rows) {
        const counts = ["High", "Medium", "Low"].map(level => ({
          label: level,
          value: data.risk_level_counts?.[level] || 0,
          color: level === "High" ? "#e0583b" : level === "Medium" ? "#d39b27" : "#0ea5a0",
        }));
        Charts.barChart(appendSprint3Chart(resultEl, sprintId, "risk-levels", "Risk level distribution"), counts);
        appendCsvButton(resultEl, `${report.id}.csv`, data.rows);
      } else if (report.id === "model-performance") {
        const metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
          .filter(key => typeof data[key] === "number")
          .map(key => ({ label: key.replace("_", " "), value: data[key] }));
        if (metrics.length) {
          Charts.barChart(appendSprint3Chart(resultEl, sprintId, "performance-metrics", "Test-set metrics"), metrics);
        }
        if (data.confusion_matrix) {
          const matrix = data.confusion_matrix;
          const table = document.createElement("div");
          table.innerHTML = `<div class="card-title" style="margin-top:18px;">Confusion matrix</div>
            ${tableFromRows([
              { actual: "Not at risk", "Predicted not at risk": matrix.true_negative, "Predicted at risk": matrix.false_positive },
              { actual: "At risk", "Predicted not at risk": matrix.false_negative, "Predicted at risk": matrix.true_positive },
            ])}`;
          resultEl.appendChild(table);
        }
      } else if (report.id === "student-risk" || report.id === "predict") {
        if (typeof data.risk_score === "number") {
          Charts.barChart(
            appendSprint3Chart(resultEl, sprintId, "student-risk", `Risk score — ${data.risk_level || ""}`),
            [{ label: data.display_student_id || data.student_id || data.dataset, value: data.risk_score, color: riskColor(data.risk_level) }]
          );
        }
      } else if (report.id === "student-shap") {
        const factors = ["academic_factors", "engagement_factors", "contextual_factors"]
          .flatMap(key => Array.isArray(data[key]) ? data[key] : []);
        if (factors.length) {
          Charts.barChart(
            appendSprint3Chart(resultEl, sprintId, "shap-factors", "SHAP factor contributions"),
            factors.map(factor => ({
              label: factor.feature,
              value: factor.contribution,
              color: factor.direction === "increases_risk" ? "#e0583b" : "#0ea5a0",
            }))
          );
        }
      } else if (report.id === "cohort-summary" && data.risk_level_counts) {
        Charts.barChart(
          appendSprint3Chart(resultEl, sprintId, "cohort-risk", "Cohort risk levels"),
          Object.entries(data.risk_level_counts).map(([label, value]) => ({ label, value, color: riskColor(label) }))
        );
      } else if (report.id === "personas" && data.rows) {
        const counts = {};
        data.rows.forEach(row => {
          const label = row.persona_name || row.persona_cluster || "Unknown";
          counts[label] = (counts[label] || 0) + 1;
        });
        Charts.barChart(
          appendSprint3Chart(resultEl, sprintId, "personas", "Students by performance persona"),
          Object.entries(counts).map(([label, value]) => ({ label, value }))
        );
        appendCsvButton(resultEl, `${report.id}.csv`, data.rows);
      } else if (report.id === "clustering-optimal-k") {
        ["inertia", "silhouette"].forEach(metric => {
          const series = data[metric] || {};
          const entries = Object.entries(series);
          if (!entries.length) return;
          const values = entries.map(([, value]) => Number(value));
          Charts.lineChart(
            appendSprint3Chart(resultEl, sprintId, `optimal-k-${metric}`, `${metric} by k`, 220),
            entries.map(([k, value]) => ({ x: k, y: Number(value) })),
            { autoScale: true, minY: Math.min(...values), maxY: Math.max(...values) }
          );
        });
      } else if (report.id === "clustering" && data.points) {
        const palette = ["#4f5bd5", "#0ea5a0", "#e0583b", "#d39b27", "#9155b5", "#3984c6", "#687384"];
        const centroids = document.createElement("div");
        centroids.innerHTML = `<div class="card-title" style="margin-top:18px;">Cluster centroids</div>${tableFromRows(data.centroids || [])}`;
        resultEl.appendChild(centroids);
        Charts.scatterChart(
          appendSprint3Chart(resultEl, sprintId, "clusters", `Student clusters (k=${data.k})`, 320),
          data.points.slice(0, 1200).map(point => ({
            x: point.pc1,
            y: point.pc2,
            color: palette[Number(point.cluster_id) % palette.length],
          })),
          { xLabel: "PCA component 1", yLabel: "PCA component 2" }
        );
        const assignmentRows = data.points.slice(0, 50);
        const assignments = document.createElement("div");
        assignments.innerHTML = `<div class="card-title" style="margin-top:18px;">Student assignments (first ${assignmentRows.length} of ${data.points.length})</div>
          ${tableFromRows(assignmentRows)}`;
        resultEl.appendChild(assignments);
        appendCsvButton(resultEl, `${report.id}-student-assignments.csv`, data.points);
      } else if (report.id === "gru-student-directory" && data.students) {
        const counts = {};
        data.students.forEach(student => {
          counts[student.final_result] = (counts[student.final_result] || 0) + 1;
        });
        Charts.barChart(
          appendSprint3Chart(resultEl, sprintId, "gru-outcomes", "Sample by final outcome"),
          Object.entries(counts).map(([label, value]) => ({ label, value }))
        );
        appendCsvButton(resultEl, `${report.id}.csv`, data.students);
      } else if (report.id === "gru-timeseries") {
        const weekly = data.weekly_engagement || [];
        if (weekly.length) {
          Charts.lineChart(
            appendSprint3Chart(resultEl, sprintId, "gru-clicks", "Weekly VLE clicks", 220),
            weekly.map(row => ({ x: row.week_number, y: row.total_clicks })),
            { autoScale: true }
          );
          Charts.lineChart(
            appendSprint3Chart(resultEl, sprintId, "gru-days", "Active days per week", 220),
            weekly.map(row => ({ x: row.week_number, y: row.days_active })),
            { autoScale: true }
          );
        }
        const trajectory = data.risk_trajectory || [];
        if (trajectory.length) {
          const scoreKey = Object.keys(trajectory[0]).find(key => key !== "week_number");
          Charts.lineChart(
            appendSprint3Chart(resultEl, sprintId, "gru-risk", "GRU risk trajectory", 220),
            trajectory.map(row => ({ x: row.week_number, y: row[scoreKey] })),
            { autoScale: true }
          );
        }
      }
    }

    function riskColor(level) {
      if (String(level || "").toLowerCase() === "high") return "#e0583b";
      if (String(level || "").toLowerCase() === "medium") return "#d39b27";
      return "#0ea5a0";
    }

    return { renderExtras: renderSprint3Extras };
  })();

  function fieldHtml(f, idPrefix) {
    const id = `${idPrefix}_${f.name}`;
    const label = `<label class="label">${Pages.escapeHtml(f.label)}</label>`;
    if (f.type === "select-class" || f.type === "select-subject") {
      const allLabel = f.allLabel || "Any";
      return `<div class="field"><label class="label">${Pages.escapeHtml(f.label)}</label>
        <select id="${id}" data-dynamic="${f.type}" data-all-label="${Pages.escapeHtml(allLabel)}"><option value="">Loading…</option></select></div>`;
    }

    if (f.type === "static-select") {
      const opts = f.options.map(o => {
        const value = typeof o === "object" ? o.value : o;
        const optLabel = typeof o === "object" ? o.label : o;
        return `<option value="${Pages.escapeHtml(value)}" ${value === f.default ? "selected" : ""}>${Pages.escapeHtml(optLabel)}</option>`;
      }).join("");
      return `<div class="field">${label}<select id="${id}">${opts}</select></div>`;
    }
    if (f.type === "static-multiselect") {
      const selected = f.default || [];
      const opts = f.options.map(o => {
        const value = typeof o === "object" ? o.value : o;
        const optLabel = typeof o === "object" ? o.label : o;
        const isSelected = selected.includes(value);
        return `<option value="${Pages.escapeHtml(value)}" ${isSelected ? "selected" : ""}>${Pages.escapeHtml(optLabel)}</option>`;
      }).join("");
      return `<div class="field">${label}<select id="${id}" multiple size="${Math.min(f.options.length, 4)}">${opts}</select></div>`;
    }
    if (f.type === "select-gru-student") {
      return `<div class="field">${label}<select id="${id}" data-dynamic="select-gru-student"><option value="">Loading sample students…</option></select></div>`;
    }
    if (f.type === "number") {
      const minAttr = f.min !== undefined ? ` min="${f.min}"` : "";
      const maxAttr = f.max !== undefined ? ` max="${f.max}"` : "";
      return `<div class="field">${label}<input type="number" id="${id}" step="${f.step || 1}"${minAttr}${maxAttr} value="${f.default ?? ""}"></div>`;
    }
    return `<div class="field">${label}<input type="text" id="${id}" placeholder="${Pages.escapeHtml(f.placeholder || "")}" value="${Pages.escapeHtml(f.default || "")}"></div>`;
  }

  async function populateDynamicSelects(root, idPrefix, sprintId) {
    const classSelects = root.querySelectorAll('select[data-dynamic="select-class"]');
    const subjectSelects = root.querySelectorAll('select[data-dynamic="select-subject"]');
    const gruStudentSelects = root.querySelectorAll('select[data-dynamic="select-gru-student"]');
    const failures = [];
    const fillOptions = (sel, items) => {
      const allLabel = sel.dataset.allLabel || "Any";
      const selectedValue = sel.value;
      sel.innerHTML = `<option value="">${Pages.escapeHtml(allLabel)}</option>` + items.map(item => {
        const value = typeof item === "object" ? item.value : item;
        const label = typeof item === "object" ? item.label : item;
        return `<option value="${Pages.escapeHtml(value)}">${Pages.escapeHtml(label)}</option>`;
      }).join("");
      sel.value = items.some(item => String(typeof item === "object" ? item.value : item) === selectedValue)
        ? selectedValue : (items.length ? String(typeof items[0] === "object" ? items[0].value : items[0]) : "");
      sel.disabled = false;
    };

    async function loadOptions(selects, label, load, getItems) {
      if (!selects.length) return;
      try {
        const data = await load();
        selects.forEach(sel => fillOptions(sel, getItems(data)));
      } catch (err) {
        selects.forEach(sel => {
          sel.innerHTML = `<option value="">Unable to load ${label}</option>`;
          sel.disabled = true;
        });
        failures.push(`${label}: ${err.message || "request failed"}`);
      }
    }

    await Promise.all([
      loadOptions(classSelects, "classes", () => Api.sprint2.classes(), data => data.classes),
      loadOptions(subjectSelects, "subjects", () => Api.sprint2.subjects(), data => data.subjects),
      loadOptions(
        gruStudentSelects,
        "sample OULAD students",
        () => Api.sprint3.studentDirectory(15),
        data => (data.students || []).map(student => ({
          value: student.id_student,
          label: `${student.id_student} · ${student.code_module} · ${student.region} · ${student.final_result}`,
        }))
      ),
    ]);
    if (sprintId === "sprint3") {
      const studentChoice = root.querySelector("#sprint3_f_student_choice");
      const studentId = root.querySelector("#sprint3_f_id_student");
      if (studentChoice?.value && studentId && !studentId.value) studentId.value = studentChoice.value;
    }

    const status = root.querySelector(`#${sprintId}_filterStatus`);
    if (status) {
      status.innerHTML = failures.length
        ? `<div class="card-sub" role="alert" style="color:var(--escalate);">Could not load report filters — ${Pages.escapeHtml(failures.join("; "))}</div>`
        : "";
    }
  }

  function readValues(root, fields, idPrefix) {
    const values = {};
    fields.forEach(f => {
      const el = root.querySelector(`#${idPrefix}_${f.name}`);
      if (!el) return;
      if (f.type === "static-multiselect") {
        const selected = Array.from(el.selectedOptions, option => option.value);
        if (!selected.length && f.optional) return;
        values[f.name] = selected;
        return;
      }
      const v = el.value;
      if (v === "" && f.optional) return;
      values[f.name] = f.type === "number" ? Number(v) : v;
    });
    return values;
  }

  /* ---------------- generic result rendering ---------------- */

  function tableFromRows(rows) {
    if (!rows || !rows.length) return `<div class="empty-state">No rows returned.</div>`;
    const cols = Object.keys(rows[0]);
    return `<div class="table-wrap"><table>
      <thead><tr>${cols.map(c => `<th>${Pages.escapeHtml(c)}</th>`).join("")}</tr></thead>
      <tbody>${rows.map(r => `<tr>${cols.map(c => `<td>${renderCell(r[c])}</td>`).join("")}</tr>`).join("")}</tbody>
    </table></div>`;
  }

  function renderCell(v) {
    if (v === null || v === undefined) return '<span style="color:var(--ink-faint)">—</span>';
    if (typeof v === "number") return Number.isInteger(v) ? v : v.toFixed(3);
    if (typeof v === "boolean") return v ? "true" : "false";
    if (Array.isArray(v) || typeof v === "object") return `<code class="inline">${Pages.escapeHtml(JSON.stringify(v))}</code>`;
    return Pages.escapeHtml(String(v));
  }

  function kvGrid(obj) {
    const entries = Object.entries(obj).filter(([k]) => !["rows", "columns"].includes(k));
    return `<div class="kv-grid">${entries.map(([k, v]) => `
      <div class="kv-item">
        <div class="kv-label">${Pages.escapeHtml(k)}</div>
        <div class="kv-value">${Array.isArray(v) || (v && typeof v === "object") ? "" : renderCell(v)}</div>
      </div>`).join("")}
    </div>`;
  }

  function renderAny(data) {
    if (data === null || data === undefined) return `<div class="empty-state">No data.</div>`;

    if (data.risk_level_counts && Array.isArray(data.top_display_student_ids)) {
      const topRows = data.top_display_student_ids.map((studentId, index) => ({
        student_id: studentId,
        risk_score: data.top_risk_scores?.[index] ?? null,
      }));
      return `<div class="kv-grid">
          ${Object.entries(data.risk_level_counts).map(([level, count]) => `<div class="kv-item"><div class="kv-label">${Pages.escapeHtml(level)} risk</div><div class="kv-value">${renderCell(count)}</div></div>`).join("")}
        </div>
        <div class="card-sub" style="margin-top:12px;">${data.total_students} students in this cohort${data.filter_applied ? ` · filter: ${Pages.escapeHtml(data.filter_applied)}` : ""}</div>
        <div class="card-title" style="margin-top:14px;">Highest-risk students</div>${tableFromRows(topRows)}`;
    }

    if (Array.isArray(data.centroids) && Array.isArray(data.points)) {
      return `<div class="card-sub">${data.points.length.toLocaleString()} students assigned across ${data.k} clusters.</div>
        <div class="card-title" style="margin-top:12px;">Cluster centroids</div>${tableFromRows(data.centroids)}`;
    }

    // {top:[...], bottom:[...]}
    if (data.top && data.bottom) {
      return `<div class="grid grid-2">
        <div><div class="card-title">Top</div>${tableFromRows(data.top)}</div>
        <div><div class="card-title">Bottom</div>${tableFromRows(data.bottom)}</div>
      </div>`;
    }

    // {overall:[...]|null, by_subject:{columns,rows}|null}  (class-comparison)
    if (("overall" in data) && ("by_subject" in data)) {
      let out = "";
      if (data.overall) out += `<div class="card-title">Overall</div>${tableFromRows(data.overall)}`;
      if (data.by_subject) out += `<div class="card-title" style="margin-top:16px;">By subject</div>${renderAny(data.by_subject)}`;
      return out || `<div class="empty-state">No data.</div>`;
    }

    // {columns:[...], rows:[...]} pivot
    if (Array.isArray(data.columns) && Array.isArray(data.rows)) {
      if (!data.rows.length) return `<div class="empty-state">No rows returned.</div>`;
      const isObjRows = typeof data.rows[0] === "object" && !Array.isArray(data.rows[0]);
      const cols = data.columns;
      return `<div class="table-wrap"><table>
        <thead><tr>${cols.map(c => `<th>${Pages.escapeHtml(c)}</th>`).join("")}</tr></thead>
        <tbody>${data.rows.map(r => `<tr>${cols.map(c => `<td>${renderCell(isObjRows ? r[c] : r[cols.indexOf(c)])}</td>`).join("")}</tr>`).join("")}</tbody>
      </table></div>`;
    }

    // {rows:[...]}
    if (Array.isArray(data.rows)) {
      const header = kvGridKeysOnly(data);
      return header + tableFromRows(data.rows);
    }

    // plain array
    if (Array.isArray(data)) return tableFromRows(data);

    // {weekly_engagement:[...], risk_trajectory:[...], ...} (GRU timeseries) -- multi-table object
    const arrayKeys = Object.keys(data).filter(k => Array.isArray(data[k]) && data[k].length && typeof data[k][0] === "object");
    if (arrayKeys.length) {
      const scalarPart = kvGrid(Object.fromEntries(Object.entries(data).filter(([k]) => !arrayKeys.includes(k))));
      const tables = arrayKeys.map(k => `<div class="card-title" style="margin-top:16px;">${Pages.escapeHtml(k)}</div>${tableFromRows(data[k])}`).join("");
      return scalarPart + tables;
    }

    // confusion_matrix / nested object with no array -> kv grid + raw json fallback
    if (typeof data === "object") {
      return kvGrid(data) + `<details style="margin-top:14px;"><summary style="cursor:pointer; font-size:12px; color:var(--ink-faint);">Raw response</summary>
        <pre style="background:var(--bg-card-soft); border:1px solid var(--border); border-radius:8px; padding:12px; font-size:11.5px; overflow-x:auto; margin-top:8px;">${Pages.escapeHtml(JSON.stringify(data, null, 2))}</pre></details>`;
    }

    return `<div class="message-box">${Pages.escapeHtml(String(data))}</div>`;
  }

  function kvGridKeysOnly(data) {
    const rest = Object.fromEntries(Object.entries(data).filter(([k]) => k !== "rows" && typeof data[k] !== "object"));
    return Object.keys(rest).length ? kvGrid(rest) : "";
  }

  /* ---------------- page shell ---------------- */

  // sprintId: "sprint2" | "sprint3". reports: config array.
  // opts.datasetToggle: true to show a uci/oulad switch at the top (sprint3).
  function renderShell(sprintId, reports, opts = {}) {
    const nav = reports.map(r => `<button class="report-nav-item" data-report="${r.id}">${Pages.escapeHtml(r.title)}</button>`).join("");
    return `
      <div class="page" id="page-${sprintId}">
        <div class="section-head">
          <h1>${opts.title}</h1>
          <p>${opts.subtitle}</p>
        </div>
        ${opts.datasetToggle ? `
        <div class="card" style="margin-bottom:16px; display:flex; align-items:center; gap:14px;">
          <span class="label" style="margin:0;">Dataset</span>
          <select id="${sprintId}_datasetToggle" style="max-width:160px;">
            <option value="uci">uci</option>
            <option value="oulad">oulad</option>
          </select>
          <span class="card-sub" style="margin:0;">Applies to every report below except the GRU time-series pages (OULAD-only).</span>
        </div>` : ""}
        <div class="grid report-layout">
          <div class="card" style="padding:12px;">
            <div class="card-title" style="padding:4px 8px 10px;">${reports.length} reports</div>
            <div id="${sprintId}_nav" style="display:flex; flex-direction:column; gap:2px; max-height:70vh; overflow-y:auto;">${nav}</div>
          </div>
          <div id="${sprintId}_body"></div>
        </div>
      </div>`;
  }

  function renderReportBody(sprintId, report, dataset) {
    const idPrefix = `${sprintId}_f`;
    const fieldsHtml = report.fields.map(f => fieldHtml(f, idPrefix)).join("");
    return `
      <div class="card">
        <div class="card-title">${Pages.escapeHtml(report.title)}</div>
        <div class="card-sub">${Pages.escapeHtml(report.desc)}</div>
        <div class="row">${fieldsHtml}</div>
        <div id="${sprintId}_filterStatus" aria-live="polite"></div>
        <button class="btn btn-primary" id="${sprintId}_run">${report.method === "POST" ? "Submit" : "Run"}</button>
        <div id="${sprintId}_result" style="margin-top:20px;"></div>
      </div>`;
  }

  /* ---------------- wiring ---------------- */

  function wire(sprintId, reports, opts = {}) {
    let currentReport = reports[0];
    const navEl = document.getElementById(`${sprintId}_nav`);
    const bodyEl = document.getElementById(`${sprintId}_body`);
    const datasetSel = opts.datasetToggle ? document.getElementById(`${sprintId}_datasetToggle`) : null;

    function currentDataset() {
      return datasetSel ? datasetSel.value : null;
    }

    function selectReport(id) {
      currentReport = reports.find(r => r.id === id) || reports[0];
      navEl.querySelectorAll(".report-nav-item").forEach(b => b.classList.toggle("active", b.dataset.report === currentReport.id));
      bodyEl.innerHTML = renderReportBody(sprintId, currentReport, currentDataset());
      if (currentReport.id === "gru-timeseries") {
        const studentChoice = document.getElementById(`${sprintId}_f_student_choice`);
        const studentId = document.getElementById(`${sprintId}_f_id_student`);
        studentChoice.addEventListener("change", () => {
          if (studentChoice.value) studentId.value = studentChoice.value;
        });
      }
      populateDynamicSelects(bodyEl, `${sprintId}_f`, sprintId);

      if (currentReport.custom === "load-data") wireLoadData(sprintId, currentReport);
      else if (currentReport.custom === "predict") wirePredict(sprintId, currentReport, currentDataset());
      else wireGenericRun(sprintId, currentReport, currentDataset());

      if (window.Assistant) window.Assistant.setPageContext(
        `${opts.title} → ${currentReport.title}${currentDataset() ? ` (dataset: ${currentDataset()})` : ""}`
      );
    }

    navEl.addEventListener("click", (e) => {
      const btn = e.target.closest(".report-nav-item");
      if (btn) selectReport(btn.dataset.report);
    });

    if (datasetSel) {
      datasetSel.addEventListener("change", () => {
        // re-filter OULAD-only reports out of the nav when uci is selected
        navEl.querySelectorAll(".report-nav-item").forEach(btn => {
          const r = reports.find(x => x.id === btn.dataset.report);
          btn.style.display = (r.ouladOnly && currentDataset() !== "oulad") ? "none" : "";
        });
        if (currentDataset() !== "oulad" && currentReport.ouladOnly) {
          selectReport(reports.find(r => !r.ouladOnly).id);
          return;
        }
        selectReport(currentReport.id);
      });
      // hide OULAD-only on initial load if needed
      navEl.querySelectorAll(".report-nav-item").forEach(btn => {
        const r = reports.find(x => x.id === btn.dataset.report);
        if (r.ouladOnly && currentDataset() !== "oulad") btn.style.display = "none";
      });
    }

    selectReport(currentReport.id);
  }

  function wireGenericRun(sprintId, report, dataset) {
    const idPrefix = `${sprintId}_f`;
    document.getElementById(`${sprintId}_run`).addEventListener("click", async () => {
      const btn = document.getElementById(`${sprintId}_run`);
      const resultEl = document.getElementById(`${sprintId}_result`);
      const values = readValues(document, report.fields, idPrefix);
      const missing = report.fields.find(field =>
        !field.optional && !field.clientOnly && document.getElementById(`${idPrefix}_${field.name}`)?.value === ""
      );
      if (missing) {
        Toast.show(`Enter a value for ${missing.label}.`, "err");
        return;
      }
      btn.disabled = true;
      btn.innerHTML = '<span class="spinner"></span> Running…';
      resultEl.innerHTML = "";
      try {
        const path = report.path(values, dataset);
        const query = { ...values };
        report.fields.filter(field => field.pathParam || field.clientOnly)
          .forEach(field => { delete query[field.name]; });
        const data = await Api.request(path, { method: report.method, query: report.method === "GET" ? query : null });
        resultEl.innerHTML = renderAny(data);
        renderExtras(resultEl, sprintId, report, data, values);
      } catch (err) {
        resultEl.innerHTML = errorBox(err);
      } finally {
        btn.disabled = false;
        btn.textContent = report.method === "POST" ? "Submit" : "Run";
      }
    });
  }

  /* ---------------- charts + CSV export (streamlit parity) ---------------- */

  function csvEscape(v) {
    if (v === null || v === undefined) return "";
    const s = String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  }

  function downloadCsv(filename, rows) {
    if (!rows || !rows.length) { Toast.show("Nothing to export.", "err"); return; }
    const cols = Object.keys(rows[0]);
    const lines = [cols.map(csvEscape).join(",")];
    rows.forEach(r => lines.push(cols.map(c => csvEscape(r[c])).join(",")));
    const blob = new Blob([lines.join("\n")], { type: "text/csv" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = filename.endsWith(".csv") ? filename : `${filename}.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  function rowsForCsv(report, data) {
    if (!data) return [];
    if (Array.isArray(data.rows)) return data.rows;
    if (data.top || data.bottom) {
      return [...(data.top || []).map(r => ({ group: "Top", ...r })), ...(data.bottom || []).map(r => ({ group: "Bottom", ...r }))];
    }
    if (data.by_subject && data.by_subject.rows) return data.by_subject.rows;
    if (data.overall) return data.overall;
    if (Array.isArray(data.scores)) return data.scores;
    if (Array.isArray(data)) return data;
    return [];
  }

  function csvButtonHtml(sprintId) {
    return `<button class="btn btn-ghost btn-sm" id="${sprintId}_csvBtn" style="margin-top:14px;">Download CSV</button>`;
  }

  function chartContainerHtml(sprintId, heightPx) {
    return `<div class="card-title" id="${sprintId}_chartTitle" style="margin-top:18px;"></div>
      <canvas id="${sprintId}_chart" style="width:100%; height:${heightPx || 220}px; display:block;"></canvas>
      <div id="${sprintId}_chartExtra"></div>`;
  }

  // Builds the right chart for a report's data and appends the CSV button --
  // the client-side equivalent of streamlit_app.py's st.pyplot(fig) +
  // export_buttons(df, fig, ...) for each page_* function.
  function renderExtras(resultEl, sprintId, report, data, values) {
    if (sprintId === "sprint3") {
      sprint3Renderer.renderExtras(resultEl, sprintId, report, data);
      return;
    }
    const csvRows = rowsForCsv(report, data);

    if (report.id === "student-report") {
      renderStudentReportChart(resultEl, data);
    } else if (report.id === "class-comparison") {
      renderClassComparisonChart(resultEl, data, values);
    } else if (report.id === "top-bottom") {
      renderTopBottomChart(resultEl, data, values);
    } else if (report.chart) {
      renderConfiguredChart(resultEl, report, data, values);
    }

    if (report.csv && csvRows.length) {
      const wrap = document.createElement("div");
      wrap.innerHTML = csvButtonHtml(sprintId);
      resultEl.appendChild(wrap.firstElementChild);
      document.getElementById(`${sprintId}_csvBtn`).addEventListener("click", () => {
        downloadCsv(`${report.id}.csv`, csvRows);
      });
    }
  }

  function appendChartShell(resultEl, title) {
    const wrap = document.createElement("div");
    wrap.innerHTML = chartContainerHtml("sprint2");
    resultEl.appendChild(wrap);
    document.getElementById("sprint2_chartTitle").textContent = title || "";
    return document.getElementById("sprint2_chart");
  }

  function renderConfiguredChart(resultEl, report, data, values) {
    const rows = Array.isArray(data.rows) ? data.rows : [];
    if (!rows.length) return;
    const spec = report.chart;
    const title = spec.title ? spec.title(values, values) : report.title;

    if (spec.type === "bar") {
      const items = rows.map(r => {
        let label;
        if (spec.labelKey === "__subject_student") label = `${r.subject} — ${r.student_name}`;
        else if (spec.labelKey === null) {
          const otherKey = Object.keys(r).find(k => k !== spec.valueKey);
          label = r[otherKey];
        } else label = r[spec.labelKey];
        return { label, value: r[spec.valueKey], highlight: spec.highlightKey ? !!r[spec.highlightKey] : false };
      });
      const canvas = appendChartShell(resultEl, title);
      Charts.barChart(canvas, items);
    } else if (spec.type === "scatter") {
      const points = rows.map(r => ({ x: r[spec.xKey], y: r[spec.yKey], highlight: !!spec.highlightAll }));
      const canvas = appendChartShell(resultEl, title);
      Charts.scatterChart(canvas, points, { xLabel: spec.xLabel, yLabel: spec.yLabel });
    }
  }

  function renderTopBottomChart(resultEl, data, values) {
    const label = values.class_name || "All Classes";
    const wrap = document.createElement("div");
    wrap.innerHTML = `
      <div class="card-title" style="margin-top:18px;">Top/Bottom Performers — ${Pages.escapeHtml(label)}${values.subject ? " — " + Pages.escapeHtml(values.subject) : ""}</div>
      <div class="grid grid-2">
        <div><div class="card-sub">Top</div><canvas id="sprint2_chartTop" style="width:100%; height:200px; display:block;"></canvas></div>
        <div><div class="card-sub">Bottom</div><canvas id="sprint2_chartBottom" style="width:100%; height:200px; display:block;"></canvas></div>
      </div>`;
    resultEl.appendChild(wrap);
    Charts.barChart(document.getElementById("sprint2_chartTop"), (data.top || []).map(r => ({ label: r.name, value: r.average_score })), { color: "#0ea5a0" });
    Charts.barChart(document.getElementById("sprint2_chartBottom"), (data.bottom || []).map(r => ({ label: r.name, value: r.average_score })), { color: "#e0583b" });
  }

  function renderClassComparisonChart(resultEl, data, values) {
    const wrap = document.createElement("div");
    wrap.innerHTML = `<div class="card-title" style="margin-top:18px;">Class vs Subject Comparison${values.subject ? " — " + Pages.escapeHtml(values.subject) : ""}</div>
      <div id="sprint2_heatmap"></div>`;
    resultEl.appendChild(wrap);
    if (data.by_subject && data.by_subject.rows && data.by_subject.rows.length) {
      Charts.heatmapTable(document.getElementById("sprint2_heatmap"), data.by_subject.columns, data.by_subject.rows, { indexCol: data.by_subject.columns[0] });
    } else if (data.overall && data.overall.length) {
      const canvas = document.createElement("canvas");
      canvas.style.cssText = "width:100%; height:200px; display:block;";
      document.getElementById("sprint2_heatmap").appendChild(canvas);
      Charts.barChart(canvas, data.overall.map(r => ({ label: r.class_name, value: r.average_score })));
    }
  }

  function renderStudentReportChart(resultEl, data) {
    if (!data || !Array.isArray(data.scores) || !data.scores.length) return;
    // Group by test_name (preserving first-seen order), mean score per test --
    // same grouping streamlit_app.py's page_student_report() does before plotting.
    const order = [];
    const sums = {};
    data.scores.forEach(r => {
      if (!(r.test_name in sums)) { sums[r.test_name] = { total: 0, n: 0 }; order.push(r.test_name); }
      sums[r.test_name].total += Number(r.score) || 0;
      sums[r.test_name].n += 1;
    });
    const points = order.map(name => ({ x: name, y: sums[name].total / sums[name].n }));

    const wrap = document.createElement("div");
    wrap.innerHTML = `
      <div class="grid grid-3" style="margin-top:4px;">
        <div class="card stat-tile"><div class="stat-num">${data.student_average ?? "—"}</div><div class="stat-label">Student average</div></div>
        <div class="card stat-tile"><div class="stat-num">${data.class_average ?? "—"}</div><div class="stat-label">Class ${Pages.escapeHtml(String(data.class_name ?? ""))} average</div></div>
        <div class="card stat-tile"><div class="stat-num">${data.zscore ?? "—"}</div><div class="stat-label">Z-score (vs own class)</div></div>
      </div>
      <p style="font-size:13px; margin-top:10px;"><strong>Trend:</strong> ${Pages.escapeHtml(data.trend || "—")}</p>
      ${data.zscore < -1 ? `<div class="message-box" style="border-color:var(--escalate); color:var(--escalate);">Flag: Consistently underperforming relative to class.</div>` : ""}
      <div class="card-title" style="margin-top:14px;">${Pages.escapeHtml(data.name || "")} — score trend by test</div>
      <canvas id="sprint2_studentTrend" style="width:100%; height:220px; display:block;"></canvas>`;
    resultEl.appendChild(wrap);
    Charts.lineChart(document.getElementById("sprint2_studentTrend"), points, { referenceY: data.class_average });
  }

  function wireLoadData(sprintId, report) {
    document.getElementById(`${sprintId}_run`).addEventListener("click", async () => {
      const btn = document.getElementById(`${sprintId}_run`);
      const resultEl = document.getElementById(`${sprintId}_result`);
      const dataDir = document.getElementById(`${sprintId}_f_data_dir`).value.trim() || null;
      btn.disabled = true;
      btn.innerHTML = '<span class="spinner"></span> Loading…';
      try {
        const data = await Api.sprint2.loadData(dataDir);
        resultEl.innerHTML = renderAny(data);
      } catch (err) {
        resultEl.innerHTML = errorBox(err);
      } finally {
        btn.disabled = false;
        btn.textContent = "Submit";
      }
    });
  }

  function wirePredict(sprintId, report, dataset) {
    const bodyEl = document.getElementById(`${sprintId}_body`);
    const features = SPRINT3_FEATURES[dataset] || [];
    const groups = dataset === "uci" ? [
      { name: "school", label: "School", options: [{ label: "GP", values: {} }, { label: "MS", values: { school_MS: 1 } }] },
      { name: "sex", label: "Sex", options: [{ label: "Female", values: {} }, { label: "Male", values: { sex_M: 1 } }] },
      { name: "address", label: "Home address", options: [{ label: "Rural", values: {} }, { label: "Urban", values: { address_U: 1 } }] },
      { name: "family_size", label: "Family size", options: [{ label: "<=3", values: { famsize_LE3: 1 } }, { label: ">3", values: {} }] },
      { name: "subject", label: "Subject", options: [{ label: "Mathematics", values: {} }, { label: "Portuguese", values: { subject_Portuguese: 1 } }] },
    ] : [
      { name: "code_module", label: "Module", options: ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"].map(module => ({
        label: module, values: module === "AAA" ? {} : { [`code_module_${module}`]: 1 },
      })) },
      { name: "region", label: "Region", options: [
        "East Anglian Region", "East Midlands Region", "Ireland", "London Region", "North Region",
        "North Western Region", "Scotland", "South East Region", "South Region", "South West Region",
        "Wales", "West Midlands Region", "Yorkshire Region",
      ].map(region => ({
        label: region,
        values: region === "East Anglian Region" ? {} : { [`region_${region}`]: 1 },
      })) },
      { name: "gender", label: "Gender", options: [{ label: "Female", values: {} }, { label: "Male", values: { gender_M: 1 } }] },
      { name: "disability", label: "Disability", options: [{ label: "No", values: {} }, { label: "Yes", values: { disability_Y: 1 } }] },
    ];
    const groupedFeatures = new Set(groups.flatMap(group => group.options.flatMap(option => Object.keys(option.values))));
    const uciInputs = {
      age: ["Age", 15, 22, 17], medu: ["Mother's education (0-4)", 0, 4, 2],
      fedu: ["Father's education (0-4)", 0, 4, 2], traveltime: ["Home-to-school travel time (1-4)", 1, 4, 1],
      studytime: ["Weekly study time", 1, 4, 2], failures: ["Past class failures", 0, 3, 0],
      famrel: ["Family relationship quality (1-5)", 1, 5, 4], freetime: ["Free time after school (1-5)", 1, 5, 3],
      goout: ["Going out with friends (1-5)", 1, 5, 3], dalc: ["Workday alcohol use (1-5)", 1, 5, 1],
      walc: ["Weekend alcohol use (1-5)", 1, 5, 1], health: ["Current health (1-5)", 1, 5, 4],
      absences: ["Absences", 0, 93, 4], g1: ["Period 1 grade (0-20)", 0, 20, 10], g2: ["Period 2 grade (0-20)", 0, 20, 10],
    };
    const ouladInputs = {
      total_clicks_sum: ["Total clicks so far", 0, 20000, 150],
      total_clicks_max: ["Peak single-week clicks", 0, 5000, 40],
      days_active_sum: ["Total distinct active days", 0, 280, 20],
      num_weeks_active: ["Weeks active so far", 0, 40, 10],
      engagement_trend: ["Engagement trend (change in weekly clicks)", -200, 200, 0],
      num_of_prev_attempts: ["Previous attempts at this module", 0, 6, 0],
      studied_credits: ["Studied credits", 30, 655, 60],
      pct_assessments_submitted: ["Assessments submitted (%)", 0, 100, 60],
      age_band_ordinal: ["Age band", 0, 2, 0],
      highest_education_ordinal: ["Highest education", 0, 4, 2],
      imd_band_ordinal: ["IMD band midpoint", 5, 95, 55],
    };
    const inputSettings = dataset === "uci" ? uciInputs : ouladInputs;
    if (dataset === "oulad") {
      groupedFeatures.add("total_clicks_mean");
      groupedFeatures.add("days_active_mean");
    }

    const featureControl = feature => {
      const setting = inputSettings[feature];
      const id = `${sprintId}_pf_${btoa(feature).replace(/=/g, "")}`;
      if (!setting) {
        const label = Pages.escapeHtml(feature.replace(/_/g, " "));
        if (dataset === "oulad" && feature === "imd_band_missing") {
          return `<div class="field"><label class="label">${label}</label><input type="checkbox" id="${id}"></div>`;
        }
        return `<div class="field"><label class="label">${label}</label><input type="number" id="${id}" min="0" max="1" step="1" value="0"></div>`;
      }
      return `<div class="field"><label class="label">${Pages.escapeHtml(setting[0])}</label>
        <input type="number" id="${id}" min="${setting[1]}" max="${setting[2]}" value="${setting[3]}"></div>`;
    };
    const groupControls = groups.map(group => `
      <div class="field">
        <label class="label">${Pages.escapeHtml(group.label)}</label>
        <select id="${sprintId}_pg_${group.name}">${group.options.map((option, index) =>
          `<option value="${index}">${Pages.escapeHtml(option.label)}</option>`).join("")}</select>
      </div>`).join("");
    const visibleFeatures = features.filter(feature => !groupedFeatures.has(feature));

    const formHtml = `
      <div class="card">
        <div class="card-title">${Pages.escapeHtml(report.title)}</div>
        <div class="card-sub">${Pages.escapeHtml(report.desc)} — dataset: <strong>${dataset}</strong></div>
        <div class="row">${groupControls}</div>
        <div class="grid grid-3" id="${sprintId}_predictFields" style="margin-bottom:14px;">
          ${visibleFeatures.map(featureControl).join("")}
        </div>
        <button class="btn btn-primary" id="${sprintId}_run">Submit</button>
        <div id="${sprintId}_result" style="margin-top:20px;"></div>
      </div>`;
    bodyEl.innerHTML = formHtml;

    document.getElementById(`${sprintId}_run`).addEventListener("click", async () => {
      const btn = document.getElementById(`${sprintId}_run`);
      const resultEl = document.getElementById(`${sprintId}_result`);
      const body = {};
      groupedFeatures.forEach(feature => { body[feature] = 0; });
      visibleFeatures.forEach(feature => {
        const el = document.getElementById(`${sprintId}_pf_${btoa(feature).replace(/=/g, "")}`);
        body[feature] = el.type === "checkbox" ? Number(el.checked) : Number(el.value);
      });
      groups.forEach(group => {
        const choice = group.options[Number(document.getElementById(`${sprintId}_pg_${group.name}`).value)];
        Object.assign(body, choice.values);
      });
      if (dataset === "oulad") {
        const weeks = Math.max(1, body.num_weeks_active);
        body.total_clicks_mean = body.total_clicks_sum / weeks;
        body.days_active_mean = body.days_active_sum / weeks;
      }
      btn.disabled = true;
      btn.innerHTML = '<span class="spinner"></span> Scoring…';
      try {
        const data = await Api.request(report.path({}, dataset), { method: "POST", body });
        resultEl.innerHTML = renderAny(data);
        renderExtras(resultEl, sprintId, report, data, {});
      } catch (err) {
        resultEl.innerHTML = errorBox(err);
      } finally {
        btn.disabled = false;
        btn.textContent = "Submit";
      }
    });
  }

  function errorBox(err) {
    // The one endpoint that can send a structured (non-string) detail is
    // /sprint2/student-report's 400 "multiple matches" response -- show the
    // actual matches so the person can narrow the search, same intent as
    // streamlit_app.py's st.selectbox("Student", ...) picker for that case.
    const detail = err.payload && err.payload.detail;
    if (detail && typeof detail === "object" && Array.isArray(detail.matches)) {
      return `<div class="card" style="border-color:var(--escalate);">
        <div class="card-title" style="color:var(--escalate);">${Pages.escapeHtml(detail.message || "Multiple students match.")}</div>
        <p style="font-size:13px; color:var(--ink-soft);">Narrow your search (try the roll number instead of the name):</p>
        ${tableFromRows(detail.matches)}
      </div>`;
    }
    const validationErrors = Array.isArray(detail)
      ? detail.map(issue => {
        const location = Array.isArray(issue.loc)
          ? issue.loc.filter(part => part !== "body").join(".")
          : "";
        return [location, issue.msg].filter(Boolean).join(": ") || JSON.stringify(issue);
      }).join("\n")
      : detail && typeof detail === "object"
        ? JSON.stringify(detail)
        : (typeof detail === "string" ? detail : err.message);
    const message = typeof validationErrors === "string" ? validationErrors : JSON.stringify(validationErrors);
    return `<div class="card" style="border-color:var(--escalate);">
      <div class="card-title" style="color:var(--escalate);">Request failed (${err.status ?? "network"})</div>
      <p style="font-size:13px; color:var(--ink-soft);">${Pages.escapeHtml(message).replace(/\n/g, "<br>")}</p>
    </div>`;
  }

  return { renderShell, wire };
})();
