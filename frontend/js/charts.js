/* ============================================================
   charts.js — tiny dependency-free canvas charts.
   Just enough for this console: a horizontal bar breakdown
   (decision counts) and nothing else. No charting library
   needed for this scope.
   ============================================================ */

const Charts = (() => {
  const COLORS = {
    nudge: "#0ea5a0",
    resource: "#4f5bd5",
    escalate: "#e0583b",
    default: "#9aa3b4",
  };

  function barBreakdown(canvas, counts) {
    // counts: [{ label, value, key }]
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = w * dpr; canvas.height = h * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    const max = Math.max(1, ...counts.map(c => c.value));
    const rowH = h / counts.length;
    const labelW = 86;
    const barMaxW = w - labelW - 50;

    ctx.font = "12px Inter, sans-serif";
    ctx.textBaseline = "middle";

    counts.forEach((c, i) => {
      const y = i * rowH + rowH / 2;
      const barW = (c.value / max) * barMaxW;
      const color = COLORS[c.key] || COLORS.default;

      ctx.fillStyle = "#5b6475";
      ctx.textAlign = "left";
      ctx.fillText(c.label, 0, y);

      ctx.fillStyle = "#eef0f5";
      roundRect(ctx, labelW, y - 8, barMaxW, 16, 8);
      ctx.fill();

      ctx.fillStyle = color;
      roundRect(ctx, labelW, y - 8, Math.max(barW, 4), 16, 8);
      ctx.fill();

      ctx.fillStyle = "#1a1f2b";
      ctx.textAlign = "left";
      ctx.font = "600 12px Inter, sans-serif";
      ctx.fillText(String(c.value), labelW + barMaxW + 10, y);
      ctx.font = "12px Inter, sans-serif";
    });
  }

  function roundRect(ctx, x, y, w, h, r) {
    r = Math.min(r, h / 2);
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  /* ------------------------------------------------------------
     barChart -- generic horizontal bar chart, the client-side
     equivalent of the Streamlit app's matplotlib st.pyplot() bars
     (class average, grade distribution, percentile ranking, etc).
     items: [{ label, value, highlight?: bool }]
     ------------------------------------------------------------ */
  function barChart(canvas, items, opts = {}) {
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const rowH = opts.rowHeight || 26;
    const w = canvas.clientWidth;
    const h = Math.max(canvas.clientHeight, items.length * rowH + 10);
    canvas.style.height = h + "px";
    canvas.width = w * dpr; canvas.height = h * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    if (!items.length) return;
    const values = items.map(c => Number(c.value) || 0);
    const max = Math.max(1, ...values.map(Math.abs));
    const labelW = Math.min(170, w * 0.32);
    const barMaxW = w - labelW - 58;

    ctx.font = "11.5px Inter, sans-serif";
    ctx.textBaseline = "middle";

    items.forEach((c, i) => {
      const y = i * rowH + rowH / 2 + 4;
      const v = Number(c.value) || 0;
      const barW = (Math.abs(v) / max) * barMaxW;
      const color = c.color || (c.highlight ? "#e0583b" : (opts.color || "#4f5bd5"));

      ctx.fillStyle = "#5b6475";
      ctx.textAlign = "left";
      ctx.fillText(truncate(String(c.label), 24), 0, y);

      ctx.fillStyle = "#eef0f5";
      roundRect(ctx, labelW, y - 7, barMaxW, 14, 7);
      ctx.fill();

      ctx.fillStyle = color;
      roundRect(ctx, labelW, y - 7, Math.max(barW, 3), 14, 7);
      ctx.fill();

      ctx.fillStyle = "#1a1f2b";
      ctx.textAlign = "left";
      ctx.font = "600 11.5px Inter, sans-serif";
      ctx.fillText(`${formatNum(v)}${opts.valueSuffix || ""}`, labelW + barMaxW + 8, y);
      ctx.font = "11.5px Inter, sans-serif";
    });
  }

  /* ------------------------------------------------------------
     lineChart -- simple connected-line chart (student score trend).
     points: [{ x: label, y: number }]
     ------------------------------------------------------------ */
  function lineChart(canvas, points, opts = {}) {
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight || 220;
    canvas.style.height = h + "px";
    canvas.width = w * dpr; canvas.height = h * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);
    if (!points.length) return;

    const padL = 42, padB = 28, padT = 14, padR = 14;
    const plotW = w - padL - padR, plotH = h - padT - padB;
    const ys = points.map(p => Number(p.y) || 0);
    const minY = opts.minY !== undefined ? opts.minY : (opts.autoScale ? Math.min(...ys) : Math.min(0, ...ys));
    const maxY = opts.maxY !== undefined ? opts.maxY : (opts.autoScale ? Math.max(...ys) : Math.max(100, ...ys));
    const xStep = points.length > 1 ? plotW / (points.length - 1) : 0;

    const toX = i => padL + i * xStep;
    const toY = v => padT + plotH - ((v - minY) / (maxY - minY || 1)) * plotH;

    // reference line (average), if given
    if (opts.referenceY !== undefined) {
      ctx.strokeStyle = "#c7cbd6";
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(padL, toY(opts.referenceY));
      ctx.lineTo(padL + plotW, toY(opts.referenceY));
      ctx.stroke();
      ctx.setLineDash([]);
    }

    ctx.strokeStyle = opts.color || "#4f5bd5";
    ctx.lineWidth = 2;
    ctx.beginPath();
    points.forEach((p, i) => {
      const x = toX(i), y = toY(Number(p.y) || 0);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();

    ctx.fillStyle = opts.color || "#4f5bd5";
    points.forEach((p, i) => {
      const x = toX(i), y = toY(Number(p.y) || 0);
      ctx.beginPath();
      ctx.arc(x, y, 3.5, 0, Math.PI * 2);
      ctx.fill();
    });

    ctx.fillStyle = "#5b6475";
    ctx.font = "10.5px Inter, sans-serif";
    points.forEach((p, i) => {
      if (points.length > 10 && i % Math.ceil(points.length / 10) !== 0) return;
      // Clamp alignment at the edges so the first/last labels don't clip
      // outside the canvas -- centered labels elsewhere.
      ctx.textAlign = i === 0 ? "left" : (i === points.length - 1 ? "right" : "center");
      ctx.fillText(truncate(String(p.x), 10), toX(i), h - 8);
    });
  }

  function multiLineChart(canvas, series, opts = {}) {
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight || 220;
    canvas.width = w * dpr; canvas.height = h * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    const points = series.flatMap(line => line.points);
    if (!points.length) return;
    const padL = 36, padB = 38, padT = 22, padR = 12;
    const plotW = w - padL - padR, plotH = h - padT - padB;
    const maxY = Math.max(1, ...points.map(point => Number(point.y) || 0));
    const toY = value => padT + plotH - (value / maxY) * plotH;
    const maxPoints = Math.max(...series.map(line => line.points.length));
    const xStep = maxPoints > 1 ? plotW / (maxPoints - 1) : 0;

    ctx.strokeStyle = "#d8dce4";
    ctx.beginPath();
    ctx.moveTo(padL, padT);
    ctx.lineTo(padL, padT + plotH);
    ctx.lineTo(padL + plotW, padT + plotH);
    ctx.stroke();

    series.forEach(line => {
      ctx.strokeStyle = line.color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      line.points.forEach((point, index) => {
        const x = padL + index * xStep;
        const y = toY(Number(point.y) || 0);
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    });

    ctx.font = "10px Inter, sans-serif";
    ctx.textBaseline = "top";
    let legendX = padL;
    series.forEach(line => {
      ctx.fillStyle = line.color;
      ctx.fillRect(legendX, 4, 9, 9);
      ctx.fillStyle = "#5b6475";
      ctx.textAlign = "left";
      ctx.fillText(line.name, legendX + 13, 2);
      legendX += ctx.measureText(line.name).width + 32;
    });

    const labels = series[0].points;
    const labelWidths = labels.map(point => ctx.measureText(truncate(String(point.x), 10)).width);
    const minLabelGap = Math.max(0, ...labelWidths) * 1.5 + 8;
    const labelStep = xStep > 0 ? Math.max(1, Math.ceil(minLabelGap / xStep)) : labels.length;
    const labelIndexes = [];
    for (let index = 0; index < labels.length; index += labelStep) {
      labelIndexes.push(index);
    }
    const lastIndex = labels.length - 1;
    if (lastIndex > 0 && labelIndexes[labelIndexes.length - 1] !== lastIndex) {
      const previousIndex = labelIndexes.length > 1
        ? labelIndexes[labelIndexes.length - 2]
        : labelIndexes[0];
      if ((lastIndex - previousIndex) * xStep >= minLabelGap) {
        if (labelIndexes.length > 1) labelIndexes[labelIndexes.length - 1] = lastIndex;
        else labelIndexes.push(lastIndex);
      }
    }

    labelIndexes.forEach(index => {
      const point = labels[index];
      ctx.textAlign = index === 0 ? "left" : (index === lastIndex ? "right" : "center");
      ctx.fillText(truncate(String(point.x), 10), padL + index * xStep, h - 25);
    });
  }

  /* ------------------------------------------------------------
     scatterChart -- (x, y) scatter plot (consistency, outliers).
     points: [{ x: number, y: number, label?: string, highlight?: bool }]
     ------------------------------------------------------------ */
  function scatterChart(canvas, points, opts = {}) {
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight || 260;
    canvas.style.height = h + "px";
    canvas.width = w * dpr; canvas.height = h * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);
    if (!points.length) return;

    const padL = 46, padB = 28, padT = 14, padR = 14;
    const plotW = w - padL - padR, plotH = h - padT - padB;
    const xs = points.map(p => Number(p.x) || 0), ys = points.map(p => Number(p.y) || 0);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(0, ...ys), maxY = Math.max(...ys, 1);
    const spanX = (maxX - minX) || 1, spanY = (maxY - minY) || 1;

    const toX = v => padL + ((v - minX) / spanX) * plotW;
    const toY = v => padT + plotH - ((v - minY) / spanY) * plotH;

    // axes
    ctx.strokeStyle = "#d8dce4";
    ctx.beginPath();
    ctx.moveTo(padL, padT); ctx.lineTo(padL, padT + plotH); ctx.lineTo(padL + plotW, padT + plotH);
    ctx.stroke();

    points.forEach(p => {
      const x = toX(Number(p.x) || 0), y = toY(Number(p.y) || 0);
      ctx.fillStyle = p.color || (p.highlight ? "#e0583b" : (opts.color || "#4f5bd5"));
      ctx.beginPath();
      ctx.arc(x, y, p.highlight ? 5 : 4, 0, Math.PI * 2);
      ctx.fill();
    });

    ctx.fillStyle = "#5b6475";
    ctx.font = "10.5px Inter, sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(opts.yLabel || "", 4, padT);
    ctx.textAlign = "center";
    ctx.fillText(opts.xLabel || "", padL + plotW / 2, h - 4);
  }

  function appendGruTimeseries(container, chartData) {
    if (!chartData) {
      return;
    }

    const section = document.createElement("section");
    section.className = "assistant-timeseries";
    const heading = document.createElement("div");
    heading.className = "card-title";
    heading.textContent = `Engagement and risk over time — student ${chartData.student_id || ""}`;
    section.appendChild(heading);

    if (chartData.image_png_base64) {
      const image = document.createElement("img");
      image.className = "assistant-timeseries-image";
      image.alt = `Weekly engagement and GRU risk trajectory for student ${chartData.student_id || ""}`;
      image.src = `data:image/png;base64,${chartData.image_png_base64}`;
      image.style.display = "block";
      image.style.width = "100%";
      image.style.height = "auto";
      section.appendChild(image);
    } else {
      return;
    }
    container.appendChild(section);

  }

  /* ------------------------------------------------------------
     heatmapTable -- DOM/CSS heatmap (class x subject comparison),
     simpler and crisper than a canvas heatmap for a small grid.
     ------------------------------------------------------------ */
  function heatmapTable(container, columns, rows, opts = {}) {
    const valueCols = columns.filter(c => c !== (opts.indexCol || columns[0]));
    const indexCol = opts.indexCol || columns[0];
    let allVals = [];
    rows.forEach(r => valueCols.forEach(c => { if (typeof r[c] === "number") allVals.push(r[c]); }));
    const min = Math.min(...allVals, 0), max = Math.max(...allVals, 100);

    const cellColor = (v) => {
      if (typeof v !== "number") return "transparent";
      const t = (v - min) / ((max - min) || 1);
      const r = Math.round(224 - t * (224 - 79));
      const g = Math.round(88 + t * (173 - 88));
      const b = Math.round(59 + t * (123 - 59));
      return `rgba(${r},${g},${b},0.22)`;
    };

    const html = `<div class="table-wrap"><table>
      <thead><tr><th>${Pages.escapeHtml(indexCol)}</th>${valueCols.map(c => `<th>${Pages.escapeHtml(c)}</th>`).join("")}</tr></thead>
      <tbody>${rows.map(r => `<tr><td><strong>${Pages.escapeHtml(String(r[indexCol]))}</strong></td>${valueCols.map(c => {
        const v = r[c];
        const display = typeof v === "number" ? v.toFixed(1) : "—";
        return `<td style="background:${cellColor(v)};">${display}</td>`;
      }).join("")}</tr>`).join("")}</tbody>
    </table></div>`;
    container.innerHTML = html;
  }

  function truncate(s, n) { return s.length > n ? s.slice(0, n - 1) + "…" : s; }
  function formatNum(v) { return Number.isInteger(v) ? String(v) : v.toFixed(2); }

  return {
    barBreakdown,
    barChart,
    lineChart,
    multiLineChart,
    scatterChart,
    appendGruTimeseries,
    heatmapTable,
  };
})();
