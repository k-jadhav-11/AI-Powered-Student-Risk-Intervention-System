/* ============================================================
   assistant.js — the floating "Ask Assistant" panel present on
   EVERY page (like Streamlit's collapsible per-page chat sub-agent),
   calling the same POST /agent/chat/stream endpoint the dedicated Chat page
   uses, but always visible regardless of which page is open.

   Context sharing: whatever page/report/student is currently on
   screen calls Assistant.setPageContext("...") (see app.js and
   reportEngine.js) and that string rides along with every message
   as ChatRequest.page_context -- the backend patch in PATCHES.md
   threads it into ai_assistant/agent.py's run_agent(), which already
   supports it end to end. The session_id (and therefore the model's
   actual conversation memory) is shared with the dedicated Chat page
   via the same sessionStorage key, so starting a conversation here
   and continuing it on the Chat page (or vice versa) is one
   continuous session, not two.
   ============================================================ */

const Assistant = (() => {
  const TEACHER_ID_KEY = "ic_teacher_id";
  let pageContext = "";
  let sessionId = null;
  let open = false;

  function mount() {
    const wrap = document.createElement("div");
    wrap.id = "assistantWidget";
    wrap.innerHTML = `
      <button id="assistantFab" class="assistant-fab" title="Ask the assistant">
        <span class="assistant-fab-icon">&#129302;</span>
      </button>
      <div id="assistantPanel" class="assistant-panel">
        <div class="assistant-head">
          <div>
            <div class="assistant-title">Ask the Assistant</div>
            <div class="assistant-context" id="assistantContextLine">—</div>
          </div>
          <div style="display:flex; gap:6px;">
            <button class="btn btn-ghost btn-sm" id="assistantNewChat" title="New session">New</button>
            <button class="btn btn-ghost btn-sm" id="assistantClose" title="Close">&times;</button>
          </div>
        </div>
        <div class="assistant-log" id="assistantLog">
          <div class="msg msg-ai">Hi! I can see whatever you're currently looking at. Ask me about a student, a cohort, or ask me to run an intervention.</div>
        </div>
        <div class="assistant-input-row">
          <input type="text" id="assistantInput" placeholder="Ask about this page…">
          <button class="btn btn-primary btn-sm" id="assistantSend">Send</button>
        </div>
      </div>`;
    document.body.appendChild(wrap);

    document.getElementById("assistantFab").addEventListener("click", togglePanel);
    document.getElementById("assistantClose").addEventListener("click", togglePanel);
    document.getElementById("assistantSend").addEventListener("click", send);
    document.getElementById("assistantInput").addEventListener("keydown", (e) => { if (e.key === "Enter") send(); });
    document.getElementById("assistantNewChat").addEventListener("click", newSession);

    sessionId = sessionStorage.getItem("ic_chat_session") || null;
  }

  function togglePanel() {
    open = !open;
    document.getElementById("assistantPanel").classList.toggle("open", open);
    document.getElementById("assistantFab").classList.toggle("active", open);
    if (open) document.getElementById("assistantInput").focus();
  }

  function setPageContext(text) {
    pageContext = text || "";
    const line = document.getElementById("assistantContextLine");
    if (line) line.textContent = pageContext ? `Context: ${pageContext}` : "—";
  }

  function capturePageContext(question = "") {
    const activePage = document.querySelector(".page.active");
    if (!activePage) return pageContext;
    const isChartQuestion = /\b(?:graph|chart|diagram|plot|image|picture|trajectory|statistic|number|numbers|value|count|percentage|percent|rate|average|metric|bar|trend|shown|above|how many|total|unique|risk[- ]level|escalation|students|this|that|these|those)\b|\b\d+(?:\.\d+)?\s*(?:%|percent)?\b/i.test(question);

    const controlContext = Array.from(activePage.querySelectorAll("input, select, textarea"))
      .filter(element => element.type !== "button" && element.type !== "submit")
      .filter(element => !(isChartQuestion && element.id === "dashboardEntrySelect"))
      .map(element => {
        const label = activePage.querySelector(`label[for="${CSS.escape(element.id)}"]`);
        const value = element.type === "checkbox" ? String(element.checked) : element.value;
        return label && value ? `${label.textContent.trim()}: ${value}` : "";
      })
      .filter(Boolean)
      .join("\n");
    const pageCopy = activePage.cloneNode(true);
    pageCopy.querySelectorAll("table").forEach(table => {
      const rowCount = table.querySelectorAll("tbody tr").length;
      table.replaceWith(`[Table omitted: ${rowCount} rows]`);
    });
    if (isChartQuestion) {
      pageCopy.querySelectorAll("#dashboardEntrySelect, #dashboardEntryDetail, #runResult, #batchSummary, #sprint2_result, #sprint3_result")
        .forEach(detail => detail.replaceWith("[Student-level detail omitted for chart/statistic question]"));
    }
    const pageText = pageCopy.innerText.replace(/\s+/g, " ").trim().slice(0, 8500);
    const selectedDetail = activePage.querySelector("#dashboardEntryDetail, #runResult, #batchSummary, #sprint2_result, #sprint3_result");
    const detailText = !isChartQuestion && selectedDetail?.innerText.replace(/\s+/g, " ").trim().slice(0, 3500) || "";

    return [
      pageContext,
      controlContext && `Current filters and inputs:\n${controlContext}`,
      `Visible page data, statistics, and chart titles:\n${pageText}`,
      detailText && `Selected result detail:\n${detailText}`,
    ].filter(Boolean).join("\n\n").slice(0, 14000);
  }

  function captureVisibleCharts() {
    const activePage = document.querySelector(".page.active");
    if (!activePage) return {};

    return Array.from(activePage.querySelectorAll("canvas"))
      .filter(canvas => canvas.clientWidth > 0 && canvas.clientHeight > 0)
      .slice(0, 4)
      .reduce((gallery, canvas, index) => {
        const title = canvas.closest(".card")?.querySelector(".card-title")?.textContent?.trim()
          || canvas.parentElement?.querySelector(".card-title")?.textContent?.trim()
          || canvas.id
          || `Chart ${index + 1}`;
        const copy = document.createElement("canvas");
        const scale = Math.min(1, 1200 / canvas.width, 800 / canvas.height);
        copy.width = Math.max(1, Math.round(canvas.width * scale));
        copy.height = Math.max(1, Math.round(canvas.height * scale));
        copy.getContext("2d").drawImage(canvas, 0, 0, copy.width, copy.height);
        gallery[title] = copy.toDataURL("image/jpeg", 0.75);
        return gallery;
      }, {});
  }

  function newSession() {
    sessionId = null;
    sessionStorage.removeItem("ic_chat_session");
    document.getElementById("assistantLog").innerHTML = `<div class="msg msg-ai">New session started.</div>`;
    Toast.show("Started a new assistant session.");
  }

  function appendMessage(role, content, meta = "") {
    const log = document.getElementById("assistantLog");
    const div = document.createElement("div");
    div.className = "msg " + (role === "human" ? "msg-human" : "msg-ai");
    div.innerHTML = (role === "human" ? Pages.escapeHtml(content).replace(/\n/g, "<br>") : Pages.formatAssistantText(content))
      + (meta ? `<div class="msg-meta">${meta}</div>` : "");
    log.appendChild(div);
    log.scrollTop = log.scrollHeight;
  }

  function createStreamingMessage() {
    const log = document.getElementById("assistantLog");
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

  async function send() {
    const input = document.getElementById("assistantInput");
    const btn = document.getElementById("assistantSend");
    const question = input.value.trim();
    if (!question) return;
    appendMessage("human", question);
    input.value = "";
    btn.disabled = true;
    const streamingMessage = createStreamingMessage();
    try {
      const res = await Api.chatStream(
        {
          question,
          session_id: sessionId,
          teacher_id: localStorage.getItem(TEACHER_ID_KEY)?.trim() || null,
          page_context: capturePageContext(question) || null,
          chart_gallery: captureVisibleCharts(),
        },
        { onStart: data => {
          sessionId = data.session_id;
          sessionStorage.setItem("ic_chat_session", sessionId);
        }, onToken: streamingMessage.append }
      );
      sessionId = res.session_id;
      sessionStorage.setItem("ic_chat_session", sessionId);
      streamingMessage.complete(res);
    } catch (err) {
      streamingMessage.fail(err);
      Toast.show(err.message, "err");
    } finally {
      btn.disabled = false;
    }
  }

  function getSessionId() { return sessionId; }
  function setSessionId(id) { sessionId = id; sessionStorage.setItem("ic_chat_session", id); }

  return { mount, setPageContext, togglePanel, getSessionId, setSessionId, captureVisibleCharts };
})();

// Top-level `const` in a classic <script> does NOT become a `window`
// property (unlike `var`), but every other file guards its calls with
// `if (window.Assistant)` -- so without this line those guards are always
// false and setPageContext/getSessionId/setSessionId silently never fire.
window.Assistant = Assistant;
