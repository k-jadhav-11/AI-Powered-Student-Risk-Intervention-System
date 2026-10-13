/* ============================================================
   api.js — thin fetch wrapper around the Sprint 5 FastAPI backend
   (backend/main.py + backend/routers/agent.py)

   Every function here maps 1:1 to a real endpoint that already
   exists in the project. Nothing here calls an endpoint that
   doesn't exist -- if a sprint's data isn't exposed over REST
   (Sprint 2/3/4's raw analytics, for example), this file does not
   pretend otherwise. The Chat endpoint is the one place Sprint
   2-4's underlying tools (get_student_risk, get_student_shap,
   get_cohort_summary, get_gru_context) are reachable, via the
   function-calling agent.
   ============================================================ */

const Api = (() => {
  const DEFAULT_BASE = "http://127.0.0.1:8000";
  const STORAGE_KEY = "ic_api_base";

  function getBase() {
    return localStorage.getItem(STORAGE_KEY) || DEFAULT_BASE;
  }

  function setBase(url) {
    localStorage.setItem(STORAGE_KEY, url.replace(/\/+$/, ""));
  }

  async function request(path, { method = "GET", body = null, query = null } = {}) {
    let url = getBase() + path;
    if (query) {
      const params = Object.entries(query)
        .filter(([, v]) => v !== null && v !== undefined && v !== "")
        .flatMap(([k, v]) => Array.isArray(v)
          ? (v.length ? v : [""]).map(item => `${encodeURIComponent(k)}=${encodeURIComponent(item)}`)
          : [`${encodeURIComponent(k)}=${encodeURIComponent(v)}`])
        .join("&");
      if (params) url += "?" + params;
    }

    const opts = { method, headers: {} };
    if (body !== null) {
      opts.headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(body);
    }

    let res;
    try {
      res = await fetch(url, opts);
    } catch (networkErr) {
      throw new ApiError(
        `Could not reach ${url}. Is the FastAPI backend running, and is CORS enabled for this origin?`,
        0,
        null
      );
    }

    let data = null;
    const text = await res.text();
    try { data = text ? JSON.parse(text) : null; } catch { data = text; }

    if (!res.ok) {
      const detail = (data && data.detail) ? data.detail : (typeof data === "string" ? data : res.statusText);
      throw new ApiError(detail, res.status, data);
    }
    return data;
  }

  class ApiError extends Error {
    constructor(message, status, payload) {
      super(message);
      this.status = status;
      this.payload = payload;
    }
  }

  async function chatStream(payload, { onStart = () => {}, onToken = () => {} } = {}) {
    let response;
    const url = getBase() + "/agent/chat/stream";
    try {
      response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json", "Accept": "text/event-stream" },
        body: JSON.stringify(payload),
      });
    } catch {
      throw new ApiError(
        `Could not reach ${url}. Is the FastAPI backend running, and is CORS enabled for this origin?`,
        0,
        null
      );
    }

    if (!response.ok) {
      const text = await response.text();
      let data = null;
      try { data = text ? JSON.parse(text) : null; } catch { data = text; }
      const detail = (data && data.detail) ? data.detail : (typeof data === "string" ? data : response.statusText);
      throw new ApiError(detail, response.status, data);
    }
    if (!response.body) throw new ApiError("The browser does not support streaming responses.", response.status, null);

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let answer = "";
    let completion = null;

    function dispatch(frame) {
      let event = "message";
      const dataLines = [];
      frame.split(/\r?\n/).forEach(line => {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
      });
      if (!dataLines.length) return;
      const data = JSON.parse(dataLines.join("\n"));
      if (event === "start") onStart(data);
      else if (event === "token") {
        answer += data.text || "";
        onToken(data.text || "");
      } else if (event === "complete") completion = data;
      else if (event === "error") throw new ApiError(data.message || "Assistant stream failed.", 500, data);
    }

    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const frames = buffer.split(/\r?\n\r?\n/);
      buffer = frames.pop();
      frames.forEach(dispatch);
      if (done) break;
    }
    if (buffer.trim()) dispatch(buffer);
    if (!completion) throw new ApiError("The assistant stream ended before completion.", 500, null);
    return { ...completion, answer };
  }

  return {
    getBase,
    setBase,
    ApiError,
    request, // exposed so reportEngine.js can call arbitrary sprint2/sprint3 paths generically

    // ---- root / health ----
    rootHealth: () => request("/health"),
    agentHealth: () => request("/agent/health"),

    // ---- Sprint 5: single-student intervention ----
    runIntervention: (payload) => request("/agent/run", { method: "POST", body: payload }),

    // ---- Sprint 5: cohort batch ----
    runBatch: (payload) => request("/agent/batch", { method: "POST", body: payload }),

    // ---- Sprint 5: audit log ----
    getAuditLog: ({ limit = 50, dataset = null, decision = null } = {}) =>
      request("/agent/audit-log", { query: { limit, dataset, decision } }),

    // ---- Chat (Sprint 2-4 reachable through the function-calling assistant,
    //      plus Sprint 5's own run_intervention_agent tool). page_context is
    //      a one-line description of whatever the person is currently
    //      looking at -- see assistant.js. ----
    chat: (payload) => request("/agent/chat", { method: "POST", body: payload }),
    chatStream,
    chatHistory: (sessionId, sessionKey = "api_chat") =>
      request("/agent/chat/history", { query: { session_id: sessionId, session_key: sessionKey } }),

    // ---- Sprint 2: marks analytics (backend/routers/sprint2.py) ----
    sprint2: {
      health: () => request("/sprint2/health"),
      classes: () => request("/sprint2/classes"),
      subjects: () => request("/sprint2/subjects"),
      loadData: (dataDir = null) => request("/sprint2/load-data", { method: "POST", query: { data_dir: dataDir } }),
    },

    // ---- Sprint 3: risk prediction (backend/routers/sprint3.py) ----
    sprint3: {
      health: () => request("/sprint3/health"),
      studentDirectory: (samplePerOutcome = 20) =>
        request("/sprint3/oulad/gru/student-directory", { query: { sample_per_outcome: samplePerOutcome } }),
    },
  };
})();
