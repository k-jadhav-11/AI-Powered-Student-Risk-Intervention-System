/* ============================================================
   toast.js — single shared toast notification, used by app.js,
   reportEngine.js, and assistant.js alike.
   ============================================================ */

const Toast = (() => {
  function show(message, kind = "") {
    const el = document.getElementById("toast");
    if (!el) return;
    el.textContent = message;
    el.className = "toast show " + kind;
    clearTimeout(el._t);
    el._t = setTimeout(() => { el.className = "toast"; }, 3600);
  }
  return { show };
})();
