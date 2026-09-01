/* ── Kivi front end ───────────────────────────────────────────────────
   Three surfaces for a normal person (Dictate, Hey Kivi, What Kivi knows)
   and one for an engineer (Engineering view, behind a footer link).
   Vanilla JS, no build step. All interpolated text goes through esc().
   ──────────────────────────────────────────────────────────────────── */

(function () {
  "use strict";

  /* ── helpers ──────────────────────────────────────────────────── */

  const $ = (sel, root) => (root || document).querySelector(sel);

  function esc(value) {
    return String(value ?? "").replace(/[&<>"']/g, (char) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      "\"": "&quot;",
      "'": "&#039;",
    }[char]));
  }

  async function api(path, options) {
    let response;
    try {
      response = await fetch(path, options);
    } catch (err) {
      throw new Error("Kivi can't reach its own memory right now. Check that the local server is still running.");
    }
    if (response.status === 404) {
      throw new Error("That part of Kivi isn't answering (" + path + " returned 404).");
    }
    if (!response.ok) {
      let detail = "";
      try { detail = (await response.text()).slice(0, 200); } catch (_) { /* ignore */ }
      throw new Error("Kivi hit a problem" + (detail ? ": " + detail : ".") );
    }
    try {
      return await response.json();
    } catch (err) {
      throw new Error("Kivi sent back something unreadable.");
    }
  }

  function postJson(path, body) {
    return api(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  }

  const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

  function fmtDate(value) {
    if (!value) return "";
    const iso = String(value).replace(" ", "T");
    const d = new Date(/(Z|[+-]\d\d:?\d\d)$/.test(iso) ? iso : iso + "Z");
    if (isNaN(d.getTime())) return String(value);
    let h = d.getHours();
    const suffix = h >= 12 ? "pm" : "am";
    h = h % 12 || 12;
    const m = String(d.getMinutes()).padStart(2, "0");
    return d.getDate() + " " + MONTHS[d.getMonth()] + " " + d.getFullYear() + ", " + h + ":" + m + suffix;
  }

  function fmtDay(value) {
    const full = fmtDate(value);
    return full.split(",")[0];
  }

  function appLabel(app) {
    const name = String(app || "dictation").trim();
    if (!name) return "Dictation";
    return name.charAt(0).toUpperCase() + name.slice(1);
  }

  function count(n, one, many) {
    return n + " " + (n === 1 ? one : many);
  }

  function loadingHtml(label) {
    return '<div class="loading"><span class="dot"></span>' + esc(label) + "</div>";
  }

  function noticeHtml(message) {
    return '<div class="notice"><strong>Kivi couldn\'t do that.</strong> ' + esc(message) + "</div>";
  }

  let seq = 0;
  const nextId = () => "t" + (++seq);

  /* ── shared state ─────────────────────────────────────────────── */

  const turns = new Map();      // turn id -> /api/ask payload
  const forgotten = new Set();  // memory ids the person has dropped this session
  let knowsLoaded = false;

  /* ── mode switching ───────────────────────────────────────────── */

  const tabs = Array.prototype.slice.call(document.querySelectorAll(".mode"));
  const panels = {
    dictate: $("#panel-dictate"),
    ask: $("#panel-ask"),
    knows: $("#panel-knows"),
  };

  function setMode(mode, focusTab) {
    tabs.forEach((tab) => {
      const on = tab.dataset.mode === mode;
      tab.classList.toggle("is-active", on);
      tab.setAttribute("aria-selected", on ? "true" : "false");
      tab.tabIndex = on ? 0 : -1;
      if (on && focusTab) tab.focus();
    });
    Object.keys(panels).forEach((key) => { panels[key].hidden = key !== mode; });
    if (mode === "ask") $("#ask-input").focus();
    if (mode === "knows" && !knowsLoaded) loadKnows();
  }

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => setMode(tab.dataset.mode));
    tab.addEventListener("keydown", (event) => {
      const order = tabs.map((t) => t.dataset.mode);
      const i = order.indexOf(tab.dataset.mode);
      if (event.key === "ArrowRight") { event.preventDefault(); setMode(order[(i + 1) % order.length], true); }
      if (event.key === "ArrowLeft") { event.preventDefault(); setMode(order[(i - 1 + order.length) % order.length], true); }
    });
  });

  /* ── Dictate ──────────────────────────────────────────────────── */

  const dictateForm = $("#dictate-form");
  const dictateRegion = $("#dictate-region");
  const dictateSubmit = $("#dictate-submit");

  dictateForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const text = $("#dictate-text").value.trim();
    const app = $("#dictate-app").value;
    if (!text) {
      dictateRegion.innerHTML = noticeHtml("There was nothing to write. Say something first.");
      return;
    }
    dictateSubmit.disabled = true;
    dictateRegion.setAttribute("aria-busy", "true");
    dictateRegion.innerHTML = loadingHtml("Writing it the way you write…");
    try {
      const payload = await postJson("/api/dictate", { text: text, app: app });
      dictateRegion.innerHTML = renderDictate(payload);
      knowsLoaded = false; // memory may have moved
      refreshKnowsCount();
    } catch (err) {
      dictateRegion.innerHTML = noticeHtml(err.message);
    } finally {
      dictateSubmit.disabled = false;
      dictateRegion.setAttribute("aria-busy", "false");
    }
  });

  function paragraphs(text) {
    const value = String(text || "").trim();
    if (!value) return '<p class="hint">Kivi didn’t get any words back for that one.</p>';
    return value
      .split(/\n{2,}/)
      .map((block) => "<p>" + esc(block.trim()) + "</p>")
      .join("");
  }

  function renderDictate(payload) {
    const applied = payload.applied || [];
    const learned = payload.learned || [];
    const ignored = payload.ignored || [];
    const written = payload.formatted || payload.raw || "";

    /* the quiet line under the text: memory's only visible job in Dictate */
    let line;
    if (applied.length) {
      line = "Written using " + count(applied.length, "thing", "things") + " Kivi knows about you";
    } else if (ignored.length) {
      line = "Written without leaning on anything Kivi knows";
    } else {
      line = "Written straight from what you said";
    }

    const expandable = applied.length > 0 || ignored.length > 0;

    let detail = "";
    if (expandable) {
      const appliedList = applied.length
        ? "<ul>" + applied.map((item) => "<li>" + esc(item.claim || item.id) + "</li>").join("") + "</ul>"
        : '<p class="hint">Kivi didn\'t need anything it knows about you here.</p>';
      const skipped = ignored.length
        ? '<div class="skipped">Heard but not kept: ' + ignored.map((item) => esc(item)).join("; ") + "</div>"
        : "";
      detail = '<div class="applied-detail" id="applied-detail" hidden>' + appliedList + skipped + "</div>";
    }

    const learnedNotes = learned.map((item) => (
      '<div class="learned-note" data-learned="' + esc(item.id) + '">' +
        '<span class="learned-dot" aria-hidden="true"></span>' +
        '<div class="learned-body">' +
          "<p>Kivi noticed " + esc(item.claim) + "</p>" +
          '<button class="link-btn" type="button" data-action="drop-learned" data-id="' + esc(item.id) + '">No, drop that</button>' +
        "</div>" +
      "</div>"
    )).join("");

    return (
      '<article class="written">' +
        '<div class="written-head">' +
          '<span class="written-label">' + esc(appLabel($("#dictate-app").value)) + "</span>" +
          '<button class="link-btn" type="button" data-action="copy">Copy</button>' +
        "</div>" +
        '<div class="written-body">' + paragraphs(written) + "</div>" +
        '<div class="written-foot">' +
          (expandable
            ? '<button class="applied-line" type="button" data-action="toggle-applied"' +
                ' aria-expanded="false" aria-controls="applied-detail">' +
                '<span class="caret" aria-hidden="true"></span>' + esc(line) +
              "</button>"
            : '<span class="applied-line static">' + esc(line) + "</span>") +
          detail +
          learnedNotes +
        "</div>" +
      "</article>"
    );
  }

  /* ── Hey Kivi ─────────────────────────────────────────────────── */

  const askForm = $("#ask-form");
  const askInput = $("#ask-input");
  const askSubmit = $("#ask-submit");
  const thread = $("#thread");

  askForm.addEventListener("submit", (event) => {
    event.preventDefault();
    ask(askInput.value.trim());
  });

  async function ask(query) {
    if (!query) return;
    const empty = $("#ask-empty");
    if (empty) empty.remove();

    const id = nextId();
    const turn = document.createElement("div");
    turn.className = "turn";
    turn.id = id;
    turn.innerHTML =
      '<div class="said">' + esc(query) + "</div>" +
      '<div class="reply-slot">' + loadingHtml("Kivi is checking what it actually knows…") + "</div>";
    thread.appendChild(turn);
    turn.scrollIntoView({ block: "nearest" });

    askInput.value = "";
    askSubmit.disabled = true;
    thread.setAttribute("aria-busy", "true");

    try {
      const payload = await postJson("/api/ask", { query: query });
      turns.set(id, payload);
      $(".reply-slot", turn).innerHTML = renderReply(id, payload);
    } catch (err) {
      $(".reply-slot", turn).innerHTML = noticeHtml(err.message);
    } finally {
      askSubmit.disabled = false;
      thread.setAttribute("aria-busy", "false");
      turn.scrollIntoView({ block: "nearest" });
    }
  }

  function renderReply(id, payload) {
    const used = payload.used_memories || [];
    const abstained = !!payload.abstained;

    const restraint = abstained
      ? '<span class="reply-restraint">Kivi didn’t answer this one</span>'
      : "";

    const actions = [];
    if (used.length) {
      actions.push('<button class="link-btn" type="button" data-action="toggle-why" data-turn="' + id +
        '" aria-expanded="false" aria-controls="why-' + id + '">Why?</button>');
      actions.push('<button class="link-btn" type="button" data-action="flag-wrong" data-turn="' + id + '">That’s wrong</button>');
    } else if (abstained && payload.reason) {
      actions.push('<button class="link-btn" type="button" data-action="toggle-why" data-turn="' + id +
        '" aria-expanded="false" aria-controls="why-' + id + '">Why not?</button>');
    }

    return (
      '<article class="reply' + (abstained ? " abstained" : "") + '">' +
        '<div class="reply-body">' + restraint + esc(payload.answer || "") + "</div>" +
        (actions.length ? '<div class="reply-actions">' + actions.join("") + "</div>" : "") +
        '<div class="why-panel" id="why-' + id + '" hidden>' + renderWhy(payload) + "</div>" +
      "</article>"
    );
  }

  function renderWhy(payload) {
    const used = payload.used_memories || [];
    const sources = payload.source_transcripts || [];
    let html = "";

    if (used.length) {
      html +=
        '<div class="why-block">' +
          '<div class="why-heading">What Kivi used</div>' +
          used.map((memory) => (
            '<div class="why-memory" data-memory-row="' + esc(memory.id) + '">' +
              "<p>" + esc(memory.canonical_text) +
                (memory.status === "tentative" ? ' <span class="forgotten-flag">(still unsure about this one)</span>' : "") +
              "</p>" +
              '<button class="link-btn danger" type="button" data-action="forget" data-id="' + esc(memory.id) + '">Forget this</button>' +
            "</div>"
          )).join("") +
        "</div>";
    }

    if (sources.length) {
      html +=
        '<div class="why-block">' +
          '<div class="why-heading">Where that came from</div>' +
          sources.slice(0, 5).map((source) => (
            '<div class="why-source">' +
              '<div class="src-meta">' + esc(appLabel(source.app)) + " · " + esc(fmtDate(source.created_at)) + "</div>" +
              '<div class="src-text">' + esc(source.formatted_text || "") + "</div>" +
            "</div>"
          )).join("") +
        "</div>";
    }

    if (!used.length && payload.reason) {
      html +=
        '<div class="why-block">' +
          '<div class="why-heading">Why Kivi held back</div>' +
          '<p class="src-text">' + esc(payload.reason) + "</p>" +
        "</div>";
    }

    return html || '<p class="src-text">Kivi answered from what you said, not from anything it remembers.</p>';
  }

  document.addEventListener("click", (event) => {
    const chip = event.target.closest("[data-starter]");
    if (chip) { setMode("ask"); ask(chip.dataset.starter); }
  });

  /* ── What Kivi knows ──────────────────────────────────────────── */

  const knowsList = $("#knows-list");
  const knowsCount = $("#knows-count");

  async function loadKnows() {
    knowsList.setAttribute("aria-busy", "true");
    knowsList.innerHTML = loadingHtml("Gathering what Kivi knows…");
    try {
      const payload = await api("/api/memories?limit=100");
      renderKnows(payload.memories || []);
      knowsLoaded = true;
    } catch (err) {
      knowsList.innerHTML = noticeHtml(err.message);
    } finally {
      knowsList.setAttribute("aria-busy", "false");
    }
  }

  function renderKnows(memories) {
    const live = memories.filter((m) => m.status !== "archived");
    const gone = memories.filter((m) => m.status === "archived");

    knowsCount.textContent = live.length ? count(live.length, "thing", "things") : "nothing yet";

    if (!memories.length) {
      knowsList.innerHTML =
        '<div class="empty-state">' +
          '<span class="empty-mark" aria-hidden="true"></span>' +
          "<h2>Kivi doesn’t know anything about you yet</h2>" +
          "<p>Dictate a few things and Kivi will start noticing how you like your writing. Whatever it picks up shows up here, in plain words, for you to keep or drop.</p>" +
        "</div>";
      return;
    }

    let html = live.map(knowCard).join("");
    if (gone.length) {
      html += '<h3 class="knows-group-label">Forgotten</h3>' + gone.map(knowCard).join("");
    }
    knowsList.innerHTML = html;
  }

  function knowCard(memory) {
    const archived = memory.status === "archived";
    const evidence = memory.evidence || [];
    const meta = [];
    if (evidence.length) meta.push("Heard in " + count(evidence.length, "dictation", "dictations"));
    if (memory.last_seen_at) meta.push("last on " + fmtDay(memory.last_seen_at));
    if (memory.status === "tentative") meta.push("Kivi isn’t sure about this one yet");

    const whyId = "know-why-" + memory.id;

    return (
      '<article class="know' + (archived ? " is-forgotten" : "") + '" data-know="' + esc(memory.id) + '">' +
        '<div class="know-top"><div>' +
          '<div class="know-text">' + esc(memory.canonical_text) + "</div>" +
          (meta.length ? '<div class="know-meta">' + esc(meta.join(" · ")) + "</div>" : "") +
        "</div></div>" +
        '<div class="know-actions">' +
          '<button class="link-btn" type="button" data-action="toggle-know-why" aria-expanded="false" aria-controls="' + esc(whyId) + '">Why do you know this?</button>' +
          (archived
            ? '<span class="forgotten-flag">Kivi has stopped using this</span>'
            : '<button class="link-btn danger" type="button" data-action="forget" data-id="' + esc(memory.id) + '">Forget this</button>') +
        "</div>" +
        '<div class="know-why" id="' + esc(whyId) + '" hidden>' +
          (evidence.length
            ? evidence.map((item) => (
                '<div class="why-source">' +
                  '<div class="src-meta">' + esc(appLabel(item.app)) + " · " + esc(fmtDate(item.created_at)) + "</div>" +
                  '<div class="src-text">' + esc(item.snippet || item.formatted_text || "") + "</div>" +
                "</div>"
              )).join("")
            : '<p class="src-text">Kivi no longer has the dictation this came from.</p>') +
        "</div>" +
      "</article>"
    );
  }

  async function refreshKnowsCount() {
    try {
      const payload = await api("/api/memories?limit=100");
      const live = (payload.memories || []).filter((m) => m.status !== "archived");
      knowsCount.textContent = live.length ? count(live.length, "thing", "things") : "nothing yet";
    } catch (err) { /* the count is a nicety; stay quiet */ }
  }

  /* ── forgetting (the one control that matters) ────────────────── */

  async function forget(id, button) {
    if (forgotten.has(id)) return;
    const label = button ? button.textContent : "";
    if (button) { button.disabled = true; button.textContent = "Forgetting…"; }
    try {
      await postJson("/api/memories/forget", { id: id });
      forgotten.add(id);
      markForgotten(id);
      knowsLoaded = false;
      refreshKnowsCount();
      if (!panels.knows.hidden) loadKnows();
    } catch (err) {
      if (button) { button.disabled = false; button.textContent = label; }
      const holder = button && button.closest(".know, .why-memory, .learned-note");
      if (holder) {
        const note = document.createElement("div");
        note.className = "notice";
        note.innerHTML = "<strong>Kivi couldn’t forget that.</strong> " + esc(err.message);
        holder.appendChild(note);
      }
    }
  }

  function markForgotten(id) {
    document.querySelectorAll('[data-learned="' + CSS.escape(id) + '"]').forEach((note) => {
      note.classList.add("dropped");
      note.innerHTML =
        '<span class="learned-dot" aria-hidden="true"></span>' +
        '<div class="learned-body"><p>Dropped. Kivi won’t use that.</p></div>';
    });
    document.querySelectorAll('[data-memory-row="' + CSS.escape(id) + '"]').forEach((row) => {
      row.innerHTML = '<p class="forgotten-flag">Forgotten. Kivi has stopped using this.</p>';
    });
    document.querySelectorAll('[data-know="' + CSS.escape(id) + '"]').forEach((card) => {
      card.classList.add("is-forgotten");
      const actions = $(".know-actions", card);
      const btn = actions && $('[data-action="forget"]', actions);
      if (btn) btn.replaceWith(Object.assign(document.createElement("span"), {
        className: "forgotten-flag",
        textContent: "Kivi has stopped using this",
      }));
    });
  }

  /* ── Engineering view ─────────────────────────────────────────── */

  const engToggle = $("#eng-toggle");
  const engPanel = $("#engineering");
  let engLoaded = false;

  engToggle.addEventListener("click", () => {
    const open = engPanel.hidden;
    engPanel.hidden = !open;
    engToggle.setAttribute("aria-expanded", open ? "true" : "false");
    engToggle.textContent = open ? "Hide engineering view" : "Engineering view";
    if (open) {
      engPanel.scrollIntoView({ block: "nearest" });
      if (!engLoaded) loadEngineering();
    }
  });

  $("#eng-refresh").addEventListener("click", loadEngineering);

  async function loadEngineering() {
    const banner = $("#eng-backends");
    const statsEl = $("#eng-stats");
    const log = $("#eng-decisions");
    banner.textContent = "Loading backends…";
    log.innerHTML = loadingHtml("Loading controller decisions…");

    const state = await api("/api/state").catch(() => api("/api/summary").catch(() => null));
    if (state) {
      renderBackends(state.backends || {});
      const counts = (state.stats && state.stats.counts) || {};
      const growth = state.growth ? " | growth: " + esc(JSON.stringify(state.growth)) : "";
      statsEl.innerHTML =
        "counts: transcripts " + esc(counts.transcripts ?? "?") +
        " | memories " + esc(counts.memories ?? "?") +
        " | evidence " + esc(counts.memory_evidence ?? "?") +
        " | decisions " + esc(counts.decisions ?? "?") + growth;
    } else {
      banner.className = "eng-banner offline";
      banner.textContent = "state unavailable (/api/state and /api/summary both failed)";
      statsEl.textContent = "";
    }

    try {
      const payload = await api("/api/decisions?limit=50");
      const decisions = payload.decisions || [];
      log.innerHTML = decisions.length
        ? decisions.map(renderDecision).join("")
        : '<div class="eng-item"><div class="eng-line">no decisions recorded</div></div>';
    } catch (err) {
      log.innerHTML = noticeHtml(err.message);
    }
    engLoaded = true;
  }

  function renderBackends(b) {
    const banner = $("#eng-backends");
    const reachable = !!(b.ollama && b.ollama.reachable);
    const parts = [
      "extractor: " + ((b.extractor && b.extractor.backend) || "?"),
      "nli: " + ((b.nli && b.nli.backend) || "?"),
      "embeddings: " + ((b.embedder && b.embedder.backend) || "?"),
      "answers: " + ((b.answerer && b.answerer.backend) || "?"),
    ];
    banner.className = "eng-banner " + (reachable ? "live" : "offline");
    banner.textContent =
      (reachable ? "local models: " : "deterministic fallback (no local model reachable) - ") +
      parts.join("  |  ");
  }

  function renderDecision(decision) {
    const text = decision.formatted_text ||
      (decision.candidate && decision.candidate.canonical_text) ||
      decision.reason || "";
    const num = (v) => Number(v ?? 0).toFixed(2);
    return (
      '<div class="eng-item">' +
        '<div class="eng-line"><span class="pill act-' + esc(decision.action) + '">' + esc(decision.action) + "</span> " +
          esc(decision.reason) + "</div>" +
        "<div>" +
          '<span class="pill">' + esc(decision.app || "memory") + "</span>" +
          '<span class="pill">confidence ' + esc(num(decision.confidence)) + "</span>" +
          '<span class="pill">utility ' + esc(num(decision.utility)) + "</span>" +
          '<span class="pill">' + esc(decision.extractor || "rule") + "</span>" +
          (decision.nli_label
            ? '<span class="pill nli">NLI ' + esc(decision.nli_label) + " " + esc(num(decision.nli_probability)) + "</span>"
            : "") +
          (decision.created_at ? '<span class="pill">' + esc(decision.created_at) + "</span>" : "") +
        "</div>" +
        '<div class="eng-text">' + esc(String(text).slice(0, 220)) + "</div>" +
      "</div>"
    );
  }

  /* ── delegated actions ────────────────────────────────────────── */

  document.addEventListener("click", (event) => {
    const el = event.target.closest("[data-action]");
    if (!el) return;
    const action = el.dataset.action;

    if (action === "toggle-applied") {
      const detail = $("#applied-detail");
      if (!detail) return;
      const open = detail.hidden;
      detail.hidden = !open;
      el.setAttribute("aria-expanded", open ? "true" : "false");
    }

    if (action === "toggle-know-why") {
      const panel = document.getElementById(el.getAttribute("aria-controls"));
      if (!panel) return;
      const open = panel.hidden;
      panel.hidden = !open;
      el.setAttribute("aria-expanded", open ? "true" : "false");
      el.textContent = open ? "Hide where this came from" : "Why do you know this?";
    }

    if (action === "toggle-why") {
      const panel = document.getElementById("why-" + el.dataset.turn);
      if (!panel) return;
      const open = panel.hidden;
      panel.hidden = !open;
      el.setAttribute("aria-expanded", open ? "true" : "false");
    }

    if (action === "flag-wrong") {
      const panel = document.getElementById("why-" + el.dataset.turn);
      if (!panel) return;
      panel.hidden = false;
      const why = panel.closest(".reply").querySelector('[data-action="toggle-why"]');
      if (why) why.setAttribute("aria-expanded", "true");
      if (!$(".wrong-prompt", panel)) {
        const prompt = document.createElement("div");
        prompt.className = "wrong-prompt";
        prompt.textContent = "Which of these is wrong? Forget it and Kivi stops using it — here and everywhere else.";
        panel.insertBefore(prompt, panel.firstChild);
      }
      panel.scrollIntoView({ block: "nearest" });
    }

    if (action === "forget" || action === "drop-learned") {
      forget(el.dataset.id, el);
    }

    if (action === "copy") {
      const body = el.closest(".written").querySelector(".written-body");
      const text = body ? body.innerText : "";
      const done = () => { el.textContent = "Copied"; setTimeout(() => { el.textContent = "Copy"; }, 1600); };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, () => { el.textContent = "Press ⌘C"; });
      } else {
        el.textContent = "Press ⌘C";
      }
    }
  });

  /* ── boot ─────────────────────────────────────────────────────── */

  setMode("dictate");
  refreshKnowsCount();
})();
