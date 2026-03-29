/**
 * Nexus-Q cockpit — particle field, orbital provider map, WebSocket stream.
 */

const $ = (sel) => document.querySelector(sel);

const canvas = $("#field");
const ctx = canvas.getContext("2d");
const orbitNodes = $("#orbitNodes");
const orbitCenter = $("#orbitCenter");
const connStatus = $("#connStatus");
const connLabel = $("#connLabel");
const eventStream = $("#eventStream");
const progressBar = $("#progressBar");
const progressText = $("#progressText");
const summaryPanel = $("#summaryPanel");
const summaryPre = $("#summaryPre");

const reqSlider = $("#reqSlider");
const budgetSlider = $("#budgetSlider");
const delaySlider = $("#delaySlider");
const reqOut = $("#reqOut");
const budgetOut = $("#budgetOut");
const delayOut = $("#delayOut");

const mEps = $("#mEps");
const mBudget = $("#mBudget");
const mReward = $("#mReward");
const mLat = $("#mLat");

const btnLaunch = $("#btnLaunch");
const btnSendPrompt = $("#btnSendPrompt");
const btnStop = $("#btnStop");
const btnClear = $("#btnClear");
const userPrompt = $("#userPrompt");
const routeOnce = $("#routeOnce");
const autoPriorityBadge = $("#autoPriorityBadge");
const autoPriorityConf = $("#autoPriorityConf");
const chosenModelName = $("#chosenModelName");
const chosenModelReason = $("#chosenModelReason");
const llmOutputBody = $("#llmOutputBody");
const llmOutputMeta = $("#llmOutputMeta");

const quantumBadge = $("#quantumBadge");
const expComplexity = $("#expComplexity");
const expHealth = $("#expHealth");
const expLatency = $("#expLatency");
const expBlocked = $("#expBlocked");
const expQChart = $("#expQChart");

let ws = null;
let aborted = false;
let runFinished = false;
let particles = [];
let hueShift = 0;
let burstIntensity = 0;

function resizeCanvas() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.floor(window.innerWidth * dpr);
  canvas.height = Math.floor(window.innerHeight * dpr);
  canvas.style.width = `${window.innerWidth}px`;
  canvas.style.height = `${window.innerHeight}px`;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  initParticles();
}

function initParticles() {
  const n = Math.floor((window.innerWidth * window.innerHeight) / 18000);
  particles = Array.from({ length: Math.max(40, n) }, () => ({
    x: Math.random() * window.innerWidth,
    y: Math.random() * window.innerHeight,
    vx: (Math.random() - 0.5) * 0.35,
    vy: (Math.random() - 0.5) * 0.35,
    r: Math.random() * 1.8 + 0.3,
    phase: Math.random() * Math.PI * 2,
  }));
}

function tickField() {
  hueShift = (hueShift + 0.15) % 360;
  burstIntensity *= 0.92;
  const w = window.innerWidth;
  const h = window.innerHeight;
  ctx.fillStyle = "rgba(7, 6, 12, 0.22)";
  ctx.fillRect(0, 0, w, h);

  const grid = 48;
  ctx.strokeStyle = `hsla(${hueShift + 180}, 25%, 18%, ${0.06 + burstIntensity * 0.08})`;
  ctx.lineWidth = 1;
  const off = (performance.now() / 50) % grid;
  for (let x = -off; x < w + grid; x += grid) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
  }
  for (let y = -off; y < h + grid; y += grid) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  for (const p of particles) {
    p.x += p.vx + Math.sin(performance.now() / 900 + p.phase) * 0.12;
    p.y += p.vy + Math.cos(performance.now() / 700 + p.phase) * 0.1;
    if (p.x < 0) p.x = w;
    if (p.x > w) p.x = 0;
    if (p.y < 0) p.y = h;
    if (p.y > h) p.y = 0;
    const alpha = 0.15 + burstIntensity * 0.35;
    ctx.beginPath();
    ctx.arc(p.x, p.y, p.r * (1 + burstIntensity * 2), 0, Math.PI * 2);
    ctx.fillStyle = `hsla(${hueShift + p.phase * 40}, 90%, 65%, ${alpha})`;
    ctx.fill();
  }

  requestAnimationFrame(tickField);
}

function setConnState(mode) {
  connStatus.classList.remove("live", "error");
  if (mode === "live") {
    connStatus.classList.add("live");
    connLabel.textContent = "Streaming";
  } else if (mode === "error") {
    connStatus.classList.add("error");
    connLabel.textContent = "Error";
  } else {
    connLabel.textContent = "Idle";
  }
}

function shortName(p) {
  if (p.length <= 12) return p;
  return p.slice(0, 10) + "…";
}

function resetRoutingDisplay() {
  chosenModelName.textContent = "—";
  chosenModelReason.textContent = "";
  chosenModelReason.hidden = true;
  llmOutputBody.textContent = "Waiting for response…";
  llmOutputMeta.textContent = "";
}

function inferenceModeLabel(mode) {
  const map = {
    live_openai: "OpenAI (live)",
    live_anthropic: "Claude (live)",
    live_google: "Gemini (live)",
    live_mistral: "Mistral (live)",
    simulated: "Simulated",
  };
  return map[mode] || mode || "";
}

function updateLlmOutputDisplay(data) {
  const text = (data.response_text || data.response_preview || "").trim();
  llmOutputBody.textContent = text || "(empty response)";
  const mode = data.inference_mode || "";
  const model = data.model || data.provider || "";
  const parts = [];
  const im = inferenceModeLabel(mode);
  if (im) parts.push(im);
  if (model) parts.push(model);
  llmOutputMeta.textContent = parts.filter(Boolean).join(" · ");
}

function updateChosenModelDisplay(data) {
  const p = data.provider || "—";
  const m = data.model || p;
  chosenModelName.textContent = m === p ? p : `${p} · ${m}`;
  const why = data.reasoning || "";
  if (why) {
    chosenModelReason.textContent = why;
    chosenModelReason.hidden = false;
  } else {
    chosenModelReason.hidden = true;
  }
}

function updateExplainability(data) {
  if (!quantumBadge || !expQChart) return;
  const ex = data.explainability;
  if (!ex) return;

  const sv = ex.state_vector || {};
  quantumBadge.setAttribute(
    "data-active",
    ex.quantum_randomization_fired ? "true" : "false",
  );
  quantumBadge.textContent = ex.quantum_randomization_fired
    ? "Quantum randomization · ON"
    : "Quantum randomization";

  expComplexity.textContent = sv.prompt_complexity || "—";

  expHealth.innerHTML = "";
  const health = sv.provider_health || {};
  for (const [k, v] of Object.entries(health)) {
    const li = document.createElement("li");
    li.innerHTML = `<code>${escapeHtml(k)}</code> — ${escapeHtml(String(v))}`;
    expHealth.appendChild(li);
  }

  expLatency.innerHTML = "";
  const lat = sv.latency_history || {};
  for (const [k, arr] of Object.entries(lat)) {
    const li = document.createElement("li");
    const s = Array.isArray(arr) && arr.length
      ? arr.map((x) => Number(x).toFixed(3)).join(", ")
      : "—";
    li.innerHTML = `<code>${escapeHtml(k)}</code> — ${escapeHtml(s)}`;
    expLatency.appendChild(li);
  }

  const blocked = ex.blocked_providers || [];
  expBlocked.innerHTML = "";
  if (blocked.length === 0) {
    const li = document.createElement("li");
    li.className = "dim";
    li.textContent = "None";
    expBlocked.appendChild(li);
  } else {
    for (const p of blocked) {
      const li = document.createElement("li");
      li.className = "blocked";
      li.textContent = p;
      expBlocked.appendChild(li);
    }
  }

  const qv = ex.q_values || {};
  const vals = Object.values(qv).map(Number);
  const maxQ = Math.max(1e-12, ...vals.map((v) => Math.abs(v)));
  const maxAbs = Math.max(...vals.map((v) => Math.abs(v)));
  const dec = maxAbs > 0 && maxAbs < 0.01 ? 5 : 4;
  expQChart.innerHTML = "";
  for (const [name, score] of Object.entries(qv)) {
    const row = document.createElement("div");
    row.className = "q-row";
    const w = Math.max(6, (Math.abs(Number(score)) / maxQ) * 100);
    row.innerHTML = `
      <span class="q-name" title="${escapeHtml(name)}">${escapeHtml(name)}</span>
      <div class="q-track"><div class="q-bar" style="width:${w}%"></div></div>
      <span class="q-val">${Number(score).toFixed(dec)}</span>`;
    expQChart.appendChild(row);
  }
}

function updateAutoPriority(data) {
  const pri = (data.priority || "").toLowerCase();
  const conf = data.priority_confidence;
  autoPriorityBadge.textContent = pri.toUpperCase();
  autoPriorityBadge.setAttribute("data-priority", pri);
  if (typeof conf === "number") {
    autoPriorityConf.textContent = `${(conf * 100).toFixed(0)}% confidence`;
  } else {
    autoPriorityConf.textContent = "";
  }
}

function buildOrbit(providers) {
  orbitNodes.innerHTML = "";
  const cx = 200;
  const cy = 200;
  const r = 132;
  const n = providers.length;
  providers.forEach((name, i) => {
    const angle = (i / n) * Math.PI * 2 - Math.PI / 2;
    const x = cx + Math.cos(angle) * r;
    const y = cy + Math.sin(angle) * r;
    const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
    g.setAttribute("class", "orbit-node dim");
    g.setAttribute("data-provider", name);
    g.innerHTML = `
      <circle cx="${x}" cy="${y}" r="14" fill="rgba(255,255,255,0.06)" stroke="rgba(255,255,255,0.2)" stroke-width="1" />
      <text x="${x}" y="${y + 26}">${shortName(name)}</text>
    `;
    orbitNodes.appendChild(g);
  });
}

function highlightProvider(name) {
  for (const g of orbitNodes.querySelectorAll(".orbit-node")) {
    const match = g.getAttribute("data-provider") === name;
    g.classList.toggle("active", match);
    g.classList.toggle("dim", !match);
  }
  orbitCenter.textContent = shortName(name);
  orbitCenter.setAttribute("title", name);
  orbitCenter.classList.add("pulse");
  setTimeout(() => orbitCenter.classList.remove("pulse"), 350);
}

function appendEvent(html, opts = {}) {
  const li = document.createElement("li");
  li.innerHTML = html;
  if (opts.spike) li.classList.add("spike");
  if (opts.quantum) li.classList.add("quantum");
  eventStream.appendChild(li);
  eventStream.scrollTop = eventStream.scrollHeight;
  if (eventStream.children.length > 200) {
    eventStream.removeChild(eventStream.firstChild);
  }
}

function wsUrl() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}/ws/simulate`;
}

function setRunningUi(running) {
  btnLaunch.disabled = running;
  btnSendPrompt.disabled = running;
  btnStop.disabled = !running;
}

/**
 * @param {{ fromSendButton?: boolean }} [opts]
 */
function runSimulation(opts = {}) {
  if (ws && ws.readyState === WebSocket.OPEN) return;
  aborted = false;
  runFinished = false;
  summaryPanel.hidden = true;
  setRunningUi(true);
  setConnState("live");
  burstIntensity = 0.3;

  const rawPrompt = userPrompt.value.trim();
  let nReq = Number(reqSlider.value);
  const singleStep =
    rawPrompt &&
    (routeOnce.checked || opts.fromSendButton === true);
  if (singleStep) {
    nReq = 1;
  }
  const budget = Number(budgetSlider.value);
  const stepDelay = Number(delaySlider.value);

  resetRoutingDisplay();
  autoPriorityBadge.textContent = "Auto-detect";
  autoPriorityBadge.removeAttribute("data-priority");
  autoPriorityConf.textContent = "";
  chosenModelName.textContent = "Scheduling…";
  llmOutputBody.textContent = "Waiting for response…";
  llmOutputMeta.textContent = "";

  progressBar.style.width = "0%";
  progressText.textContent = `0 / ${nReq}`;

  ws = new WebSocket(wsUrl());
  ws.onopen = () => {
    const modeEl = document.querySelector('input[name="promptMode"]:checked');
    const payload = {
      n_requests: nReq,
      daily_budget: budget,
      step_delay_ms: stepDelay,
      prompt_mode: modeEl ? modeEl.value : "single",
    };
    if (rawPrompt) {
      payload.user_prompt = rawPrompt;
    }
    ws.send(JSON.stringify(payload));
  };
  ws.onmessage = (ev) => {
    const data = JSON.parse(ev.data);
    if (data.type === "error") {
      appendEvent(`<span class="idx">!</span> ${data.message}`);
      finishRun("error");
      return;
    }
    if (data.type === "start") {
      buildOrbit(data.providers);
      const src = data.prompt_source === "user" ? "user prompts" : "built-in corpus";
      const variants =
        data.user_prompt_variants > 0
          ? ` · ${data.user_prompt_variants} variant(s)`
          : "";
      const openaiHint =
        data.openai_live_configured === true
          ? " · OpenAI: <strong>live</strong> when routed (user prompts only)"
          : "";
      const claudeHint =
        data.claude_live_configured === true
          ? " · Claude: <strong>live</strong> when routed (user prompts only)"
          : "";
      const googleHint =
        data.google_live_configured === true
          ? " · Gemini: <strong>live</strong> when routed (user prompts only)"
          : "";
      const mistralHint =
        data.mistral_live_configured === true
          ? " · Mistral: <strong>live</strong> when routed (user prompts only)"
          : "";
      appendEvent(
        `<span class="idx">●</span> Session · ${data.n_requests} requests · budget $${data.daily_budget.toFixed(2)} · <strong>${src}</strong>${variants} · priority: <strong>auto-detected</strong>${openaiHint}${claudeHint}${googleHint}${mistralHint}`,
      );
      return;
    }
    if (data.type === "step") {
      const pct = (data.index / nReq) * 100;
      progressBar.style.width = `${pct}%`;
      progressText.textContent = `${data.index} / ${nReq}`;
      const modelName = data.model || data.provider;
      updateChosenModelDisplay(data);
      updateAutoPriority(data);
      updateExplainability(data);
      updateLlmOutputDisplay(data);
      highlightProvider(data.provider || modelName);
      if (data.is_spike) burstIntensity = Math.min(1, burstIntensity + 0.15);

      mEps.textContent = data.rl_epsilon.toFixed(4);
      mBudget.textContent = `$${data.budget_remaining.toFixed(4)}`;
      mReward.textContent = data.reward.toFixed(4);
      mLat.textContent = `${data.latency.toFixed(3)}s`;

      const q = data.quantum_resolved ? " ⚛" : "";
      const spike = data.is_spike ? " [SPIKE]" : "";
      appendEvent(
        `<span class="idx">#${data.index}</span>${spike} <strong>Model: ${escapeHtml(modelName)}</strong>${q} · ${data.priority} · ` +
          `${data.complexity} · ${data.pacemaker} · reward ${data.reward.toFixed(3)} · ` +
          `<span style="opacity:0.85">${escapeHtml(data.prompt.slice(0, 96))}${data.prompt.length > 96 ? "…" : ""}</span>`,
        { spike: data.is_spike, quantum: data.quantum_resolved },
      );
      const liveModes = {
        live_openai: "OpenAI",
        live_anthropic: "Claude",
        live_google: "Gemini",
        live_mistral: "Mistral",
      };
      const liveLabel = liveModes[data.inference_mode];
      const liveBody = data.response_text || data.response_preview || "";
      if (liveLabel && liveBody) {
        const cap = 4000;
        const body = liveBody.length > cap ? `${liveBody.slice(0, cap)}…` : liveBody;
        appendEvent(
          `<span class="idx">↳</span> <span style="opacity:0.88"><strong>${liveLabel} (live)</strong> · ${escapeHtml(body)}</span>`,
        );
      }
      return;
    }
    if (data.type === "complete") {
      progressBar.style.width = "100%";
      progressText.textContent = `${nReq} / ${nReq}`;
      summaryPanel.hidden = false;
      summaryPre.textContent = [
        data.savings_summary,
        "",
        `Rate-limit (429) events: ${data.rate_limit_events}`,
        `Quantum tie-breaks: ${data.quantum_resolutions}`,
        `RL steps: ${data.rl_steps} · ε final: ${data.rl_epsilon.toFixed(4)}`,
        `Budget: $${data.budget_spent.toFixed(4)} / $${data.budget_limit.toFixed(2)} (${data.utilization_pct.toFixed(1)}%)`,
      ].join("\n");
      appendEvent(`<span class="idx">✓</span> Complete`);
      finishRun("done");
    }
  };
  ws.onerror = () => {
    setConnState("error");
    appendEvent(`<span class="idx">!</span> WebSocket error`);
    finishRun("error");
  };
  ws.onclose = () => {
    finishRun("close");
  };
}

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

/**
 * @param {"done"|"error"|"close"} reason
 */
function finishRun(reason) {
  if (runFinished) return;
  if (reason === "close") {
    if (ws === null) return;
    runFinished = true;
    setRunningUi(false);
    ws = null;
    if (!aborted) {
      setConnState("idle");
      connLabel.textContent = "Idle";
    }
    return;
  }

  runFinished = true;
  setRunningUi(false);
  if (ws) {
    ws.close();
    ws = null;
  }
  connStatus.classList.remove("live");

  if (reason === "done") {
    connLabel.textContent = "Complete";
    setConnState("idle");
  } else if (reason === "error") {
    connLabel.textContent = "Error";
  }
}

function abortRun() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  aborted = true;
  appendEvent(`<span class="idx">×</span> Aborted`);
  ws.close();
  runFinished = true;
  setRunningUi(false);
  ws = null;
  setConnState("idle");
  connLabel.textContent = "Aborted";
}

function sendPromptFromButton() {
  const text = userPrompt.value.trim();
  if (!text) {
    userPrompt.focus();
    return;
  }
  runSimulation({ fromSendButton: true });
}

reqSlider.addEventListener("input", () => {
  reqOut.textContent = reqSlider.value;
});
budgetSlider.addEventListener("input", () => {
  budgetOut.textContent = Number(budgetSlider.value).toFixed(2);
});
delaySlider.addEventListener("input", () => {
  delayOut.textContent = delaySlider.value;
});

btnLaunch.addEventListener("click", () => runSimulation({}));
btnSendPrompt.addEventListener("click", sendPromptFromButton);
btnStop.addEventListener("click", abortRun);
userPrompt.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    sendPromptFromButton();
  }
});
btnClear.addEventListener("click", () => {
  eventStream.innerHTML = "";
});

document.addEventListener("keydown", (e) => {
  const tag = document.activeElement?.tagName;
  const inField = tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";
  if (e.code === "Space" && !inField) {
    e.preventDefault();
    runSimulation();
  }
  if (e.code === "Escape") {
    abortRun();
  }
});

window.addEventListener("resize", resizeCanvas);
resizeCanvas();
requestAnimationFrame(tickField);
