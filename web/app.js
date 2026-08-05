import {
  PAYOUT_PRESETS,
  STACK_PRESETS,
  AUTOMATIC_PRESETS,
  percentsToStacks,
} from "./presets.js";

const DEFAULT_MAX_N = 10000;
const HARD_MAX_N = 100000;
const CHUNK_THRESHOLD = 500;
const CHUNK_SIZE = 300;
const SVG_NS = "http://www.w3.org/2000/svg";

const MAX_N = getMaxN();

function getMaxN() {
  const params = new URLSearchParams(location.search);
  let maxN = DEFAULT_MAX_N;
  if (params.has("maxn")) {
    const v = parseInt(params.get("maxn"), 10);
    if (Number.isFinite(v) && v > 0) {
      maxN = Math.min(v, HARD_MAX_N);
    }
  }
  return maxN;
}

const errorBanner = document.getElementById("errorBanner");
const errorBannerText = document.getElementById("errorBannerText");
const errorBannerDismiss = document.getElementById("errorBannerDismiss");

const browsePresetsBtn = document.getElementById("browsePresetsBtn");
const togglePasteBtn = document.getElementById("togglePasteBtn");
const pasteArea = document.getElementById("pasteArea");
const pasteTextarea = document.getElementById("pasteTextarea");
const applyPasteBtn = document.getElementById("applyPasteBtn");
const cancelPasteBtn = document.getElementById("cancelPasteBtn");
const prizesTableBody = document.getElementById("prizesTableBody");
const addRowBtn = document.getElementById("addRowBtn");
const clearPrizesBtn = document.getElementById("clearPrizesBtn");
const prizesMeta = document.getElementById("prizesMeta");

const stackModeToggle = document.getElementById("stackModeToggle");
const automaticStacks = document.getElementById("automaticStacks");
const manualStacks = document.getElementById("manualStacks");
const automaticPresetsRow = document.getElementById("automaticPresetsRow");
const manualPresetsRow = document.getElementById("manualPresetsRow");
const playersRemainingInput = document.getElementById("playersRemaining");
const averageStackInput = document.getElementById("averageStack");
const spreadSlider = document.getElementById("spreadSlider");
const spreadValue = document.getElementById("spreadValue");
const stackCurveSvg = document.getElementById("stackCurveSvg");
const manualStacksTextarea = document.getElementById("manualStacksTextarea");
const manualStacksMeta = document.getElementById("manualStacksMeta");

const calculateBtn = document.getElementById("calculateBtn");
const resultsSection = document.getElementById("resultsSection");
const resultsSummary = document.getElementById("resultsSummary");
const equityTableBody = document.getElementById("equityTableBody");
const equityChartSvg = document.getElementById("equityChartSvg");

const presetsOverlay = document.getElementById("presetsOverlay");
const presetsList = document.getElementById("presetsList");
const closePresetsBtn = document.getElementById("closePresetsBtn");

let prizes = [];
let stackMode = "automatic";
let computing = false;
let pendingStacks = null;
let pendingPayouts = null;

const worker = new Worker("worker.js", { type: "module" });

function showError(message) {
  errorBannerText.textContent = message;
  errorBanner.hidden = false;
}

function hideError() {
  errorBanner.hidden = true;
  errorBannerText.textContent = "";
}

errorBannerDismiss.addEventListener("click", hideError);

function formatMoney(v) {
  return v.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

function parseNumberTokens(text) {
  return text
    .split(/[\s,]+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
    .map(Number);
}

const renderGenerations = new WeakMap();

function renderChunked(tbody, count, buildRowFn) {
  const generation = (renderGenerations.get(tbody) || 0) + 1;
  renderGenerations.set(tbody, generation);
  tbody.textContent = "";
  if (count === 0) return;
  if (count <= CHUNK_THRESHOLD) {
    const frag = document.createDocumentFragment();
    for (let i = 0; i < count; i++) frag.appendChild(buildRowFn(i));
    tbody.appendChild(frag);
    return;
  }
  let i = 0;
  function step() {
    if (renderGenerations.get(tbody) !== generation) return;
    const end = Math.min(i + CHUNK_SIZE, count);
    const frag = document.createDocumentFragment();
    for (; i < end; i++) frag.appendChild(buildRowFn(i));
    tbody.appendChild(frag);
    if (i < count) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

function poolTotal() {
  return prizes.reduce((a, b) => a + b, 0);
}

function recomputePrizePercents() {
  const pool = poolTotal();
  const pctEls = prizesTableBody.querySelectorAll(".prize-pct");
  pctEls.forEach((el, i) => {
    const pct = pool > 0 ? (prizes[i] / pool) * 100 : 0;
    el.textContent = pct.toFixed(1);
  });
}

function updatePrizesMeta() {
  prizesMeta.textContent = `${prizes.length} places, pool ${formatMoney(poolTotal())}`;
}

function buildPrizeRow(i) {
  const tr = document.createElement("tr");

  const tdPlace = document.createElement("td");
  tdPlace.textContent = String(i + 1);

  const tdPrize = document.createElement("td");
  const input = document.createElement("input");
  input.type = "number";
  input.step = "any";
  input.min = "0";
  input.value = prizes[i];
  input.addEventListener("input", () => {
    const v = parseFloat(input.value);
    prizes[i] = Number.isFinite(v) ? v : 0;
    recomputePrizePercents();
    updatePrizesMeta();
  });
  tdPrize.appendChild(input);

  const tdPct = document.createElement("td");
  tdPct.className = "prize-pct";
  const pool = poolTotal();
  tdPct.textContent = pool > 0 ? ((prizes[i] / pool) * 100).toFixed(1) : "0.0";

  const tdAction = document.createElement("td");
  const removeBtn = document.createElement("button");
  removeBtn.type = "button";
  removeBtn.className = "row-remove-btn";
  removeBtn.textContent = "\u00D7";
  removeBtn.setAttribute("aria-label", "Remove row");
  removeBtn.addEventListener("click", () => {
    prizes.splice(i, 1);
    renderPrizesTable();
  });
  tdAction.appendChild(removeBtn);

  tr.append(tdPlace, tdPrize, tdPct, tdAction);
  return tr;
}

function renderPrizesTable() {
  renderChunked(prizesTableBody, prizes.length, buildPrizeRow);
  updatePrizesMeta();
}

addRowBtn.addEventListener("click", () => {
  prizes.push(0);
  renderPrizesTable();
});

clearPrizesBtn.addEventListener("click", () => {
  prizes = [];
  renderPrizesTable();
});

togglePasteBtn.addEventListener("click", () => {
  pasteArea.hidden = !pasteArea.hidden;
});

cancelPasteBtn.addEventListener("click", () => {
  pasteArea.hidden = true;
  pasteTextarea.value = "";
});

applyPasteBtn.addEventListener("click", () => {
  const tokens = parseNumberTokens(pasteTextarea.value).filter((v) =>
    Number.isFinite(v),
  );
  prizes = tokens;
  renderPrizesTable();
  pasteArea.hidden = true;
  pasteTextarea.value = "";
});

function applyPayoutPercents(percents) {
  const pool = poolTotal() > 0 ? poolTotal() : 1000;
  prizes = percents.map((p) => Math.round(((pool * p) / 100) * 100) / 100);
  renderPrizesTable();
}

function renderPresetsList() {
  presetsList.textContent = "";
  for (const preset of PAYOUT_PRESETS) {
    const item = document.createElement("div");
    item.className = "preset-item";

    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "preset-select";
    btn.textContent = preset.name;

    if (preset.generator) {
      const fieldInput = document.createElement("input");
      fieldInput.type = "number";
      fieldInput.min = "2";
      fieldInput.step = "1";
      fieldInput.value = String(preset.defaultFieldSize);
      btn.addEventListener("click", () => {
        const fieldSize = parseInt(fieldInput.value, 10);
        if (!Number.isFinite(fieldSize) || fieldSize < 1) return;
        applyPayoutPercents(preset.generator(fieldSize));
        presetsOverlay.hidden = true;
      });
      item.append(btn, fieldInput);
    } else {
      btn.addEventListener("click", () => {
        applyPayoutPercents(preset.percents);
        presetsOverlay.hidden = true;
      });
      item.append(btn);
    }

    presetsList.appendChild(item);
  }
}

browsePresetsBtn.addEventListener("click", () => {
  presetsOverlay.hidden = false;
});

closePresetsBtn.addEventListener("click", () => {
  presetsOverlay.hidden = true;
});

presetsOverlay.addEventListener("click", (ev) => {
  if (ev.target === presetsOverlay) presetsOverlay.hidden = true;
});

function renderAutomaticPresets() {
  automaticPresetsRow.textContent = "";
  for (const preset of AUTOMATIC_PRESETS) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "preset-chip";
    chip.textContent = preset.name;
    chip.addEventListener("click", () => {
      playersRemainingInput.value = String(preset.playersRemaining);
      averageStackInput.value = String(preset.averageStack);
      spreadSlider.value = String(preset.spread);
      spreadValue.textContent = preset.spread.toFixed(2);
      redrawStackCurve();
    });
    automaticPresetsRow.appendChild(chip);
  }
}

function renderManualPresets() {
  manualPresetsRow.textContent = "";
  for (const preset of STACK_PRESETS) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "preset-chip";
    chip.textContent = preset.name;
    chip.addEventListener("click", () => {
      const avg = parseFloat(averageStackInput.value) || preset.defaultAvgStack;
      const stacks = percentsToStacks(preset.percents, avg);
      manualStacksTextarea.value = stacks.join("\n");
      updateManualMeta();
    });
    manualPresetsRow.appendChild(chip);
  }
}

stackModeToggle.addEventListener("click", (ev) => {
  const btn = ev.target.closest(".segmented-option");
  if (!btn) return;
  const mode = btn.dataset.mode;
  if (mode === stackMode) return;
  stackMode = mode;
  for (const opt of stackModeToggle.querySelectorAll(".segmented-option")) {
    opt.classList.toggle("active", opt.dataset.mode === mode);
  }
  automaticStacks.hidden = mode !== "automatic";
  manualStacks.hidden = mode !== "manual";
});

function generateAutomaticStacks(n, avg, spread) {
  n = Math.max(1, Math.floor(n));
  const lambda = spread * Math.log(1e6);
  const raw = new Float64Array(n);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const r = i / n;
    const v = Math.exp(-lambda * r);
    raw[i] = v;
    sum += v;
  }
  const mean = sum / n;
  const stacks = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let v = mean > 0 ? (avg * raw[i]) / mean : 0;
    v = Math.round(v * 100) / 100;
    if (v <= 0) v = 0.01;
    stacks[i] = v;
  }
  return stacks;
}

function redrawStackCurve() {
  const n = parseInt(playersRemainingInput.value, 10);
  const avg = parseFloat(averageStackInput.value);
  const spread = parseFloat(spreadSlider.value);
  stackCurveSvg.textContent = "";
  if (!Number.isFinite(n) || n < 1 || !Number.isFinite(avg) || avg <= 0) return;

  const sampleCount = Math.max(2, Math.min(n, 200));
  const stacks = generateAutomaticStacks(sampleCount, avg, spread);
  const logVals = Array.from(stacks, (v) => Math.log10(Math.max(v, 1e-6)));
  const minLog = Math.min(...logVals);
  const maxLog = Math.max(...logVals);
  const range = maxLog - minLog || 1;

  const w = 400;
  const h = 160;
  const pad = 8;
  const points = [];
  for (let i = 0; i < stacks.length; i++) {
    const x = pad + (i / (stacks.length - 1 || 1)) * (w - 2 * pad);
    const y = pad + (1 - (logVals[i] - minLog) / range) * (h - 2 * pad);
    points.push(`${x.toFixed(1)},${y.toFixed(1)}`);
  }
  const poly = document.createElementNS(SVG_NS, "polyline");
  poly.setAttribute("points", points.join(" "));
  poly.setAttribute("style", "fill:none;stroke:var(--accent);stroke-width:2");
  stackCurveSvg.appendChild(poly);
}

[playersRemainingInput, averageStackInput].forEach((el) =>
  el.addEventListener("input", redrawStackCurve),
);
spreadSlider.addEventListener("input", () => {
  spreadValue.textContent = parseFloat(spreadSlider.value).toFixed(2);
  redrawStackCurve();
});

function updateManualMeta() {
  const tokens = parseNumberTokens(manualStacksTextarea.value);
  const nonzero = tokens.filter((v) => Number.isFinite(v) && v !== 0).length;
  manualStacksMeta.textContent = `${tokens.length} players, ${nonzero} nonzero`;
}

manualStacksTextarea.addEventListener("input", updateManualMeta);

function getCurrentStacks() {
  if (stackMode === "automatic") {
    const n = parseInt(playersRemainingInput.value, 10);
    const avg = parseFloat(averageStackInput.value);
    const spread = parseFloat(spreadSlider.value);
    if (!Number.isFinite(n) || n < 1) {
      throw new Error("Players remaining must be a positive integer");
    }
    if (!Number.isFinite(avg) || avg <= 0) {
      throw new Error("Average stack must be a positive number");
    }
    if (!Number.isFinite(spread) || spread < 0 || spread > 1) {
      throw new Error("Spread must be between 0 and 1");
    }
    return generateAutomaticStacks(n, avg, spread);
  }
  const tokens = parseNumberTokens(manualStacksTextarea.value);
  if (tokens.length === 0) {
    throw new Error("Enter at least one stack");
  }
  return Float64Array.from(tokens);
}

function getCurrentPayouts() {
  if (prizes.length === 0) {
    throw new Error("Add at least one prize");
  }
  return Float64Array.from(prizes);
}

calculateBtn.addEventListener("click", () => {
  if (computing) return;
  hideError();

  let stacks;
  let payouts;
  try {
    stacks = getCurrentStacks();
    payouts = getCurrentPayouts();
  } catch (err) {
    showError(err.message);
    return;
  }

  if (stacks.length > MAX_N) {
    showError(
      `Players (${stacks.length}) exceed the ${MAX_N} cap set for measured browser performance. ` +
        `Add ?maxn=<value> to the URL to raise it, up to ${HARD_MAX_N}.`,
    );
    return;
  }
  if (payouts.length > MAX_N) {
    showError(
      `Payout places (${payouts.length}) exceed the ${MAX_N} cap set for measured browser performance. ` +
        `Add ?maxn=<value> to the URL to raise it, up to ${HARD_MAX_N}.`,
    );
    return;
  }

  computing = true;
  calculateBtn.disabled = true;
  calculateBtn.textContent = "computing, Q=256";
  pendingStacks = stacks;
  pendingPayouts = payouts;
  worker.postMessage({ type: "compute", stacks, payouts });
});

worker.onmessage = (ev) => {
  const msg = ev.data;
  if (msg.type === "progress") {
    calculateBtn.textContent = `computing, Q=${msg.q}`;
  } else if (msg.type === "result") {
    computing = false;
    calculateBtn.disabled = false;
    calculateBtn.textContent = "CALCULATE";
    renderResults(msg, pendingStacks, pendingPayouts);
  } else if (msg.type === "error") {
    computing = false;
    calculateBtn.disabled = false;
    calculateBtn.textContent = "CALCULATE";
    showError(msg.message);
  }
};

worker.onerror = (ev) => {
  computing = false;
  calculateBtn.disabled = false;
  calculateBtn.textContent = "CALCULATE";
  showError(ev.message || "Worker failed unexpectedly");
};

function buildEquityRow(row, i, pool) {
  const tr = document.createElement("tr");
  if (row.stack === 0) tr.className = "muted-row";

  const tdRank = document.createElement("td");
  tdRank.textContent = String(i + 1);

  const tdStack = document.createElement("td");
  tdStack.textContent = formatMoney(row.stack);

  const tdEquity = document.createElement("td");
  tdEquity.textContent = formatMoney(row.equity);

  const tdPct = document.createElement("td");
  tdPct.textContent = pool > 0 ? ((row.equity / pool) * 100).toFixed(1) : "0.0";

  tr.append(tdRank, tdStack, tdEquity, tdPct);
  return tr;
}

function downsample(arr, maxPoints) {
  if (arr.length <= maxPoints) {
    return arr.map((v, i) => [i, v]);
  }
  const out = [];
  const step = (arr.length - 1) / (maxPoints - 1);
  for (let i = 0; i < maxPoints; i++) {
    const idx = Math.round(i * step);
    out.push([idx, arr[idx]]);
  }
  return out;
}

function drawEquityChart(rows, payouts) {
  equityChartSvg.textContent = "";
  if (rows.length === 0) return;

  const w = 800;
  const h = 300;
  const padL = 55;
  const padR = 20;
  const padT = 16;
  const padB = 30;
  const n = rows.length;
  const equities = rows.map((r) => r.equity);
  const maxEq = Math.max(...equities, 1e-9);

  function xOf(rank) {
    return padL + (rank / Math.max(n - 1, 1)) * (w - padL - padR);
  }
  function yOf(eq) {
    return padT + (1 - eq / maxEq) * (h - padT - padB);
  }

  const axis = document.createElementNS(SVG_NS, "line");
  axis.setAttribute("x1", String(padL));
  axis.setAttribute("y1", String(h - padB));
  axis.setAttribute("x2", String(w - padR));
  axis.setAttribute("y2", String(h - padB));
  axis.setAttribute("style", "stroke:var(--border);stroke-width:1");
  equityChartSvg.appendChild(axis);

  const kEff = Math.min(payouts.length, n);
  let prevPayout = null;
  let markerCount = 0;
  for (let i = 0; i < kEff && markerCount < 200; i++) {
    if (payouts[i] !== prevPayout) {
      const x = xOf(i);
      const line = document.createElementNS(SVG_NS, "line");
      line.setAttribute("x1", x.toFixed(1));
      line.setAttribute("y1", String(padT));
      line.setAttribute("x2", x.toFixed(1));
      line.setAttribute("y2", String(h - padB));
      line.setAttribute(
        "style",
        "stroke:var(--accent-dim);stroke-width:1;opacity:0.5",
      );
      equityChartSvg.appendChild(line);
      markerCount++;
    }
    prevPayout = payouts[i];
  }

  const pts = downsample(equities, 2000);
  const points = pts
    .map(([rank, eq]) => `${xOf(rank).toFixed(1)},${yOf(eq).toFixed(1)}`)
    .join(" ");
  const poly = document.createElementNS(SVG_NS, "polyline");
  poly.setAttribute("points", points);
  poly.setAttribute("style", "fill:none;stroke:var(--accent);stroke-width:2");
  equityChartSvg.appendChild(poly);
}

function renderResults(result, stacks, payouts) {
  resultsSection.hidden = false;
  const pool = payouts.reduce((a, b) => a + b, 0);

  const rows = [];
  for (let i = 0; i < stacks.length; i++) {
    rows.push({ stack: stacks[i], equity: result.equity[i] });
  }
  rows.sort((a, b) => b.stack - a.stack);

  const parts = [
    `<strong>${result.nNonzero}</strong> players`,
    `<strong>${result.kEffective}</strong> paid`,
    `pool <strong>${formatMoney(pool)}</strong>`,
    `<strong>${result.elapsedMs.toFixed(0)}</strong> ms`,
    `accuracy <strong>${result.residual.toExponential(1)}</strong>, Q=${result.q}`,
  ];
  let html = parts.join(", ") + ".";
  if (result.notice) {
    html += `<span class="notice">${result.notice}.</span>`;
  }
  resultsSummary.innerHTML = html;

  renderChunked(equityTableBody, rows.length, (i) =>
    buildEquityRow(rows[i], i, pool),
  );

  drawEquityChart(rows, payouts);
}

function init() {
  const defaultPayout = PAYOUT_PRESETS.find((p) => p.id === "sng9max");
  prizes = defaultPayout.percents.map(
    (p) => Math.round(p * 10 * 100) / 100,
  );
  renderPrizesTable();
  renderAutomaticPresets();
  renderManualPresets();
  renderPresetsList();
  redrawStackCurve();
  updateManualMeta();
}

init();
