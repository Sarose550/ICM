import createIcmModule from "./dist/icm.mjs";

const Q_LADDER = [256, 512, 1024, 2048, 4096];
const RESIDUAL_TARGET = 1e-9;
const RATIO_LIMIT = 1e9;

const modulePromise = createIcmModule().then((mod) => {
  mod._icm_init(0);
  return mod;
});

function isNonNegativeFinite(arr) {
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i];
    if (!Number.isFinite(v) || v < 0) return false;
  }
  return true;
}

function computeRung(M, stacks, payouts, Q) {
  const n = stacks.length;
  const k = payouts.length;
  const pS = M._malloc(n * 8);
  M.HEAPF64.set(stacks, pS / 8);
  const pP = M._malloc(k * 8);
  M.HEAPF64.set(payouts, pP / 8);
  const pE = M._malloc(n * 8);
  M._icm_equity(n, pS, Q, pP, k, pE);
  const equity = M.HEAPF64.slice(pE / 8, pE / 8 + n);
  M._free(pS);
  M._free(pP);
  M._free(pE);
  return equity;
}

function residualOf(equity, payouts) {
  let eqSum = 0;
  for (let i = 0; i < equity.length; i++) eqSum += equity[i];
  let paySum = 0;
  for (let i = 0; i < payouts.length; i++) paySum += payouts[i];
  return Math.abs(eqSum - paySum) / paySum;
}

async function runCompute(M, stacksIn, payoutsIn) {
  const t0 = performance.now();

  if (!isNonNegativeFinite(stacksIn)) {
    throw new Error("Stacks must be non-negative finite numbers");
  }
  if (!isNonNegativeFinite(payoutsIn)) {
    throw new Error("Payouts must be non-negative finite numbers");
  }

  const nonzeroIndices = [];
  for (let i = 0; i < stacksIn.length; i++) {
    if (stacksIn[i] !== 0) nonzeroIndices.push(i);
  }
  const nNonzero = nonzeroIndices.length;
  if (nNonzero === 0) {
    throw new Error("At least one player must have a nonzero stack");
  }

  let maxStack = -Infinity;
  let minStack = Infinity;
  for (const idx of nonzeroIndices) {
    const v = stacksIn[idx];
    if (v > maxStack) maxStack = v;
    if (v < minStack) minStack = v;
  }
  if (maxStack / minStack > RATIO_LIMIT) {
    throw new Error(
      "Stack ratio exceeds 1e9 (largest / smallest nonzero stack); reduce the spread.",
    );
  }

  const payoutsTrimmed = Array.from(payoutsIn);
  while (
    payoutsTrimmed.length > 0 &&
    payoutsTrimmed[payoutsTrimmed.length - 1] === 0
  ) {
    payoutsTrimmed.pop();
  }
  if (payoutsTrimmed.length === 0) {
    throw new Error("At least one positive payout is required");
  }
  for (let i = 0; i < payoutsTrimmed.length; i++) {
    if (payoutsTrimmed[i] <= 0) {
      throw new Error(
        "Payout values must be positive, trailing zeros are dropped automatically",
      );
    }
  }

  let notice = null;
  let kEffective = payoutsTrimmed.length;
  if (kEffective > nNonzero) {
    kEffective = nNonzero;
    notice = `Payout list truncated to k=${kEffective} because only ${nNonzero} players have a nonzero stack`;
  }
  const payoutsEffective = Float64Array.from(
    payoutsTrimmed.slice(0, kEffective),
  );

  const nonzeroStacks = new Float64Array(nNonzero);
  for (let j = 0; j < nNonzero; j++) {
    nonzeroStacks[j] = stacksIn[nonzeroIndices[j]];
  }

  let best = null;
  let prevResidual = null;
  let stallCount = 0;
  for (const q of Q_LADDER) {
    self.postMessage({ type: "progress", q });
    const equity = computeRung(M, nonzeroStacks, payoutsEffective, q);
    const residual = residualOf(equity, payoutsEffective);
    if (best === null || residual < best.residual) {
      best = { q, residual, equity };
    }
    if (residual <= RESIDUAL_TARGET) break;
    if (prevResidual !== null && prevResidual / residual < 10) {
      stallCount++;
      if (stallCount >= 2) break;
    } else {
      stallCount = 0;
    }
    prevResidual = residual;
  }

  const fullEquity = new Float64Array(stacksIn.length);
  for (let j = 0; j < nNonzero; j++) {
    fullEquity[nonzeroIndices[j]] = best.equity[j];
  }

  const elapsedMs = performance.now() - t0;

  return {
    type: "result",
    equity: fullEquity,
    q: best.q,
    residual: best.residual,
    elapsedMs,
    kEffective,
    nNonzero,
    notice,
  };
}

self.onmessage = async (ev) => {
  const msg = ev.data;
  if (!msg || msg.type !== "compute") return;
  try {
    const M = await modulePromise;
    const result = await runCompute(M, msg.stacks, msg.payouts);
    self.postMessage(result, [result.equity.buffer]);
  } catch (err) {
    self.postMessage({
      type: "error",
      message: err && err.message ? err.message : String(err),
    });
  }
};
