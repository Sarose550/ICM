/*
 * WASM timing sweep used to choose the widget's default input cap.
 * V8 wasm performance is representative of Chromium; Safari runs the same
 * module within a small factor. Run from web/verify/:
 *   node perf_node.mjs
 */
import createIcmModule from "../dist/icm.mjs";

const M = await createIcmModule();
M._icm_init(0);

function timeCase(n, k, reps) {
  const stacks = new Float64Array(n);
  let s = 12345n;
  for (let i = 0; i < n; i++) {
    s = (s * 6364136223846793005n + 1442695040888963407n) & 0xffffffffffffffffn;
    stacks[i] = 500 + Number(s >> 40n) / 16777.216;
  }
  const payouts = new Float64Array(k);
  let sum = 0;
  for (let i = 0; i < k; i++) {
    payouts[i] = 1 / (1 + 0.5 * i);
    sum += payouts[i];
  }
  for (let i = 0; i < k; i++) payouts[i] /= sum;

  const pS = M._malloc(n * 8);
  const pP = M._malloc(k * 8);
  const pE = M._malloc(n * 8);
  M.HEAPF64.set(stacks, pS / 8);
  M.HEAPF64.set(payouts, pP / 8);

  let best = Infinity;
  for (let r = 0; r < reps; r++) {
    const t0 = performance.now();
    M._icm_equity(n, pS, 256, pP, k, pE);
    const ms = performance.now() - t0;
    if (ms < best) best = ms;
  }
  M._free(pS);
  M._free(pP);
  M._free(pE);
  return best;
}

timeCase(64, 8, 3);

console.log("n,k,best_ms");
const sweeps = [
  { label: "k=n (all paid, worst case)", pairs: [50, 100, 200, 400, 800, 1200, 1600, 2400, 3200, 4800, 6400, 9600, 12800, 19200, 25600].map((n) => [n, n]) },
  { label: "k=n/6 (typical MTT paid fraction)", pairs: [300, 600, 1200, 2400, 4800, 9600, 19200, 38400].map((n) => [n, Math.round(n / 6)]) },
  { label: "k=15 (final tables from a big field)", pairs: [1000, 4000, 16000, 64000, 128000].map((n) => [n, 15]) },
];

for (const sweep of sweeps) {
  console.log(`# ${sweep.label}`);
  for (const [n, k] of sweep.pairs) {
    const ms = timeCase(n, k, ms_budget(n, k));
    console.log(`${n},${k},${ms.toFixed(1)}`);
    if (ms > 4000) {
      console.log(`# stopping sweep: ${ms.toFixed(0)}ms exceeds budget`);
      break;
    }
  }
}

function ms_budget(n, k) {
  return n * k > 4e6 ? 2 : 3;
}
