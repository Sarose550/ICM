/*
 * Replays web/verify/reference.json through the WASM module and compares
 * against the native library's equities. Run from web/verify/:
 *   node verify_node.mjs
 * Exit 0 iff every case's max relative error is within TOL.
 */
import { readFileSync } from "node:fs";
import createIcmModule from "../dist/icm.mjs";

const TOL = 5e-12;

const cases = JSON.parse(
  readFileSync(new URL("./reference.json", import.meta.url), "utf8"),
);

const M = await createIcmModule();
M._icm_init(0);

function computeEquity(stacks, payouts, Q) {
  const n = stacks.length;
  const k = payouts.length;
  const pS = M._malloc(n * 8);
  const pP = M._malloc(k * 8);
  const pE = M._malloc(n * 8);
  M.HEAPF64.set(stacks, pS / 8);
  M.HEAPF64.set(payouts, pP / 8);
  M._icm_equity(n, pS, Q, pP, k, pE);
  const out = Array.from(M.HEAPF64.subarray(pE / 8, pE / 8 + n));
  M._free(pS);
  M._free(pP);
  M._free(pE);
  return out;
}

let worst = 0;
let failed = 0;
for (const c of cases) {
  const eq = computeEquity(c.stacks, c.payouts, c.Q);
  const scale = Math.max(...c.equity.map(Math.abs));
  let maxErr = 0;
  for (let i = 0; i < c.n; i++) {
    const err = Math.abs(eq[i] - c.equity[i]) / scale;
    if (err > maxErr) maxErr = err;
  }
  const ok = maxErr <= TOL;
  if (!ok) failed++;
  if (maxErr > worst) worst = maxErr;
  console.log(
    `${ok ? "PASS" : "FAIL"} ${c.name.padEnd(12)} n=${String(c.n).padEnd(6)} k=${String(c.k).padEnd(6)} max rel err = ${maxErr.toExponential(2)}`,
  );
}

console.log(`\nworst case rel err: ${worst.toExponential(3)} (tol ${TOL})`);
if (failed > 0) {
  console.log(`${failed} case(s) FAILED`);
  process.exit(1);
}
console.log("ALL WASM VERIFICATION CASES PASSED");
