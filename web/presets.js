/*
Payout and stack presets for the Prefill overlay.
PAYOUT_PRESETS: { id, name, percents } for fixed structures, or
  { id, name, generator, defaultFieldSize } where generator(fieldSize) returns
  percents. percents are percent of pool, in place order, summing to 100.
STACK_PRESETS: { id, name, percents, defaultAvgStack }. percents are percent
  of total chips in play; percentsToStacks converts them to absolute stacks.
AUTOMATIC_PRESETS: { id, name, playersRemaining, averageStack, spread }, fills
  the Automatic mode inputs directly.
*/

function mtt15Percent(fieldSize) {
  const places = Math.max(1, Math.round(fieldSize * 0.15));
  const weights = [];
  for (let i = 0; i < places; i++) {
    weights.push(1 / (i + 1));
  }
  const sum = weights.reduce((a, b) => a + b, 0);
  return weights.map((w) => (w / sum) * 100);
}

export function percentsToStacks(percents, averageStack) {
  const n = percents.length;
  const total = averageStack * n;
  return percents.map((p) => Math.round(((total * p) / 100) * 100) / 100);
}

export const PAYOUT_PRESETS = [
  { id: "headsup", name: "Heads up, winner takes all", percents: [100] },
  { id: "sng9max", name: "9 max SNG (50 / 30 / 20)", percents: [50, 30, 20] },
  {
    id: "final45",
    name: "45 man final table (40 / 23 / 16 / 12 / 9)",
    percents: [40, 23, 16, 12, 9],
  },
  {
    id: "top180",
    name: "180 man, top 27 paid",
    percents: mtt15Percent(180),
  },
  {
    id: "mtt15",
    name: "MTT, top 15% paid",
    generator: mtt15Percent,
    defaultFieldSize: 500,
  },
];

export const STACK_PRESETS = [
  {
    id: "ft9",
    name: "9 max final table chip distribution",
    percents: [25, 18, 14, 11, 9, 7, 6, 5, 5],
    defaultAvgStack: 300000,
  },
];

export const AUTOMATIC_PRESETS = [
  {
    id: "auto-final-table",
    name: "Final table (9 left)",
    playersRemaining: 9,
    averageStack: 300000,
    spread: 0.4,
  },
  {
    id: "auto-bubble",
    name: "Bubble (50 left)",
    playersRemaining: 50,
    averageStack: 50000,
    spread: 0.55,
  },
  {
    id: "auto-early",
    name: "Early field (500 left)",
    playersRemaining: 500,
    averageStack: 15000,
    spread: 0.35,
  },
];
