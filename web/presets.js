/*
Payout and stack presets for the Prefill overlay.
PAYOUT_PRESETS entries are one of:
  { id, name, percents }: fixed structure, percents of pool summing to 100,
    scaled to the current pool on apply
  { id, name, amounts }: real published payouts in dollars, applied
    verbatim (the pool becomes the sum of the amounts)
  { id, name, generator, defaultFieldSize }: generator(n) returns percents
Structures with real-world provenance:
  wsop2026ft: WSOP Main Event 2026 final table, fixed pay table
    ($10M, $6M, $3.75M, $2.75M, $2.25M, $1.75M, $1.5M, $1.25M, $1M),
    normalized to the money at the table.
  sng180 is the long-cited classic PokerStars 180-man table (18 paid,
    10th-18th tied); labeled classic because current lobbies may differ.
  Double-or-nothing follows from its defining rule: top half each receive
    2/N of the pool.
STACK_PRESETS: { id, name, percents, defaultAvgStack }, percent of total
  chips in play; percentsToStacks converts to absolute stacks.
AUTOMATIC_PRESETS fill the Automatic mode inputs directly.
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

function satelliteSeats(seats) {
  const n = Math.max(1, Math.round(seats));
  return new Array(n).fill(100 / n);
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
    id: "don10",
    name: "Double or nothing, 10 players (top 5 double)",
    percents: [20, 20, 20, 20, 20],
  },
  {
    id: "wsop2026ft",
    name: "WSOP Main Event 2026 final table",
    amounts: [
      10000000, 6000000, 3750000, 2750000, 2250000, 1750000, 1500000,
      1250000, 1000000,
    ],
  },
  {
    id: "sng180",
    name: "180 man SNG, top 18 paid (classic)",
    percents: [30, 20, 11.9, 8, 6.5, 5, 3.5, 2.6, 1.7,
      1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2],
  },
  {
    id: "mtt15",
    name: "MTT, top 15% paid (generic curve)",
    generator: mtt15Percent,
    defaultFieldSize: 500,
  },
  {
    id: "satellite",
    name: "Satellite, equal seats",
    generator: satelliteSeats,
    defaultFieldSize: 10,
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
