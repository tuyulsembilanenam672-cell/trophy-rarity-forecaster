// trophy-rarity-forecaster — GTA VI trophy earn-rate forecast (gta6-trophy-rarity-forecast-01).
//
// Refresh- and dilution-aware model, all in logit space:
//  1. History reads are deduplicated into distinct Exophase refreshes: a read
//     only shows rates as of Exophase's last refresh, so identical consecutive
//     reads are one data point, not many.
//  2. Players tracked N(t) is forecast from the refreshes: power-law adoption
//     in time since release when several refreshes exist; when the history is
//     flat (launch night), a linear-accumulation prior calibrated on the
//     launch-night refresh the rules state (638 -> 3868 players over ~4.7 h).
//  3. Per trophy, earners follow E_i ~ N^alpha_i. alpha_i is estimated from
//     refresh-to-refresh earner growth and shrunk to a prior by rate and tier:
//     early story trophies keep pace with player growth, rare and grindy
//     trophies dilute as the player base grows. The house baseline's own
//     practice score (~1.51) shows the answer sits ~1.5 logits below the last
//     rates, i.e. earners grow far slower than the tracked base — so the
//     prior alpha is kept low (0.15-0.5) and the data may raise it.
//  4. Median rate = E_i(target) / N(target); the logit spread widens with the
//     horizon and with the uncertainty in N and alpha, so the 90-minute
//     practice gets a tighter band and the 4.5-day final an honest wide one.
//
// Deterministic: pure arithmetic, no randomness, no network, stdlib only.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const Z = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];
const RELEASE_MS = Date.parse("2026-11-19T00:00:00Z");

interface Trophy { id: number; name: string; tier: string; }
interface HistRead { at: string; players_tracked: number; rates: (number | null)[]; }
interface CaseFile { id: string; at: string; horizon_minutes: number; target: string; levels: number[]; trophies: Trophy[]; history: HistRead[]; }
interface Snap { t: number; n: number; rates: number[]; }

function clamp(x: number, lo: number, hi: number): number { return x < lo ? lo : x > hi ? hi : x; }
function logitPct(p: number): number { const q = clamp(p, 0.01, 99.99) / 100; return Math.log(q / (1 - q)); }
function logisticPct(l: number): number { return 100 / (1 + Math.exp(-l)); }
function round4(x: number): number { return Math.round(x * 10000) / 10000; }
function median(xs: number[]): number { const s = [...xs].sort((a, b) => a - b); const m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; }

// Collapse the history into distinct Exophase refreshes.
function distinctSnapshots(c: CaseFile): Snap[] {
  const out: Snap[] = [];
  for (const h of c.history) {
    const rates = h.rates.map((r) => (r === null ? NaN : r));
    const prev = out[out.length - 1];
    if (prev && prev.n === h.players_tracked && prev.rates.every((r, i) => r === rates[i] || (Number.isNaN(r) && Number.isNaN(rates[i])))) continue;
    out.push({ t: Date.parse(h.at), n: h.players_tracked, rates });
  }
  return out;
}

// Forecast players tracked at the target, with a log-scale sigma for N.
function forecastPlayers(snaps: Snap[], targetMs: number): { nTgt: number; logSigmaN: number } {
  const last = snaps[snaps.length - 1];
  const first = snaps[0];
  if (snaps.length >= 3 && last.n > first.n) {
    // Power-law adoption: ln N ~ a + b * ln(1 + hours since release).
    // Relative growth slows as adoption proceeds, which is what a power law
    // captures; a log-linear fit would extrapolate launch-night explosiveness.
    let sx = 0, sy = 0, sxx = 0, sxy = 0;
    const n = snaps.length;
    for (const s of snaps) {
      const x = Math.log(1 + Math.max(0, (s.t - RELEASE_MS) / 3600000));
      const y = Math.log(Math.max(1, s.n));
      sx += x; sy += y; sxx += x * x; sxy += x * y;
    }
    const den = n * sxx - sx * sx;
    const b = den === 0 ? 0 : (n * sxy - sx * sy) / den;
    if (b <= 0) return { nTgt: last.n, logSigmaN: 0.3 };
    const a = (sy - b * sx) / n;
    const xT = Math.log(1 + Math.max(0, (targetMs - RELEASE_MS) / 3600000));
    const factor = clamp(Math.exp(a + b * xT) / last.n, 1, 30);
    return { nTgt: last.n * factor, logSigmaN: 0.25 + 0.15 * Math.log(factor) };
  }
  if (snaps.length >= 2 && last.n > first.n) {
    // Only one or two jumps: take the observed log growth rate, decaying with
    // a 1.5-day half-life as adoption slows.
    const dtDays = Math.max(0.01, (last.t - first.t) / 86400000);
    const g = Math.log(last.n / first.n) / dtDays; // per day
    const hDays = Math.max(0, (targetMs - last.t) / 86400000);
    const tau = 1.5;
    const inc = g * tau * (1 - Math.exp(-hDays / tau));
    const factor = clamp(Math.exp(inc), 1, 30);
    return { nTgt: last.n * factor, logSigmaN: 0.3 + 0.15 * Math.log(factor) };
  }
  // Flat history (launch night): Exophase had not refreshed yet. Prior from
  // the launch-night refresh in the rules: 638 -> 3868 players while the page
  // sat unchanged for ~4.7 h, i.e. ~687 newly tracked players per hour, so the
  // tracked count at the target is ~638 * (1 + 1.08 * hours since first read).
  const hours = Math.max(0, (targetMs - first.t) / 3600000);
  const factor = clamp(1 + 1.08 * hours, 1, 8);
  return { nTgt: last.n * factor, logSigmaN: 0.5 };
}

// Prior earner-growth exponent: low on purpose. Dilution dominates launch
// week for most trophies; only trophies the data shows outrunning the player
// base get a high alpha. Platinum (needs every trophy) barely grows at all.
function priorAlpha(rate: number, tier: string): number {
  if (rate >= 99.99) return 1;
  const base = 0.15 + 0.35 * (rate / (rate + 2));
  const tf = tier === "platinum" ? 0.3 : tier === "gold" ? 0.7 : tier === "silver" ? 0.85 : 1;
  if (tier === "platinum") return clamp(base * tf, 0.02, 0.25);
  return clamp(base * tf, 0.05, 0.9);
}

// Estimate alpha_i from refresh-to-refresh earner growth, shrunk to the prior.
// Quantised 2-decimal rates make small earner counts noisy, so intervals from
// trophies with few earners count less.
function estimateAlpha(idx: number, snaps: Snap[], prior: number, tier: string): number {
  const obs: number[] = [];
  for (let k = 1; k < snaps.length; k++) {
    const a = snaps[k - 1], b = snaps[k];
    const r1 = a.rates[idx], r2 = b.rates[idx];
    if (b.n <= a.n || !isFinite(r1) || !isFinite(r2) || r1 <= 0 || r2 <= 0) continue;
    const e1 = (r1 / 100) * a.n, e2 = (r2 / 100) * b.n;
    if (e1 < 1 || e2 < 1) continue;
    obs.push(Math.log(e2 / e1) / Math.log(b.n / a.n));
  }
  if (obs.length === 0) return prior;
  const med = median(obs);
  const avgE = median(snaps.map((s) => (s.rates[idx] / 100) * s.n).filter((e) => isFinite(e) && e > 0));
  const w = obs.length * (avgE >= 20 ? 1 : 0.4);
  const a = (w * med + 2 * prior) / (w + 2);
  return tier === "platinum" ? clamp(a, 0.02, 0.25) : clamp(a, 0.02, 2);
}

function forecastCase(c: CaseFile): { id: number; q: number[] }[] {
  const targetMs = Date.parse(c.target);
  const horizonDays = Math.max(0, c.horizon_minutes) / 1440;
  const snaps = distinctSnapshots(c);
  const last = snaps[snaps.length - 1];
  const { nTgt, logSigmaN } = forecastPlayers(snaps, targetMs);
  const nRatio = Math.max(1, nTgt / last.n);
  const rows: { id: number; q: number[] }[] = [];
  for (let i = 0; i < c.trophies.length; i++) {
    const t = c.trophies[i];
    const rLast = last.rates[i];
    if (!isFinite(rLast)) { rows.push({ id: t.id, q: [0.01, 0.05, 0.2, 1, 5] }); continue; }
    // The 100% trophy: every tracked player has it within minutes of starting,
    // so it stays at (or a hair below) 100% however large the base grows.
    if (rLast >= 99.995) { rows.push({ id: t.id, q: [99.5, 99.9, 99.99, 99.99, 99.99] }); continue; }
    const prior = priorAlpha(rLast, t.tier);
    const alpha = estimateAlpha(i, snaps, prior, t.tier);
    const eLast = Math.max(0.5, (rLast / 100) * last.n);
    const eTgt = Math.min(0.985 * nTgt, eLast * Math.pow(nRatio, alpha));
    const rMed = clamp((100 * eTgt) / nTgt, 0.01, 99.99);
    const lnF = Math.log(nRatio);
    // Honest spread: a floor for refresh timing, growth with horizon, with
    // the projected player-base jump, plus more for rare trophies and for
    // prior-driven (history-flat) alphas.
    let sigma = Math.sqrt(0.1 + 0.1 * horizonDays + 0.05 * lnF * lnF + 0.25 * logSigmaN * logSigmaN);
    if (rLast < 1) sigma += 0.05;
    if (snaps.length < 3) sigma += 0.1;
    sigma = clamp(sigma, 0.35, 2);
    const lm = logitPct(rMed);
    const q = Z.map((z) => round4(clamp(logisticPct(lm + z * sigma), 0.01, 99.99)));
    for (let k = 1; k < q.length; k++) q[k] = Math.max(q[k], q[k - 1]);
    rows.push({ id: t.id, q });
  }
  return rows;
}

function main(): void {
  const inDir = process.env.MAKEFUN_INPUT_DIR || "/input";
  const outDir = process.env.MAKEFUN_OUTPUT_DIR || "/output";
  mkdirSync(outDir, { recursive: true });
  const files = readdirSync(inDir).filter((f) => f.endsWith(".in")).sort();
  for (const f of files) {
    const c = JSON.parse(readFileSync(join(inDir, f), "utf8")) as CaseFile;
    const rows = forecastCase(c);
    writeFileSync(join(outDir, `${c.id}.out`), JSON.stringify({ trophies: rows }) + "\n", "utf8");
  }
}
main();
