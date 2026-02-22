/**
 * Settlement logic for Asian Handicap and Totals bets.
 * Shared between daily-simulator.tsx (client) and settle API route (server).
 */

export const ASSUMED_ODDS = 1.90;

export function settleAhHalf(
  goalDiff: number,
  line: number,
  betHome: boolean,
  stake: number,
): { pnl: number; result: "W" | "L" | "P" } {
  const adjusted = goalDiff + line;
  const perspective = betHome ? adjusted : -adjusted;

  if (perspective > 0) return { pnl: stake * (ASSUMED_ODDS - 1), result: "W" };
  if (perspective < 0) return { pnl: -stake, result: "L" };
  return { pnl: 0, result: "P" };
}

export function settleAh(
  homeScore: number,
  awayScore: number,
  closingLine: number,
  betHome: boolean,
  stake: number,
): { pnl: number; result: "W" | "L" | "P" } {
  const goalDiff = homeScore - awayScore;

  // Check if quarter line (e.g. 0.25, 0.75, -0.25, etc.)
  const frac = Math.abs(closingLine) % 0.5;
  const isQuarter = frac > 0.01 && frac < 0.49;

  if (!isQuarter) {
    return settleAhHalf(goalDiff, closingLine, betHome, stake);
  }

  // Quarter line: split into 2 half-bets
  const halfStake = stake / 2;
  const floorLine = closingLine > 0
    ? Math.floor(closingLine * 2) / 2
    : -Math.ceil(Math.abs(closingLine) * 2) / 2;
  const ceilLine = closingLine > 0
    ? Math.ceil(closingLine * 2) / 2
    : -Math.floor(Math.abs(closingLine) * 2) / 2;

  const h1 = settleAhHalf(goalDiff, floorLine, betHome, halfStake);
  const h2 = settleAhHalf(goalDiff, ceilLine, betHome, halfStake);

  const totalPnl = h1.pnl + h2.pnl;
  const result: "W" | "L" | "P" =
    totalPnl > 0 ? "W" : totalPnl < 0 ? "L" : "P";
  return { pnl: totalPnl, result };
}

export function settleTotalsHalf(
  totalGoals: number,
  line: number,
  betOver: boolean,
  stake: number,
): { pnl: number; result: "W" | "L" | "P" } {
  const diff = totalGoals - line;

  if (diff > 0) {
    return betOver
      ? { pnl: stake * (ASSUMED_ODDS - 1), result: "W" }
      : { pnl: -stake, result: "L" };
  }
  if (diff < 0) {
    return betOver
      ? { pnl: -stake, result: "L" }
      : { pnl: stake * (ASSUMED_ODDS - 1), result: "W" };
  }
  return { pnl: 0, result: "P" };
}

export function settleTotals(
  homeScore: number,
  awayScore: number,
  betOver: boolean,
  totalsLine: number,
  stake: number,
): { pnl: number; result: "W" | "L" | "P" } {
  const totalGoals = homeScore + awayScore;

  const frac = (totalsLine * 10) % 5;
  const isQuarter = frac > 0.1 && frac < 4.9;

  if (!isQuarter) {
    return settleTotalsHalf(totalGoals, totalsLine, betOver, stake);
  }

  const halfStake = stake / 2;
  const floorLine = Math.floor(totalsLine * 2) / 2;
  const ceilLine = Math.ceil(totalsLine * 2) / 2;

  const h1 = settleTotalsHalf(totalGoals, floorLine, betOver, halfStake);
  const h2 = settleTotalsHalf(totalGoals, ceilLine, betOver, halfStake);

  const totalPnl = h1.pnl + h2.pnl;
  const result: "W" | "L" | "P" =
    totalPnl > 0 ? "W" : totalPnl < 0 ? "L" : "P";
  return { pnl: totalPnl, result };
}
