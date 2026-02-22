"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import type { PredictionRow, TodayFixture, LockedBet, DailyPick } from "@/lib/types";
import { settleAh, settleTotals, ASSUMED_ODDS } from "@/lib/settlement";
import {
  buildMatchCards,
  getVerdict,
  type MatchCard,
} from "@/lib/pick-engine";

const TZ = "Asia/Bangkok";

// Staking constants (THB) — Kelly-proportional: T0/Pink/T1 get more, T2/T3/Purple base
const STAKE_BY_CAT: Record<string, number> = {
  T0: 1000,
  T1: 1000,
  T2: 1000,
  T3: 1000,
  Purple: 1000,
  Pink: 1000,
};
const DEFAULT_STAKE = 1000;

function getStake(category: string, _leagueKey: string): number {
  return STAKE_BY_CAT[category] ?? DEFAULT_STAKE;
}

// ---------------------------------------------------------------------------
// Bet types
// ---------------------------------------------------------------------------

type BetCategory = "T0" | "T1" | "T2" | "T3" | "Purple" | "Pink";

interface Bet {
  fixtureId: number;
  category: BetCategory;
  side: string; // team name or "Over X" / "Under X"
  homeTeam: string;
  awayTeam: string;
  kickoff: string;
  leagueName: string;
  leagueKey: string;
  line: string; // e.g. "AH +0.25" or "O/U 2.75"
  stake: number;
  ahClosingLine: number | null;
  totalsLine: number | null;
  signalsJson: Record<string, unknown>;
  // Settlement (null = pending)
  pnl: number | null;
  result: "W" | "L" | "P" | null; // Win/Loss/Push/pending
  score: string | null; // e.g. "2-1"
}

// ---------------------------------------------------------------------------
// Category styling
// ---------------------------------------------------------------------------

const CAT_STYLE: Record<BetCategory, { bg: string; text: string; border: string }> = {
  T0: { bg: "bg-orange-500/10", text: "text-orange-400", border: "border-orange-500/30" },
  T1: { bg: "bg-green-500/10", text: "text-green-400", border: "border-green-500/30" },
  T2: { bg: "bg-blue-500/10", text: "text-blue-400", border: "border-blue-500/30" },
  T3: { bg: "bg-yellow-500/10", text: "text-yellow-400", border: "border-yellow-500/30" },
  Purple: { bg: "bg-purple-500/10", text: "text-purple-400", border: "border-purple-500/30" },
  Pink: { bg: "bg-pink-500/10", text: "text-pink-400", border: "border-pink-500/30" },
};

// ---------------------------------------------------------------------------
// Build bets from cards (live tab)
// ---------------------------------------------------------------------------

function buildBets(
  cards: MatchCard[],
  totalsLines: Record<number, number>,
): Bet[] {
  const bets: Bet[] = [];

  for (const card of cards) {
    const pred = card.prediction;
    if (!pred) continue;

    const isFinished = card.status === "finished";
    const hasScore = pred.home_score != null && pred.away_score != null;
    const score = hasScore ? `${pred.home_score}-${pred.away_score}` : null;

    // --- AH tiers ---
    const verdict = getVerdict(card);
    if (verdict && verdict.tier >= 0) {
      const category: BetCategory =
        verdict.tier === 0 ? "T0"
          : verdict.tier === 1 ? "T1"
          : verdict.tier === 2 ? "T2" : "T3";
      const betSide =
        verdict.tier <= 1
          ? card.pick!.side
          : verdict.tier === 2
            ? card.modelFilter!.side
            : card.pick!.side;
      const teamName = betSide === "home" ? card.homeTeam : card.awayTeam;
      const stake = getStake(category, card.leagueKey);
      const ahLine = pred.ah_closing_line;
      const line = ahLine != null
        ? (ahLine > 0 ? "+" : "") + ahLine.toFixed(2)
        : "N/A";

      let pnl: number | null = null;
      let result: "W" | "L" | "P" | null = null;

      if (isFinished && hasScore && ahLine != null) {
        const settlement = settleAh(
          pred.home_score!,
          pred.away_score!,
          ahLine,
          betSide === "home",
          stake,
        );
        pnl = settlement.pnl;
        result = settlement.result;
      }

      // Build signals JSON for storage
      const signalsJson: Record<string, unknown> = {
        tier: category,
        side: betSide,
        signals: card.pick?.signals ?? null,
        total: card.pick?.total ?? null,
        edge: card.pick?.edge ?? null,
        filters: card.modelFilter?.filters ?? [],
        filterCount: card.modelFilter?.count ?? 0,
        modelFilterSide: card.modelFilter?.side ?? null,
        ahFairLine: pred.ah_fair_line,
        ahClosingLine: pred.ah_closing_line,
      };

      bets.push({
        fixtureId: card.fixtureId,
        category,
        side: teamName,
        homeTeam: card.homeTeam,
        awayTeam: card.awayTeam,
        kickoff: card.kickoff,
        leagueName: card.leagueName,
        leagueKey: card.leagueKey,
        line: `AH ${line}`,
        stake,
        ahClosingLine: ahLine,
        totalsLine: null,
        signalsJson,
        pnl,
        result,
        score,
      });
    }

    // --- Purple (totals fade) ---
    if (card.totalsPick) {
      const betOver = card.totalsPick.side === "over";
      const marketLine = totalsLines[card.fixtureId] ?? 2.5;
      let pnl: number | null = null;
      let result: "W" | "L" | "P" | null = null;

      const purpleStake = getStake("Purple", card.leagueKey);
      if (isFinished && hasScore) {
        const settlement = settleTotals(
          pred.home_score!,
          pred.away_score!,
          betOver,
          marketLine,
          purpleStake,
        );
        pnl = settlement.pnl;
        result = settlement.result;
      }

      bets.push({
        fixtureId: card.fixtureId,
        category: "Purple",
        side: betOver ? `Over ${marketLine}` : `Under ${marketLine}`,
        homeTeam: card.homeTeam,
        awayTeam: card.awayTeam,
        kickoff: card.kickoff,
        leagueName: card.leagueName,
        leagueKey: card.leagueKey,
        line: `O/U ${marketLine}`,
        stake: purpleStake,
        ahClosingLine: null,
        totalsLine: marketLine,
        signalsJson: {
          tier: "Purple",
          totalsSide: card.totalsPick.side,
          edge: card.totalsPick.edge,
        },
        pnl,
        result,
        score,
      });
    }

    // --- Pink (agree over) ---
    if (card.agreeOverPick) {
      const marketLine = totalsLines[card.fixtureId] ?? 2.5;
      let pnl: number | null = null;
      let result: "W" | "L" | "P" | null = null;

      const pinkStake = getStake("Pink", card.leagueKey);
      if (isFinished && hasScore) {
        const settlement = settleTotals(
          pred.home_score!,
          pred.away_score!,
          true,
          marketLine,
          pinkStake,
        );
        pnl = settlement.pnl;
        result = settlement.result;
      }

      bets.push({
        fixtureId: card.fixtureId,
        category: "Pink",
        side: `Over ${marketLine}`,
        homeTeam: card.homeTeam,
        awayTeam: card.awayTeam,
        kickoff: card.kickoff,
        leagueName: card.leagueName,
        leagueKey: card.leagueKey,
        line: `O/U ${marketLine}`,
        stake: pinkStake,
        ahClosingLine: null,
        totalsLine: marketLine,
        signalsJson: {
          tier: "Pink",
          edge: card.agreeOverPick.edge,
        },
        pnl,
        result,
        score,
      });
    }
  }

  return bets;
}

// ---------------------------------------------------------------------------
// Convert locked DB rows to Bet[] (for session tabs)
// ---------------------------------------------------------------------------

function lockedToBets(
  locked: LockedBet[],
  predictions: PredictionRow[],
): Bet[] {
  const predMap = new Map<number, PredictionRow>();
  for (const p of predictions) predMap.set(p.fixture_id, p);

  return locked.map((lb) => {
    const pred = predMap.get(lb.fixture_id);

    // Use DB-settled results if available, otherwise compute client-side
    let pnl: number | null = lb.pnl != null ? Number(lb.pnl) : null;
    let result: "W" | "L" | "P" | null = lb.result ?? null;
    const dbScore = lb.home_score != null && lb.away_score != null
      ? `${lb.home_score}-${lb.away_score}` : null;
    const predScore = pred?.home_score != null && pred?.away_score != null
      ? `${pred.home_score}-${pred.away_score}` : null;
    const score = dbScore ?? predScore;

    // Fallback: compute client-side if not yet settled in DB
    if (result == null && pred?.status === "finished" && pred?.home_score != null && pred?.away_score != null) {
      const isAh = lb.line.startsWith("AH ");
      if (isAh && lb.ah_closing_line != null) {
        const settlement = settleAh(pred.home_score, pred.away_score, lb.ah_closing_line, lb.side === lb.home_team, lb.stake);
        pnl = settlement.pnl;
        result = settlement.result;
      } else if (!isAh && lb.totals_line != null) {
        const settlement = settleTotals(pred.home_score, pred.away_score, lb.side.startsWith("Over"), lb.totals_line, lb.stake);
        pnl = settlement.pnl;
        result = settlement.result;
      }
    }

    return {
      fixtureId: lb.fixture_id,
      category: lb.category as BetCategory,
      side: lb.side,
      homeTeam: lb.home_team,
      awayTeam: lb.away_team,
      kickoff: lb.kickoff,
      leagueName: lb.league_name,
      leagueKey: lb.league_key,
      line: lb.line,
      stake: lb.stake,
      ahClosingLine: lb.ah_closing_line,
      totalsLine: lb.totals_line,
      signalsJson: {},
      pnl,
      result,
      score,
    };
  });
}

// ---------------------------------------------------------------------------
// Convert DailyPick DB rows to Bet[] (for stored picks display)
// ---------------------------------------------------------------------------

function dailyPickToBet(dp: DailyPick): Bet {
  const score = dp.home_score != null && dp.away_score != null
    ? `${dp.home_score}-${dp.away_score}`
    : null;

  return {
    fixtureId: dp.fixture_id,
    category: dp.category as BetCategory,
    side: dp.side,
    homeTeam: dp.home_team,
    awayTeam: dp.away_team,
    kickoff: dp.kickoff,
    leagueName: dp.league_name,
    leagueKey: dp.league_key,
    line: dp.line,
    stake: Number(dp.stake),
    ahClosingLine: dp.ah_closing_line,
    totalsLine: dp.totals_line,
    signalsJson: dp.signals_json,
    pnl: dp.pnl != null ? Number(dp.pnl) : null,
    result: dp.result,
    score,
  };
}

// ---------------------------------------------------------------------------
// Summary computation
// ---------------------------------------------------------------------------

interface CategorySummary {
  category: BetCategory;
  bets: number;
  stake: number;
  wins: number;
  losses: number;
  pushes: number;
  pending: number;
  pendingStake: number;
  pnl: number;
}

function computeSummary(bets: Bet[]): CategorySummary[] {
  const cats: BetCategory[] = ["T0", "T1", "T2", "T3", "Purple", "Pink"];
  const summaries: CategorySummary[] = [];

  for (const cat of cats) {
    const catBets = bets.filter((b) => b.category === cat);
    if (catBets.length === 0) continue;

    const pendingBets = catBets.filter((b) => b.result === null);
    summaries.push({
      category: cat,
      bets: catBets.length,
      stake: catBets.reduce((s, b) => s + b.stake, 0),
      wins: catBets.filter((b) => b.result === "W").length,
      losses: catBets.filter((b) => b.result === "L").length,
      pushes: catBets.filter((b) => b.result === "P").length,
      pending: pendingBets.length,
      pendingStake: pendingBets.reduce((s, b) => s + b.stake, 0),
      pnl: catBets.reduce((s, b) => s + (b.pnl ?? 0), 0),
    });
  }

  return summaries;
}

// ---------------------------------------------------------------------------
// Diff computation (morning vs closing)
// ---------------------------------------------------------------------------

type DiffStatus = "added" | "dropped" | "side_changed" | "line_changed" | "unchanged";

interface DiffRow {
  fixtureId: number;
  category: string;
  homeTeam: string;
  awayTeam: string;
  kickoff: string;
  leagueKey: string;
  status: DiffStatus;
  morningSide: string | null;
  morningLine: string | null;
  closingSide: string | null;
  closingLine: string | null;
}

function computeDiff(morning: Bet[], closing: Bet[]): DiffRow[] {
  const morningMap = new Map<string, Bet>();
  for (const b of morning) morningMap.set(`${b.fixtureId}-${b.category}`, b);

  const closingMap = new Map<string, Bet>();
  for (const b of closing) closingMap.set(`${b.fixtureId}-${b.category}`, b);

  const allKeys = new Set([...morningMap.keys(), ...closingMap.keys()]);
  const rows: DiffRow[] = [];

  for (const key of allKeys) {
    const m = morningMap.get(key);
    const c = closingMap.get(key);

    if (c && !m) {
      rows.push({
        fixtureId: c.fixtureId,
        category: c.category,
        homeTeam: c.homeTeam,
        awayTeam: c.awayTeam,
        kickoff: c.kickoff,
        leagueKey: c.leagueKey,
        status: "added",
        morningSide: null,
        morningLine: null,
        closingSide: c.side,
        closingLine: c.line,
      });
    } else if (m && !c) {
      rows.push({
        fixtureId: m.fixtureId,
        category: m.category,
        homeTeam: m.homeTeam,
        awayTeam: m.awayTeam,
        kickoff: m.kickoff,
        leagueKey: m.leagueKey,
        status: "dropped",
        morningSide: m.side,
        morningLine: m.line,
        closingSide: null,
        closingLine: null,
      });
    } else if (m && c) {
      let status: DiffStatus = "unchanged";
      if (m.side !== c.side) status = "side_changed";
      else if (m.line !== c.line) status = "line_changed";

      rows.push({
        fixtureId: m.fixtureId,
        category: m.category,
        homeTeam: m.homeTeam,
        awayTeam: m.awayTeam,
        kickoff: m.kickoff,
        leagueKey: m.leagueKey,
        status,
        morningSide: m.side,
        morningLine: m.line,
        closingSide: c.side,
        closingLine: c.line,
      });
    }
  }

  rows.sort((a, b) => new Date(a.kickoff).getTime() - new Date(b.kickoff).getTime());
  return rows;
}

const DIFF_STYLE: Record<DiffStatus, { bg: string; label: string }> = {
  added: { bg: "bg-green-500/10 text-green-400", label: "Added" },
  dropped: { bg: "bg-red-500/10 text-red-400", label: "Dropped" },
  side_changed: { bg: "bg-yellow-500/10 text-yellow-400", label: "Side changed" },
  line_changed: { bg: "bg-yellow-500/10 text-yellow-400", label: "Line changed" },
  unchanged: { bg: "bg-muted/30 text-muted-foreground", label: "Unchanged" },
};

// ---------------------------------------------------------------------------
// Streaming helper (shared by settle / update-odds)
// ---------------------------------------------------------------------------

async function runStream(
  url: string,
  body: Record<string, unknown>,
  onMsg: (msg: string) => void,
): Promise<boolean> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.body) throw new Error("No response stream");

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let success = false;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        const msg = JSON.parse(line);
        if (msg.message) onMsg(msg.message);
        if (msg.success === true) success = true;
        if (msg.success === false) throw new Error(msg.error || "Failed");
      } catch (e) {
        if (e instanceof Error && e.message !== "Failed") continue;
        throw e;
      }
    }
  }

  return success;
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function SummaryTable({ bets }: { bets: Bet[] }) {
  const summaries = computeSummary(bets);
  if (summaries.length === 0) return null;

  const totalStake = summaries.reduce((s, c) => s + c.stake, 0);
  const totalPnl = summaries.reduce((s, c) => s + c.pnl, 0);
  const totalWins = summaries.reduce((s, c) => s + c.wins, 0);
  const totalLosses = summaries.reduce((s, c) => s + c.losses, 0);
  const totalPushes = summaries.reduce((s, c) => s + c.pushes, 0);
  const totalPending = summaries.reduce((s, c) => s + c.pending, 0);

  return (
    <div className="overflow-x-auto rounded-md border border-border">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border bg-muted/30">
            <th className="px-3 py-2 text-left font-medium text-muted-foreground">Category</th>
            <th className="px-3 py-2 text-right font-medium text-muted-foreground">Bets</th>
            <th className="px-3 py-2 text-right font-medium text-muted-foreground">Stake</th>
            <th className="px-3 py-2 text-right font-medium text-muted-foreground">W-L-P</th>
            <th className="px-3 py-2 text-right font-medium text-muted-foreground">P&amp;L</th>
            <th className="px-3 py-2 text-right font-medium text-muted-foreground">ROI</th>
          </tr>
        </thead>
        <tbody>
          {summaries.map((s) => {
            const style = CAT_STYLE[s.category];
            const settledStake = (s.wins + s.losses + s.pushes) > 0
              ? s.stake - s.pendingStake
              : 0;
            const displayRoi = settledStake > 0 ? (s.pnl / settledStake) * 100 : 0;
            return (
              <tr key={s.category} className="border-b border-border/50">
                <td className="px-3 py-2">
                  <span className={`rounded px-2 py-0.5 text-xs font-bold ${style.bg} ${style.text} border ${style.border}`}>
                    {s.category}
                  </span>
                </td>
                <td className="px-3 py-2 text-right tabular-nums">{s.bets}</td>
                <td className="px-3 py-2 text-right tabular-nums">{s.stake.toLocaleString()}</td>
                <td className="px-3 py-2 text-right tabular-nums">
                  {s.wins}-{s.losses}-{s.pushes}
                  {s.pending > 0 && <span className="text-muted-foreground"> ({s.pending}p)</span>}
                </td>
                <td className={`px-3 py-2 text-right tabular-nums font-semibold ${s.pnl > 0 ? "text-green-400" : s.pnl < 0 ? "text-red-400" : ""}`}>
                  {s.pnl >= 0 ? "+" : ""}{s.pnl.toLocaleString()}
                </td>
                <td className={`px-3 py-2 text-right tabular-nums ${displayRoi > 0 ? "text-green-400" : displayRoi < 0 ? "text-red-400" : ""}`}>
                  {s.pnl === 0 && (s.wins + s.losses) === 0 ? "-" : `${displayRoi >= 0 ? "+" : ""}${displayRoi.toFixed(1)}%`}
                </td>
              </tr>
            );
          })}
          {summaries.length > 1 && (
            <tr className="border-t-2 border-border font-semibold">
              <td className="px-3 py-2 text-muted-foreground">TOTAL</td>
              <td className="px-3 py-2 text-right tabular-nums">{bets.length}</td>
              <td className="px-3 py-2 text-right tabular-nums">{totalStake.toLocaleString()}</td>
              <td className="px-3 py-2 text-right tabular-nums">
                {totalWins}-{totalLosses}-{totalPushes}
                {totalPending > 0 && <span className="text-muted-foreground"> ({totalPending}p)</span>}
              </td>
              <td className={`px-3 py-2 text-right tabular-nums ${totalPnl > 0 ? "text-green-400" : totalPnl < 0 ? "text-red-400" : ""}`}>
                {totalPnl >= 0 ? "+" : ""}{totalPnl.toLocaleString()}
              </td>
              <td className={`px-3 py-2 text-right tabular-nums ${totalPnl > 0 ? "text-green-400" : totalPnl < 0 ? "text-red-400" : ""}`}>
                {(() => {
                  const settledTotal = bets
                    .filter((b) => b.result !== null)
                    .reduce((s, b) => s + b.stake, 0);
                  if (settledTotal === 0) return "-";
                  const r = (totalPnl / settledTotal) * 100;
                  return `${r >= 0 ? "+" : ""}${r.toFixed(1)}%`;
                })()}
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

function BetTable({ bets }: { bets: Bet[] }) {
  if (bets.length === 0) {
    return (
      <div className="rounded-lg border border-border p-8 text-center text-muted-foreground">
        No bets in this session.
      </div>
    );
  }

  return (
    <div className="overflow-x-auto rounded-md border border-border">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border bg-muted/30">
            <th className="px-3 py-2 text-left font-medium text-muted-foreground">Time</th>
            <th className="px-3 py-2 text-left font-medium text-muted-foreground">Match</th>
            <th className="px-3 py-2 text-left font-medium text-muted-foreground">League</th>
            <th className="px-3 py-2 text-center font-medium text-muted-foreground">Cat</th>
            <th className="px-3 py-2 text-left font-medium text-muted-foreground">Side</th>
            <th className="px-3 py-2 text-left font-medium text-muted-foreground">Line</th>
            <th className="px-3 py-2 text-right font-medium text-muted-foreground">Stake</th>
            <th className="px-3 py-2 text-center font-medium text-muted-foreground">Score</th>
            <th className="px-3 py-2 text-right font-medium text-muted-foreground">P&amp;L</th>
          </tr>
        </thead>
        <tbody>
          {bets.map((bet, i) => {
            const style = CAT_STYLE[bet.category];
            const time = new Date(bet.kickoff).toLocaleTimeString("en-GB", {
              hour: "2-digit",
              minute: "2-digit",
              timeZone: TZ,
            });
            return (
              <tr
                key={`${bet.fixtureId}-${bet.category}-${i}`}
                className="border-b border-border/50 hover:bg-muted/20"
              >
                <td className="px-3 py-2 tabular-nums text-muted-foreground whitespace-nowrap">
                  {time}
                </td>
                <td className="px-3 py-2 whitespace-nowrap">
                  {bet.homeTeam} <span className="text-muted-foreground">vs</span> {bet.awayTeam}
                </td>
                <td className="px-3 py-2 text-xs text-muted-foreground whitespace-nowrap">
                  {bet.leagueKey}
                </td>
                <td className="px-3 py-2 text-center">
                  <span className={`rounded px-2 py-0.5 text-xs font-bold ${style.bg} ${style.text} border ${style.border}`}>
                    {bet.category}
                  </span>
                </td>
                <td className="px-3 py-2 font-medium whitespace-nowrap">
                  {bet.side}
                </td>
                <td className="px-3 py-2 text-xs text-muted-foreground whitespace-nowrap">
                  {bet.line}
                </td>
                <td className="px-3 py-2 text-right tabular-nums">
                  {bet.stake.toLocaleString()}
                </td>
                <td className="px-3 py-2 text-center tabular-nums">
                  {bet.score ?? (
                    <span className="text-xs text-muted-foreground">pending</span>
                  )}
                </td>
                <td className={`px-3 py-2 text-right tabular-nums font-semibold ${
                  bet.pnl == null
                    ? "text-muted-foreground"
                    : bet.pnl > 0
                      ? "text-green-400"
                      : bet.pnl < 0
                        ? "text-red-400"
                        : ""
                }`}>
                  {bet.pnl == null
                    ? "-"
                    : `${bet.pnl >= 0 ? "+" : ""}${bet.pnl.toLocaleString()}`}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function DiffTable({ rows }: { rows: DiffRow[] }) {
  if (rows.length === 0) {
    return (
      <div className="rounded-lg border border-border p-8 text-center text-muted-foreground">
        No data to compare. Lock both Morning and Closing sessions first.
      </div>
    );
  }

  return (
    <div className="overflow-x-auto rounded-md border border-border">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border bg-muted/30">
            <th className="px-3 py-2 text-left font-medium text-muted-foreground">Time</th>
            <th className="px-3 py-2 text-left font-medium text-muted-foreground">Match</th>
            <th className="px-3 py-2 text-left font-medium text-muted-foreground">Cat</th>
            <th className="px-3 py-2 text-center font-medium text-muted-foreground">Status</th>
            <th className="px-3 py-2 text-left font-medium text-muted-foreground">Morning</th>
            <th className="px-3 py-2 text-left font-medium text-muted-foreground">Closing</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => {
            const ds = DIFF_STYLE[row.status];
            const time = new Date(row.kickoff).toLocaleTimeString("en-GB", {
              hour: "2-digit",
              minute: "2-digit",
              timeZone: TZ,
            });
            return (
              <tr key={`${row.fixtureId}-${row.category}-${i}`} className="border-b border-border/50">
                <td className="px-3 py-2 tabular-nums text-muted-foreground whitespace-nowrap">{time}</td>
                <td className="px-3 py-2 whitespace-nowrap">
                  {row.homeTeam} <span className="text-muted-foreground">vs</span> {row.awayTeam}
                </td>
                <td className="px-3 py-2 text-xs text-muted-foreground">{row.category}</td>
                <td className="px-3 py-2 text-center">
                  <span className={`rounded px-2 py-0.5 text-xs font-bold ${ds.bg}`}>
                    {ds.label}
                  </span>
                </td>
                <td className="px-3 py-2 text-xs whitespace-nowrap">
                  {row.morningSide ? (
                    <span>{row.morningSide} <span className="text-muted-foreground">{row.morningLine}</span></span>
                  ) : (
                    <span className="text-muted-foreground">-</span>
                  )}
                </td>
                <td className="px-3 py-2 text-xs whitespace-nowrap">
                  {row.closingSide ? (
                    <span>{row.closingSide} <span className="text-muted-foreground">{row.closingLine}</span></span>
                  ) : (
                    <span className="text-muted-foreground">-</span>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------

type TabKey = "live" | "morning" | "closing" | "changes";

interface Props {
  date: string;
  predictions?: PredictionRow[];
  allPredictions?: PredictionRow[];
  fixtures?: TodayFixture[];
  leagues?: { league_key: string; name: string }[];
  totalsLines?: Record<number, number>;
  lockedBets: { morning: LockedBet[]; closing: LockedBet[] };
  storedPicks?: DailyPick[];
}

export function DailySimulator({
  date,
  predictions = [],
  allPredictions = [],
  fixtures = [],
  leagues = [],
  totalsLines = {},
  lockedBets,
  storedPicks,
}: Props) {
  const router = useRouter();
  const [tab, setTab] = useState<TabKey>("live");
  const [settling, setSettling] = useState(false);
  const [settleMsg, setSettleMsg] = useState<string | null>(null);
  const [settleErr, setSettleErr] = useState<string | null>(null);
  const [updatingOdds, setUpdatingOdds] = useState(false);
  const [oddsMsg, setOddsMsg] = useState<string | null>(null);
  const [oddsErr, setOddsErr] = useState<string | null>(null);
  const [locking, setLocking] = useState<"morning" | "closing" | null>(null);
  const [lockErr, setLockErr] = useState<string | null>(null);

  // Track whether this is a stored-picks-only view (past date fast path)
  const isStoredOnly = storedPicks != null && predictions.length === 0;

  // Build live bets (only when we have predictions data)
  const cards = useMemo(
    () => isStoredOnly ? [] : buildMatchCards(fixtures, predictions, allPredictions),
    [fixtures, predictions, allPredictions, isStoredOnly],
  );
  const liveBets = useMemo(
    () => isStoredOnly ? [] : buildBets(cards, totalsLines),
    [cards, totalsLines, isStoredOnly],
  );

  // Stored picks converted to Bet[] (for past-date fast path)
  const storedBets = useMemo(
    () => (storedPicks || []).map(dailyPickToBet),
    [storedPicks],
  );

  // Use stored bets as the "live" tab data when in stored-only mode
  const displayBets = isStoredOnly ? storedBets : liveBets;

  // Build session bets from locked DB data (settled using snapshotted lines)
  const morningBets = useMemo(
    () => lockedToBets(lockedBets.morning, predictions),
    [lockedBets.morning, predictions],
  );
  const closingBets = useMemo(
    () => lockedToBets(lockedBets.closing, predictions),
    [lockedBets.closing, predictions],
  );

  // Diff (morning vs closing locked)
  const diffRows = useMemo(
    () => computeDiff(morningBets, closingBets),
    [morningBets, closingBets],
  );

  // Live vs morning diff (auto-compare without needing to lock closing)
  const liveDiff = useMemo(
    () => morningBets.length > 0 ? computeDiff(morningBets, liveBets) : [],
    [morningBets, liveBets],
  );

  // Locked timestamps
  const morningLockedAt = lockedBets.morning[0]?.locked_at ?? null;
  const closingLockedAt = lockedBets.closing[0]?.locked_at ?? null;

  // Auto-store picks to daily_picks table (fire and forget)
  const lastStoredRef = useRef<string>("");
  useEffect(() => {
    if (liveBets.length === 0 || isStoredOnly) return;

    // Dedup: only store when bets change (keyed by fixture+category combos)
    const betsKey = liveBets.map((b) => `${b.fixtureId}-${b.category}-${b.side}-${b.line}`).join("|");
    if (betsKey === lastStoredRef.current) return;
    lastStoredRef.current = betsKey;

    const payload = liveBets.map((b) => ({
      fixtureId: b.fixtureId,
      category: b.category,
      side: b.side,
      homeTeam: b.homeTeam,
      awayTeam: b.awayTeam,
      kickoff: b.kickoff,
      leagueName: b.leagueName,
      leagueKey: b.leagueKey,
      line: b.line,
      stake: b.stake,
      ahClosingLine: b.ahClosingLine,
      totalsLine: b.totalsLine,
      signalsJson: b.signalsJson,
    }));

    fetch("/api/store-picks", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ date, bets: payload }),
    }).catch(console.error);
  }, [liveBets, date, isStoredOnly]);

  // Date navigation
  function navigate(offset: number) {
    const d = new Date(date + "T12:00:00Z");
    d.setUTCDate(d.getUTCDate() + offset);
    const next = d.toISOString().slice(0, 10);
    router.push(`/daily?date=${next}`);
  }

  const displayDate = new Date(date + "T12:00:00Z").toLocaleDateString("en-GB", {
    weekday: "short",
    day: "numeric",
    month: "short",
    year: "numeric",
    timeZone: TZ,
  });

  // Lock handler
  async function handleLock(session: "morning" | "closing") {
    setLocking(session);
    setLockErr(null);
    try {
      const payload = liveBets.map((b) => ({
        fixtureId: b.fixtureId,
        category: b.category,
        side: b.side,
        homeTeam: b.homeTeam,
        awayTeam: b.awayTeam,
        kickoff: b.kickoff,
        leagueName: b.leagueName,
        leagueKey: b.leagueKey,
        line: b.line,
        stake: b.stake,
        ahClosingLine: b.ahClosingLine,
        totalsLine: b.totalsLine,
      }));

      const res = await fetch("/api/lock-bets", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ date, session, bets: payload }),
      });
      const data = await res.json();
      if (!data.success) {
        setLockErr(data.error || "Lock failed");
        return;
      }
      setTab(session);
      router.refresh();
    } catch (e) {
      setLockErr(e instanceof Error ? e.message : "Network error");
    } finally {
      setLocking(null);
    }
  }

  // Settle handler
  async function handleSettle() {
    setSettling(true);
    setSettleErr(null);
    setSettleMsg("Settling...");
    try {
      const success = await runStream(
        "/api/settle-results",
        { date },
        (msg) => setSettleMsg(msg),
      );
      if (success) {
        setSettleMsg(null);
        router.refresh();
      }
    } catch (e) {
      setSettleErr(e instanceof Error ? e.message : "Network error");
    } finally {
      setSettling(false);
      if (!settleErr) setSettleMsg(null);
    }
  }

  // Update odds handler
  async function handleUpdateOdds() {
    setUpdatingOdds(true);
    setOddsErr(null);
    setOddsMsg("Refreshing odds...");
    try {
      const success = await runStream(
        "/api/update-odds",
        { date },
        (msg) => setOddsMsg(msg),
      );
      if (success) {
        setOddsMsg(null);
        router.refresh();
      }
    } catch (e) {
      setOddsErr(e instanceof Error ? e.message : "Network error");
    } finally {
      setUpdatingOdds(false);
      if (!oddsErr) setOddsMsg(null);
    }
  }

  // Tab content
  const tabBets: Record<TabKey, Bet[]> = {
    live: displayBets,
    morning: morningBets,
    closing: closingBets,
    changes: [], // not used
  };

  const TABS: { key: TabKey; label: string }[] = [
    { key: "live", label: isStoredOnly ? "Picks" : "Live" },
    {
      key: "morning",
      label: morningLockedAt
        ? `Morning (${new Date(morningLockedAt).toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", timeZone: TZ })})`
        : "Morning",
    },
    {
      key: "closing",
      label: closingLockedAt
        ? `Closing (${new Date(closingLockedAt).toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", timeZone: TZ })})`
        : "Closing",
    },
    { key: "changes", label: "Changes" },
  ];

  return (
    <div className="space-y-6">
      {/* Date navigation + action buttons */}
      <div className="flex flex-wrap items-center gap-3">
        <Button variant="outline" size="sm" onClick={() => navigate(-1)}>
          &larr; Prev
        </Button>
        <span className="text-sm font-semibold">{displayDate}</span>
        <Button variant="outline" size="sm" onClick={() => navigate(1)}>
          Next &rarr;
        </Button>

        {isStoredOnly && (
          <span className="text-xs text-muted-foreground bg-muted/50 px-2 py-1 rounded">
            Loaded from DB
          </span>
        )}

        <div className="flex-1" />

        {!isStoredOnly && (
          <Button
            variant="outline"
            size="sm"
            onClick={handleUpdateOdds}
            disabled={updatingOdds}
          >
            {updatingOdds ? "Updating..." : "Update Odds"}
          </Button>
        )}

        <Button
          variant="default"
          size="sm"
          onClick={handleSettle}
          disabled={settling}
        >
          {settling ? "Settling..." : "Settle Results"}
        </Button>
      </div>

      {/* Status messages */}
      {(settleErr || settleMsg || oddsErr || oddsMsg || lockErr) && (
        <div className="flex flex-wrap gap-3 text-sm">
          {settleErr && <span className="text-destructive">{settleErr}</span>}
          {!settleErr && settleMsg && <span className="text-muted-foreground">{settleMsg}</span>}
          {oddsErr && <span className="text-destructive">{oddsErr}</span>}
          {!oddsErr && oddsMsg && <span className="text-muted-foreground">{oddsMsg}</span>}
          {lockErr && <span className="text-destructive">{lockErr}</span>}
        </div>
      )}

      {/* Tabs */}
      <div className="flex items-center gap-1 border-b border-border">
        {TABS.map((t) => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
              tab === t.key
                ? "border-primary text-primary"
                : "border-transparent text-muted-foreground hover:text-foreground"
            }`}
          >
            {t.label}
            {t.key !== "changes" && t.key !== "live" && (
              <span className="ml-1 text-xs text-muted-foreground">
                ({(t.key === "morning" ? morningBets : closingBets).length})
              </span>
            )}
          </button>
        ))}

        <span className="ml-auto text-sm text-muted-foreground">
          {displayBets.length} {isStoredOnly ? "stored" : "live"} bet{displayBets.length !== 1 ? "s" : ""}
          {!isStoredOnly && ` from ${cards.length} match${cards.length !== 1 ? "es" : ""}`}
        </span>
      </div>

      {/* Tab content */}
      {tab === "changes" ? (
        <div className="space-y-4">
          {/* Comparison summary */}
          {morningBets.length > 0 && closingBets.length > 0 && (
            <div className="grid grid-cols-2 gap-4">
              {([
                { label: "Morning", bets: morningBets },
                { label: "Closing", bets: closingBets },
              ] as const).map(({ label, bets: sessionBets }) => {
                const totalPnl = sessionBets.reduce((s, b) => s + (b.pnl ?? 0), 0);
                const settledStake = sessionBets
                  .filter((b) => b.result !== null)
                  .reduce((s, b) => s + b.stake, 0);
                const roi = settledStake > 0 ? (totalPnl / settledStake) * 100 : 0;
                return (
                  <div key={label} className="rounded-md border border-border p-4">
                    <div className="text-sm font-medium text-muted-foreground">{label}</div>
                    <div className="mt-1 text-lg font-bold tabular-nums">
                      {sessionBets.length} bets
                    </div>
                    <div className={`text-sm font-semibold tabular-nums ${totalPnl > 0 ? "text-green-400" : totalPnl < 0 ? "text-red-400" : ""}`}>
                      {totalPnl >= 0 ? "+" : ""}{totalPnl.toLocaleString()} THB
                      {settledStake > 0 && (
                        <span className="ml-2">({roi >= 0 ? "+" : ""}{roi.toFixed(1)}% ROI)</span>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
          <DiffTable rows={diffRows} />
        </div>
      ) : (
        <div className="space-y-4">
          {/* Closing Confirmation Card — compare morning direction vs current closing price */}
          {tab === "live" && !isStoredOnly && morningLockedAt && liveDiff.length > 0 && (() => {
            const dropped = liveDiff.filter((d) => d.status === "dropped");
            const sideChanged = liveDiff.filter((d) => d.status === "side_changed");
            const valid = liveDiff.filter((d) => d.status === "unchanged" || d.status === "line_changed");
            const hasProblems = dropped.length > 0 || sideChanged.length > 0;

            // Build a lookup: fixtureId-category → current live bet (for closing line/stake)
            const liveMap = new Map<string, Bet>();
            for (const b of liveBets) liveMap.set(`${b.fixtureId}-${b.category}`, b);
            // Morning bet lookup
            const mornMap = new Map<string, Bet>();
            for (const b of morningBets) mornMap.set(`${b.fixtureId}-${b.category}`, b);

            return (
              <div className="rounded-md border border-border bg-muted/20 p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <div className="text-sm font-bold">
                    Closing Confirmation
                  </div>
                  <div className="text-xs text-muted-foreground">
                    Morning direction + closing price
                  </div>
                </div>

                {/* Valid picks — BET these */}
                {valid.length > 0 && (
                  <div className="space-y-1">
                    {valid.map((d) => {
                      const live = liveMap.get(`${d.fixtureId}-${d.category}`);
                      const morn = mornMap.get(`${d.fixtureId}-${d.category}`);
                      const style = CAT_STYLE[(d.category as BetCategory)] ?? CAT_STYLE.T3;
                      return (
                        <div key={`${d.fixtureId}-${d.category}`} className="flex items-center gap-2 rounded bg-green-500/10 border border-green-500/20 px-3 py-2 text-sm">
                          <span className="text-green-400 font-bold text-xs w-5">BET</span>
                          <span className={`rounded px-1.5 py-0.5 text-[10px] font-bold ${style.bg} ${style.text} border ${style.border}`}>
                            {d.category}
                          </span>
                          <span className="font-medium truncate">{d.homeTeam} vs {d.awayTeam}</span>
                          <span className="text-xs text-muted-foreground ml-auto whitespace-nowrap">
                            {d.morningSide}
                          </span>
                          <span className="text-xs font-mono tabular-nums whitespace-nowrap">
                            {live?.line ?? d.closingLine ?? d.morningLine}
                          </span>
                          <span className="text-xs tabular-nums font-semibold text-green-400 whitespace-nowrap">
                            {(morn?.stake ?? 1000).toLocaleString()}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                )}

                {/* Dropped picks — SKIP */}
                {dropped.length > 0 && (
                  <div className="space-y-1">
                    {dropped.map((d) => {
                      const morn = mornMap.get(`${d.fixtureId}-${d.category}`);
                      return (
                        <div key={`${d.fixtureId}-${d.category}`} className="flex items-center gap-2 rounded bg-red-500/10 border border-red-500/20 px-3 py-2 text-sm">
                          <span className="text-red-400 font-bold text-xs w-5">SKIP</span>
                          <span className="rounded px-1.5 py-0.5 text-[10px] font-bold bg-red-500/10 text-red-400 border border-red-500/30">
                            {d.category}
                          </span>
                          <span className="font-medium truncate text-red-300 line-through">{d.homeTeam} vs {d.awayTeam}</span>
                          <span className="text-xs text-red-400 ml-auto whitespace-nowrap">
                            {d.morningSide} {d.morningLine} — dropped
                          </span>
                        </div>
                      );
                    })}
                  </div>
                )}

                {/* Side flipped — SKIP */}
                {sideChanged.length > 0 && (
                  <div className="space-y-1">
                    {sideChanged.map((d) => (
                      <div key={`${d.fixtureId}-${d.category}`} className="flex items-center gap-2 rounded bg-yellow-500/10 border border-yellow-500/20 px-3 py-2 text-sm">
                        <span className="text-yellow-400 font-bold text-xs w-5">SKIP</span>
                        <span className="rounded px-1.5 py-0.5 text-[10px] font-bold bg-yellow-500/10 text-yellow-400 border border-yellow-500/30">
                          {d.category}
                        </span>
                        <span className="font-medium truncate text-yellow-300">{d.homeTeam} vs {d.awayTeam}</span>
                        <span className="text-xs text-yellow-400 ml-auto whitespace-nowrap">
                          {d.morningSide} → {d.closingSide} FLIPPED
                        </span>
                      </div>
                    ))}
                  </div>
                )}

                {/* Summary */}
                <div className="text-xs text-muted-foreground pt-1 border-t border-border">
                  {valid.length} bet{valid.length !== 1 ? "s" : ""} confirmed
                  {hasProblems && (
                    <span className="text-red-400 ml-2">
                      ({dropped.length + sideChanged.length} skipped)
                    </span>
                  )}
                  {!hasProblems && (
                    <span className="text-green-400 ml-2">
                      — all morning picks valid
                    </span>
                  )}
                </div>
              </div>
            );
          })()}

          {/* Lock buttons (only on Live tab when not in stored-only mode) */}
          {tab === "live" && !isStoredOnly && liveBets.length > 0 && (
            <div className="flex gap-3">
              <Button
                variant="outline"
                size="sm"
                onClick={() => handleLock("morning")}
                disabled={locking !== null}
              >
                {locking === "morning" ? "Locking..." : morningLockedAt ? "Re-lock Morning" : "Lock as Morning"}
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => handleLock("closing")}
                disabled={locking !== null}
              >
                {locking === "closing" ? "Locking..." : closingLockedAt ? "Re-lock Closing" : "Lock as Closing"}
              </Button>
            </div>
          )}

          <SummaryTable bets={tabBets[tab]} />
          <BetTable bets={tabBets[tab]} />
        </div>
      )}
    </div>
  );
}
