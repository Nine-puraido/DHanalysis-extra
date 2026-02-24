"use client";

import { useState, useMemo } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { Card, CardContent } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import type { PredictionRow, TodayFixture } from "@/lib/types";
import {
  MIN_DECIDED,
  WEAK_LEAGUES,
  TOP6_LEAGUES,
  TOTALS_EXCLUDE,
  TOTALS_MIN_EDGE,
  AGREE_OVER_MIN_EDGE,
  AGREE_O_EXCLUDE,
  S1_MODEL_LEAGUES,
  S1_MARKET_LEAGUES,
  T2_EXCLUDE,
  FULL_EXCLUDE,
  IPL_T1_ONLY,
  computeTeamAccuracy,
  computePriceChainSignal,
  computeCardPick,
  computeModelFilterPick,
  computeTotalsPick,
  computeAgreeOverPick,
  getVerdict,
  groupByLeague,
  type MatchCard,
  type ModelFilterPick,
  type TotalsPick,
  type AgreeOverPick,
} from "@/lib/pick-engine";

// Europe: IDs 1-14, World: IDs 15-22
const EUROPE_IDS = new Set([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]);
const TZ = "Asia/Bangkok";

interface Props {
  predictions: PredictionRow[];
  allPredictions: PredictionRow[];
  fixtures: TodayFixture[];
  leagues: { league_key: string; name: string }[];
  days: number;
}

const DAY_OPTIONS = [1, 2, 3, 4, 5, 6, 7] as const;

// Compute historical hit rates for each category from finished predictions.
// Processes matches CHRONOLOGICALLY so team accuracy is built incrementally
// (no look-ahead bias), matching the backend simulation approach.
function computeCategoryStats(allPredictions: PredictionRow[]) {
  type CatStat = { wins: number; total: number; byLeague: Map<string, { wins: number; total: number }> };
  const makeStat = (): CatStat => ({ wins: 0, total: 0, byLeague: new Map() });
  const addTo = (cat: CatStat, lk: string, win: boolean) => {
    cat.total++;
    if (win) cat.wins++;
    if (!cat.byLeague.has(lk)) cat.byLeague.set(lk, { wins: 0, total: 0 });
    const l = cat.byLeague.get(lk)!;
    l.total++;
    if (win) l.wins++;
  };

  // Pre-group by league for price chain signal (already time-filtered internally)
  const byLeague = groupByLeague(allPredictions);

  const stats = {
    t0: makeStat(),
    t1: makeStat(),
    t2: makeStat(),
    t3: makeStat(),
    purple: makeStat(),
    pink: makeStat(),
  };

  // Sort finished predictions chronologically for incremental team accuracy
  const finished = allPredictions
    .filter(r => r.home_score != null && r.away_score != null)
    .sort((a, b) => new Date(a.kickoff_at).getTime() - new Date(b.kickoff_at).getTime());

  // Running team accuracy — sliding window of last 15 decided results per team
  const teamAcc = new Map<string, { results: boolean[] }>();

  for (const r of finished) {
    // Skip fully excluded leagues
    if (FULL_EXCLUDE.has(r.league_key)) continue;

    // --- AH tiers ---
    if (r.ah_fair_line != null && r.ah_closing_line != null && r.ah_fair_line !== r.ah_closing_line) {
      const goalDiff = r.home_score! - r.away_score!;
      const modelBetsHome = r.ah_fair_line < r.ah_closing_line;
      const adjusted = goalDiff + r.ah_closing_line;

      if (adjusted !== 0) {
        const modelWin = modelBetsHome ? adjusted > 0 : adjusted < 0;

        // Compute filter count
        const filt = computeModelFilterPick(r, r.league_key);
        const filtCount = filt?.count ?? 0;

        // Signal 1: League-conditional model direction
        const s1: "model" | "market" | null = S1_MODEL_LEAGUES.has(r.league_key) ? "model"
          : S1_MARKET_LEAGUES.has(r.league_key) ? "market"
          : null;

        // Signal 2: Price chain (already time-filtered to < kickoff internally)
        const leaguePreds = byLeague.get(r.league_key);
        const s2: "model" | "market" | null = leaguePreds
          ? computePriceChainSignal(r, leaguePreds)
          : null;

        // Signal 3: Team trends from CURRENT running counters (no look-ahead, last 15)
        const homeStats = teamAcc.get(r.home_team);
        const awayStats = teamAcc.get(r.away_team);
        let s3: "model" | "market" | null = null;
        if (homeStats && awayStats) {
          const hWins = homeStats.results.filter(x => x).length;
          const hDec = homeStats.results.length;
          const aWins = awayStats.results.filter(x => x).length;
          const aDec = awayStats.results.length;
          if (hDec >= MIN_DECIDED && aDec >= MIN_DECIDED) {
            const hPct = hWins / hDec;
            const aPct = aWins / aDec;
            const wPct = (hPct * hDec + aPct * aDec) / (hDec + aDec);
            s3 = wPct > 0.5 ? "model" : "market";
          }
        }

        // Signal 4: Totals agreement (model+market agree on O/U 2.5 → model; disagree → market)
        let s4: "model" | "market" | null = null;
        if (r.prob_over25 != null && r.closing_over25 != null) {
          const mOver = r.prob_over25 > 0.5;
          const mkOver = 1 / r.closing_over25 > 0.5;
          s4 = mOver === mkOver ? "model" : "market";
        }

        const sigs = [s1, s2, s3, s4].filter((s): s is "model" | "market" => s !== null);
        const mCnt = sigs.filter(s => s === "model").length;
        const mkCnt = sigs.filter(s => s === "market").length;
        let has3sig = false;
        let sigSide: "home" | "away" | null = null;
        if (mCnt !== mkCnt && sigs.length > 0) {
          has3sig = true;
          const followModel = mCnt > mkCnt;
          sigSide = followModel ? (modelBetsHome ? "home" : "away") : (modelBetsHome ? "away" : "home");
        }

        // Classify tier (aligned exactly with getVerdict)
        const hasFilt = filtCount >= 2;
        const sigAgree = has3sig && filt != null && sigSide === filt.side;
        const sigCount = Math.max(mCnt, mkCnt);
        const all4 = sigs.length === 4;
        const unanimous = sigCount === sigs.length;

        // Dog trap: backing underdog on thin gap is negative EV
        const filtFilters = filt?.filters ?? [];
        const isDog = filtFilters.includes("Dog");
        const hasGap = filtFilters.includes("Gap+Top6");
        const dogTrap = isDog && !hasGap;

        const isIplOnly = IPL_T1_ONLY.has(r.league_key);

        // T0: 4 signals + filter ≥2 + unanimous + agree (gold) — no dog trap, IPL excluded
        if (has3sig && all4 && unanimous && hasFilt && sigAgree && !dogTrap && !isIplOnly) {
          const betHome = sigSide === "home";
          const win = betHome ? adjusted > 0 : adjusted < 0;
          addTo(stats.t0, r.league_key, win);
        }
        // T1: 4 signals + filter ≥2 + agree (green) — no dog trap
        else if (has3sig && all4 && hasFilt && sigAgree && !dogTrap) {
          const betHome = sigSide === "home";
          const win = betHome ? adjusted > 0 : adjusted < 0;
          addTo(stats.t1, r.league_key, win);
        }
        // T2: filter ≥2 + signal exists + agree (blue) — skip bleeding leagues, no dog trap, IPL excluded
        else if (hasFilt && has3sig && sigAgree && !T2_EXCLUDE.has(r.league_key) && !dogTrap && !isIplOnly) {
          const betHome = sigSide === "home";
          const win = betHome ? adjusted > 0 : adjusted < 0;
          addTo(stats.t2, r.league_key, win);
        }
        // T3: 4 signals active, no filter required (yellow) — IPL excluded
        // Dog-trap bets only allowed if line gap >= 0.5
        else if (has3sig && all4 && (!dogTrap || Math.abs(r.ah_fair_line! - r.ah_closing_line!) >= 0.5) && !isIplOnly) {
          const betHome = sigSide === "home";
          const win = betHome ? adjusted > 0 : adjusted < 0;
          addTo(stats.t3, r.league_key, win);
        }

        // Update running team accuracy AFTER classification (sliding window of 15)
        for (const team of [r.home_team, r.away_team]) {
          if (!teamAcc.has(team)) teamAcc.set(team, { results: [] });
          const t = teamAcc.get(team)!;
          t.results.push(modelWin);
          if (t.results.length > 15) t.results.shift();
        }
      }
    }

    // --- Totals (skip IPL-T1-only leagues) ---
    if (r.prob_over25 != null && r.closing_over25 != null && !IPL_T1_ONLY.has(r.league_key)) {
      const modelProb = r.prob_over25;
      const marketProb = 1 / r.closing_over25;
      const edge = Math.abs(modelProb - marketProb);
      const modelOver = modelProb > 0.5;
      const marketOver = marketProb > 0.5;
      const actualOver = (r.home_score! + r.away_score!) > 2.5;

      // Purple: disagree, edge >= 15%, excl leagues
      if (modelOver !== marketOver && edge >= TOTALS_MIN_EDGE && !TOTALS_EXCLUDE.has(r.league_key)) {
        const betOver = marketOver;
        const win = betOver === actualOver;
        addTo(stats.purple, r.league_key, win);
      }

      // Pink: agree over + market more confident by >= 10%, excl leagues
      if (modelOver && marketOver && !AGREE_O_EXCLUDE.has(r.league_key) && (marketProb - modelProb) >= AGREE_OVER_MIN_EDGE) {
        const win = actualOver;
        addTo(stats.pink, r.league_key, win);
      }
    }
  }

  return stats;
}

export function TodayMatchList({ predictions, allPredictions, fixtures, leagues, days }: Props) {
  const router = useRouter();
  const [updating, setUpdating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<string | null>(null);

  const predictionMap = new Map<number, PredictionRow>();
  for (const p of predictions) {
    predictionMap.set(p.fixture_id, p);
  }

  // Compute historical hit rates
  const categoryStats = useMemo(() => computeCategoryStats(allPredictions), [allPredictions]);

  function handleDaysChange(newDays: number) {
    router.push(`/today?days=${newDays}`);
  }

  async function handleUpdate() {
    setUpdating(true);
    setError(null);
    setProgress("Starting...");
    try {
      const res = await fetch("/api/update-today", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ days }),
      });

      if (!res.body) {
        setError("No response stream");
        return;
      }

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
            if (msg.message) setProgress(msg.message);
            if (msg.success === true) success = true;
            if (msg.success === false) setError(msg.error || "Update failed");
          } catch {
            // ignore non-JSON lines
          }
        }
      }

      if (success) {
        setProgress(null);
        router.refresh();
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Network error");
    } finally {
      setUpdating(false);
      if (!error) setProgress(null);
    }
  }

  // Build match cards from fixtures + predictions
  const byLeagueMap = groupByLeague(allPredictions);
  const cards: MatchCard[] = [];

  for (const f of fixtures) {
    if (FULL_EXCLUDE.has(f.league_key)) continue;
    const pred = predictionMap.get(f.id) || null;
    const leaguePreds = byLeagueMap.get(f.league_key);
    cards.push({
      fixtureId: f.id,
      kickoff: f.kickoff_at,
      homeTeam: f.home_team,
      awayTeam: f.away_team,
      leagueName: f.league_name,
      leagueId: f.league_id,
      status: f.status,
      prediction: pred,
      pick: pred ? computeCardPick(pred, allPredictions, leaguePreds, f.league_key) : null,
      modelFilter: pred ? computeModelFilterPick(pred, f.league_key) : null,
      totalsPick: pred ? computeTotalsPick(pred, f.league_key) : null,
      agreeOverPick: pred ? computeAgreeOverPick(pred, f.league_key) : null,
      leagueKey: f.league_key,
    });
  }

  // Add any predictions without a matching fixture
  for (const p of predictions) {
    if (FULL_EXCLUDE.has(p.league_key)) continue;
    if (!fixtures.some((f) => f.id === p.fixture_id)) {
      const leaguePreds = byLeagueMap.get(p.league_key);
      cards.push({
        fixtureId: p.fixture_id,
        kickoff: p.kickoff_at,
        homeTeam: p.home_team,
        awayTeam: p.away_team,
        leagueName: p.league_name,
        leagueId: 0,
        status: p.status,
        prediction: p,
        pick: computeCardPick(p, allPredictions, leaguePreds, p.league_key),
        modelFilter: computeModelFilterPick(p, p.league_key),
        totalsPick: computeTotalsPick(p, p.league_key),
        agreeOverPick: computeAgreeOverPick(p, p.league_key),
        leagueKey: p.league_key,
      });
    }
  }

  // Sort by kickoff
  cards.sort(
    (a, b) => new Date(a.kickoff).getTime() - new Date(b.kickoff).getTime()
  );

  // Split into Europe vs World
  const europeCards = cards.filter((c) => EUROPE_IDS.has(c.leagueId));
  const worldCards = cards.filter((c) => !EUROPE_IDS.has(c.leagueId));

  // Group by date, then by league
  function groupByDateLeague(list: MatchCard[]) {
    const map = new Map<string, Map<string, MatchCard[]>>();
    for (const c of list) {
      const dateKey = new Date(c.kickoff).toLocaleDateString("en-GB", {
        weekday: "short",
        day: "numeric",
        month: "short",
        timeZone: TZ,
      });
      if (!map.has(dateKey)) map.set(dateKey, new Map());
      const leagueMap = map.get(dateKey)!;
      if (!leagueMap.has(c.leagueName)) leagueMap.set(c.leagueName, []);
      leagueMap.get(c.leagueName)!.push(c);
    }
    return map;
  }

  const europeByDateLeague = groupByDateLeague(europeCards);
  const worldByDateLeague = groupByDateLeague(worldCards);

  return (
    <div className="space-y-6">
      {/* Controls row */}
      <div className="flex flex-wrap items-center gap-3">
        {/* Days selector */}
        <div className="flex items-center gap-1 rounded-md border border-border p-1">
          {DAY_OPTIONS.map((d) => (
            <button
              key={d}
              onClick={() => handleDaysChange(d)}
              className={`rounded px-2.5 py-1 text-xs font-medium transition-colors ${
                d === days
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:bg-accent hover:text-accent-foreground"
              }`}
            >
              {d}d
            </button>
          ))}
        </div>

        {/* Update button */}
        <Button
          onClick={handleUpdate}
          disabled={updating}
          variant="default"
          size="sm"
        >
          {updating ? (
            <>
              <svg
                className="mr-2 h-4 w-4 animate-spin"
                viewBox="0 0 24 24"
                fill="none"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                />
              </svg>
              Updating {days}d...
            </>
          ) : (
            `Fetch & Predict (${days}d)`
          )}
        </Button>

        {error && <p className="text-sm text-destructive">{error}</p>}
        {!error && progress && (
          <p className="text-sm text-muted-foreground">{progress}</p>
        )}

        {!updating && (
          <span className="text-sm text-muted-foreground">
            {cards.length} match{cards.length !== 1 ? "es" : ""}
            {predictions.length > 0 &&
              ` | ${predictions.length} with predictions`}
          </span>
        )}
      </div>

      {/* Category hit rates legend */}
      <CategoryStatsBar stats={categoryStats} leagues={leagues} />

      {/* Empty state */}
      {cards.length === 0 && (
        <div className="rounded-lg border border-border p-8 text-center text-muted-foreground">
          No fixtures found. Click{" "}
          <strong>Fetch &amp; Predict</strong> to fetch matches.
        </div>
      )}

      {/* Match cards in Europe / World tabs */}
      {cards.length > 0 && (
        <Tabs defaultValue="europe">
          <TabsList>
            <TabsTrigger value="europe">
              Europe ({europeCards.length})
            </TabsTrigger>
            <TabsTrigger value="world">
              World ({worldCards.length})
            </TabsTrigger>
          </TabsList>

          <TabsContent value="europe" className="mt-4">
            <MatchCardGrid groups={europeByDateLeague} showDate={days > 1} />
          </TabsContent>

          <TabsContent value="world" className="mt-4">
            <MatchCardGrid groups={worldByDateLeague} showDate={days > 1} />
          </TabsContent>
        </Tabs>
      )}
    </div>
  );
}

type CatStat = { wins: number; total: number; byLeague: Map<string, { wins: number; total: number }> };

function fmtHr(s: CatStat): string {
  if (s.total === 0) return "-";
  return `${(s.wins / s.total * 100).toFixed(0)}%`;
}

const CAT_CONFIG: { key: string; label: string; border: string; bg: string; text: string }[] = [
  { key: "t0", label: "T0", border: "border-orange-500/30", bg: "bg-orange-500/10", text: "text-orange-400" },
  { key: "t1", label: "T1", border: "border-green-500/30", bg: "bg-green-500/10", text: "text-green-400" },
  { key: "t2", label: "T2", border: "border-blue-500/30", bg: "bg-blue-500/10", text: "text-blue-400" },
  { key: "t3", label: "T3", border: "border-yellow-500/30", bg: "bg-yellow-500/10", text: "text-yellow-400" },
  { key: "purple", label: "O/U", border: "border-purple-500/30", bg: "bg-purple-500/10", text: "text-purple-400" },
  { key: "pink", label: "Agree O", border: "border-pink-500/30", bg: "bg-pink-500/10", text: "text-pink-400" },
];

const LEAGUE_FULL_NAME: Record<string, string> = {
  POL: "Poland - Ekstraklasa",
  GSL: "Greece - Super League",
  CFL: "Czech First League",
  DSL: "Denmark - Superliga",
  RSL: "Romania - SuperLiga",
  UPL: "Ukraine - Premier League",
  HNL: "Croatia - HNL",
  SRBL: "Serbia - Superliga",
  IPL: "Israel - Premier League",
  SB: "Italy - Serie B",
  LL2: "Spain - LaLiga 2",
  L2: "France - Ligue 2",
  EL1: "England - League One",
  GL3: "Germany - 3. Liga",
  SPL_SA: "Saudi Pro League",
  MLS: "USA - MLS",
  J1: "Japan - J1 League",
  ARG: "Argentina - Liga Profesional",
  ALM: "Australia - A-League",
  TL1: "Thailand - Thai League 1",
  QSL: "Qatar - Stars League",
  UAE: "UAE - Pro League",
};

function CategoryStatsBar({ stats, leagues }: { stats: Record<string, CatStat>; leagues: { league_key: string; name: string }[] }) {
  const [expanded, setExpanded] = useState(false);

  // Build league name map from props + fallback
  const leagueNameMap = new Map<string, string>();
  for (const l of leagues) leagueNameMap.set(l.league_key, l.name);

  // Collect all league keys across all categories
  const allLeagues = new Set<string>();
  for (const cat of Object.values(stats)) {
    for (const lk of cat.byLeague.keys()) allLeagues.add(lk);
  }
  const leagueKeys = Array.from(allLeagues).sort();

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-2 text-xs">
        {CAT_CONFIG.map(({ key, label, border, bg, text }) => {
          const s = stats[key];
          if (!s || s.total === 0) return null;
          return (
            <span key={key} className={`rounded-md border ${border} ${bg} px-2 py-1 ${text}`}>
              {label} {fmtHr(s)} ({s.wins}/{s.total})
            </span>
          );
        })}
        <button
          onClick={() => setExpanded(!expanded)}
          className="rounded-md border border-border bg-muted/50 px-2 py-1 text-muted-foreground hover:text-foreground transition-colors"
        >
          {expanded ? "Hide" : "Leagues"}
        </button>
      </div>

      {expanded && (
        <div className="overflow-x-auto rounded-md border border-border">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-border bg-muted/30">
                <th className="px-2 py-1.5 text-left font-medium text-muted-foreground">League</th>
                {CAT_CONFIG.map(({ key, label, text }) => {
                  const s = stats[key];
                  if (!s || s.total === 0) return null;
                  return (
                    <th key={key} className={`px-2 py-1.5 text-right font-medium ${text}`}>
                      {label}
                    </th>
                  );
                })}
              </tr>
            </thead>
            <tbody>
              {leagueKeys.map((lk) => (
                <tr key={lk} className="border-b border-border/50">
                  <td className="px-2 py-1 font-medium whitespace-nowrap">
                    {LEAGUE_FULL_NAME[lk] || leagueNameMap.get(lk) || lk}
                  </td>
                  {CAT_CONFIG.map(({ key, text }) => {
                    const s = stats[key];
                    if (!s || s.total === 0) return null;
                    const lg = s.byLeague.get(lk);
                    if (!lg || lg.total === 0) {
                      return <td key={key} className="px-2 py-1 text-right text-muted-foreground/40">-</td>;
                    }
                    const hr = lg.wins / lg.total * 100;
                    const clr = hr >= 55 ? text : hr >= 50 ? "text-muted-foreground" : "text-red-400/80";
                    return (
                      <td key={key} className={`px-2 py-1 text-right tabular-nums ${clr}`}>
                        {hr.toFixed(0)}% <span className="text-muted-foreground/60">{lg.wins}/{lg.total}</span>
                      </td>
                    );
                  })}
                </tr>
              ))}
              {/* Totals row */}
              <tr className="border-t-2 border-border font-semibold">
                <td className="px-2 py-1">Total</td>
                {CAT_CONFIG.map(({ key, text }) => {
                  const s = stats[key];
                  if (!s || s.total === 0) return null;
                  return (
                    <td key={key} className={`px-2 py-1 text-right tabular-nums ${text}`}>
                      {fmtHr(s)} <span className="text-muted-foreground/60">{s.wins}/{s.total}</span>
                    </td>
                  );
                })}
              </tr>
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function MatchCardGrid({
  groups,
  showDate,
}: {
  groups: Map<string, Map<string, MatchCard[]>>;
  showDate: boolean;
}) {
  if (groups.size === 0) {
    return (
      <p className="py-6 text-center text-sm text-muted-foreground">
        No matches in this category.
      </p>
    );
  }

  return (
    <div className="space-y-4">
      {Array.from(groups.entries()).map(([dateKey, leagueMap]) => (
        <div key={dateKey} className="space-y-4">
          {showDate && (
            <h2 className="border-b border-border pb-1 text-base font-semibold">
              {dateKey}
            </h2>
          )}
          {Array.from(leagueMap.entries()).map(([leagueName, leagueCards]) => (
            <div key={`${dateKey}-${leagueName}`} className="space-y-3">
              <h3 className="text-sm font-semibold text-muted-foreground">
                {leagueName}
              </h3>
              <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
                {leagueCards.map((card) => (
                  <MatchCardComponent key={card.fixtureId} card={card} />
                ))}
              </div>
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}

function Pill({ children, variant = "model" }: { children: React.ReactNode; variant?: "model" | "market" | "na" }) {
  const cls =
    variant === "model"
      ? "bg-blue-500/20 text-blue-400"
      : variant === "market"
        ? "bg-green-500/20 text-green-400"
        : "bg-muted text-muted-foreground";
  return (
    <span className={`inline-block min-w-[3rem] rounded px-2 py-0.5 text-center text-xs font-medium ${cls}`}>
      {children}
    </span>
  );
}

function fmtOdds(prob: number | null): string {
  if (prob == null || prob <= 0) return "N/A";
  return (1 / prob).toFixed(2);
}

function fmtLine(line: number | null): string {
  if (line == null) return "N/A";
  return (line > 0 ? "+" : "") + line.toFixed(2);
}

// getVerdict is imported from pick-engine

function MatchCardComponent({
  card,
}: {
  card: MatchCard;
}) {
  const p = card.prediction;
  const kickoffDate = new Date(card.kickoff);
  const timeStr = kickoffDate.toLocaleTimeString("en-GB", {
    hour: "2-digit",
    minute: "2-digit",
    timeZone: TZ,
  });

  const hasPredictions = p !== null && p.lambda_home !== null;

  // Model P(O2.5) as %
  const modelOver25Pct = p?.prob_over25 != null ? p.prob_over25 * 100 : null;

  const verdict = getVerdict(card);

  return (
    <Link href={`/today/${card.fixtureId}`}>
      <Card className={`cursor-pointer transition-colors hover:border-primary/50 ${
        verdict?.tier === 0 ? "border-orange-500/30" : verdict?.tier === 1 ? "border-green-500/30" : ""
      }`}>
        <CardContent className="flex flex-col p-4">
          {/* Header: time + teams */}
          <div className="mb-2 flex items-center justify-between">
            <span className="rounded bg-muted px-1.5 py-0.5 text-[10px] font-medium text-muted-foreground">
              {timeStr}
            </span>
            {card.status === "finished" && (
              <span className="rounded bg-muted px-1.5 py-0.5 text-[10px] text-muted-foreground">FT</span>
            )}
          </div>
          <div className="mb-3 text-center text-sm font-semibold">
            {card.homeTeam}
            <span className="mx-1.5 text-xs font-normal text-muted-foreground">vs</span>
            {card.awayTeam}
          </div>

          {hasPredictions ? (
            <div className="w-full space-y-3 border-t border-border pt-3 text-xs">
              {/* 1X2 ODDS */}
              {p!.prob_home != null && (
                <div className="flex flex-col items-center">
                  <div className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                    1x2 Odds
                  </div>
                  <div className="space-y-1">
                    <div className="flex items-center justify-center gap-2">
                      <span className="w-11 shrink-0 text-right text-muted-foreground">Model</span>
                      <div className="flex gap-1">
                        <Pill variant="model">{fmtOdds(p!.prob_home)}</Pill>
                        <Pill variant="model">{fmtOdds(p!.prob_draw)}</Pill>
                        <Pill variant="model">{fmtOdds(p!.prob_away)}</Pill>
                      </div>
                    </div>
                    <div className="flex items-center justify-center gap-2">
                      <span className="w-11 shrink-0 text-right text-muted-foreground">Market</span>
                      <div className="flex gap-1">
                        <Pill variant={p!.closing_home != null ? "market" : "na"}>
                          {p!.closing_home != null ? p!.closing_home.toFixed(2) : "N/A"}
                        </Pill>
                        <Pill variant={p!.closing_draw != null ? "market" : "na"}>
                          {p!.closing_draw != null ? p!.closing_draw.toFixed(2) : "N/A"}
                        </Pill>
                        <Pill variant={p!.closing_away != null ? "market" : "na"}>
                          {p!.closing_away != null ? p!.closing_away.toFixed(2) : "N/A"}
                        </Pill>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* ASIAN HANDICAP */}
              <div className="flex flex-col items-center">
                <div className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                  Asian Handicap
                </div>
                <div className="space-y-1">
                  <div className="flex items-center justify-center gap-2">
                    <span className="w-11 shrink-0 text-right text-muted-foreground">Model</span>
                    <div className="flex gap-1">
                      <Pill variant={p!.ah_fair_line != null ? "model" : "na"}>
                        {fmtLine(p!.ah_fair_line)}
                      </Pill>
                      <Pill variant={p!.ah_home_prob != null ? "model" : "na"}>
                        {fmtOdds(p!.ah_home_prob)}
                      </Pill>
                    </div>
                  </div>
                  <div className="flex items-center justify-center gap-2">
                    <span className="w-11 shrink-0 text-right text-muted-foreground">Market</span>
                    <div className="flex gap-1">
                      <Pill variant={p!.ah_closing_line != null ? "market" : "na"}>
                        {fmtLine(p!.ah_closing_line)}
                      </Pill>
                      <Pill variant={p!.ah_closing_home != null ? "market" : "na"}>
                        {p!.ah_closing_home != null ? p!.ah_closing_home.toFixed(2) : "N/A"}
                      </Pill>
                    </div>
                  </div>
                </div>
              </div>

              {/* TOTAL GOALS O/U 2.5 */}
              <div className="flex flex-col items-center">
                <div className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                  Total Goals O/U 2.5
                </div>
                <div className="space-y-1">
                  <div className="flex items-center justify-center gap-2">
                    <span className="w-11 shrink-0 text-right text-muted-foreground">Model</span>
                    <div className="flex gap-1">
                      <Pill variant={p!.prob_over25 != null ? "model" : "na"}>
                        {fmtOdds(p!.prob_over25)}
                      </Pill>
                      <Pill variant={p!.prob_under25 != null ? "model" : "na"}>
                        {fmtOdds(p!.prob_under25)}
                      </Pill>
                    </div>
                  </div>
                  <div className="flex items-center justify-center gap-2">
                    <span className="w-11 shrink-0 text-right text-muted-foreground">Market</span>
                    <div className="flex gap-1">
                      <Pill variant={p!.closing_over25 != null ? "market" : "na"}>
                        {p!.closing_over25 != null ? p!.closing_over25.toFixed(2) : "N/A"}
                      </Pill>
                      <Pill variant={p!.closing_under25 != null ? "market" : "na"}>
                        {p!.closing_under25 != null ? p!.closing_under25.toFixed(2) : "N/A"}
                      </Pill>
                    </div>
                  </div>
                </div>
              </div>

              {/* Expected Goals footer */}
              <div className="border-t border-border pt-2 text-center text-xs text-muted-foreground">
                xG{" "}
                <span className="font-semibold text-foreground">
                  {p!.lambda_home!.toFixed(2)} - {p!.lambda_away!.toFixed(2)}
                </span>
                {modelOver25Pct != null && (
                  <span className="ml-2">
                    O2.5{" "}
                    <span className="font-semibold text-foreground">
                      {modelOver25Pct.toFixed(0)}%
                    </span>
                  </span>
                )}
              </div>

              {/* Combined verdict */}
              {(verdict || card.totalsPick || card.agreeOverPick) && (
                <div className="border-t border-border pt-2 text-center space-y-2">
                  {verdict && (
                    <>
                      <span className={`inline-block rounded-md px-3 py-1 text-xs font-bold ${verdict.cls}`}>
                        {verdict.label}
                      </span>
                      {/* Sub-detail */}
                      <div className="flex flex-wrap items-center justify-center gap-1.5">
                        {card.pick && (
                          <span className="rounded bg-yellow-500/15 px-1.5 py-0.5 text-[10px] text-yellow-400">
                            3sig: {card.pick.side === "home" ? "H" : "A"} ({card.pick.signals}/{card.pick.total}) EV+{(card.pick.edge * 100).toFixed(0)}%
                          </span>
                        )}
                        {card.modelFilter && (
                          <span className={`rounded px-1.5 py-0.5 text-[10px] ${
                            card.modelFilter.count >= 2 ? "bg-blue-500/15 text-blue-400" : "bg-muted text-muted-foreground"
                          }`}>
                            Filt: {card.modelFilter.side === "home" ? "H" : "A"} ({card.modelFilter.count}/4)
                          </span>
                        )}
                        {verdict.tier >= 0 && (
                          <span className="rounded bg-muted px-1.5 py-0.5 text-[10px] text-muted-foreground">
                            {verdict.tier === 0 ? "T0" : verdict.tier === 1 ? "T1" : verdict.tier === 2 ? "T2" : "T3"}
                          </span>
                        )}
                      </div>
                    </>
                  )}
                  {/* Totals picks */}
                  <div className="flex flex-wrap items-center justify-center gap-1.5">
                    {card.totalsPick && (
                      <span className="inline-block rounded-md px-3 py-1 text-xs font-bold bg-purple-500/20 text-purple-400 border border-purple-500/30">
                        {card.totalsPick.side === "over" ? "Over" : "Under"} ({(card.totalsPick.edge * 100).toFixed(0)}%)
                      </span>
                    )}
                    {card.agreeOverPick && (
                      <span className="inline-block rounded-md px-3 py-1 text-xs font-bold bg-pink-500/20 text-pink-400 border border-pink-500/30">
                        Over ({(card.agreeOverPick.edge * 100).toFixed(0)}%)
                      </span>
                    )}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <p className="border-t border-border pt-3 text-center text-xs text-muted-foreground">
              No predictions yet. Click Fetch &amp; Predict.
            </p>
          )}
        </CardContent>
      </Card>
    </Link>
  );
}
