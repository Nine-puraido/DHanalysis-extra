"use client";

import { useMemo } from "react";
import Link from "next/link";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { TrendsChart } from "@/components/trends-chart";
import { TotalsTrendsChart } from "@/components/totals-trends-chart";
import type {
  FixtureDetail,
  PredictionRow,
  LeagueFixtureForChain,
  StandingsRow,
  ProxyMatch,
} from "@/lib/types";
import { S1_MODEL_LEAGUES, S1_MARKET_LEAGUES } from "@/lib/pick-engine";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const TZ = "Asia/Bangkok";
const WEAK_LEAGUES = new Set(["BL", "PPL", "ABL"]);
const TOP6_LEAGUES = new Set(["EPL", "LL", "SPL", "TSL", "SA", "L1"]);

// Totals: leagues where fading model is negative ROI → exclude
const TOTALS_EXCLUDE = new Set(["L1", "BPL", "TSL"]);
const TOTALS_MIN_EDGE = 0.15;
const AGREE_OVER_MIN_EDGE = 0.10;

interface AgreeOverData {
  modelProb: number;
  marketProb: number;
  edge: number;       // market_prob - model_prob
  eligible: boolean;
}

interface TotalsStrategyData {
  modelProb: number;      // model P(Over 2.5)
  marketProb: number;     // implied from closing odds
  modelFavorsOver: boolean;
  marketFavorsOver: boolean;
  disagree: boolean;
  edge: number;           // |model - market|
  eligible: boolean;      // passes edge + league filter
  recommendation: "over" | "under" | null;
  excludeReason: string | null; // why not eligible
}

interface ModelFilterResult {
  exclWeak: boolean;
  top6: boolean;
  gapTop6: boolean;
  exclWeakDog: boolean;
  matchCount: number; // how many filters pass
  pick: "home" | "away" | null; // always model side
  pickTeam: string | null;
  detail: {
    leagueKey: string;
    modelLine: number;
    marketLine: number;
    gap: number;
    modelBetsHome: boolean;
    backingDog: boolean;
  } | null;
}

// ---------------------------------------------------------------------------
// Betting signal types
// ---------------------------------------------------------------------------

interface TeamAccuracy {
  modelWins: number;
  marketWins: number;
  pushes: number;
  decided: number;
  modelPct: number; // modelWins / decided (0-1)
}

type SignalResult = "model" | "market" | null; // null = insufficient data

interface BettingSignalData {
  signal1: SignalResult;
  signal1Detail: { modelLine: number; marketLine: number; modelBetsHome: boolean } | null;
  signal2: SignalResult;
  signal2Detail: { avgImplied: number; modelLine: number; marketLine: number } | null;
  signal3: SignalResult;
  signal3Detail: { homeAcc: TeamAccuracy; awayAcc: TeamAccuracy } | null;
  signal4: SignalResult;
  signal4Detail: { modelOver: boolean; marketOver: boolean } | null;
  modelSignals: number;
  marketSignals: number;
  totalSignals: number;
  recommendation: "home" | "away" | null; // null = insufficient
}

interface Props {
  detail: FixtureDetail;
  allPredictions: PredictionRow[];
  todayPredictions: PredictionRow[];
  leagues: { league_key: string; name: string }[];
  leagueFixtures: LeagueFixtureForChain[];
}

export function TodayMatchDetail({
  detail,
  allPredictions,
  todayPredictions,
  leagues,
  leagueFixtures,
}: Props) {
  const { fixture, league, home_team, away_team, result, predictions, odds } =
    detail;

  // Find this match's prediction row
  const thisPrediction = todayPredictions.find(
    (p) => p.fixture_id === fixture.id
  );

  // Filter predictions involving either team
  const homePredictions = useMemo(
    () =>
      allPredictions.filter(
        (p) =>
          p.home_team === home_team.name || p.away_team === home_team.name
      ),
    [allPredictions, home_team.name]
  );

  const awayPredictions = useMemo(
    () =>
      allPredictions.filter(
        (p) =>
          p.home_team === away_team.name || p.away_team === away_team.name
      ),
    [allPredictions, away_team.name]
  );

  // Upcoming predictions for this team
  const homeUpcoming = useMemo(
    () =>
      todayPredictions.filter(
        (p) =>
          p.home_team === home_team.name || p.away_team === home_team.name
      ),
    [todayPredictions, home_team.name]
  );

  const awayUpcoming = useMemo(
    () =>
      todayPredictions.filter(
        (p) =>
          p.home_team === away_team.name || p.away_team === away_team.name
      ),
    [todayPredictions, away_team.name]
  );

  // Compute standings from league fixtures
  const standings = useMemo(
    () => computeStandings(leagueFixtures),
    [leagueFixtures]
  );

  // Compute proxy matches for price chain
  const proxyMatches = useMemo(
    () =>
      computeProxyMatches(
        leagueFixtures,
        fixture.home_team_id,
        fixture.away_team_id,
        standings
      ),
    [leagueFixtures, fixture.home_team_id, fixture.away_team_id, standings]
  );

  // Compute model filter picks
  const modelFilters = useMemo(
    () =>
      computeModelFilters(
        thisPrediction,
        league.league_key,
        home_team.name,
        away_team.name
      ),
    [thisPrediction, league.league_key, home_team.name, away_team.name]
  );

  // Compute 4-signal betting recommendation
  const bettingSignals = useMemo(
    () =>
      computeBettingSignals(
        thisPrediction,
        proxyMatches,
        homePredictions,
        awayPredictions,
        home_team.name,
        away_team.name,
        league.league_key
      ),
    [thisPrediction, proxyMatches, homePredictions, awayPredictions, home_team.name, away_team.name, league.league_key]
  );

  // Compute totals O/U 2.5 strategy
  const totalsStrategy = useMemo(
    () => computeTotalsStrategy(thisPrediction, league.league_key),
    [thisPrediction, league.league_key]
  );

  // Compute agree over strategy
  const agreeOver = useMemo(
    () => computeAgreeOver(thisPrediction),
    [thisPrediction]
  );

  const kickoffTime = new Date(fixture.kickoff_at).toLocaleTimeString(
    "en-GB",
    { hour: "2-digit", minute: "2-digit", timeZone: TZ }
  );
  const kickoffDate = new Date(fixture.kickoff_at).toLocaleDateString(
    "en-GB",
    { day: "numeric", month: "short", year: "numeric", timeZone: TZ }
  );

  return (
    <div className="space-y-6">
      {/* Back link */}
      <Link
        href="/today"
        className="text-sm text-muted-foreground hover:text-foreground"
      >
        &larr; Back to Today
      </Link>

      {/* Match header */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Badge variant="outline">{league.name}</Badge>
            <span className="text-sm text-muted-foreground">
              {kickoffDate} {kickoffTime} (TH)
            </span>
          </div>
          <CardTitle className="text-2xl">
            {home_team.name} vs {away_team.name}
          </CardTitle>
        </CardHeader>
        <CardContent>
          {thisPrediction && thisPrediction.lambda_home != null ? (
            <div className="grid grid-cols-2 gap-4 text-sm sm:grid-cols-4">
              <div>
                <p className="text-muted-foreground">xG (model)</p>
                <p className="text-lg font-semibold">
                  {thisPrediction.lambda_home.toFixed(2)} -{" "}
                  {thisPrediction.lambda_away!.toFixed(2)}
                </p>
              </div>
              {thisPrediction.ah_fair_line != null && (
                <div>
                  <p className="text-muted-foreground">AH Line</p>
                  <p className="text-lg font-semibold">
                    <span className="text-blue-400">
                      {thisPrediction.ah_fair_line > 0 ? "+" : ""}
                      {thisPrediction.ah_fair_line.toFixed(2)}
                    </span>
                    {thisPrediction.ah_closing_line != null && (
                      <>
                        {" / "}
                        <span className="text-green-400">
                          {thisPrediction.ah_closing_line > 0 ? "+" : ""}
                          {thisPrediction.ah_closing_line.toFixed(2)}
                        </span>
                      </>
                    )}
                  </p>
                </div>
              )}
              {thisPrediction.prob_over25 != null && (
                <div>
                  <p className="text-muted-foreground">P(O2.5)</p>
                  <p className="text-lg font-semibold">
                    <span className="text-blue-400">
                      {(thisPrediction.prob_over25 * 100).toFixed(0)}%
                    </span>
                    {thisPrediction.closing_over25 != null && (
                      <>
                        {" / "}
                        <span className="text-green-400">
                          {((1 / thisPrediction.closing_over25) * 100).toFixed(
                            0
                          )}
                          %
                        </span>
                      </>
                    )}
                  </p>
                </div>
              )}
              {thisPrediction.prob_home != null && (
                <div>
                  <p className="text-muted-foreground">1x2</p>
                  <p className="text-lg font-semibold text-blue-400">
                    {(thisPrediction.prob_home * 100).toFixed(0)} /{" "}
                    {(thisPrediction.prob_draw! * 100).toFixed(0)} /{" "}
                    {(thisPrediction.prob_away! * 100).toFixed(0)}
                  </p>
                </div>
              )}
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">
              No predictions available for this match.
            </p>
          )}

          {result && (
            <div className="mt-4 border-t border-border pt-4">
              <p className="text-muted-foreground">Result</p>
              <p className="text-xl font-bold">
                {result.home_score} - {result.away_score}
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Strategy Picks */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
        {/* 3-Signal Checklist */}
        <BettingSignals
          signals={bettingSignals}
          homeTeam={home_team.name}
          awayTeam={away_team.name}
        />

        {/* Model Filter Picks */}
        <ModelFilters
          filters={modelFilters}
          homeTeam={home_team.name}
          awayTeam={away_team.name}
        />

        {/* Totals O/U 2.5 */}
        <TotalsStrategy
          data={totalsStrategy}
          leagueKey={league.league_key}
        />

        {/* Agree Over */}
        <AgreeOverStrategy data={agreeOver} />
      </div>

      {/* Tabs */}
      <Tabs defaultValue="price-chain">
        <TabsList>
          <TabsTrigger value="home-trends">{home_team.name}</TabsTrigger>
          <TabsTrigger value="away-trends">{away_team.name}</TabsTrigger>
          <TabsTrigger value="price-chain">Price Chain</TabsTrigger>
        </TabsList>

        <TabsContent value="home-trends" className="space-y-6 pt-4">
          {homePredictions.length > 0 ? (
            <>
              <TrendsChart
                data={homePredictions}
                upcoming={homeUpcoming}
                leagues={leagues}
              />
              <TotalsTrendsChart
                data={homePredictions}
                upcoming={homeUpcoming}
                leagues={leagues}
              />
            </>
          ) : (
            <p className="text-sm text-muted-foreground">
              No historical predictions found for {home_team.name}.
            </p>
          )}
        </TabsContent>

        <TabsContent value="away-trends" className="space-y-6 pt-4">
          {awayPredictions.length > 0 ? (
            <>
              <TrendsChart
                data={awayPredictions}
                upcoming={awayUpcoming}
                leagues={leagues}
              />
              <TotalsTrendsChart
                data={awayPredictions}
                upcoming={awayUpcoming}
                leagues={leagues}
              />
            </>
          ) : (
            <p className="text-sm text-muted-foreground">
              No historical predictions found for {away_team.name}.
            </p>
          )}
        </TabsContent>

        <TabsContent value="price-chain" className="space-y-6 pt-4">
          <PriceChainSection
            proxyMatches={proxyMatches}
            homeTeam={home_team.name}
            awayTeam={away_team.name}
            currentAhLine={thisPrediction?.ah_closing_line ?? null}
            modelAhLine={thisPrediction?.ah_fair_line ?? null}
          />
        </TabsContent>
      </Tabs>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Betting Signals
// ---------------------------------------------------------------------------

function computeTeamAccuracy(
  predictions: PredictionRow[],
  teamName: string,
  beforeKickoff?: string,
): TeamAccuracy {
  // Filter to eligible decided matches
  let eligible = predictions.filter((r) => {
    if (r.ah_fair_line == null || r.ah_closing_line == null) return false;
    if (r.home_score == null || r.away_score == null) return false;
    if (r.ah_fair_line === r.ah_closing_line) return false;
    if (r.home_team !== teamName && r.away_team !== teamName) return false;
    if (beforeKickoff && new Date(r.kickoff_at).getTime() >= new Date(beforeKickoff).getTime()) return false;
    return true;
  });

  // Sort descending by kickoff and take last 15
  eligible.sort((a, b) => new Date(b.kickoff_at).getTime() - new Date(a.kickoff_at).getTime());
  eligible = eligible.slice(0, 15);

  let modelWins = 0;
  let marketWins = 0;
  let pushes = 0;

  for (const r of eligible) {
    const goalDiff = r.home_score! - r.away_score!;
    const modelBetsHome = r.ah_fair_line! < r.ah_closing_line!;
    const adjusted = goalDiff + r.ah_closing_line!;

    if (adjusted === 0) {
      pushes++;
    } else if (modelBetsHome) {
      if (adjusted > 0) modelWins++;
      else marketWins++;
    } else {
      if (adjusted < 0) modelWins++;
      else marketWins++;
    }
  }

  const decided = modelWins + marketWins;
  return {
    modelWins,
    marketWins,
    pushes,
    decided,
    modelPct: decided > 0 ? modelWins / decided : 0,
  };
}

function computeBettingSignals(
  thisPrediction: PredictionRow | undefined,
  proxyMatches: ProxyMatch[],
  homePredictions: PredictionRow[],
  awayPredictions: PredictionRow[],
  homeTeamName: string,
  awayTeamName: string,
  leagueKey?: string,
): BettingSignalData {
  const empty: BettingSignalData = {
    signal1: null,
    signal1Detail: null,
    signal2: null,
    signal2Detail: null,
    signal3: null,
    signal3Detail: null,
    signal4: null,
    signal4Detail: null,
    modelSignals: 0,
    marketSignals: 0,
    totalSignals: 0,
    recommendation: null,
  };

  if (
    !thisPrediction ||
    thisPrediction.ah_fair_line == null ||
    thisPrediction.ah_closing_line == null ||
    thisPrediction.ah_fair_line === thisPrediction.ah_closing_line
  ) {
    return empty;
  }

  const modelLine = thisPrediction.ah_fair_line;
  const marketLine = thisPrediction.ah_closing_line;
  const modelBetsHome = modelLine < marketLine;

  // Signal 1: League-conditional model direction
  const signal1: SignalResult = leagueKey
    ? S1_MODEL_LEAGUES.has(leagueKey) ? "model"
      : S1_MARKET_LEAGUES.has(leagueKey) ? "market"
      : null
    : "model"; // fallback if no leagueKey
  const signal1Detail = { modelLine, marketLine, modelBetsHome };

  // Signal 2: Price chain (requires ≥3 proxy teams)
  let signal2: SignalResult = null;
  let signal2Detail: BettingSignalData["signal2Detail"] = null;
  const gapsWithValues = proxyMatches.filter((p) => p.implied_ah_gap !== null);
  if (gapsWithValues.length >= 3) {
    const avgImplied =
      gapsWithValues.reduce((s, p) => s + p.implied_ah_gap!, 0) /
      gapsWithValues.length;
    const distToModel = Math.abs(avgImplied - modelLine);
    const distToMarket = Math.abs(avgImplied - marketLine);
    signal2 = distToModel < distToMarket ? "model" : "market";
    signal2Detail = { avgImplied, modelLine, marketLine };
  }

  // Signal 3: Team trends (weighted average, last 15 matches)
  let signal3: SignalResult = null;
  let signal3Detail: BettingSignalData["signal3Detail"] = null;
  const homeAcc = computeTeamAccuracy(homePredictions, homeTeamName, thisPrediction.kickoff_at);
  const awayAcc = computeTeamAccuracy(awayPredictions, awayTeamName, thisPrediction.kickoff_at);
  const MIN_DECIDED = 3;
  if (homeAcc.decided >= MIN_DECIDED && awayAcc.decided >= MIN_DECIDED) {
    const totalDecided = homeAcc.decided + awayAcc.decided;
    const weightedModelPct =
      (homeAcc.modelPct * homeAcc.decided +
        awayAcc.modelPct * awayAcc.decided) /
      totalDecided;
    signal3 = weightedModelPct > 0.5 ? "model" : "market";
    signal3Detail = { homeAcc, awayAcc };
  }

  // Signal 4: Totals agreement (model+market agree on O/U → model; disagree → market)
  let signal4: SignalResult = null;
  let signal4Detail: BettingSignalData["signal4Detail"] = null;
  if (thisPrediction.prob_over25 != null && thisPrediction.closing_over25 != null) {
    const modelOver = thisPrediction.prob_over25 > 0.5;
    const marketOver = 1 / thisPrediction.closing_over25 > 0.5;
    signal4 = modelOver === marketOver ? "model" : "market";
    signal4Detail = { modelOver, marketOver };
  }

  // Majority vote
  const signals = [signal1, signal2, signal3, signal4].filter(
    (s) => s !== null
  ) as ("model" | "market")[];
  const modelCount = signals.filter((s) => s === "model").length;
  const marketCount = signals.filter((s) => s === "market").length;

  let recommendation: "home" | "away" | null = null;
  if (signals.length >= 2) {
    if (modelCount > marketCount) {
      recommendation = modelBetsHome ? "home" : "away";
    } else if (marketCount > modelCount) {
      recommendation = modelBetsHome ? "away" : "home";
    }
    // tie → null (no recommendation)
  }

  return {
    signal1,
    signal1Detail,
    signal2,
    signal2Detail,
    signal3,
    signal3Detail,
    signal4,
    signal4Detail,
    modelSignals: modelCount,
    marketSignals: marketCount,
    totalSignals: signals.length,
    recommendation,
  };
}

function BettingSignals({
  signals,
  homeTeam,
  awayTeam,
}: {
  signals: BettingSignalData;
  homeTeam: string;
  awayTeam: string;
}) {
  if (!signals.signal1Detail) {
    return null; // No AH lines → nothing to show
  }

  const { recommendation, modelSignals, marketSignals, totalSignals } = signals;
  const agreedCount = Math.max(modelSignals, marketSignals);

  // Border color
  const borderClass =
    recommendation && agreedCount === totalSignals && totalSignals >= 3
      ? "border-green-500/60"
      : recommendation && agreedCount >= 2
        ? "border-yellow-500/50"
        : "border-border";

  const betLabel =
    recommendation === "home"
      ? `Bet ${homeTeam} (HOME)`
      : recommendation === "away"
        ? `Bet ${awayTeam} (AWAY)`
        : "Insufficient signals";

  return (
    <Card className={`${borderClass} border-2`}>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Betting Signals</CardTitle>
        {/* Prominent recommendation banner */}
        {recommendation ? (
          <div
            className={`mt-2 rounded-md px-4 py-3 text-center text-lg font-bold ${
              agreedCount === totalSignals && totalSignals >= 3
                ? "bg-green-600/20 text-green-400"
                : "bg-yellow-600/20 text-yellow-400"
            }`}
          >
            {betLabel} ({agreedCount}/{totalSignals} signals)
          </div>
        ) : (
          <div className="mt-2 rounded-md bg-muted/50 px-4 py-3 text-center text-sm text-muted-foreground">
            {betLabel}
          </div>
        )}
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        {/* Signal 1: AH Line */}
        <div className="flex items-start justify-between gap-2">
          <div className="flex-1">
            <div className="flex items-center gap-2">
              <span className="font-medium">1. AH Line (League)</span>
              <SignalBadge result={signals.signal1} />
            </div>
            {signals.signal1Detail && (
              <p className="mt-0.5 text-xs text-muted-foreground">
                Model{" "}
                <span className="text-blue-400">
                  {fmtSignedLine(signals.signal1Detail.modelLine)}
                </span>
                {" vs Market "}
                <span className="text-green-400">
                  {fmtSignedLine(signals.signal1Detail.marketLine)}
                </span>
                {" → "}
                {signals.signal1Detail.modelBetsHome
                  ? "Model says Home undervalued"
                  : "Model says Away undervalued"}
              </p>
            )}
          </div>
        </div>

        {/* Signal 2: Price Chain */}
        <div className="flex items-start justify-between gap-2">
          <div className="flex-1">
            <div className="flex items-center gap-2">
              <span className="font-medium">2. Price Chain</span>
              <SignalBadge result={signals.signal2} />
            </div>
            {signals.signal2Detail ? (
              <p className="mt-0.5 text-xs text-muted-foreground">
                Avg implied:{" "}
                <span className="text-orange-400">
                  {fmtSignedLine(signals.signal2Detail.avgImplied)}
                </span>
                {" → Closer to "}
                {signals.signal2 === "model" ? (
                  <span className="text-blue-400">model</span>
                ) : (
                  <span className="text-green-400">market</span>
                )}
              </p>
            ) : (
              <p className="mt-0.5 text-xs text-muted-foreground">
                Fewer than 3 proxy matches (need {"\u2265"}3)
              </p>
            )}
          </div>
        </div>

        {/* Signal 3: Team Trends */}
        <div className="flex items-start justify-between gap-2">
          <div className="flex-1">
            <div className="flex items-center gap-2">
              <span className="font-medium">3. Team Trends (L15)</span>
              <SignalBadge result={signals.signal3} />
            </div>
            {signals.signal3Detail ? (
              <div className="mt-0.5 text-xs text-muted-foreground">
                <span>
                  {homeTeam}: Model{" "}
                  {(signals.signal3Detail.homeAcc.modelPct * 100).toFixed(0)}%
                  ({signals.signal3Detail.homeAcc.modelWins}/
                  {signals.signal3Detail.homeAcc.decided})
                </span>
                <span className="mx-2">|</span>
                <span>
                  {awayTeam}: Model{" "}
                  {(signals.signal3Detail.awayAcc.modelPct * 100).toFixed(0)}%
                  ({signals.signal3Detail.awayAcc.modelWins}/
                  {signals.signal3Detail.awayAcc.decided})
                </span>
              </div>
            ) : (
              <p className="mt-0.5 text-xs text-muted-foreground">
                Insufficient historical data (min 3 decided matches per team)
              </p>
            )}
          </div>
        </div>

        {/* Signal 4: Totals Agreement */}
        <div className="flex items-start justify-between gap-2">
          <div className="flex-1">
            <div className="flex items-center gap-2">
              <span className="font-medium">4. Totals Agreement</span>
              <SignalBadge result={signals.signal4} />
            </div>
            {signals.signal4Detail ? (
              <p className="mt-0.5 text-xs text-muted-foreground">
                Model{" "}
                <span className="text-blue-400">
                  {signals.signal4Detail.modelOver ? "Over" : "Under"}
                </span>
                {" vs Market "}
                <span className="text-green-400">
                  {signals.signal4Detail.marketOver ? "Over" : "Under"}
                </span>
                {" → "}
                {signals.signal4 === "model" ? "Agree (model)" : "Disagree (market)"}
              </p>
            ) : (
              <p className="mt-0.5 text-xs text-muted-foreground">
                No totals data available
              </p>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Totals O/U 2.5 Strategy
// ---------------------------------------------------------------------------

function computeTotalsStrategy(
  thisPrediction: PredictionRow | undefined,
  leagueKey: string,
): TotalsStrategyData {
  const empty: TotalsStrategyData = {
    modelProb: 0,
    marketProb: 0,
    modelFavorsOver: false,
    marketFavorsOver: false,
    disagree: false,
    edge: 0,
    eligible: false,
    recommendation: null,
    excludeReason: "No totals data available",
  };

  if (!thisPrediction || thisPrediction.prob_over25 == null || thisPrediction.closing_over25 == null) {
    return empty;
  }

  const modelProb = thisPrediction.prob_over25;
  const marketProb = 1 / thisPrediction.closing_over25;
  const modelFavorsOver = modelProb > 0.5;
  const marketFavorsOver = marketProb > 0.5;
  const disagree = modelFavorsOver !== marketFavorsOver;
  const edge = Math.abs(modelProb - marketProb);

  let excludeReason: string | null = null;
  if (!disagree) {
    excludeReason = "Model and market agree";
  } else if (TOTALS_EXCLUDE.has(leagueKey)) {
    excludeReason = `League ${leagueKey} excluded (negative ROI)`;
  } else if (edge < TOTALS_MIN_EDGE) {
    excludeReason = `Edge ${(edge * 100).toFixed(1)}% < ${(TOTALS_MIN_EDGE * 100).toFixed(0)}% threshold`;
  }

  const eligible = disagree && !TOTALS_EXCLUDE.has(leagueKey) && edge >= TOTALS_MIN_EDGE;
  const recommendation = eligible ? (marketFavorsOver ? "over" : "under") : null;

  return {
    modelProb,
    marketProb,
    modelFavorsOver,
    marketFavorsOver,
    disagree,
    edge,
    eligible,
    recommendation,
    excludeReason,
  };
}

function TotalsStrategy({
  data,
  leagueKey,
}: {
  data: TotalsStrategyData;
  leagueKey: string;
}) {
  if (data.modelProb === 0 && data.marketProb === 0) {
    return (
      <Card className="border-border">
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Totals O/U 2.5</CardTitle>
          <div className="mt-2 rounded-md bg-muted/50 px-4 py-3 text-center text-sm text-muted-foreground">
            No totals data available
          </div>
        </CardHeader>
      </Card>
    );
  }

  const borderClass = data.eligible
    ? "border-purple-500/60"
    : "border-border";

  return (
    <Card className={`${borderClass} border-2`}>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Totals O/U 2.5</CardTitle>
        {data.recommendation ? (
          <div className="mt-2 rounded-md bg-purple-600/20 px-4 py-3 text-center text-lg font-bold text-purple-400">
            {data.recommendation === "over" ? "Over 2.5" : "Under 2.5"} ({(data.edge * 100).toFixed(0)}% edge)
          </div>
        ) : (
          <div className="mt-2 rounded-md bg-muted/50 px-4 py-3 text-center text-sm text-muted-foreground">
            {data.excludeReason || "No recommendation"}
          </div>
        )}
      </CardHeader>
      <CardContent className="space-y-3 text-sm">
        {/* Model vs Market */}
        <div className="flex items-center justify-between">
          <span className="text-muted-foreground">Model P(O2.5)</span>
          <span className="tabular-nums font-medium">
            <span className="text-blue-400">{(data.modelProb * 100).toFixed(1)}%</span>
            {" → "}
            {data.modelFavorsOver ? "Over" : "Under"}
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-muted-foreground">Market P(O2.5)</span>
          <span className="tabular-nums font-medium">
            <span className="text-green-400">{(data.marketProb * 100).toFixed(1)}%</span>
            {" → "}
            {data.marketFavorsOver ? "Over" : "Under"}
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-muted-foreground">Disagree?</span>
          <span className={`font-medium ${data.disagree ? "text-orange-400" : "text-muted-foreground"}`}>
            {data.disagree ? "Yes" : "No"}
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-muted-foreground">Edge</span>
          <span className={`tabular-nums font-medium ${data.edge >= TOTALS_MIN_EDGE ? "text-purple-400" : "text-muted-foreground"}`}>
            {(data.edge * 100).toFixed(1)}%
          </span>
        </div>

        {/* Strategy explanation */}
        <div className="border-t border-border pt-2 text-xs text-muted-foreground">
          Strategy: fade model when it disagrees with market (58.5% HR, +16.9% ROI at {"\u2265"}15% edge).
          {TOTALS_EXCLUDE.has(leagueKey) && (
            <span className="text-red-400/80"> League excluded.</span>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Agree Over Strategy
// ---------------------------------------------------------------------------

function computeAgreeOver(
  thisPrediction: PredictionRow | undefined,
): AgreeOverData {
  const empty: AgreeOverData = {
    modelProb: 0,
    marketProb: 0,
    edge: 0,
    eligible: false,
  };

  if (!thisPrediction || thisPrediction.prob_over25 == null || thisPrediction.closing_over25 == null) {
    return empty;
  }

  const modelProb = thisPrediction.prob_over25;
  const marketProb = 1 / thisPrediction.closing_over25;

  // Both must favor Over
  if (modelProb <= 0.5 || marketProb <= 0.5) {
    return { modelProb, marketProb, edge: 0, eligible: false };
  }

  // Market must be more confident
  const edge = marketProb - modelProb;
  const eligible = edge >= AGREE_OVER_MIN_EDGE;

  return { modelProb, marketProb, edge, eligible };
}

function AgreeOverStrategy({ data }: { data: AgreeOverData }) {
  // Don't show if neither side favors Over
  if (data.modelProb === 0 && data.marketProb === 0) {
    return null;
  }

  // Only show card when both favor Over
  const bothOver = data.modelProb > 0.5 && data.marketProb > 0.5;
  if (!bothOver) return null;

  const borderClass = data.eligible ? "border-pink-500/60" : "border-border";

  return (
    <Card className={`${borderClass} border-2`}>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Agree Over</CardTitle>
        {data.eligible ? (
          <div className="mt-2 rounded-md bg-pink-600/20 px-4 py-3 text-center text-lg font-bold text-pink-400">
            Over 2.5 ({(data.edge * 100).toFixed(0)}% edge)
          </div>
        ) : (
          <div className="mt-2 rounded-md bg-muted/50 px-4 py-3 text-center text-sm text-muted-foreground">
            Edge {(data.edge * 100).toFixed(1)}% &lt; {(AGREE_OVER_MIN_EDGE * 100).toFixed(0)}% threshold
          </div>
        )}
      </CardHeader>
      <CardContent className="space-y-3 text-sm">
        <div className="flex items-center justify-between">
          <span className="text-muted-foreground">Model P(O2.5)</span>
          <span className="tabular-nums font-medium text-blue-400">
            {(data.modelProb * 100).toFixed(1)}%
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-muted-foreground">Market P(O2.5)</span>
          <span className="tabular-nums font-medium text-green-400">
            {(data.marketProb * 100).toFixed(1)}%
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-muted-foreground">Market edge</span>
          <span className={`tabular-nums font-medium ${data.eligible ? "text-pink-400" : "text-muted-foreground"}`}>
            +{(data.edge * 100).toFixed(1)}%
          </span>
        </div>
        <div className="border-t border-border pt-2 text-xs text-muted-foreground">
          Both agree Over, market more confident. 66% HR at {"\u2265"}10% edge (flat unit).
        </div>
      </CardContent>
    </Card>
  );
}

function SignalBadge({ result }: { result: SignalResult }) {
  if (result === null) {
    return (
      <Badge variant="outline" className="text-xs text-muted-foreground">
        N/A
      </Badge>
    );
  }
  return result === "model" ? (
    <Badge className="bg-blue-500/20 text-xs text-blue-400 hover:bg-blue-500/20">
      Model
    </Badge>
  ) : (
    <Badge className="bg-green-500/20 text-xs text-green-400 hover:bg-green-500/20">
      Market
    </Badge>
  );
}

function fmtSignedLine(line: number): string {
  return (line > 0 ? "+" : "") + line.toFixed(2);
}

// ---------------------------------------------------------------------------
// Model Filters
// ---------------------------------------------------------------------------

function computeModelFilters(
  thisPrediction: PredictionRow | undefined,
  leagueKey: string,
  homeTeamName: string,
  awayTeamName: string
): ModelFilterResult {
  const empty: ModelFilterResult = {
    exclWeak: false,
    top6: false,
    gapTop6: false,
    exclWeakDog: false,
    matchCount: 0,
    pick: null,
    pickTeam: null,
    detail: null,
  };

  if (
    !thisPrediction ||
    thisPrediction.ah_fair_line == null ||
    thisPrediction.ah_closing_line == null ||
    thisPrediction.ah_fair_line === thisPrediction.ah_closing_line
  ) {
    return empty;
  }

  const modelLine = thisPrediction.ah_fair_line;
  const marketLine = thisPrediction.ah_closing_line;
  const gap = Math.abs(modelLine - marketLine);
  const modelBetsHome = modelLine < marketLine;

  // Model backs underdog?
  // If model bets home: backing dog when market AH >= 0 (home is dog)
  // If model bets away: backing dog when market AH < 0 (away is dog)
  const backingDog = modelBetsHome ? marketLine >= 0 : marketLine < 0;

  const isWeak = WEAK_LEAGUES.has(leagueKey);
  const isTop6 = TOP6_LEAGUES.has(leagueKey);

  const exclWeak = !isWeak;
  const top6 = isTop6;
  const gapTop6 = isTop6 && gap >= 0.5;
  const exclWeakDog = !isWeak && backingDog;

  const matchCount = [exclWeak, top6, gapTop6, exclWeakDog].filter(Boolean).length;

  const pick: "home" | "away" = modelBetsHome ? "home" : "away";
  const pickTeam = modelBetsHome ? homeTeamName : awayTeamName;

  return {
    exclWeak,
    top6,
    gapTop6,
    exclWeakDog,
    matchCount,
    pick,
    pickTeam,
    detail: {
      leagueKey,
      modelLine,
      marketLine,
      gap,
      modelBetsHome,
      backingDog,
    },
  };
}

function ModelFilters({
  filters,
  homeTeam,
  awayTeam,
}: {
  filters: ModelFilterResult;
  homeTeam: string;
  awayTeam: string;
}) {
  if (!filters.detail) {
    return (
      <Card className="border-border">
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Model Filters</CardTitle>
          <div className="mt-2 rounded-md bg-muted/50 px-4 py-3 text-center text-sm text-muted-foreground">
            No AH lines available
          </div>
        </CardHeader>
      </Card>
    );
  }

  const { matchCount, pick, pickTeam, detail } = filters;

  const borderClass =
    matchCount >= 4
      ? "border-green-500/60"
      : matchCount >= 2
        ? "border-yellow-500/50"
        : "border-border";

  const filterItems: { key: string; label: string; desc: string; pass: boolean; roi: string }[] = [
    {
      key: "exclWeak",
      label: "Excl Weak",
      desc: "League not in BL, PPL, ABL",
      pass: filters.exclWeak,
      roi: "+10.4%",
    },
    {
      key: "top6",
      label: "Top 6",
      desc: "League in EPL, LL, SPL, TSL, SA, L1",
      pass: filters.top6,
      roi: "+17.5%",
    },
    {
      key: "gapTop6",
      label: "Gap + Top 6",
      desc: `Top 6 + gap >= 0.50 (gap: ${detail.gap.toFixed(2)})`,
      pass: filters.gapTop6,
      roi: "+23.1%",
    },
    {
      key: "exclWeakDog",
      label: "Excl Weak + Dog",
      desc: `Not weak + model backs ${detail.backingDog ? "underdog" : "favorite"}`,
      pass: filters.exclWeakDog,
      roi: "+20.4%",
    },
  ];

  return (
    <Card className={`${borderClass} border-2`}>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Model Filters</CardTitle>
        {matchCount > 0 ? (
          <div
            className={`mt-2 rounded-md px-4 py-3 text-center text-lg font-bold ${
              matchCount >= 4
                ? "bg-green-600/20 text-green-400"
                : matchCount >= 2
                  ? "bg-yellow-600/20 text-yellow-400"
                  : "bg-muted/50 text-muted-foreground"
            }`}
          >
            {pickTeam} ({pick === "home" ? "HOME" : "AWAY"}) — {matchCount}/4 filters
          </div>
        ) : (
          <div className="mt-2 rounded-md bg-muted/50 px-4 py-3 text-center text-sm text-muted-foreground">
            No filters match
          </div>
        )}
      </CardHeader>
      <CardContent className="space-y-3 text-sm">
        {filterItems.map((f) => (
          <div key={f.key} className="flex items-start justify-between gap-2">
            <div className="flex-1">
              <div className="flex items-center gap-2">
                <span className={f.pass ? "text-green-400" : "text-muted-foreground/50"}>
                  {f.pass ? "\u2713" : "\u2717"}
                </span>
                <span className={`font-medium ${!f.pass ? "text-muted-foreground/50" : ""}`}>
                  {f.label}
                </span>
                <Badge
                  variant="outline"
                  className={`text-[10px] ${f.pass ? "text-foreground" : "text-muted-foreground/40"}`}
                >
                  ROI {f.roi}
                </Badge>
              </div>
              <p className={`mt-0.5 text-xs ${f.pass ? "text-muted-foreground" : "text-muted-foreground/40"}`}>
                {f.desc}
              </p>
            </div>
          </div>
        ))}

        {/* AH line summary */}
        <div className="border-t border-border pt-2 text-xs text-muted-foreground">
          Model{" "}
          <span className="text-blue-400">{fmtSignedLine(detail.modelLine)}</span>
          {" vs Market "}
          <span className="text-green-400">{fmtSignedLine(detail.marketLine)}</span>
          {" | Gap "}
          <span className="font-medium text-foreground">{detail.gap.toFixed(2)}</span>
          {" | "}
          <span className="font-medium text-foreground">
            {detail.backingDog ? "Underdog" : "Favorite"}
          </span>
        </div>
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Price Chain
// ---------------------------------------------------------------------------

function PriceChainSection({
  proxyMatches,
  homeTeam,
  awayTeam,
  currentAhLine,
  modelAhLine,
}: {
  proxyMatches: ProxyMatch[];
  homeTeam: string;
  awayTeam: string;
  currentAhLine: number | null;
  modelAhLine: number | null;
}) {
  if (proxyMatches.length === 0) {
    return (
      <Card>
        <CardContent className="py-8 text-center text-muted-foreground">
          No proxy teams found within 2 months. Price chain requires common
          opponents with AH odds data.
        </CardContent>
      </Card>
    );
  }

  // Compute average implied gap
  const gapsWithValues = proxyMatches.filter(
    (p) => p.implied_ah_gap !== null
  );
  const avgGap =
    gapsWithValues.length > 0
      ? gapsWithValues.reduce((s, p) => s + p.implied_ah_gap!, 0) /
        gapsWithValues.length
      : null;

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="text-base">
            Price Chain: {homeTeam} vs {awayTeam}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Proxy (Pos.)</TableHead>
                  <TableHead>vs {homeTeam}</TableHead>
                  <TableHead className="text-right">AH</TableHead>
                  <TableHead className="text-right">Total</TableHead>
                  <TableHead>vs {awayTeam}</TableHead>
                  <TableHead className="text-right">AH</TableHead>
                  <TableHead className="text-right">Total</TableHead>
                  <TableHead className="text-right">Implied AH</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {proxyMatches.map((pm) => (
                  <TableRow key={pm.proxy_team_id}>
                    <TableCell className="font-medium">
                      {pm.proxy_team_name} ({ordinal(pm.proxy_position)})
                    </TableCell>
                    <TableCell className="text-xs">
                      <div>{pm.vs_a_home_team} vs {pm.vs_a_away_team}</div>
                      <div className="text-muted-foreground">
                        {pm.vs_a_score} |{" "}
                        {new Date(pm.vs_a_date).toLocaleDateString("en-GB", {
                          day: "numeric",
                          month: "short",
                          timeZone: TZ,
                        })}
                      </div>
                    </TableCell>
                    <TableCell className="text-right text-sm">
                      {pm.vs_a_ah_line != null
                        ? `${pm.vs_a_ah_line > 0 ? "+" : ""}${pm.vs_a_ah_line.toFixed(2)}`
                        : "-"}
                    </TableCell>
                    <TableCell className="text-right text-sm">
                      {pm.vs_a_totals_line != null
                        ? pm.vs_a_totals_line.toFixed(1)
                        : "-"}
                    </TableCell>
                    <TableCell className="text-xs">
                      <div>{pm.vs_b_home_team} vs {pm.vs_b_away_team}</div>
                      <div className="text-muted-foreground">
                        {pm.vs_b_score} |{" "}
                        {new Date(pm.vs_b_date).toLocaleDateString("en-GB", {
                          day: "numeric",
                          month: "short",
                          timeZone: TZ,
                        })}
                      </div>
                    </TableCell>
                    <TableCell className="text-right text-sm">
                      {pm.vs_b_ah_line != null
                        ? `${pm.vs_b_ah_line > 0 ? "+" : ""}${pm.vs_b_ah_line.toFixed(2)}`
                        : "-"}
                    </TableCell>
                    <TableCell className="text-right text-sm">
                      {pm.vs_b_totals_line != null
                        ? pm.vs_b_totals_line.toFixed(1)
                        : "-"}
                    </TableCell>
                    <TableCell className="text-right font-medium">
                      {pm.implied_ah_gap != null
                        ? `${pm.implied_ah_gap > 0 ? "+" : ""}${pm.implied_ah_gap.toFixed(2)}`
                        : "-"}
                    </TableCell>
                  </TableRow>
                ))}

                {/* Summary row */}
                <TableRow className="border-t-2 font-semibold">
                  <TableCell colSpan={7} className="text-right">
                    Avg Implied AH
                  </TableCell>
                  <TableCell className="text-right">
                    {avgGap != null
                      ? `${avgGap > 0 ? "+" : ""}${avgGap.toFixed(2)}`
                      : "-"}
                  </TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </div>

          {/* Reference lines */}
          <div className="mt-4 flex flex-wrap gap-4 border-t border-border pt-4 text-sm">
            {currentAhLine != null && (
              <div>
                <span className="text-muted-foreground">Market AH: </span>
                <span className="font-semibold text-green-400">
                  {currentAhLine > 0 ? "+" : ""}
                  {currentAhLine.toFixed(2)}
                </span>
              </div>
            )}
            {modelAhLine != null && (
              <div>
                <span className="text-muted-foreground">Model AH: </span>
                <span className="font-semibold text-blue-400">
                  {modelAhLine > 0 ? "+" : ""}
                  {modelAhLine.toFixed(2)}
                </span>
              </div>
            )}
            {avgGap != null && (
              <div>
                <span className="text-muted-foreground">Chain AH: </span>
                <span className="font-semibold text-orange-400">
                  {avgGap > 0 ? "+" : ""}
                  {avgGap.toFixed(2)}
                </span>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function ordinal(n: number): string {
  const s = ["th", "st", "nd", "rd"];
  const v = n % 100;
  return n + (s[(v - 20) % 10] || s[v] || s[0]);
}

function computeStandings(
  fixtures: LeagueFixtureForChain[]
): StandingsRow[] {
  const teamMap = new Map<
    number,
    {
      team_id: number;
      team_name: string;
      played: number;
      won: number;
      drawn: number;
      lost: number;
      gf: number;
      ga: number;
    }
  >();

  for (const f of fixtures) {
    if (f.home_score == null || f.away_score == null) continue;

    // Ensure both teams exist
    for (const [tid, tname] of [
      [f.home_team_id, f.home_team],
      [f.away_team_id, f.away_team],
    ] as [number, string][]) {
      if (!teamMap.has(tid)) {
        teamMap.set(tid, {
          team_id: tid,
          team_name: tname,
          played: 0,
          won: 0,
          drawn: 0,
          lost: 0,
          gf: 0,
          ga: 0,
        });
      }
    }

    const home = teamMap.get(f.home_team_id)!;
    const away = teamMap.get(f.away_team_id)!;

    home.played++;
    away.played++;
    home.gf += f.home_score;
    home.ga += f.away_score;
    away.gf += f.away_score;
    away.ga += f.home_score;

    if (f.home_score > f.away_score) {
      home.won++;
      away.lost++;
    } else if (f.home_score < f.away_score) {
      away.won++;
      home.lost++;
    } else {
      home.drawn++;
      away.drawn++;
    }
  }

  const rows: StandingsRow[] = Array.from(teamMap.values()).map((t) => ({
    ...t,
    gd: t.gf - t.ga,
    points: t.won * 3 + t.drawn,
    position: 0,
  }));

  // Sort by points, then GD, then GF
  rows.sort((a, b) => b.points - a.points || b.gd - a.gd || b.gf - a.gf);

  // Assign positions
  rows.forEach((r, i) => {
    r.position = i + 1;
  });

  return rows;
}

// Home Field Advantage in AH terms (~0.5 goals is industry standard for top European leagues)
const HFA = 0.5;

function computeProxyMatches(
  fixtures: LeagueFixtureForChain[],
  teamAId: number, // home team in the actual match
  teamBId: number, // away team in the actual match
  standings: StandingsRow[]
): ProxyMatch[] {
  // Only consider fixtures from the last 2 months
  const twoMonthsAgo = new Date();
  twoMonthsAgo.setMonth(twoMonthsAgo.getMonth() - 2);
  const recentFixtures = fixtures.filter(
    (f) => new Date(f.kickoff_at) >= twoMonthsAgo
  );

  // Find matches involving team A
  const teamAMatches = recentFixtures.filter(
    (f) => f.home_team_id === teamAId || f.away_team_id === teamAId
  );

  // Find matches involving team B
  const teamBMatches = recentFixtures.filter(
    (f) => f.home_team_id === teamBId || f.away_team_id === teamBId
  );

  // Find common opponents (proxy teams)
  const teamAOpponents = new Set(
    teamAMatches.flatMap((f) =>
      f.home_team_id === teamAId ? [f.away_team_id] : [f.home_team_id]
    )
  );

  const teamBOpponents = new Set(
    teamBMatches.flatMap((f) =>
      f.home_team_id === teamBId ? [f.away_team_id] : [f.home_team_id]
    )
  );

  const proxyTeamIds = [...teamAOpponents].filter(
    (id) => teamBOpponents.has(id) && id !== teamAId && id !== teamBId
  );

  const standingsMap = new Map(standings.map((s) => [s.team_id, s]));

  const results: ProxyMatch[] = [];

  for (const proxyId of proxyTeamIds) {
    // Find match of proxy vs team A
    const vsA = teamAMatches.find(
      (f) => f.home_team_id === proxyId || f.away_team_id === proxyId
    );
    // Find match of proxy vs team B
    const vsB = teamBMatches.find(
      (f) => f.home_team_id === proxyId || f.away_team_id === proxyId
    );

    if (!vsA || !vsB) continue;

    const proxyStanding = standingsMap.get(proxyId);
    const proxyName =
      vsA.home_team_id === proxyId ? vsA.home_team : vsA.away_team;

    // --- HFA-adjusted implied AH line ---
    // 1. Compute each team's NEUTRAL strength advantage over the proxy
    //    AH line is from home perspective (negative = home favored)
    //    Expected home margin = -ah_line
    //    Neutral margin for home = -ah_line - HFA
    //    Team's advantage over proxy:
    //      If team was home: advantage = -ah_line - HFA
    //      If team was away: advantage = ah_line + HFA
    let impliedAh: number | null = null;

    if (vsA.ah_line != null && vsB.ah_line != null) {
      const teamA_vs_proxy =
        vsA.home_team_id === teamAId
          ? -(vsA.ah_line) - HFA  // teamA was home
          : vsA.ah_line + HFA;    // teamA was away

      const teamB_vs_proxy =
        vsB.home_team_id === teamBId
          ? -(vsB.ah_line) - HFA  // teamB was home
          : vsB.ah_line + HFA;    // teamB was away

      // 2. Neutral strength diff: positive = teamA stronger
      const neutralDiff = teamA_vs_proxy - teamB_vs_proxy;

      // 3. Implied AH for actual match (teamA is home)
      //    Expected margin = neutralDiff + HFA
      //    AH line = -expected_margin
      impliedAh = -(neutralDiff + HFA);
    }

    results.push({
      proxy_team_id: proxyId,
      proxy_team_name: proxyName,
      proxy_position: proxyStanding?.position ?? 0,
      vs_a_fixture_id: vsA.fixture_id,
      vs_a_date: vsA.kickoff_at,
      vs_a_home_team: vsA.home_team,
      vs_a_away_team: vsA.away_team,
      vs_a_score:
        vsA.home_score != null
          ? `${vsA.home_score}-${vsA.away_score}`
          : "vs",
      vs_a_ah_line: vsA.ah_line,
      vs_a_totals_line: vsA.totals_line,
      vs_b_fixture_id: vsB.fixture_id,
      vs_b_date: vsB.kickoff_at,
      vs_b_home_team: vsB.home_team,
      vs_b_away_team: vsB.away_team,
      vs_b_score:
        vsB.home_score != null
          ? `${vsB.home_score}-${vsB.away_score}`
          : "vs",
      vs_b_ah_line: vsB.ah_line,
      vs_b_totals_line: vsB.totals_line,
      implied_ah_gap: impliedAh,
    });
  }

  // Sort by proxy position
  results.sort((a, b) => a.proxy_position - b.proxy_position);

  return results;
}
