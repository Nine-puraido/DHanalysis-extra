/**
 * Shared pick classification logic used by both /today and /daily pages.
 * Extracted from today-match-list.tsx to avoid duplication.
 */
import type { PredictionRow, TodayFixture } from "@/lib/types";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

export const MIN_DECIDED = 3;

/** All 22 extra league keys */
const ALL_LEAGUES = [
  "ALM","ARG","CFL","DSL","EL1","GL3","GSL","HNL","IPL","J1",
  "L2","LL2","MLS","POL","QSL","RSL","SB","SPL_SA","SRBL","TL1","UAE","UPL",
];

/** Leagues where model filter is not applied (weak signal) */
export const WEAK_LEAGUES = new Set<string>([]);

/** Top-tier leagues for filter logic */
export const TOP6_LEAGUES = new Set<string>(ALL_LEAGUES);

/** Leagues excluded from totals fade strategy (bad O/U accuracy) */
export const TOTALS_EXCLUDE = new Set<string>(["EL1", "UPL"]);

/** S1 league-conditional zones — default all to model zone */
export const S1_MODEL_LEAGUES = new Set<string>(ALL_LEAGUES);
/** Market zone — none yet (will tune after accumulating accuracy data) */
export const S1_MARKET_LEAGUES = new Set<string>([]);

/** Leagues excluded from T2 bets (<45% hit rate with 15+ samples) */
export const T2_EXCLUDE = new Set<string>(["LL2", "EL1", "UPL", "POL"]);

/** Leagues excluded from Agree Over picks (<53% hit rate) */
export const AGREE_O_EXCLUDE = new Set<string>(["GSL", "POL"]);

/** Minimum edge for totals fade pick */
export const TOTALS_MIN_EDGE = 0.15;

/** Minimum edge for agree-over pick (EV-based: model_prob × odds - 1) */
/** Minimum probability gap for agree-over pick (market implied - model) */
export const AGREE_OVER_MIN_EDGE = 0.10;

/** Minimum EV edge for AH pick (edge = model_prob × decimal_odds - 1) */
export const MIN_AH_EDGE = 0.01;

// ---------------------------------------------------------------------------
// Interfaces
// ---------------------------------------------------------------------------

export interface ModelFilterPick {
  side: "home" | "away";
  filters: string[];
  count: number;
}

export interface TotalsPick {
  side: "over" | "under";
  edge: number;
}

export interface AgreeOverPick {
  edge: number;
}

export interface MatchCard {
  fixtureId: number;
  kickoff: string;
  homeTeam: string;
  awayTeam: string;
  leagueName: string;
  leagueId: number;
  status: string;
  prediction: PredictionRow | null;
  pick: { side: "home" | "away"; signals: number; total: number; edge: number } | null;
  modelFilter: ModelFilterPick | null;
  totalsPick: TotalsPick | null;
  agreeOverPick: AgreeOverPick | null;
  leagueKey: string;
}

// ---------------------------------------------------------------------------
// Price chain (Signal 2)
// ---------------------------------------------------------------------------

/** Home-field advantage in AH terms (~0.5 goals) */
const HFA = 0.5;

export function computePriceChainSignal(
  p: PredictionRow,
  leaguePredictions: PredictionRow[],
): "model" | "market" | null {
  if (p.ah_fair_line == null || p.ah_closing_line == null) return null;

  const modelLine = p.ah_fair_line;
  const marketLine = p.ah_closing_line;

  // Recent finished fixtures in same league (last 2 months before this match)
  const kickoff = new Date(p.kickoff_at).getTime();
  const cutoff = kickoff - 60 * 24 * 60 * 60 * 1000; // 60 days

  const recent = leaguePredictions.filter(
    (r) =>
      r.fixture_id !== p.fixture_id &&
      r.status === "finished" &&
      r.ah_closing_line != null &&
      r.home_score != null &&
      new Date(r.kickoff_at).getTime() >= cutoff &&
      new Date(r.kickoff_at).getTime() < kickoff,
  );

  // Find matches involving home team and away team
  const homeMatches = recent.filter(
    (r) => r.home_team === p.home_team || r.away_team === p.home_team,
  );
  const awayMatches = recent.filter(
    (r) => r.home_team === p.away_team || r.away_team === p.away_team,
  );

  // Common opponents
  const homeOpponents = new Set(
    homeMatches.flatMap((r) =>
      r.home_team === p.home_team ? [r.away_team] : [r.home_team],
    ),
  );
  const awayOpponents = new Set(
    awayMatches.flatMap((r) =>
      r.home_team === p.away_team ? [r.away_team] : [r.home_team],
    ),
  );

  const proxyTeams = [...homeOpponents].filter(
    (t) => awayOpponents.has(t) && t !== p.home_team && t !== p.away_team,
  );

  if (proxyTeams.length < 3) return null;

  const impliedGaps: number[] = [];

  for (const proxy of proxyTeams) {
    const vsHome = homeMatches.find(
      (r) => r.home_team === proxy || r.away_team === proxy,
    );
    const vsAway = awayMatches.find(
      (r) => r.home_team === proxy || r.away_team === proxy,
    );
    if (!vsHome || !vsAway) continue;
    if (vsHome.ah_closing_line == null || vsAway.ah_closing_line == null)
      continue;

    // HFA-adjusted neutral strength
    const homeTeamVsProxy =
      vsHome.home_team === p.home_team
        ? -(vsHome.ah_closing_line) - HFA
        : vsHome.ah_closing_line + HFA;

    const awayTeamVsProxy =
      vsAway.home_team === p.away_team
        ? -(vsAway.ah_closing_line) - HFA
        : vsAway.ah_closing_line + HFA;

    const neutralDiff = homeTeamVsProxy - awayTeamVsProxy;
    const impliedAh = -(neutralDiff + HFA);
    impliedGaps.push(impliedAh);
  }

  if (impliedGaps.length === 0) return null;

  const avgImplied =
    impliedGaps.reduce((s, v) => s + v, 0) / impliedGaps.length;
  const distToModel = Math.abs(avgImplied - modelLine);
  const distToMarket = Math.abs(avgImplied - marketLine);

  return distToModel < distToMarket ? "model" : "market";
}

// ---------------------------------------------------------------------------
// Totals agreement (Signal 4)
// ---------------------------------------------------------------------------

/**
 * S4: Model and market agree on Over/Under 2.5 → vote "model" on AH direction.
 * Disagree → "market". Missing data → null.
 */
export function computeTotalsAgreementSignal(
  p: PredictionRow,
): "model" | "market" | null {
  if (p.prob_over25 == null || p.closing_over25 == null) return null;

  const modelOver = p.prob_over25 > 0.5;
  const marketOver = 1 / p.closing_over25 > 0.5;

  return modelOver === marketOver ? "model" : "market";
}

// ---------------------------------------------------------------------------
// Classification functions
// ---------------------------------------------------------------------------

export function computeTeamAccuracy(
  predictions: PredictionRow[],
  teamName: string,
  beforeKickoff?: string,
) {
  // Filter to decided matches for this team, optionally before a kickoff time
  let eligible = predictions.filter((r) => {
    if (r.ah_fair_line == null || r.ah_closing_line == null) return false;
    if (r.home_score == null || r.away_score == null) return false;
    if (r.ah_fair_line === r.ah_closing_line) return false;
    if (r.home_team !== teamName && r.away_team !== teamName) return false;
    if (beforeKickoff && new Date(r.kickoff_at).getTime() >= new Date(beforeKickoff).getTime()) return false;

    const goalDiff = r.home_score - r.away_score;
    const adjusted = goalDiff + r.ah_closing_line;
    return adjusted !== 0; // skip pushes
  });

  // Sort descending by kickoff and take last 15
  eligible.sort((a, b) => new Date(b.kickoff_at).getTime() - new Date(a.kickoff_at).getTime());
  eligible = eligible.slice(0, 15);

  let modelWins = 0;
  let marketWins = 0;

  for (const r of eligible) {
    const goalDiff = r.home_score! - r.away_score!;
    const modelBetsHome = r.ah_fair_line! < r.ah_closing_line!;
    const adjusted = goalDiff + r.ah_closing_line!;

    if (modelBetsHome) {
      if (adjusted > 0) modelWins++;
      else marketWins++;
    } else {
      if (adjusted < 0) modelWins++;
      else marketWins++;
    }
  }

  const decided = modelWins + marketWins;
  return { modelWins, decided, modelPct: decided > 0 ? modelWins / decided : 0 };
}

export function computeCardPick(
  p: PredictionRow,
  allPredictions: PredictionRow[],
  leaguePredictions?: PredictionRow[],
  leagueKey?: string,
): { side: "home" | "away"; signals: number; total: number; edge: number } | null {
  if (p.ah_fair_line == null || p.ah_closing_line == null) return null;
  if (p.ah_fair_line === p.ah_closing_line) return null;

  const modelBetsHome = p.ah_fair_line < p.ah_closing_line;

  // Signal 1: League-conditional model direction
  // Model zone (>53% accuracy): vote model
  // Market zone (<49% accuracy): vote market
  // Dead zone (49-53%): abstain (null)
  const s1: "model" | "market" | null = leagueKey
    ? S1_MODEL_LEAGUES.has(leagueKey) ? "model"
      : S1_MARKET_LEAGUES.has(leagueKey) ? "market"
      : null
    : "model"; // fallback if no leagueKey provided

  // Signal 2: Price chain (proxy matches)
  const s2: "model" | "market" | null = leaguePredictions
    ? computePriceChainSignal(p, leaguePredictions)
    : null;

  // Signal 3: Team trends (weighted accuracy, last 15 matches)
  const homeAcc = computeTeamAccuracy(allPredictions, p.home_team, p.kickoff_at);
  const awayAcc = computeTeamAccuracy(allPredictions, p.away_team, p.kickoff_at);
  let s3: "model" | "market" | null = null;
  if (homeAcc.decided >= MIN_DECIDED && awayAcc.decided >= MIN_DECIDED) {
    const totalDecided = homeAcc.decided + awayAcc.decided;
    const weightedPct =
      (homeAcc.modelPct * homeAcc.decided + awayAcc.modelPct * awayAcc.decided) / totalDecided;
    s3 = weightedPct > 0.5 ? "model" : "market";
  }

  // Signal 4: Totals agreement (model+market agree on O/U → model; disagree → market)
  const s4 = computeTotalsAgreementSignal(p);

  const signals = [s1, s2, s3, s4].filter((s) => s !== null) as ("model" | "market")[];
  const modelCount = signals.filter((s) => s === "model").length;
  const marketCount = signals.filter((s) => s === "market").length;

  if (modelCount === marketCount) return null;

  const followModel = modelCount > marketCount;
  const side = followModel
    ? (modelBetsHome ? "home" : "away")
    : (modelBetsHome ? "away" : "home");

  // Compute EV-based edge: model_prob × decimal_odds - 1
  let edge: number | null = null;
  if (side === "home" && p.ah_home_prob != null && p.ah_closing_home != null) {
    edge = p.ah_home_prob * p.ah_closing_home - 1;
  } else if (side === "away" && p.ah_away_prob != null && p.ah_closing_away != null) {
    edge = p.ah_away_prob * p.ah_closing_away - 1;
  }

  // Skip picks without sufficient EV edge
  if (edge == null || edge < MIN_AH_EDGE) return null;

  return { side, signals: Math.max(modelCount, marketCount), total: signals.length, edge };
}

export function computeModelFilterPick(
  p: PredictionRow,
  leagueKey: string,
): ModelFilterPick | null {
  if (p.ah_fair_line == null || p.ah_closing_line == null) return null;
  if (p.ah_fair_line === p.ah_closing_line) return null;

  const modelBetsHome = p.ah_fair_line < p.ah_closing_line;
  const gap = Math.abs(p.ah_fair_line - p.ah_closing_line);
  const marketLine = p.ah_closing_line;

  const backingDog = modelBetsHome ? marketLine >= 0 : marketLine < 0;

  const isWeak = WEAK_LEAGUES.has(leagueKey);
  const isTop6 = TOP6_LEAGUES.has(leagueKey);

  const filters: string[] = [];
  if (!isWeak) filters.push("Excl Weak");
  if (isTop6) filters.push("Top 6");
  if (isTop6 && gap >= 0.5) filters.push("Gap+Top6");
  if (!isWeak && backingDog) filters.push("Dog");

  if (filters.length === 0) return null;

  return {
    side: modelBetsHome ? "home" : "away",
    filters,
    count: filters.length,
  };
}

export function computeTotalsPick(
  p: PredictionRow,
  leagueKey: string,
): TotalsPick | null {
  if (TOTALS_EXCLUDE.has(leagueKey)) return null;
  if (p.prob_over25 == null || p.closing_over25 == null) return null;

  const modelProb = p.prob_over25;
  const marketProb = 1 / p.closing_over25;

  const modelFavorsOver = modelProb > 0.5;
  const marketFavorsOver = marketProb > 0.5;

  if (modelFavorsOver === marketFavorsOver) return null;

  const edge = Math.abs(modelProb - marketProb);
  if (edge < TOTALS_MIN_EDGE) return null;

  const side: "over" | "under" = marketFavorsOver ? "over" : "under";
  return { side, edge };
}

export function computeAgreeOverPick(
  p: PredictionRow,
  leagueKey?: string,
): AgreeOverPick | null {
  if (leagueKey && AGREE_O_EXCLUDE.has(leagueKey)) return null;
  if (p.prob_over25 == null || p.closing_over25 == null) return null;

  const modelProb = p.prob_over25;
  const marketProb = 1 / p.closing_over25;

  if (modelProb <= 0.5 || marketProb <= 0.5) return null;

  // Market must be more confident about over by >= 10%
  const edge = marketProb - modelProb;
  if (edge < AGREE_OVER_MIN_EDGE) return null;

  return { edge };
}

export function getVerdict(
  card: MatchCard,
): { label: string; cls: string; tier: number } | null {
  const hasPick = card.pick !== null;
  const filtCount = card.modelFilter?.count ?? 0;
  const hasFilt = filtCount >= 2;
  const agree = hasPick && card.modelFilter != null && card.pick!.side === card.modelFilter.side;
  const all4 = hasPick && card.pick!.total === 4;
  const unanimous = hasPick && card.pick!.signals === card.pick!.total;

  // Dog trap: backing underdog on a thin line gap is negative EV.
  // Skip filter-based tiers (T0/T1/T2) when Dog=ON but GapTop6=OFF.
  // For T3: allow dog-trap bets only if line gap >= 0.5.
  const filters = card.modelFilter?.filters ?? [];
  const isDog = filters.includes("Dog");
  const hasGap = filters.includes("Gap+Top6");
  const dogTrap = isDog && !hasGap;
  const lineGap = card.prediction
    ? Math.abs((card.prediction.ah_fair_line ?? 0) - (card.prediction.ah_closing_line ?? 0))
    : 0;

  // T0: 4 signals + filter ≥2 + unanimous + agree (gold)
  if (all4 && unanimous && hasFilt && agree && !dogTrap) {
    const team = card.pick!.side === "home" ? card.homeTeam : card.awayTeam;
    return {
      label: `BET ${team}`,
      cls: "bg-orange-500/20 text-orange-400 border border-orange-500/30",
      tier: 0,
    };
  }

  // T1: 4 signals + filter ≥2 + agree (green)
  if (all4 && hasFilt && agree && !dogTrap) {
    const team = card.pick!.side === "home" ? card.homeTeam : card.awayTeam;
    return {
      label: `BET ${team}`,
      cls: "bg-green-500/20 text-green-400 border border-green-500/30",
      tier: 1,
    };
  }

  // T2: Filter ≥2 + signal exists + agree (blue) — skip bleeding leagues
  // Dog trap waived when all signals unanimous
  if (hasFilt && hasPick && agree && !T2_EXCLUDE.has(card.leagueKey) && (!dogTrap || unanimous)) {
    const team = card.pick!.side === "home" ? card.homeTeam : card.awayTeam;
    return {
      label: `BET ${team}`,
      cls: "bg-blue-500/20 text-blue-400 border border-blue-500/30",
      tier: 2,
    };
  }

  // T3: All active signals unanimous (3/3 or 4/4), no filter requirement (yellow)
  // Dog-trap bets only allowed if line gap >= 0.5
  if (unanimous && (!dogTrap || lineGap >= 0.5)) {
    const team = card.pick!.side === "home" ? card.homeTeam : card.awayTeam;
    return {
      label: `BET ${team}`,
      cls: "bg-yellow-500/20 text-yellow-400 border border-yellow-500/30",
      tier: 3,
    };
  }

  return null;
}

// ---------------------------------------------------------------------------
// Card builder — builds MatchCards from fixtures + predictions
// ---------------------------------------------------------------------------

/** Group predictions by league_key for price chain lookups */
export function groupByLeague(
  predictions: PredictionRow[],
): Map<string, PredictionRow[]> {
  const map = new Map<string, PredictionRow[]>();
  for (const p of predictions) {
    const arr = map.get(p.league_key) ?? [];
    arr.push(p);
    map.set(p.league_key, arr);
  }
  return map;
}

export function buildMatchCards(
  fixtures: TodayFixture[],
  predictions: PredictionRow[],
  allPredictions: PredictionRow[],
): MatchCard[] {
  const predictionMap = new Map<number, PredictionRow>();
  for (const p of predictions) {
    predictionMap.set(p.fixture_id, p);
  }

  // Pre-group by league for price chain signal
  const byLeague = groupByLeague(allPredictions);

  const cards: MatchCard[] = [];

  for (const f of fixtures) {
    const pred = predictionMap.get(f.id) || null;
    const leaguePreds = byLeague.get(f.league_key);
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

  // Add predictions without a matching fixture
  for (const p of predictions) {
    if (!fixtures.some((f) => f.id === p.fixture_id)) {
      const leaguePreds = byLeague.get(p.league_key);
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

  cards.sort(
    (a, b) => new Date(a.kickoff).getTime() - new Date(b.kickoff).getTime(),
  );

  return cards;
}
