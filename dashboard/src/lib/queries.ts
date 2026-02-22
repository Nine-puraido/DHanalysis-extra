import { createClient } from "@/lib/supabase/server";
import type {
  PredictionRow,
  FixtureRow,
  FixtureDetail,
  ModelVersion,
  CalibrationBin,
  Prediction,
  OddsSnapshot,
  TodayFixture,
  LeagueFixtureForChain,
  LockedBet,
  DailyPick,
  DailyPnlSummary,
} from "./types";

// ---------------------------------------------------------------------------
// Predictions
// ---------------------------------------------------------------------------

export async function getPredictions(): Promise<PredictionRow[]> {
  const supabase = await createClient();

  // RPC results are capped at 1000 rows by PostgREST — paginate to get all.
  const PAGE = 1000;
  const rows: PredictionRow[] = [];
  let offset = 0;

  while (true) {
    const { data } = await supabase
      .rpc("get_predictions_summary")
      .range(offset, offset + PAGE - 1);

    if (!data || data.length === 0) break;
    rows.push(...(data as PredictionRow[]));
    if (data.length < PAGE) break;
    offset += PAGE;
  }

  return rows;
}

export async function getUpcomingPredictions(): Promise<PredictionRow[]> {
  const supabase = await createClient();
  const { data } = await supabase.rpc("get_predictions_summary", {
    p_exclude_status: "finished",
  });
  return (data as PredictionRow[]) || [];
}

// ---------------------------------------------------------------------------
// Model Versions / Evaluation
// ---------------------------------------------------------------------------

export async function getModelVersions(): Promise<ModelVersion[]> {
  const supabase = await createClient();
  const { data } = await supabase
    .from("model_versions")
    .select("*")
    .order("created_at", { ascending: false });
  return (data as ModelVersion[]) || [];
}

export async function getCalibrationData(
  modelVersionId: number
): Promise<CalibrationBin[]> {
  const supabase = await createClient();

  // Get predictions for this model version
  const { data: predictions } = await supabase
    .from("predictions")
    .select("fixture_id, market, selection, probability")
    .eq("model_version_id", modelVersionId)
    .eq("market", "1x2");

  if (!predictions || predictions.length === 0) return [];

  // Get results for these fixtures
  const fixtureIds = [...new Set(predictions.map((p) => p.fixture_id))];
  const { data: results } = await supabase
    .from("results")
    .select("fixture_id, home_score, away_score")
    .in("fixture_id", fixtureIds);

  const resultMap = new Map<
    number,
    { home_score: number; away_score: number }
  >();
  results?.forEach((r) => resultMap.set(r.fixture_id, r));

  // Bin predictions
  const bins: { predicted: number; actual: number }[][] = Array.from(
    { length: 10 },
    () => []
  );

  for (const p of predictions) {
    const result = resultMap.get(p.fixture_id);
    if (!result) continue;

    const prob = Number(p.probability);
    const binIdx = Math.min(Math.floor(prob * 10), 9);

    // Determine if the selection actually won
    let hit = false;
    if (p.selection === "home") hit = result.home_score > result.away_score;
    else if (p.selection === "draw")
      hit = result.home_score === result.away_score;
    else if (p.selection === "away") hit = result.away_score > result.home_score;

    bins[binIdx].push({ predicted: prob, actual: hit ? 1 : 0 });
  }

  return bins
    .map((bin, i) => {
      if (bin.length === 0) return null;
      const avgPred = bin.reduce((s, b) => s + b.predicted, 0) / bin.length;
      const actualRate = bin.reduce((s, b) => s + b.actual, 0) / bin.length;
      return {
        bin_start: i * 0.1,
        bin_end: (i + 1) * 0.1,
        bin_label: `${(i * 10).toString()}–${((i + 1) * 10).toString()}%`,
        predicted_avg: Math.round(avgPred * 1000) / 1000,
        actual_rate: Math.round(actualRate * 1000) / 1000,
        count: bin.length,
      };
    })
    .filter((b): b is CalibrationBin => b !== null);
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const FIXTURES_PER_PAGE = 50;

export async function getFixtures(
  page: number = 1,
  leagueKey?: string
): Promise<{ fixtures: FixtureRow[]; total: number }> {
  const supabase = await createClient();

  let query = supabase
    .from("fixtures")
    .select(
      `
      id, kickoff_at, status, league_id,
      leagues!inner(league_key, name),
      home_team:teams!fixtures_home_team_id_fkey(name),
      away_team:teams!fixtures_away_team_id_fkey(name),
      results(home_score, away_score),
      match_statistics(home_xg, away_xg)
    `,
      { count: "exact" }
    )
    .eq("status", "finished")
    .order("kickoff_at", { ascending: false });

  if (leagueKey) {
    query = query.eq("leagues.league_key", leagueKey);
  }

  const from = (page - 1) * FIXTURES_PER_PAGE;
  const to = from + FIXTURES_PER_PAGE - 1;
  query = query.range(from, to);

  const { data, count } = await query;

  const fixtures: FixtureRow[] = (data || []).map((f) => {
    const league = f.leagues as unknown as { league_key: string; name: string };
    const homeTeam = f.home_team as unknown as { name: string };
    const awayTeam = f.away_team as unknown as { name: string };
    // PostgREST returns object (not array) for one-to-one relations (PK = fixture_id)
    const resultRaw = f.results as unknown;
    const result = Array.isArray(resultRaw)
      ? (resultRaw as { home_score: number; away_score: number }[])[0]
      : (resultRaw as { home_score: number; away_score: number } | null);
    const statsRaw = f.match_statistics as unknown;
    const stats = Array.isArray(statsRaw)
      ? (statsRaw as { home_xg: number; away_xg: number }[])[0]
      : (statsRaw as { home_xg: number; away_xg: number } | null);

    return {
      id: f.id,
      kickoff_at: f.kickoff_at,
      status: f.status,
      league_key: league.league_key,
      league_name: league.name,
      home_team: homeTeam.name,
      away_team: awayTeam.name,
      home_score: result?.home_score ?? null,
      away_score: result?.away_score ?? null,
      home_xg: stats?.home_xg ? Number(stats.home_xg) : null,
      away_xg: stats?.away_xg ? Number(stats.away_xg) : null,
    };
  });

  return { fixtures, total: count || 0 };
}

export async function getFixtureDetail(
  id: number
): Promise<FixtureDetail | null> {
  const supabase = await createClient();

  const { data: fixture } = await supabase
    .from("fixtures")
    .select("*")
    .eq("id", id)
    .single();

  if (!fixture) return null;

  const [
    { data: league },
    { data: homeTeam },
    { data: awayTeam },
    { data: resultArr },
    { data: stats },
    { data: predictions },
    { data: odds },
  ] = await Promise.all([
    supabase.from("leagues").select("*").eq("id", fixture.league_id).single(),
    supabase.from("teams").select("*").eq("id", fixture.home_team_id).single(),
    supabase.from("teams").select("*").eq("id", fixture.away_team_id).single(),
    supabase.from("results").select("*").eq("fixture_id", id),
    supabase
      .from("match_statistics")
      .select("*")
      .eq("fixture_id", id)
      .single(),
    supabase
      .from("vw_latest_predictions")
      .select("*")
      .eq("fixture_id", id),
    supabase
      .from("vw_odds_latest_pre_kickoff")
      .select("*")
      .eq("fixture_id", id),
  ]);

  return {
    fixture,
    league: league!,
    home_team: homeTeam!,
    away_team: awayTeam!,
    result: (resultArr as unknown as FixtureDetail["result"][])?.[0] ?? null,
    stats: (stats as unknown as FixtureDetail["stats"]) ?? null,
    predictions: (predictions as Prediction[]) || [],
    odds: (odds as OddsSnapshot[]) || [],
  };
}

// ---------------------------------------------------------------------------
// Today's Matches
// ---------------------------------------------------------------------------

export async function getTodayPredictions(
  days: number = 1
): Promise<PredictionRow[]> {
  const supabase = await createClient();
  const now = new Date();
  const startOfDay = new Date(
    Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate())
  ).toISOString();
  const endOfRange = new Date(
    Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate() + days)
  ).toISOString();

  const { data } = await supabase.rpc("get_predictions_summary", {
    p_start: startOfDay,
    p_end: endOfRange,
  });

  return (data as PredictionRow[]) || [];
}

export async function getTodayFixtures(
  days: number = 1
): Promise<TodayFixture[]> {
  const supabase = await createClient();
  const now = new Date();
  const startOfDay = new Date(
    Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate())
  ).toISOString();
  const endOfRange = new Date(
    Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate() + days)
  ).toISOString();

  const { data } = await supabase
    .from("fixtures")
    .select(
      `
      id, league_id, home_team_id, away_team_id, kickoff_at, status,
      leagues!inner(league_key, name),
      home_team:teams!fixtures_home_team_id_fkey(name),
      away_team:teams!fixtures_away_team_id_fkey(name)
    `
    )
    .gte("kickoff_at", startOfDay)
    .lt("kickoff_at", endOfRange)
    .order("kickoff_at", { ascending: true });

  return (data || []).map((f) => {
    const league = f.leagues as unknown as { league_key: string; name: string };
    const homeTeam = f.home_team as unknown as { name: string };
    const awayTeam = f.away_team as unknown as { name: string };
    return {
      id: f.id as number,
      league_id: f.league_id as number,
      home_team_id: f.home_team_id as number,
      away_team_id: f.away_team_id as number,
      kickoff_at: f.kickoff_at as string,
      status: f.status as string,
      league_key: league.league_key,
      league_name: league.name,
      home_team: homeTeam.name,
      away_team: awayTeam.name,
    };
  });
}

export async function getLeagueFixturesForPriceChain(
  leagueId: number
): Promise<LeagueFixtureForChain[]> {
  const supabase = await createClient();

  // Fetch finished fixtures for the current season (since 2025-08-01)
  const { data } = await supabase
    .from("fixtures")
    .select(
      `
      id, league_id, home_team_id, away_team_id, kickoff_at, status,
      home_team:teams!fixtures_home_team_id_fkey(name),
      away_team:teams!fixtures_away_team_id_fkey(name),
      results(home_score, away_score)
    `
    )
    .eq("league_id", leagueId)
    .eq("status", "finished")
    .gte("kickoff_at", "2025-08-01T00:00:00Z")
    .order("kickoff_at", { ascending: true });

  if (!data || data.length === 0) return [];

  // Fetch AH home lines and totals over lines from odds view (bookmaker_id=6 = sofascore_avg)
  const fixtureIds = data.map((f) => f.id as number);

  // Batch fetch odds in chunks of 200
  const oddsMap = new Map<number, { ah_line: number | null; totals_line: number | null }>();
  for (let i = 0; i < fixtureIds.length; i += 200) {
    const chunk = fixtureIds.slice(i, i + 200);

    const [ahResult, totalsResult] = await Promise.all([
      supabase
        .from("vw_odds_latest_pre_kickoff")
        .select("fixture_id, line")
        .in("fixture_id", chunk)
        .eq("bookmaker_id", 6)
        .eq("market", "ah")
        .eq("selection", "home"),
      supabase
        .from("vw_odds_latest_pre_kickoff")
        .select("fixture_id, line")
        .in("fixture_id", chunk)
        .eq("bookmaker_id", 6)
        .eq("market", "totals")
        .eq("selection", "over")
        .eq("is_main", true),
    ]);

    for (const row of ahResult.data || []) {
      const existing = oddsMap.get(row.fixture_id) || { ah_line: null, totals_line: null };
      existing.ah_line = row.line != null ? Number(row.line) : null;
      oddsMap.set(row.fixture_id, existing);
    }
    for (const row of totalsResult.data || []) {
      const existing = oddsMap.get(row.fixture_id) || { ah_line: null, totals_line: null };
      existing.totals_line = row.line != null ? Number(row.line) : null;
      oddsMap.set(row.fixture_id, existing);
    }
  }

  return data.map((f) => {
    const homeTeam = f.home_team as unknown as { name: string };
    const awayTeam = f.away_team as unknown as { name: string };
    // PostgREST returns object (not array) for one-to-one relations (results PK = fixture_id)
    const resultRaw = f.results as unknown;
    const result = Array.isArray(resultRaw)
      ? (resultRaw as { home_score: number; away_score: number }[])[0]
      : (resultRaw as { home_score: number; away_score: number } | null);
    const odds = oddsMap.get(f.id as number);

    return {
      fixture_id: f.id as number,
      league_id: f.league_id as number,
      home_team_id: f.home_team_id as number,
      away_team_id: f.away_team_id as number,
      home_team: homeTeam.name,
      away_team: awayTeam.name,
      kickoff_at: f.kickoff_at as string,
      status: f.status as string,
      home_score: result?.home_score ?? null,
      away_score: result?.away_score ?? null,
      ah_line: odds?.ah_line ?? null,
      totals_line: odds?.totals_line ?? null,
    };
  });
}

// ---------------------------------------------------------------------------
// Date-specific queries (for /daily page, Bangkok timezone)
// ---------------------------------------------------------------------------

export async function getDatePredictions(
  dateStr: string,
): Promise<PredictionRow[]> {
  const supabase = await createClient();
  const startUTC = dateStr + "T00:00:00Z";
  const endUTC = new Date(
    new Date(startUTC).getTime() + 24 * 60 * 60 * 1000,
  ).toISOString();

  const { data } = await supabase.rpc("get_predictions_summary", {
    p_start: startUTC,
    p_end: endUTC,
  });

  return (data as PredictionRow[]) || [];
}

export async function getDateFixtures(
  dateStr: string,
): Promise<TodayFixture[]> {
  const supabase = await createClient();
  const startUTC = dateStr + "T00:00:00Z";
  const endUTC = new Date(
    new Date(startUTC).getTime() + 24 * 60 * 60 * 1000,
  ).toISOString();

  const { data } = await supabase
    .from("fixtures")
    .select(
      `
      id, league_id, home_team_id, away_team_id, kickoff_at, status,
      leagues!inner(league_key, name),
      home_team:teams!fixtures_home_team_id_fkey(name),
      away_team:teams!fixtures_away_team_id_fkey(name)
    `,
    )
    .gte("kickoff_at", startUTC)
    .lt("kickoff_at", endUTC)
    .order("kickoff_at", { ascending: true });

  return (data || []).map((f) => {
    const league = f.leagues as unknown as { league_key: string; name: string };
    const homeTeam = f.home_team as unknown as { name: string };
    const awayTeam = f.away_team as unknown as { name: string };
    return {
      id: f.id as number,
      league_id: f.league_id as number,
      home_team_id: f.home_team_id as number,
      away_team_id: f.away_team_id as number,
      kickoff_at: f.kickoff_at as string,
      status: f.status as string,
      league_key: league.league_key,
      league_name: league.name,
      home_team: homeTeam.name,
      away_team: awayTeam.name,
    };
  });
}

export async function getMainTotalsLines(
  fixtureIds: number[],
): Promise<Record<number, number>> {
  if (fixtureIds.length === 0) return {};
  const supabase = await createClient();

  // Fetch ALL totals lines (over + under) to infer the balanced market line.
  // SofaScore only marks 2.5 as is_main, but the real market line varies.
  type OddsRow = { fixture_id: number; selection: string; line: string; price_decimal: string };
  const allRows: OddsRow[] = [];

  for (let i = 0; i < fixtureIds.length; i += 200) {
    const chunk = fixtureIds.slice(i, i + 200);
    const { data } = await supabase
      .from("vw_odds_latest_pre_kickoff")
      .select("fixture_id, selection, line, price_decimal")
      .in("fixture_id", chunk)
      .eq("bookmaker_id", 6)
      .eq("market", "totals")
      .in("selection", ["over", "under"]);

    if (data) allRows.push(...(data as OddsRow[]));
  }

  // Group by fixture_id → line → { over_price, under_price }
  const byFixture = new Map<number, Map<number, { over: number; under: number }>>();
  for (const row of allRows) {
    if (row.line == null || row.price_decimal == null) continue;
    const fid = row.fixture_id;
    const line = Number(row.line);
    const price = Number(row.price_decimal);
    if (!byFixture.has(fid)) byFixture.set(fid, new Map());
    const lineMap = byFixture.get(fid)!;
    if (!lineMap.has(line)) lineMap.set(line, { over: 0, under: 0 });
    const entry = lineMap.get(line)!;
    if (row.selection === "over") entry.over = price;
    else entry.under = price;
  }

  const result: Record<number, number> = {};

  for (const [fid, lineMap] of byFixture) {
    // Build sorted list of (line, fair P(Over)) where both prices exist
    const points: { line: number; fairOver: number }[] = [];
    for (const [line, prices] of lineMap) {
      if (prices.over > 0 && prices.under > 0) {
        const implOver = 1 / prices.over;
        const implUnder = 1 / prices.under;
        const fairOver = implOver / (implOver + implUnder);
        points.push({ line, fairOver });
      }
    }

    if (points.length === 0) continue;
    points.sort((a, b) => a.line - b.line);

    // Find where fairOver crosses 0.5 (decreasing as line increases)
    let inferred: number | null = null;

    // If fairOver > 0.5 at the highest line, use that line
    if (points[points.length - 1].fairOver >= 0.5) {
      inferred = points[points.length - 1].line;
    }
    // If fairOver < 0.5 at the lowest line, use that line
    else if (points[0].fairOver <= 0.5) {
      inferred = points[0].line;
    }
    // Otherwise interpolate between adjacent lines
    else {
      for (let j = 0; j < points.length - 1; j++) {
        const lo = points[j]; // lower line, higher P(Over)
        const hi = points[j + 1]; // higher line, lower P(Over)
        if (lo.fairOver >= 0.5 && hi.fairOver < 0.5) {
          // Linear interpolation: find line where fairOver = 0.5
          const t = (0.5 - lo.fairOver) / (hi.fairOver - lo.fairOver);
          inferred = lo.line + t * (hi.line - lo.line);
          break;
        }
      }
    }

    if (inferred != null) {
      // Round to nearest 0.25
      result[fid] = Math.round(inferred * 4) / 4;
    }
  }

  return result;
}

// ---------------------------------------------------------------------------
// Locked session bets (for /daily two-session tracking)
// ---------------------------------------------------------------------------

export async function getLockedSessionBets(
  dateStr: string,
): Promise<{ morning: LockedBet[]; closing: LockedBet[] }> {
  const supabase = await createClient();
  const { data } = await supabase
    .from("daily_session_bets")
    .select("*")
    .eq("bet_date", dateStr)
    .order("kickoff", { ascending: true });

  const rows = (data as LockedBet[]) || [];
  return {
    morning: rows.filter((r) => r.session === "morning"),
    closing: rows.filter((r) => r.session === "closing"),
  };
}

// ---------------------------------------------------------------------------
// Daily Picks (persisted computed bets)
// ---------------------------------------------------------------------------

export async function getDailyPicks(dateStr: string): Promise<DailyPick[]> {
  const supabase = await createClient();
  const { data } = await supabase
    .from("daily_picks")
    .select("*")
    .eq("bet_date", dateStr)
    .order("kickoff", { ascending: true });

  return (data as DailyPick[]) || [];
}

export async function getDailyPnlSummary(
  dateStr: string,
): Promise<DailyPnlSummary[]> {
  const supabase = await createClient();
  const { data } = await supabase
    .from("vw_daily_pnl_summary")
    .select("*")
    .eq("bet_date", dateStr);

  return (data as DailyPnlSummary[]) || [];
}

// ---------------------------------------------------------------------------
// Leagues (for filter dropdowns)
// ---------------------------------------------------------------------------

export async function getLeagues(): Promise<
  { league_key: string; name: string }[]
> {
  const supabase = await createClient();
  const { data } = await supabase
    .from("leagues")
    .select("league_key, name")
    .order("name");
  return data || [];
}
