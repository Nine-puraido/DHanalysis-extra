import { createClient } from "@supabase/supabase-js";

export async function POST(request: Request) {
  const body = await request.json().catch(() => null);
  if (!body || !body.date || !Array.isArray(body.bets)) {
    return Response.json(
      { success: false, error: "Missing date or bets" },
      { status: 400 },
    );
  }

  const { date, bets } = body as {
    date: string;
    bets: {
      fixtureId: number;
      category: string;
      side: string;
      homeTeam: string;
      awayTeam: string;
      kickoff: string;
      leagueName: string;
      leagueKey: string;
      line: string;
      stake: number;
      ahClosingLine?: number | null;
      totalsLine?: number | null;
      signalsJson?: Record<string, unknown>;
    }[];
  };

  const supabase = createClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    { db: { schema: "dhx" } },
  );

  // Delete unsettled rows for this date that are NOT in the incoming batch
  // (picks that were dropped). Never delete already-settled rows.
  const incomingKeys = new Set(
    bets.map((b) => `${b.fixtureId}-${b.category}`),
  );

  const { data: existing } = await supabase
    .from("daily_picks")
    .select("id, fixture_id, category, result")
    .eq("bet_date", date);

  const toDelete = (existing || [])
    .filter(
      (row) =>
        row.result === null &&
        !incomingKeys.has(`${row.fixture_id}-${row.category}`),
    )
    .map((row) => row.id);

  if (toDelete.length > 0) {
    await supabase.from("daily_picks").delete().in("id", toDelete);
  }

  if (bets.length === 0) {
    return Response.json({ success: true, stored: 0, settled: 0 });
  }

  // Upsert each bet on (bet_date, fixture_id, category)
  // Only update non-settlement fields (never overwrite result/pnl if settled)
  const rows = bets.map((b) => ({
    bet_date: date,
    fixture_id: b.fixtureId,
    category: b.category,
    side: b.side,
    home_team: b.homeTeam,
    away_team: b.awayTeam,
    kickoff: b.kickoff,
    league_name: b.leagueName,
    league_key: b.leagueKey,
    line: b.line,
    stake: b.stake,
    ah_closing_line: b.ahClosingLine ?? null,
    totals_line: b.totalsLine ?? null,
    signals_json: b.signalsJson ?? {},
    computed_at: new Date().toISOString(),
  }));

  const { error } = await supabase.from("daily_picks").upsert(rows, {
    onConflict: "bet_date,fixture_id,category",
    ignoreDuplicates: false,
  });

  if (error) {
    return Response.json(
      { success: false, error: `Upsert failed: ${error.message}` },
      { status: 500 },
    );
  }

  // Count how many are already settled
  const settledCount = (existing || []).filter((r) => r.result !== null).length;

  return Response.json({
    success: true,
    stored: bets.length,
    settled: settledCount,
  });
}
