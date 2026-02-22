import { createClient } from "@supabase/supabase-js";

export async function POST(request: Request) {
  const body = await request.json().catch(() => null);
  if (!body || !body.date || !body.session || !Array.isArray(body.bets)) {
    return Response.json(
      { success: false, error: "Missing date, session, or bets" },
      { status: 400 },
    );
  }

  const { date, session, bets } = body as {
    date: string;
    session: "morning" | "closing";
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
    }[];
  };

  if (!["morning", "closing"].includes(session)) {
    return Response.json(
      { success: false, error: "session must be 'morning' or 'closing'" },
      { status: 400 },
    );
  }

  // Use service-level client (anon key with RLS policy allows all)
  const supabase = createClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    { db: { schema: "dhx" } },
  );

  // Delete existing rows for this (date, session), then insert new ones
  const { error: delError } = await supabase
    .from("daily_session_bets")
    .delete()
    .eq("bet_date", date)
    .eq("session", session);

  if (delError) {
    return Response.json(
      { success: false, error: `Delete failed: ${delError.message}` },
      { status: 500 },
    );
  }

  if (bets.length === 0) {
    return Response.json({ success: true, count: 0 });
  }

  const rows = bets.map((b) => ({
    bet_date: date,
    session,
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
  }));

  const { error: insError } = await supabase
    .from("daily_session_bets")
    .insert(rows);

  if (insError) {
    return Response.json(
      { success: false, error: `Insert failed: ${insError.message}` },
      { status: 500 },
    );
  }

  return Response.json({ success: true, count: bets.length });
}
