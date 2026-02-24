import { execFile } from "child_process";
import { promisify } from "util";
import path from "path";
import { createClient } from "@supabase/supabase-js";
import { settleAh, settleTotals } from "@/lib/settlement";

const execFileAsync = promisify(execFile);

export async function POST(request: Request) {
  const backendDir = path.resolve(process.cwd(), "..", "backend");
  const pythonBin = path.join(backendDir, ".venv", "bin", "python");

  const body = await request.json().catch(() => ({}));
  const date = typeof body.date === "string" && /^\d{4}-\d{2}-\d{2}$/.test(body.date)
    ? body.date
    : new Date().toISOString().slice(0, 10);

  const encoder = new TextEncoder();
  let closed = false;
  const stream = new ReadableStream({
    async start(controller) {
      function send(data: Record<string, unknown>) {
        if (closed) return;
        try {
          controller.enqueue(encoder.encode(JSON.stringify(data) + "\n"));
        } catch { closed = true; }
      }
      function close() {
        if (closed) return;
        try { controller.close(); } catch { /* already closed */ }
        closed = true;
      }

      try {
        // Step 1: Settle results using Python backend
        send({ step: "settle", status: "running", message: `Settling results for ${date}...` });
        const settleResult = await execFileAsync(
          pythonBin,
          [
            "-m", "dhx.ingestion.runner",
            "settle",
            "--source", "sofascore",
            "--date", date,
          ],
          { cwd: backendDir, timeout: 120_000 },
        );
        send({
          step: "settle",
          status: "done",
          message: "Results settled",
          stdout: settleResult.stdout.slice(-500),
        });

        // Step 2: Settle session bets (morning + closing) and daily_picks
        send({ step: "picks", status: "running", message: "Settling session bets..." });

        const supabase = createClient(
          process.env.NEXT_PUBLIC_SUPABASE_URL!,
          process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
          { db: { schema: "dhx" } },
        );

        // Helper to settle a list of bet rows and write results back to a table
        async function settleBetRows(
          table: string,
          rows: Array<{
            id: number; fixture_id: number; side: string; home_team: string;
            line: string; stake: number; ah_closing_line: number | null; totals_line: number | null;
          }>,
        ): Promise<number> {
          if (rows.length === 0) return 0;
          const fixtureIds = [...new Set(rows.map((p) => p.fixture_id))];
          const { data: results } = await supabase
            .from("results")
            .select("fixture_id, home_score, away_score")
            .in("fixture_id", fixtureIds);

          const resultMap = new Map<number, { home_score: number; away_score: number }>();
          for (const r of results || []) resultMap.set(r.fixture_id, r);

          let count = 0;
          for (const pick of rows) {
            const res = resultMap.get(pick.fixture_id);
            if (!res) continue;

            const isAh = pick.line.startsWith("AH ");
            let settlement: { pnl: number; result: "W" | "L" | "P" } | null = null;

            if (isAh && pick.ah_closing_line != null) {
              settlement = settleAh(res.home_score, res.away_score, pick.ah_closing_line, pick.side === pick.home_team, Number(pick.stake));
            } else if (!isAh && pick.totals_line != null) {
              settlement = settleTotals(res.home_score, res.away_score, pick.side.startsWith("Over"), pick.totals_line, Number(pick.stake));
            }

            if (settlement) {
              await supabase.from(table).update({
                home_score: res.home_score,
                away_score: res.away_score,
                result: settlement.result,
                pnl: settlement.pnl,
                settled_at: new Date().toISOString(),
              }).eq("id", pick.id);
              count++;
            }
          }
          return count;
        }

        // Settle daily_session_bets (both morning & closing — both get results for display)
        const { data: unsettledSessions } = await supabase
          .from("daily_session_bets")
          .select("id, fixture_id, side, home_team, line, stake, ah_closing_line, totals_line")
          .eq("bet_date", date)
          .is("result", null);

        const sessionCount = await settleBetRows("daily_session_bets", unsettledSessions || []);

        // Also settle daily_picks (for live tab display)
        const { data: unsettledPicks } = await supabase
          .from("daily_picks")
          .select("id, fixture_id, side, home_team, line, stake, ah_closing_line, totals_line")
          .eq("bet_date", date)
          .is("result", null);

        const picksCount = await settleBetRows("daily_picks", unsettledPicks || []);

        send({
          step: "picks",
          status: "done",
          message: `Settled: ${sessionCount} session bets, ${picksCount} daily picks`,
        });

        // Step 3: Refresh materialized view so the live tab sees updated scores
        send({ step: "matview", status: "running", message: "Refreshing predictions view..." });
        try {
          await execFileAsync(
            pythonBin,
            [
              "-c",
              "from dhx.db import get_client; get_client().schema('dhx').rpc('refresh_predictions_summary', {}).execute(); print('OK')",
            ],
            { cwd: backendDir, timeout: 60_000 },
          );
          send({ step: "matview", status: "done", message: "Predictions view refreshed" });
        } catch (mvErr) {
          // Non-fatal — settlement still succeeded
          const mvMsg = mvErr instanceof Error ? mvErr.message : "Unknown error";
          send({ step: "matview", status: "warning", message: `Matview refresh failed: ${mvMsg}` });
        }

        send({ step: "complete", status: "done", success: true });
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : "Unknown error";
        const stderr =
          err && typeof err === "object" && "stderr" in err
            ? String((err as { stderr: unknown }).stderr).slice(-500)
            : "";
        console.error("settle-results failed:", message, stderr);
        send({ step: "error", status: "failed", success: false, error: message, stderr });
      } finally {
        close();
      }
    },
    cancel() { closed = true; },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/plain; charset=utf-8",
      "Transfer-Encoding": "chunked",
      "Cache-Control": "no-cache",
    },
  });
}
