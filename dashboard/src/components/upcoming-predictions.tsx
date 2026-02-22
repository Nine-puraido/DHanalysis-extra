"use client";

import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { PredictionRow } from "@/lib/types";
import { cn } from "@/lib/utils";

interface Props {
  data: PredictionRow[];
}

function formatProb(p: number | null): string {
  if (p === null) return "-";
  return (p * 100).toFixed(1) + "%";
}

function formatOdds(o: number | null): string {
  if (o === null) return "-";
  return o.toFixed(2);
}

function formatLine(l: number | null): string {
  if (l === null) return "-";
  return l >= 0 ? `+${l}` : `${l}`;
}

function isValue(fair: number | null, closing: number | null): boolean {
  if (fair === null || closing === null) return false;
  return fair < closing;
}

function formatKickoff(iso: string): { date: string; time: string; relative: string } {
  const d = new Date(iso);
  const date = d.toLocaleDateString("en-GB", {
    weekday: "short",
    day: "2-digit",
    month: "short",
    timeZone: "Asia/Bangkok",
  });
  const time = d.toLocaleTimeString("en-GB", {
    hour: "2-digit",
    minute: "2-digit",
    timeZone: "Asia/Bangkok",
  });

  const now = Date.now();
  const diff = d.getTime() - now;
  let relative: string;
  if (diff < 0) {
    relative = "started";
  } else if (diff < 3600_000) {
    const mins = Math.floor(diff / 60_000);
    relative = `in ${mins}m`;
  } else if (diff < 86400_000) {
    const hrs = Math.floor(diff / 3600_000);
    const mins = Math.floor((diff % 3600_000) / 60_000);
    relative = `in ${hrs}h ${mins}m`;
  } else {
    const days = Math.floor(diff / 86400_000);
    relative = `in ${days}d`;
  }

  return { date, time, relative };
}

function ProbabilityBar({
  home,
  draw,
  away,
}: {
  home: number | null;
  draw: number | null;
  away: number | null;
}) {
  if (home === null || draw === null || away === null) return null;
  const h = home * 100;
  const d = draw * 100;
  const a = away * 100;

  return (
    <div className="flex h-3 w-full overflow-hidden rounded-full text-[10px] font-medium">
      <div
        className="flex items-center justify-center bg-blue-500 text-white"
        style={{ width: `${h}%` }}
      >
        {h >= 15 ? `${h.toFixed(0)}` : ""}
      </div>
      <div
        className="flex items-center justify-center bg-zinc-400 text-white dark:bg-zinc-500"
        style={{ width: `${d}%` }}
      >
        {d >= 15 ? `${d.toFixed(0)}` : ""}
      </div>
      <div
        className="flex items-center justify-center bg-red-500 text-white"
        style={{ width: `${a}%` }}
      >
        {a >= 15 ? `${a.toFixed(0)}` : ""}
      </div>
    </div>
  );
}

function ValueHighlight({
  label,
  prob,
  fairOdds,
  closingOdds,
}: {
  label: string;
  prob: number | null;
  fairOdds: number | null;
  closingOdds: number | null;
}) {
  const value = isValue(fairOdds, closingOdds);
  return (
    <div
      className={cn(
        "flex items-center justify-between text-sm",
        value && "text-green-400 font-semibold"
      )}
    >
      <span className="text-muted-foreground">{label}</span>
      <span className="tabular-nums">
        {formatProb(prob)}
        {closingOdds !== null && (
          <span className="text-muted-foreground ml-2 text-xs">
            @ {formatOdds(closingOdds)}
          </span>
        )}
        {value && <span className="ml-1 text-xs">VALUE</span>}
      </span>
    </div>
  );
}

function MatchCard({ row }: { row: PredictionRow }) {
  const kickoff = formatKickoff(row.kickoff_at);

  return (
    <Card>
      <CardHeader className="pb-0">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Badge variant="outline">{row.league_key}</Badge>
            <span className="text-xs text-muted-foreground">
              {kickoff.date} {kickoff.time}
            </span>
          </div>
          <Badge variant="secondary" className="text-xs">
            {kickoff.relative}
          </Badge>
        </div>
        <CardTitle className="text-lg mt-2">
          {row.home_team} vs {row.away_team}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* 1x2 Probability Bar */}
        <div className="space-y-1">
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>Home {formatProb(row.prob_home)}</span>
            <span>Draw {formatProb(row.prob_draw)}</span>
            <span>Away {formatProb(row.prob_away)}</span>
          </div>
          <ProbabilityBar
            home={row.prob_home}
            draw={row.prob_draw}
            away={row.prob_away}
          />
        </div>

        {/* Key Predictions */}
        <div className="grid grid-cols-2 gap-x-6 gap-y-1">
          <ValueHighlight
            label="P(O2.5)"
            prob={row.prob_over25}
            fairOdds={row.prob_over25 ? 1 / row.prob_over25 : null}
            closingOdds={row.closing_over25}
          />
          <ValueHighlight
            label="P(BTTS)"
            prob={row.prob_btts_yes}
            fairOdds={row.prob_btts_yes ? 1 / row.prob_btts_yes : null}
            closingOdds={null}
          />
          <ValueHighlight
            label="Home"
            prob={row.prob_home}
            fairOdds={row.fair_home}
            closingOdds={row.closing_home}
          />
          <ValueHighlight
            label="Away"
            prob={row.prob_away}
            fairOdds={row.fair_away}
            closingOdds={row.closing_away}
          />
        </div>

        {/* Lambda and AH */}
        <div className="flex items-center justify-between border-t border-border pt-3 text-xs text-muted-foreground">
          <span className="tabular-nums">
            {"\u03BB"} Home: {row.lambda_home?.toFixed(2) ?? "-"} | Away:{" "}
            {row.lambda_away?.toFixed(2) ?? "-"}
          </span>
          <span className="tabular-nums">
            AH Fair: {formatLine(row.ah_fair_line)}
            {row.ah_closing_line !== null && (
              <span className="ml-2">Mkt: {formatLine(row.ah_closing_line)}</span>
            )}
          </span>
        </div>
      </CardContent>
    </Card>
  );
}

export function UpcomingPredictions({ data }: Props) {
  if (data.length === 0) {
    return (
      <div className="rounded-lg border border-dashed border-border p-6 text-center text-muted-foreground">
        <p className="text-sm">No upcoming predictions.</p>
        <p className="text-xs mt-1">
          Run{" "}
          <code className="bg-muted px-1 py-0.5 rounded text-xs">
            python -m dh.modeling predict-upcoming
          </code>{" "}
          to generate.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <h2 className="text-lg font-semibold">Upcoming</h2>
        <Badge variant="secondary">{data.length}</Badge>
      </div>
      <div className="grid gap-4 sm:grid-cols-1 md:grid-cols-2 lg:grid-cols-3">
        {data.map((row) => (
          <MatchCard key={row.fixture_id} row={row} />
        ))}
      </div>
    </div>
  );
}
