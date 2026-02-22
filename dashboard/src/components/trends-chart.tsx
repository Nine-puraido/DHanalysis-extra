"use client";

import { useState, useMemo } from "react";
import * as echarts from "echarts/core";
import ReactEChartsCore from "echarts-for-react/lib/core";
import { LineChart as ELineChart } from "echarts/charts";
import {
  TooltipComponent,
  GridComponent,
  LegendComponent,
  DataZoomComponent,
  DataZoomInsideComponent,
  DataZoomSliderComponent,
  MarkLineComponent,
  MarkAreaComponent,
} from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { PredictionRow } from "@/lib/types";
import { cn } from "@/lib/utils";

// Register ECharts modules (tree-shaken)
echarts.use([
  ELineChart,
  TooltipComponent,
  GridComponent,
  LegendComponent,
  DataZoomComponent,
  DataZoomInsideComponent,
  DataZoomSliderComponent,
  MarkLineComponent,
  MarkAreaComponent,
  CanvasRenderer,
]);

// ---------------------------------------------------------------------------
// Colors
// ---------------------------------------------------------------------------

const COLOR_MODEL = "#3b82f6"; // blue-500
const COLOR_MARKET = "#22c55e"; // green-500
const COLOR_RESULT = "#f97316"; // orange-500
const COLOR_UPCOMING_BG = "rgba(168,85,247,0.08)"; // purple tint for upcoming zone

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface TrendPoint {
  index: number;
  fixture_id: number;
  kickoff_at: string;
  league_key: string;
  home_team: string;
  away_team: string;
  home_score: number | null;
  away_score: number | null;
  goal_diff: number | null;
  model_y: number;
  market_y: number;
  ah_fair_line: number;
  ah_closing_line: number;
  line_diff: number;
  model_bets_home: boolean;
  who_right: "model" | "market" | "push" | "upcoming";
  running_model_pct: number;
  is_upcoming: boolean;
}

interface Props {
  data: PredictionRow[];
  upcoming?: PredictionRow[];
  leagues: { league_key: string; name: string }[];
}

// ---------------------------------------------------------------------------
// Core computation
// ---------------------------------------------------------------------------

function buildTrendPoints(
  data: PredictionRow[],
  upcoming: PredictionRow[] = []
): TrendPoint[] {
  // Historical: require scores + line disagreement
  const eligible = data
    .filter(
      (r) =>
        r.ah_fair_line !== null &&
        r.ah_closing_line !== null &&
        r.home_score !== null &&
        r.away_score !== null &&
        r.ah_fair_line !== r.ah_closing_line
    )
    .sort(
      (a, b) =>
        new Date(a.kickoff_at).getTime() - new Date(b.kickoff_at).getTime()
    );

  let modelCorrect = 0;
  let decided = 0;

  const historical: TrendPoint[] = eligible.map((r, i) => {
    const goalDiff = r.home_score! - r.away_score!;
    const modelLine = r.ah_fair_line!;
    const marketLine = r.ah_closing_line!;

    const model_y = -modelLine;
    const market_y = -marketLine;

    // Betting accuracy: model's directional signal tested at market prices
    const modelBetsHome = modelLine < marketLine;
    const adjusted = goalDiff + marketLine;

    let who_right: "model" | "market" | "push";
    if (adjusted === 0) {
      who_right = "push";
    } else if (modelBetsHome) {
      who_right = adjusted > 0 ? "model" : "market";
    } else {
      who_right = adjusted < 0 ? "model" : "market";
    }

    if (who_right !== "push") {
      decided++;
      if (who_right === "model") modelCorrect++;
    }

    const runPct = decided > 0 ? (modelCorrect / decided) * 100 : 50;

    return {
      index: i + 1,
      fixture_id: r.fixture_id,
      kickoff_at: r.kickoff_at,
      league_key: r.league_key,
      home_team: r.home_team,
      away_team: r.away_team,
      home_score: r.home_score!,
      away_score: r.away_score!,
      goal_diff: goalDiff,
      model_y,
      market_y,
      ah_fair_line: modelLine,
      ah_closing_line: marketLine,
      line_diff: Math.abs(model_y - market_y),
      model_bets_home: modelBetsHome,
      who_right,
      running_model_pct: runPct,
      is_upcoming: false,
    };
  });

  // Upcoming: require at least a model line
  const upcomingEligible = upcoming
    .filter((r) => r.ah_fair_line !== null)
    .sort(
      (a, b) =>
        new Date(a.kickoff_at).getTime() - new Date(b.kickoff_at).getTime()
    );

  const lastRunPct =
    historical.length > 0
      ? historical[historical.length - 1].running_model_pct
      : 50;
  const startIdx = historical.length;

  const upcomingPoints: TrendPoint[] = upcomingEligible.map((r, i) => {
    const modelLine = r.ah_fair_line!;
    const marketLine = r.ah_closing_line ?? modelLine;
    const model_y = -modelLine;
    const market_y = -marketLine;

    return {
      index: startIdx + i + 1,
      fixture_id: r.fixture_id,
      kickoff_at: r.kickoff_at,
      league_key: r.league_key,
      home_team: r.home_team,
      away_team: r.away_team,
      home_score: null,
      away_score: null,
      goal_diff: null,
      model_y,
      market_y,
      ah_fair_line: modelLine,
      ah_closing_line: marketLine,
      line_diff: Math.abs(model_y - market_y),
      model_bets_home: modelLine < marketLine,
      who_right: "upcoming" as const,
      running_model_pct: lastRunPct,
      is_upcoming: true,
    };
  });

  return [...historical, ...upcomingPoints];
}

function formatLine(l: number): string {
  return l >= 0 ? `+${l}` : `${l}`;
}

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString("en-GB", {
    day: "2-digit",
    month: "short",
  });
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function TrendsChart({ data, upcoming = [], leagues }: Props) {
  // Filters
  const [leagueFilter, setLeagueFilter] = useState("all");
  const [teamFilter, setTeamFilter] = useState("all");
  const [directionFilter, setDirectionFilter] = useState("all");
  const [minLineDiff, setMinLineDiff] = useState("0");
  const [whoRightFilter, setWhoRightFilter] = useState("all");

  // Series toggles
  const [showModel, setShowModel] = useState(true);
  const [showMarket, setShowMarket] = useState(true);
  const [showResult, setShowResult] = useState(true);

  // Build all points once (historical + upcoming)
  const allPoints = useMemo(
    () => buildTrendPoints(data, upcoming),
    [data, upcoming]
  );

  // Extract unique teams
  const teams = useMemo(() => {
    const set = new Set<string>();
    for (const p of allPoints) {
      set.add(p.home_team);
      set.add(p.away_team);
    }
    return Array.from(set).sort();
  }, [allPoints]);

  // Apply filters
  const filtered = useMemo(() => {
    let pts = allPoints;

    if (leagueFilter !== "all") {
      pts = pts.filter((p) => p.league_key === leagueFilter);
    }
    if (teamFilter !== "all") {
      pts = pts.filter(
        (p) => p.home_team === teamFilter || p.away_team === teamFilter
      );
    }
    if (directionFilter !== "all") {
      const betsHome = directionFilter === "model_home";
      pts = pts.filter((p) => p.is_upcoming || p.model_bets_home === betsHome);
    }
    if (minLineDiff !== "0") {
      const min = parseFloat(minLineDiff);
      pts = pts.filter((p) => p.is_upcoming || p.line_diff >= min);
    }
    if (whoRightFilter !== "all") {
      if (whoRightFilter === "upcoming") {
        pts = pts.filter((p) => p.is_upcoming);
      } else {
        pts = pts.filter(
          (p) => p.is_upcoming || p.who_right === whoRightFilter
        );
      }
    }

    // Re-index and recompute running accuracy after filtering
    let modelCorrect = 0;
    let decided = 0;
    return pts.map((p, i) => {
      if (!p.is_upcoming && p.who_right !== "push") {
        decided++;
        if (p.who_right === "model") modelCorrect++;
      }
      return {
        ...p,
        index: i + 1,
        running_model_pct: decided > 0 ? (modelCorrect / decided) * 100 : 50,
      };
    });
  }, [
    allPoints,
    leagueFilter,
    teamFilter,
    directionFilter,
    minLineDiff,
    whoRightFilter,
  ]);

  // Summary stats
  const stats = useMemo(() => {
    const historical = filtered.filter((p) => !p.is_upcoming);
    const upcomingCount = filtered.filter((p) => p.is_upcoming).length;
    const total = historical.length;
    const modelWins = historical.filter((p) => p.who_right === "model").length;
    const marketWins = historical.filter(
      (p) => p.who_right === "market"
    ).length;
    const pushes = historical.filter((p) => p.who_right === "push").length;
    const decidedCount = modelWins + marketWins;
    const avgLineDiff =
      total > 0
        ? historical.reduce((s, p) => s + p.line_diff, 0) / total
        : 0;
    return {
      total,
      upcomingCount,
      modelPct: decidedCount > 0 ? (modelWins / decidedCount) * 100 : 0,
      marketPct: decidedCount > 0 ? (marketWins / decidedCount) * 100 : 0,
      pushes,
      avgLineDiff,
    };
  }, [filtered]);

  // ---------------------------------------------------------------------------
  // Main chart ECharts option
  // ---------------------------------------------------------------------------
  const mainOption = useMemo(() => {
    const xData = filtered.map((p) => p.index);
    const smallSymbol = filtered.length > 200 ? 3 : 5;

    const whoRightLabel = (w: string) =>
      w === "model"
        ? "Model"
        : w === "market"
          ? "Market"
          : w === "upcoming"
            ? "Upcoming"
            : "Push";

    // Find the index where upcoming zone starts (for markArea)
    const firstUpcomingIdx = filtered.findIndex((p) => p.is_upcoming);

    // Build markArea for upcoming zone
    const upcomingMarkArea =
      firstUpcomingIdx >= 0
        ? {
            silent: true,
            data: [
              [
                {
                  xAxis: filtered[firstUpcomingIdx].index,
                  itemStyle: { color: COLOR_UPCOMING_BG },
                },
                { xAxis: filtered[filtered.length - 1].index },
              ],
            ],
            label: {
              show: true,
              position: "insideTop" as const,
              formatter: "Upcoming",
              color: "#a78bfa",
              fontSize: 11,
            },
          }
        : undefined;

    return {
      tooltip: {
        trigger: "axis" as const,
        backgroundColor: "rgba(24, 24, 27, 0.95)",
        borderColor: "#3f3f46",
        textStyle: { color: "#fafafa", fontSize: 12 },
        /* eslint-disable @typescript-eslint/no-explicit-any */
        formatter: (params: any) => {
          const idx = params[0]?.dataIndex;
          if (idx == null) return "";
          const p = filtered[idx];
          if (!p) return "";
          const lines = [
            `<b>${p.home_team} vs ${p.away_team}</b>`,
            `${formatDate(p.kickoff_at)} · ${p.league_key}`,
          ];
          if (p.is_upcoming) {
            lines.push(`<span style="color:#a78bfa">Upcoming</span>`);
          } else {
            lines.push(`Score: ${p.home_score}–${p.away_score}`);
          }
          lines.push(
            `Model AH: <b style="font-family:monospace">${formatLine(p.ah_fair_line)}</b>`,
            `Market AH: <b style="font-family:monospace">${formatLine(p.ah_closing_line)}</b>`
          );
          if (!p.is_upcoming) {
            lines.push(
              `Bet: <b>${p.model_bets_home ? "Home" : "Away"}</b>`,
              `Profitable: <b>${whoRightLabel(p.who_right)}</b>`
            );
          } else {
            lines.push(`Signal: <b>${p.model_bets_home ? "Home" : "Away"}</b>`);
          }
          return lines.join("<br/>");
        },
        /* eslint-enable @typescript-eslint/no-explicit-any */
      },
      legend: {
        top: 0,
        textStyle: { color: "#a1a1aa" },
        selected: {
          Model: showModel,
          Market: showMarket,
          Result: showResult,
        },
      },
      grid: {
        left: 55,
        right: 20,
        top: 40,
        bottom: 80,
      },
      xAxis: {
        type: "category" as const,
        data: xData,
        name: "Match #",
        nameLocation: "center" as const,
        nameGap: 30,
        nameTextStyle: { color: "#a1a1aa" },
        axisLabel: { color: "#a1a1aa" },
      },
      yAxis: {
        type: "value" as const,
        name: "← Away | Home →",
        nameTextStyle: { color: "#a1a1aa", fontSize: 11 },
        axisLabel: { color: "#a1a1aa" },
        splitLine: { lineStyle: { color: "#3f3f46", opacity: 0.4 } },
      },
      dataZoom: [
        {
          type: "inside" as const,
          xAxisIndex: 0,
          start: 0,
          end: 100,
          zoomOnMouseWheel: true,
          moveOnMouseMove: true,
          moveOnMouseWheel: false,
        },
        {
          type: "slider" as const,
          xAxisIndex: 0,
          start: 0,
          end: 100,
          height: 24,
          bottom: 10,
          borderColor: "#3f3f46",
          fillerColor: "rgba(59,130,246,0.15)",
          handleStyle: { color: "#3b82f6" },
          textStyle: { color: "#a1a1aa" },
          dataBackground: {
            lineStyle: { color: "#3f3f46" },
            areaStyle: { color: "rgba(63,63,70,0.3)" },
          },
        },
      ],
      series: [
        {
          name: "Model",
          type: "line" as const,
          data: filtered.map((p) => ({
            value: p.model_y,
            symbol: p.is_upcoming ? "diamond" : "circle",
            symbolSize: p.is_upcoming ? smallSymbol + 3 : smallSymbol,
            itemStyle: {
              color: p.is_upcoming ? "#a78bfa" : COLOR_MODEL,
            },
          })),
          symbol: "circle",
          symbolSize: smallSymbol,
          lineStyle: { width: 1.5, color: COLOR_MODEL },
          itemStyle: { color: COLOR_MODEL },
          markLine: {
            silent: true,
            symbol: "none",
            lineStyle: { color: "#71717a", type: "dashed" as const, width: 1 },
            data: [{ yAxis: 0 }],
            label: { show: false },
          },
          ...(upcomingMarkArea ? { markArea: upcomingMarkArea } : {}),
        },
        {
          name: "Market",
          type: "line" as const,
          data: filtered.map((p) => ({
            value: p.market_y,
            symbol: p.is_upcoming ? "diamond" : "circle",
            symbolSize: p.is_upcoming ? smallSymbol + 3 : smallSymbol,
            itemStyle: {
              color: p.is_upcoming ? "#a78bfa" : COLOR_MARKET,
            },
          })),
          symbol: "circle",
          symbolSize: smallSymbol,
          lineStyle: { width: 1.5, color: COLOR_MARKET },
          itemStyle: { color: COLOR_MARKET },
        },
        {
          name: "Result",
          type: "line" as const,
          data: filtered.map((p) => (p.is_upcoming ? null : p.goal_diff)),
          symbol: "circle",
          symbolSize: smallSymbol,
          lineStyle: {
            width: 1.5,
            color: COLOR_RESULT,
            type: "dashed" as const,
          },
          itemStyle: { color: COLOR_RESULT },
          connectNulls: false,
        },
      ],
    };
  }, [filtered, showModel, showMarket, showResult]);

  // ---------------------------------------------------------------------------
  // Running accuracy mini chart option
  // ---------------------------------------------------------------------------
  const runningOption = useMemo(() => {
    const xData = filtered.map((p) => p.index);

    return {
      tooltip: {
        trigger: "axis" as const,
        backgroundColor: "rgba(24, 24, 27, 0.95)",
        borderColor: "#3f3f46",
        textStyle: { color: "#fafafa", fontSize: 12 },
        /* eslint-disable @typescript-eslint/no-explicit-any */
        formatter: (params: any) => {
          const idx = params[0]?.dataIndex;
          if (idx == null) return "";
          const p = filtered[idx];
          if (!p) return "";
          return `Match #${p.index}<br/>Model: <b>${p.running_model_pct.toFixed(1)}%</b>`;
        },
        /* eslint-enable @typescript-eslint/no-explicit-any */
      },
      grid: {
        left: 55,
        right: 20,
        top: 10,
        bottom: 55,
      },
      xAxis: {
        type: "category" as const,
        data: xData,
        axisLabel: { color: "#a1a1aa" },
      },
      yAxis: {
        type: "value" as const,
        min: 0,
        max: 100,
        axisLabel: {
          color: "#a1a1aa",
          formatter: (v: number) => `${v}%`,
        },
        splitLine: { lineStyle: { color: "#3f3f46", opacity: 0.4 } },
      },
      dataZoom: [
        {
          type: "inside" as const,
          xAxisIndex: 0,
          start: 0,
          end: 100,
        },
        {
          type: "slider" as const,
          xAxisIndex: 0,
          start: 0,
          end: 100,
          height: 20,
          bottom: 5,
          borderColor: "#3f3f46",
          fillerColor: "rgba(59,130,246,0.15)",
          handleStyle: { color: "#3b82f6" },
          textStyle: { color: "#a1a1aa" },
          dataBackground: {
            lineStyle: { color: "#3f3f46" },
            areaStyle: { color: "rgba(63,63,70,0.3)" },
          },
        },
      ],
      series: [
        {
          type: "line" as const,
          data: filtered.map((p) => p.running_model_pct),
          symbol: "none",
          lineStyle: { width: 2, color: COLOR_MODEL },
          itemStyle: { color: COLOR_MODEL },
          markLine: {
            silent: true,
            symbol: "none",
            lineStyle: { color: "#71717a", type: "dashed" as const, width: 1 },
            data: [{ yAxis: 50 }],
            label: { show: false },
          },
        },
      ],
    };
  }, [filtered]);

  return (
    <div className="space-y-6">
      {/* Filter Panel */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Filters</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap items-center gap-3">
            <Select value={leagueFilter} onValueChange={setLeagueFilter}>
              <SelectTrigger className="w-40">
                <SelectValue placeholder="All leagues" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All leagues</SelectItem>
                {leagues.map((l) => (
                  <SelectItem key={l.league_key} value={l.league_key}>
                    {l.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            <Select value={teamFilter} onValueChange={setTeamFilter}>
              <SelectTrigger className="w-48">
                <SelectValue placeholder="All teams" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All teams</SelectItem>
                {teams.map((t) => (
                  <SelectItem key={t} value={t}>
                    {t}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            <Select
              value={directionFilter}
              onValueChange={setDirectionFilter}
            >
              <SelectTrigger className="w-48">
                <SelectValue placeholder="All directions" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All directions</SelectItem>
                <SelectItem value="model_home">
                  Model bets Home
                </SelectItem>
                <SelectItem value="model_away">
                  Model bets Away
                </SelectItem>
              </SelectContent>
            </Select>

            <Select value={minLineDiff} onValueChange={setMinLineDiff}>
              <SelectTrigger className="w-36">
                <SelectValue placeholder="Any diff" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="0">Any diff</SelectItem>
                <SelectItem value="0.25">0.25+</SelectItem>
                <SelectItem value="0.5">0.50+</SelectItem>
                <SelectItem value="0.75">0.75+</SelectItem>
              </SelectContent>
            </Select>

            <Select value={whoRightFilter} onValueChange={setWhoRightFilter}>
              <SelectTrigger className="w-36">
                <SelectValue placeholder="All results" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All results</SelectItem>
                <SelectItem value="model">Model profitable</SelectItem>
                <SelectItem value="market">Market profitable</SelectItem>
                <SelectItem value="push">Push</SelectItem>
                {stats.upcomingCount > 0 && (
                  <SelectItem value="upcoming">Upcoming only</SelectItem>
                )}
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* Summary Cards */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Matches</CardDescription>
            <CardTitle className="text-3xl tabular-nums">
              {stats.total}
              {stats.upcomingCount > 0 && (
                <span className="text-base font-normal text-purple-400 ml-2">
                  +{stats.upcomingCount}
                </span>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-xs text-muted-foreground">
              With AH line disagreement
              {stats.upcomingCount > 0 && ` · ${stats.upcomingCount} upcoming`}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Model Accuracy</CardDescription>
            <CardTitle className="text-3xl tabular-nums">
              {stats.modelPct.toFixed(1)}%
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-xs text-muted-foreground">
              Excl. {stats.pushes} pushes
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Market Accuracy</CardDescription>
            <CardTitle className="text-3xl tabular-nums">
              {stats.marketPct.toFixed(1)}%
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-xs text-muted-foreground">
              Excl. {stats.pushes} pushes
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Avg Line Diff</CardDescription>
            <CardTitle className="text-3xl tabular-nums">
              {stats.avgLineDiff.toFixed(2)}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-xs text-muted-foreground">
              |model &minus; market|
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Series Toggle Bar */}
      <div className="flex items-center gap-2">
        <span className="text-sm text-muted-foreground mr-1">Series:</span>
        <button
          onClick={() => setShowModel((v) => !v)}
          className={cn(
            "inline-flex items-center rounded-full border px-3 py-1 text-xs font-medium transition-colors",
            showModel
              ? "border-transparent text-white"
              : "border-border text-muted-foreground"
          )}
          style={showModel ? { backgroundColor: COLOR_MODEL } : undefined}
        >
          Model
        </button>
        <button
          onClick={() => setShowMarket((v) => !v)}
          className={cn(
            "inline-flex items-center rounded-full border px-3 py-1 text-xs font-medium transition-colors",
            showMarket
              ? "border-transparent text-white"
              : "border-border text-muted-foreground"
          )}
          style={showMarket ? { backgroundColor: COLOR_MARKET } : undefined}
        >
          Market
        </button>
        <button
          onClick={() => setShowResult((v) => !v)}
          className={cn(
            "inline-flex items-center rounded-full border px-3 py-1 text-xs font-medium transition-colors",
            showResult
              ? "border-transparent text-white"
              : "border-border text-muted-foreground"
          )}
          style={showResult ? { backgroundColor: COLOR_RESULT } : undefined}
        >
          Result
        </button>
        <span className="ml-2 text-xs text-muted-foreground">
          {filtered.length} points
          {stats.upcomingCount > 0 && (
            <span className="text-purple-400">
              {" "}({stats.upcomingCount} upcoming)
            </span>
          )}
        </span>
        <span className="ml-auto text-xs text-muted-foreground">
          Scroll to zoom · Drag slider to pan
        </span>
      </div>

      {/* Main Chart */}
      <Card>
        <CardContent className="pt-6">
          {filtered.length === 0 ? (
            <div className="flex h-[500px] items-center justify-center text-muted-foreground">
              No matches for these filters.
            </div>
          ) : (
            <ReactEChartsCore
              echarts={echarts}
              option={mainOption}
              theme="dark"
              notMerge
              lazyUpdate
              style={{ height: 500, width: "100%" }}
            />
          )}
        </CardContent>
      </Card>

      {/* Running Accuracy Mini Chart */}
      {filtered.length > 0 && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">
              Running Model Accuracy
            </CardTitle>
            <CardDescription>
              Cumulative model accuracy % over time (excl. pushes)
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ReactEChartsCore
              echarts={echarts}
              option={runningOption}
              theme="dark"
              notMerge
              lazyUpdate
              style={{ height: 200, width: "100%" }}
            />
          </CardContent>
        </Card>
      )}
    </div>
  );
}
