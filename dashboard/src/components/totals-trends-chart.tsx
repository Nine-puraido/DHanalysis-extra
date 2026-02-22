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
const COLOR_GOALS = "#f97316"; // orange-500
const COLOR_UPCOMING_BG = "rgba(168,85,247,0.08)";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface TotalsTrendPoint {
  index: number;
  fixture_id: number;
  kickoff_at: string;
  league_key: string;
  home_team: string;
  away_team: string;
  home_score: number | null;
  away_score: number | null;
  total_goals: number | null;
  model_prob_over: number;
  market_prob_over: number;
  prob_diff: number;
  direction: "model_over" | "model_under";
  who_right: "model" | "market" | "upcoming";
  running_model_pct: number;
  is_upcoming: boolean;
  ah_fair_line: number | null;
  ah_closing_line: number | null;
}

interface Props {
  data: PredictionRow[];
  upcoming?: PredictionRow[];
  leagues: { league_key: string; name: string }[];
}

// ---------------------------------------------------------------------------
// Core computation
// ---------------------------------------------------------------------------

function buildTotalsTrendPoints(
  data: PredictionRow[],
  upcoming: PredictionRow[] = []
): TotalsTrendPoint[] {
  // Historical: require scores + prob data + disagreement
  const eligible = data
    .filter(
      (r) =>
        r.prob_over25 !== null &&
        r.closing_over25 !== null &&
        r.closing_over25 > 1 &&
        r.home_score !== null &&
        r.away_score !== null
    )
    .sort(
      (a, b) =>
        new Date(a.kickoff_at).getTime() - new Date(b.kickoff_at).getTime()
    );

  let modelCorrect = 0;
  let decided = 0;

  const historical: TotalsTrendPoint[] = eligible
    .filter((r) => {
      const modelProb = r.prob_over25!;
      const marketProb = 1 / r.closing_over25!;
      return Math.abs(modelProb - marketProb) > 0.001; // skip agreement
    })
    .map((r, i) => {
      const modelProb = r.prob_over25!;
      const marketProb = 1 / r.closing_over25!;
      const totalGoals = r.home_score! + r.away_score!;
      const modelMoreOver = modelProb > marketProb;

      // Who was right?
      let who_right: "model" | "market";
      if (totalGoals > 2.5) {
        // Over hit — whoever had higher P(O2.5) was right
        who_right = modelMoreOver ? "model" : "market";
      } else {
        // Under hit — whoever had lower P(O2.5) was right
        who_right = modelMoreOver ? "market" : "model";
      }

      decided++;
      if (who_right === "model") modelCorrect++;

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
        total_goals: totalGoals,
        model_prob_over: modelProb,
        market_prob_over: marketProb,
        prob_diff: Math.abs(modelProb - marketProb),
        direction: modelMoreOver ? "model_over" : "model_under",
        who_right,
        running_model_pct: runPct,
        is_upcoming: false,
        ah_fair_line: r.ah_fair_line,
        ah_closing_line: r.ah_closing_line,
      };
    });

  // Upcoming: require at least model prob
  const upcomingEligible = upcoming
    .filter((r) => r.prob_over25 !== null)
    .sort(
      (a, b) =>
        new Date(a.kickoff_at).getTime() - new Date(b.kickoff_at).getTime()
    );

  const lastRunPct =
    historical.length > 0
      ? historical[historical.length - 1].running_model_pct
      : 50;
  const startIdx = historical.length;

  const upcomingPoints: TotalsTrendPoint[] = upcomingEligible.map((r, i) => {
    const modelProb = r.prob_over25!;
    const marketProb =
      r.closing_over25 !== null && r.closing_over25 > 1
        ? 1 / r.closing_over25
        : modelProb;
    const modelMoreOver = modelProb > marketProb;

    return {
      index: startIdx + i + 1,
      fixture_id: r.fixture_id,
      kickoff_at: r.kickoff_at,
      league_key: r.league_key,
      home_team: r.home_team,
      away_team: r.away_team,
      home_score: null,
      away_score: null,
      total_goals: null,
      model_prob_over: modelProb,
      market_prob_over: marketProb,
      prob_diff: Math.abs(modelProb - marketProb),
      direction: modelMoreOver ? "model_over" : "model_under",
      who_right: "upcoming" as const,
      running_model_pct: lastRunPct,
      is_upcoming: true,
      ah_fair_line: r.ah_fair_line,
      ah_closing_line: r.ah_closing_line,
    };
  });

  return [...historical, ...upcomingPoints];
}

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString("en-GB", {
    day: "2-digit",
    month: "short",
  });
}

function formatPct(v: number): string {
  return `${(v * 100).toFixed(1)}%`;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function TotalsTrendsChart({ data, upcoming = [], leagues }: Props) {
  // Filters
  const [leagueFilter, setLeagueFilter] = useState("all");
  const [teamFilter, setTeamFilter] = useState("all");
  const [directionFilter, setDirectionFilter] = useState("all");
  const [minProbDiff, setMinProbDiff] = useState("0");
  const [whoRightFilter, setWhoRightFilter] = useState("all");
  const [ahLineFilter, setAhLineFilter] = useState("all");

  // Series toggles
  const [showModel, setShowModel] = useState(true);
  const [showMarket, setShowMarket] = useState(true);
  const [showGoals, setShowGoals] = useState(true);

  // Build all points once (historical + upcoming)
  const allPoints = useMemo(
    () => buildTotalsTrendPoints(data, upcoming),
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

  // Extract unique AH push line values (where model line == market line)
  const ahPushLines = useMemo(() => {
    const set = new Set<number>();
    for (const p of allPoints) {
      if (
        p.ah_fair_line !== null &&
        p.ah_closing_line !== null &&
        p.ah_fair_line === p.ah_closing_line
      ) {
        set.add(p.ah_fair_line);
      }
    }
    return Array.from(set).sort((a, b) => a - b);
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
      pts = pts.filter(
        (p) => p.is_upcoming || p.direction === directionFilter
      );
    }
    if (minProbDiff !== "0") {
      const min = parseFloat(minProbDiff) / 100;
      pts = pts.filter((p) => p.is_upcoming || p.prob_diff >= min);
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
    if (ahLineFilter !== "all") {
      if (ahLineFilter === "push") {
        // All push matches: model line == market line
        pts = pts.filter(
          (p) =>
            p.is_upcoming ||
            (p.ah_fair_line !== null &&
              p.ah_closing_line !== null &&
              p.ah_fair_line === p.ah_closing_line)
        );
      } else {
        // Specific push line value
        const lineVal = parseFloat(ahLineFilter);
        pts = pts.filter(
          (p) =>
            p.is_upcoming ||
            (p.ah_fair_line === lineVal && p.ah_closing_line === lineVal)
        );
      }
    }

    // Re-index and recompute running accuracy after filtering
    let modelCorrect = 0;
    let decided = 0;
    return pts.map((p, i) => {
      if (!p.is_upcoming) {
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
    minProbDiff,
    whoRightFilter,
    ahLineFilter,
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
    const decidedCount = modelWins + marketWins;
    const avgProbDiff =
      total > 0
        ? historical.reduce((s, p) => s + p.prob_diff, 0) / total
        : 0;
    return {
      total,
      upcomingCount,
      modelPct: decidedCount > 0 ? (modelWins / decidedCount) * 100 : 0,
      marketPct: decidedCount > 0 ? (marketWins / decidedCount) * 100 : 0,
      avgProbDiff,
    };
  }, [filtered]);

  // AH Push breakdown: group by line, split by direction, count over/under
  const ahPushBreakdown = useMemo(() => {
    // Use filtered historical points that are AH push matches
    const pushPts = filtered.filter(
      (p) =>
        !p.is_upcoming &&
        p.ah_fair_line !== null &&
        p.ah_closing_line !== null &&
        p.ah_fair_line === p.ah_closing_line &&
        p.total_goals !== null
    );

    // Group by AH line
    const byLine = new Map<
      number,
      { mktOver: number; mktUnder: number; modelOver: number; modelUnder: number }
    >();

    for (const p of pushPts) {
      const line = p.ah_fair_line!;
      if (!byLine.has(line)) {
        byLine.set(line, { mktOver: 0, mktUnder: 0, modelOver: 0, modelUnder: 0 });
      }
      const bucket = byLine.get(line)!;
      const isOver = p.total_goals! > 2.5;

      if (p.direction === "model_under") {
        // Market expects more goals
        if (isOver) bucket.mktOver++;
        else bucket.mktUnder++;
      } else {
        // Model expects more goals
        if (isOver) bucket.modelOver++;
        else bucket.modelUnder++;
      }
    }

    return Array.from(byLine.entries())
      .sort((a, b) => a[0] - b[0])
      .map(([line, b]) => ({
        line,
        mktOver: b.mktOver,
        mktUnder: b.mktUnder,
        mktTotal: b.mktOver + b.mktUnder,
        mktOverPct:
          b.mktOver + b.mktUnder > 0
            ? (b.mktOver / (b.mktOver + b.mktUnder)) * 100
            : 0,
        modelOver: b.modelOver,
        modelUnder: b.modelUnder,
        modelTotal: b.modelOver + b.modelUnder,
        modelOverPct:
          b.modelOver + b.modelUnder > 0
            ? (b.modelOver / (b.modelOver + b.modelUnder)) * 100
            : 0,
      }));
  }, [filtered]);

  // ---------------------------------------------------------------------------
  // Main chart ECharts option (dual Y-axis)
  // ---------------------------------------------------------------------------
  const mainOption = useMemo(() => {
    const xData = filtered.map((p) => p.index);
    const smallSymbol = filtered.length > 200 ? 3 : 5;

    const whoRightLabel = (w: string) =>
      w === "model" ? "Model" : w === "market" ? "Market" : "Upcoming";

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
            lines.push(
              `Score: ${p.home_score}–${p.away_score} (${p.total_goals} goals)`
            );
          }
          lines.push(
            `Model P(O2.5): <b>${formatPct(p.model_prob_over)}</b>`,
            `Market P(O2.5): <b>${formatPct(p.market_prob_over)}</b>`
          );
          if (!p.is_upcoming) {
            lines.push(`Right: <b>${whoRightLabel(p.who_right)}</b>`);
          }
          return lines.join("<br/>");
        },
        /* eslint-enable @typescript-eslint/no-explicit-any */
      },
      legend: {
        top: 0,
        textStyle: { color: "#a1a1aa" },
        selected: {
          "Model P(O2.5)": showModel,
          "Market P(O2.5)": showMarket,
          "Total Goals": showGoals,
        },
      },
      grid: {
        left: 55,
        right: 55,
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
      yAxis: [
        {
          type: "value" as const,
          name: "Probability",
          nameTextStyle: { color: "#a1a1aa", fontSize: 11 },
          min: 0,
          max: 100,
          axisLabel: {
            color: "#a1a1aa",
            formatter: (v: number) => `${v}%`,
          },
          splitLine: { lineStyle: { color: "#3f3f46", opacity: 0.4 } },
        },
        {
          type: "value" as const,
          name: "Goals",
          nameTextStyle: { color: "#a1a1aa", fontSize: 11 },
          min: 0,
          max: 8,
          axisLabel: { color: "#a1a1aa" },
          splitLine: { show: false },
        },
      ],
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
          name: "Model P(O2.5)",
          type: "line" as const,
          yAxisIndex: 0,
          data: filtered.map((p) => ({
            value: p.model_prob_over * 100,
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
            lineStyle: {
              color: "#71717a",
              type: "dashed" as const,
              width: 1,
            },
            data: [{ yAxis: 50 }],
            label: { show: false },
          },
          ...(upcomingMarkArea ? { markArea: upcomingMarkArea } : {}),
        },
        {
          name: "Market P(O2.5)",
          type: "line" as const,
          yAxisIndex: 0,
          data: filtered.map((p) => ({
            value: p.market_prob_over * 100,
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
          name: "Total Goals",
          type: "line" as const,
          yAxisIndex: 1,
          data: filtered.map((p) =>
            p.is_upcoming ? null : p.total_goals
          ),
          symbol: "circle",
          symbolSize: smallSymbol,
          lineStyle: {
            width: 1.5,
            color: COLOR_GOALS,
            type: "dashed" as const,
          },
          itemStyle: { color: COLOR_GOALS },
          connectNulls: false,
          markLine: {
            silent: true,
            symbol: "none",
            lineStyle: {
              color: "#71717a",
              type: "dashed" as const,
              width: 1,
            },
            data: [{ yAxis: 2.5 }],
            label: { show: false },
          },
        },
      ],
    };
  }, [filtered, showModel, showMarket, showGoals]);

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
        right: 55,
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
            lineStyle: {
              color: "#71717a",
              type: "dashed" as const,
              width: 1,
            },
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
                <SelectItem value="model_over">
                  Model more over
                </SelectItem>
                <SelectItem value="model_under">
                  Model more under
                </SelectItem>
              </SelectContent>
            </Select>

            <Select value={minProbDiff} onValueChange={setMinProbDiff}>
              <SelectTrigger className="w-36">
                <SelectValue placeholder="Any diff" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="0">Any diff</SelectItem>
                <SelectItem value="5">5%+</SelectItem>
                <SelectItem value="10">10%+</SelectItem>
                <SelectItem value="15">15%+</SelectItem>
              </SelectContent>
            </Select>

            <Select value={whoRightFilter} onValueChange={setWhoRightFilter}>
              <SelectTrigger className="w-36">
                <SelectValue placeholder="All results" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All results</SelectItem>
                <SelectItem value="model">Model right</SelectItem>
                <SelectItem value="market">Market right</SelectItem>
                {stats.upcomingCount > 0 && (
                  <SelectItem value="upcoming">Upcoming only</SelectItem>
                )}
              </SelectContent>
            </Select>

            <Select value={ahLineFilter} onValueChange={setAhLineFilter}>
              <SelectTrigger className="w-44">
                <SelectValue placeholder="AH line" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All AH lines</SelectItem>
                <SelectItem value="push">AH Push only</SelectItem>
                {ahPushLines.map((line) => (
                  <SelectItem key={line} value={String(line)}>
                    Push AH {line > 0 ? "+" : ""}
                    {line}
                  </SelectItem>
                ))}
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
              With O/U disagreement
              {stats.upcomingCount > 0 &&
                ` · ${stats.upcomingCount} upcoming`}
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
              O/U direction accuracy
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
              O/U direction accuracy
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Avg Prob Diff</CardDescription>
            <CardTitle className="text-3xl tabular-nums">
              {formatPct(stats.avgProbDiff)}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-xs text-muted-foreground">
              |model &minus; market| P(O2.5)
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
          Model P(O2.5)
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
          Market P(O2.5)
        </button>
        <button
          onClick={() => setShowGoals((v) => !v)}
          className={cn(
            "inline-flex items-center rounded-full border px-3 py-1 text-xs font-medium transition-colors",
            showGoals
              ? "border-transparent text-white"
              : "border-border text-muted-foreground"
          )}
          style={showGoals ? { backgroundColor: COLOR_GOALS } : undefined}
        >
          Total Goals
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
              Cumulative model accuracy % for O/U calls
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

      {/* AH Push O/U Breakdown */}
      {ahPushBreakdown.length > 0 && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">
              AH Push &mdash; Over/Under Breakdown
            </CardTitle>
            <CardDescription>
              On matches where model AH line = market AH line (push), split by
              who expects more goals (higher P(O2.5)) and actual over/under
              outcome.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border text-muted-foreground">
                    <th className="py-2 pr-4 text-left font-medium" rowSpan={2}>
                      AH Line
                    </th>
                    <th
                      className="px-2 py-1 text-center font-medium"
                      colSpan={4}
                      style={{ color: COLOR_MARKET }}
                    >
                      Market expects more goals
                    </th>
                    <th
                      className="px-2 py-1 text-center font-medium"
                      colSpan={4}
                      style={{ color: COLOR_MODEL }}
                    >
                      Model expects more goals
                    </th>
                  </tr>
                  <tr className="border-b border-border text-muted-foreground text-xs">
                    <th className="px-2 py-1 text-center">Over</th>
                    <th className="px-2 py-1 text-center">Under</th>
                    <th className="px-2 py-1 text-center">N</th>
                    <th className="px-2 py-1 text-center">Over%</th>
                    <th className="px-2 py-1 text-center">Over</th>
                    <th className="px-2 py-1 text-center">Under</th>
                    <th className="px-2 py-1 text-center">N</th>
                    <th className="px-2 py-1 text-center">Over%</th>
                  </tr>
                </thead>
                <tbody>
                  {ahPushBreakdown.map((row) => (
                    <tr
                      key={row.line}
                      className="border-b border-border/50 hover:bg-muted/30"
                    >
                      <td className="py-2 pr-4 font-mono tabular-nums">
                        {row.line > 0 ? "+" : ""}
                        {row.line}
                      </td>
                      <td className="px-2 py-2 text-center tabular-nums">
                        {row.mktOver}
                      </td>
                      <td className="px-2 py-2 text-center tabular-nums">
                        {row.mktUnder}
                      </td>
                      <td className="px-2 py-2 text-center tabular-nums text-muted-foreground">
                        {row.mktTotal}
                      </td>
                      <td
                        className={cn(
                          "px-2 py-2 text-center tabular-nums font-medium",
                          row.mktTotal > 0 && row.mktOverPct > 55
                            ? "text-green-400"
                            : row.mktTotal > 0 && row.mktOverPct < 45
                              ? "text-red-400"
                              : ""
                        )}
                      >
                        {row.mktTotal > 0
                          ? `${row.mktOverPct.toFixed(0)}%`
                          : "-"}
                      </td>
                      <td className="px-2 py-2 text-center tabular-nums">
                        {row.modelOver}
                      </td>
                      <td className="px-2 py-2 text-center tabular-nums">
                        {row.modelUnder}
                      </td>
                      <td className="px-2 py-2 text-center tabular-nums text-muted-foreground">
                        {row.modelTotal}
                      </td>
                      <td
                        className={cn(
                          "px-2 py-2 text-center tabular-nums font-medium",
                          row.modelTotal > 0 && row.modelOverPct > 55
                            ? "text-green-400"
                            : row.modelTotal > 0 && row.modelOverPct < 45
                              ? "text-red-400"
                              : ""
                        )}
                      >
                        {row.modelTotal > 0
                          ? `${row.modelOverPct.toFixed(0)}%`
                          : "-"}
                      </td>
                    </tr>
                  ))}
                  {/* Totals row */}
                  {ahPushBreakdown.length > 1 && (() => {
                    const totals = ahPushBreakdown.reduce(
                      (acc, r) => ({
                        mktOver: acc.mktOver + r.mktOver,
                        mktUnder: acc.mktUnder + r.mktUnder,
                        modelOver: acc.modelOver + r.modelOver,
                        modelUnder: acc.modelUnder + r.modelUnder,
                      }),
                      { mktOver: 0, mktUnder: 0, modelOver: 0, modelUnder: 0 }
                    );
                    const mktTotal = totals.mktOver + totals.mktUnder;
                    const modelTotal = totals.modelOver + totals.modelUnder;
                    return (
                      <tr className="border-t border-border font-medium">
                        <td className="py-2 pr-4">Total</td>
                        <td className="px-2 py-2 text-center tabular-nums">
                          {totals.mktOver}
                        </td>
                        <td className="px-2 py-2 text-center tabular-nums">
                          {totals.mktUnder}
                        </td>
                        <td className="px-2 py-2 text-center tabular-nums text-muted-foreground">
                          {mktTotal}
                        </td>
                        <td
                          className={cn(
                            "px-2 py-2 text-center tabular-nums",
                            mktTotal > 0 && (totals.mktOver / mktTotal) * 100 > 55
                              ? "text-green-400"
                              : mktTotal > 0 && (totals.mktOver / mktTotal) * 100 < 45
                                ? "text-red-400"
                                : ""
                          )}
                        >
                          {mktTotal > 0
                            ? `${((totals.mktOver / mktTotal) * 100).toFixed(0)}%`
                            : "-"}
                        </td>
                        <td className="px-2 py-2 text-center tabular-nums">
                          {totals.modelOver}
                        </td>
                        <td className="px-2 py-2 text-center tabular-nums">
                          {totals.modelUnder}
                        </td>
                        <td className="px-2 py-2 text-center tabular-nums text-muted-foreground">
                          {modelTotal}
                        </td>
                        <td
                          className={cn(
                            "px-2 py-2 text-center tabular-nums",
                            modelTotal > 0 && (totals.modelOver / modelTotal) * 100 > 55
                              ? "text-green-400"
                              : modelTotal > 0 && (totals.modelOver / modelTotal) * 100 < 45
                                ? "text-red-400"
                                : ""
                          )}
                        >
                          {modelTotal > 0
                            ? `${((totals.modelOver / modelTotal) * 100).toFixed(0)}%`
                            : "-"}
                        </td>
                      </tr>
                    );
                  })()}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
