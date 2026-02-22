import {
  getDatePredictions,
  getDateFixtures,
  getLeagues,
  getPredictions,
  getMainTotalsLines,
  getLockedSessionBets,
  getDailyPicks,
} from "@/lib/queries";
import { DailySimulator } from "@/components/daily-simulator";

export const dynamic = "force-dynamic";

interface Props {
  searchParams: Promise<{ date?: string }>;
}

function todayUTC(): string {
  return new Date().toISOString().slice(0, 10);
}

export default async function DailyPage({ searchParams }: Props) {
  const { date: dateParam } = await searchParams;
  const date = dateParam && /^\d{4}-\d{2}-\d{2}$/.test(dateParam)
    ? dateParam
    : todayUTC();

  const today = todayUTC();
  const isPast = date < today;

  if (isPast) {
    // Fast path: read from daily_picks (no allPredictions needed)
    const [storedPicks, lockedBets, predictions] = await Promise.all([
      getDailyPicks(date),
      getLockedSessionBets(date),
      getDatePredictions(date),
    ]);

    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Daily Simulator</h1>
          <p className="text-sm text-muted-foreground">
            Track daily P&amp;L across all betting categories
          </p>
        </div>
        <DailySimulator
          date={date}
          predictions={predictions}
          lockedBets={lockedBets}
          storedPicks={storedPicks}
        />
      </div>
    );
  }

  // Live path: full computation (today or future dates)
  const [predictions, fixtures, leagues, allPredictions, lockedBets] = await Promise.all([
    getDatePredictions(date),
    getDateFixtures(date),
    getLeagues(),
    getPredictions(),
    getLockedSessionBets(date),
  ]);

  // Collect all fixture IDs from both fixtures and predictions
  const allFixtureIds = [
    ...new Set([
      ...fixtures.map((f) => f.id),
      ...predictions.map((p) => p.fixture_id),
    ]),
  ];
  const totalsLines = await getMainTotalsLines(allFixtureIds);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Daily Simulator</h1>
        <p className="text-sm text-muted-foreground">
          Track daily P&amp;L across all betting categories
        </p>
      </div>
      <DailySimulator
        date={date}
        predictions={predictions}
        allPredictions={allPredictions}
        fixtures={fixtures}
        leagues={leagues}
        totalsLines={totalsLines}
        lockedBets={lockedBets}
      />
    </div>
  );
}
