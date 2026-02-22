import {
  getPredictions,
  getTodayPredictions,
  getTodayFixtures,
  getLeagues,
} from "@/lib/queries";
import { TodayMatchList } from "@/components/today-match-list";

export const dynamic = "force-dynamic";

interface Props {
  searchParams: Promise<{ days?: string }>;
}

export default async function TodayPage({ searchParams }: Props) {
  const { days: daysParam } = await searchParams;
  const days = Math.min(Math.max(Number(daysParam) || 1, 1), 7);

  const [predictions, fixtures, leagues, allPredictions] = await Promise.all([
    getTodayPredictions(days),
    getTodayFixtures(days),
    getLeagues(),
    getPredictions(),
  ]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">
          {days === 1 ? "Today\u2019s Matches" : `Matches (Next ${days} Days)`}
        </h1>
        <p className="text-sm text-muted-foreground">
          {new Date().toLocaleDateString("en-GB", {
            weekday: "long",
            day: "numeric",
            month: "long",
            year: "numeric",
            timeZone: "Asia/Bangkok",
          })}
          {days > 1 && (
            <>
              {" \u2013 "}
              {new Date(
                Date.now() + (days - 1) * 86_400_000
              ).toLocaleDateString("en-GB", {
                weekday: "long",
                day: "numeric",
                month: "long",
                timeZone: "Asia/Bangkok",
              })}
            </>
          )}
          {" (Thailand)"}
        </p>
      </div>
      <TodayMatchList
        predictions={predictions}
        allPredictions={allPredictions}
        fixtures={fixtures}
        leagues={leagues}
        days={days}
      />
    </div>
  );
}
