import { notFound } from "next/navigation";
import {
  getFixtureDetail,
  getPredictions,
  getTodayPredictions,
  getLeagues,
  getLeagueFixturesForPriceChain,
} from "@/lib/queries";
import { TodayMatchDetail } from "@/components/today-match-detail";

export const dynamic = "force-dynamic";

interface Props {
  params: Promise<{ id: string }>;
}

export default async function TodayMatchDetailPage({ params }: Props) {
  const { id } = await params;
  const fixtureId = parseInt(id, 10);
  if (isNaN(fixtureId)) notFound();

  const detail = await getFixtureDetail(fixtureId);
  if (!detail) notFound();

  const [allPredictions, todayPredictions, leagues, leagueFixtures] =
    await Promise.all([
      getPredictions(),
      getTodayPredictions(7),
      getLeagues(),
      getLeagueFixturesForPriceChain(detail.fixture.league_id),
    ]);

  return (
    <TodayMatchDetail
      detail={detail}
      allPredictions={allPredictions}
      todayPredictions={todayPredictions}
      leagues={leagues}
      leagueFixtures={leagueFixtures}
    />
  );
}
