export interface League {
  id: number;
  league_key: string;
  name: string;
  country: string;
}

export interface Team {
  id: number;
  name: string;
  short_name: string | null;
  country: string | null;
}

export interface Fixture {
  id: number;
  league_id: number;
  season_id: number | null;
  home_team_id: number;
  away_team_id: number;
  kickoff_at: string;
  status: string;
  venue: string | null;
}

export interface Result {
  fixture_id: number;
  home_score: number;
  away_score: number;
  home_ht_score: number | null;
  away_ht_score: number | null;
  result_status: string;
}

export interface MatchStatistics {
  fixture_id: number;
  home_xg: number | null;
  away_xg: number | null;
  home_shots: number | null;
  away_shots: number | null;
  home_shots_on_target: number | null;
  away_shots_on_target: number | null;
  home_shots_off_target: number | null;
  away_shots_off_target: number | null;
  home_blocked_shots: number | null;
  away_blocked_shots: number | null;
  home_possession: number | null;
  away_possession: number | null;
  home_corner_kicks: number | null;
  away_corner_kicks: number | null;
  home_offsides: number | null;
  away_offsides: number | null;
  home_fouls: number | null;
  away_fouls: number | null;
  home_yellow_cards: number | null;
  away_yellow_cards: number | null;
  home_red_cards: number | null;
  away_red_cards: number | null;
  home_passes: number | null;
  away_passes: number | null;
  home_accurate_passes: number | null;
  away_accurate_passes: number | null;
  home_saves: number | null;
  away_saves: number | null;
  home_big_chances: number | null;
  away_big_chances: number | null;
  home_big_chances_missed: number | null;
  away_big_chances_missed: number | null;
}

export interface ModelVersion {
  id: number;
  model_name: string;
  model_type: string;
  market: string;
  feature_set_version_id: number | null;
  artifact_path: string | null;
  training_window: string | null;
  metrics_json: Record<string, number>;
  params_json: Record<string, unknown>;
  is_active: boolean;
  created_at: string;
  activated_at: string | null;
}

export interface Prediction {
  id: number;
  fixture_id: number;
  model_version_id: number;
  bookmaker_id: number | null;
  market: string;
  selection: string;
  line: number | null;
  probability: number;
  fair_odds: number | null;
  predicted_at: string;
  context_json: Record<string, unknown>;
}

export interface OddsSnapshot {
  fixture_id: number;
  bookmaker_id: number;
  market: string;
  selection: string;
  line: number | null;
  price_decimal: number;
  implied_prob: number | null;
  is_main: boolean;
  is_suspended: boolean;
  pulled_at: string;
}

// Joined types for dashboard views

export interface PredictionRow {
  fixture_id: number;
  kickoff_at: string;
  status: string;
  league_key: string;
  league_name: string;
  home_team: string;
  away_team: string;
  home_score: number | null;
  away_score: number | null;
  // Poisson lambdas from context_json
  lambda_home: number | null;
  lambda_away: number | null;
  // 1x2 predictions
  prob_home: number | null;
  prob_draw: number | null;
  prob_away: number | null;
  fair_home: number | null;
  fair_draw: number | null;
  fair_away: number | null;
  // Totals
  prob_over25: number | null;
  prob_under25: number | null;
  // BTTS
  prob_btts_yes: number | null;
  prob_btts_no: number | null;
  // Asian Handicap
  ah_fair_line: number | null;
  ah_home_prob: number | null;
  ah_away_prob: number | null;
  ah_closing_line: number | null;
  ah_closing_home: number | null;
  ah_closing_away: number | null;
  // Closing odds for value detection
  closing_home: number | null;
  closing_draw: number | null;
  closing_away: number | null;
  closing_over25: number | null;
  closing_under25: number | null;
}

export interface FixtureRow {
  id: number;
  kickoff_at: string;
  status: string;
  league_key: string;
  league_name: string;
  home_team: string;
  away_team: string;
  home_score: number | null;
  away_score: number | null;
  home_xg: number | null;
  away_xg: number | null;
}

export interface FixtureDetail {
  fixture: Fixture;
  league: League;
  home_team: Team;
  away_team: Team;
  result: Result | null;
  stats: MatchStatistics | null;
  predictions: Prediction[];
  odds: OddsSnapshot[];
}

export interface AhLinePrediction {
  line: number;
  side: string;
  win: number;
  push: number;
  loss: number;
  effective_prob: number;
  fair_odds: number;
  closing_odds: number | null;
}

export interface TotalsLinePrediction {
  line: number;
  prob_over: number;
  prob_under: number;
  fair_over: number;
  fair_under: number;
  closing_over: number | null;
  closing_under: number | null;
}

export interface MarginBar {
  margin: number;
  probability: number;
}

export interface CalibrationBin {
  bin_start: number;
  bin_end: number;
  bin_label: string;
  predicted_avg: number;
  actual_rate: number;
  count: number;
}

export interface LeagueFixtureForChain {
  fixture_id: number;
  league_id: number;
  home_team_id: number;
  away_team_id: number;
  home_team: string;
  away_team: string;
  kickoff_at: string;
  status: string;
  home_score: number | null;
  away_score: number | null;
  ah_line: number | null;
  totals_line: number | null;
}

export interface StandingsRow {
  team_id: number;
  team_name: string;
  played: number;
  won: number;
  drawn: number;
  lost: number;
  gf: number;
  ga: number;
  gd: number;
  points: number;
  position: number;
}

export interface ProxyMatch {
  proxy_team_id: number;
  proxy_team_name: string;
  proxy_position: number;
  vs_a_fixture_id: number;
  vs_a_date: string;
  vs_a_home_team: string;
  vs_a_away_team: string;
  vs_a_score: string;
  vs_a_ah_line: number | null;
  vs_a_totals_line: number | null;
  vs_b_fixture_id: number;
  vs_b_date: string;
  vs_b_home_team: string;
  vs_b_away_team: string;
  vs_b_score: string;
  vs_b_ah_line: number | null;
  vs_b_totals_line: number | null;
  implied_ah_gap: number | null;
}

export interface TodayFixture {
  id: number;
  league_id: number;
  home_team_id: number;
  away_team_id: number;
  kickoff_at: string;
  status: string;
  league_key: string;
  league_name: string;
  home_team: string;
  away_team: string;
}

export interface LockedBet {
  id: number;
  bet_date: string;
  session: "morning" | "closing";
  fixture_id: number;
  category: string;
  side: string;
  home_team: string;
  away_team: string;
  kickoff: string;
  league_name: string;
  league_key: string;
  line: string;
  stake: number;
  ah_closing_line: number | null;
  totals_line: number | null;
  locked_at: string;
  // Settlement (written by settle API)
  home_score: number | null;
  away_score: number | null;
  result: "W" | "L" | "P" | null;
  pnl: number | null;
  settled_at: string | null;
}

export interface DailyPick {
  id: number;
  bet_date: string;
  fixture_id: number;
  category: string;
  side: string;
  home_team: string;
  away_team: string;
  kickoff: string;
  league_name: string;
  league_key: string;
  line: string;
  stake: number;
  assumed_odds: number;
  ah_closing_line: number | null;
  totals_line: number | null;
  signals_json: Record<string, unknown>;
  home_score: number | null;
  away_score: number | null;
  result: "W" | "L" | "P" | null;
  pnl: number | null;
  settled_at: string | null;
  computed_at: string;
  created_at: string;
  updated_at: string;
}

export interface DailyPnlSummary {
  bet_date: string;
  category: string;
  bets: number;
  total_stake: number;
  wins: number;
  losses: number;
  pushes: number;
  pending: number;
  total_pnl: number;
  roi_pct: number;
}
