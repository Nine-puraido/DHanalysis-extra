"""Extended market math: Asian Handicap (Skellam) and multi-line totals (Poisson).

Uses scipy.stats for Skellam and Poisson distributions.
All functions take lambda_home / lambda_away as inputs — no model retraining needed.
"""

from __future__ import annotations

from scipy.stats import poisson, skellam


def ah_probability(
    lambda_h: float,
    lambda_a: float,
    line: float,
    side: str = "home",
) -> dict:
    """Compute AH probabilities for a given handicap line.

    Args:
        lambda_h: expected home goals.
        lambda_a: expected away goals.
        line: handicap line from home perspective (e.g. -0.5, -1, -0.25).
        side: "home" or "away".

    Returns:
        {win, push, loss, effective_prob, fair_odds} for the requested side.
    """
    # Quarter lines are split into two half-line bets
    frac = line - int(line) if line >= 0 else line - int(line)
    # Normalise fractional part
    frac_abs = abs(frac)

    if frac_abs in (0.25, 0.75):
        return _quarter_line(lambda_h, lambda_a, line, side)

    return _single_line(lambda_h, lambda_a, line, side)


def _single_line(
    lambda_h: float,
    lambda_a: float,
    line: float,
    side: str,
) -> dict:
    """AH probability for whole or half lines (no split stake)."""
    # Home side: home wins if (home_goals - away_goals) > -line
    # i.e. goal_diff > -line (from home perspective, line is the handicap applied to home)
    threshold = -line  # home needs diff > threshold to win

    if threshold == int(threshold):
        # Whole line — push possible
        t = int(threshold)
        p_win = 1.0 - skellam.cdf(t, lambda_h, lambda_a)  # P(diff > t)
        p_push = float(skellam.pmf(t, lambda_h, lambda_a))  # P(diff == t)
        p_loss = float(skellam.cdf(t - 1, lambda_h, lambda_a))  # P(diff < t)
    else:
        # Half line — no push
        t = int(threshold) if threshold > 0 else int(threshold) - 1
        # For half lines, diff > threshold is equivalent to diff >= ceil(threshold)
        import math

        ceil_t = math.ceil(threshold)
        p_win = 1.0 - skellam.cdf(ceil_t - 1, lambda_h, lambda_a)  # P(diff >= ceil_t)
        p_push = 0.0
        p_loss = float(skellam.cdf(ceil_t - 1, lambda_h, lambda_a))  # P(diff < ceil_t)

    if side == "away":
        p_win, p_loss = p_loss, p_win

    eff = float(p_win) + 0.5 * float(p_push)
    eff = max(min(eff, 0.999), 0.001)

    return {
        "win": round(float(p_win), 6),
        "push": round(float(p_push), 6),
        "loss": round(float(p_loss), 6),
        "effective_prob": round(eff, 6),
        "fair_odds": round(1.0 / eff, 4),
    }


def _quarter_line(
    lambda_h: float,
    lambda_a: float,
    line: float,
    side: str,
) -> dict:
    """Quarter-line AH: average of two adjacent half-line results (split stake)."""
    import math

    # -0.25 splits into 0 and -0.5; -0.75 splits into -0.5 and -1
    # +0.25 splits into 0 and +0.5; +0.75 splits into +0.5 and +1
    lower = math.floor(line * 2) / 2
    upper = math.ceil(line * 2) / 2

    r1 = _single_line(lambda_h, lambda_a, lower, side)
    r2 = _single_line(lambda_h, lambda_a, upper, side)

    win = (r1["win"] + r2["win"]) / 2
    push = (r1["push"] + r2["push"]) / 2
    loss = (r1["loss"] + r2["loss"]) / 2
    eff = win + 0.5 * push
    eff = max(min(eff, 0.999), 0.001)

    return {
        "win": round(win, 6),
        "push": round(push, 6),
        "loss": round(loss, 6),
        "effective_prob": round(eff, 6),
        "fair_odds": round(1.0 / eff, 4),
    }


def totals_probability(
    lambda_h: float,
    lambda_a: float,
    line: float,
) -> dict:
    """Compute over/under probabilities for a given total goals line.

    Args:
        lambda_h: expected home goals.
        lambda_a: expected away goals.
        line: total goals line (e.g. 2.5, 3.5).

    Returns:
        {over, under, push, fair_over, fair_under}.
    """
    total_lambda = lambda_h + lambda_a

    if line == int(line):
        # Whole line — push possible
        t = int(line)
        p_under = float(poisson.cdf(t - 1, total_lambda))  # P(goals < t)
        p_push = float(poisson.pmf(t, total_lambda))  # P(goals == t)
        p_over = 1.0 - p_under - p_push  # P(goals > t)
    else:
        # Half line — no push
        import math

        t = math.floor(line)
        p_under = float(poisson.cdf(t, total_lambda))  # P(goals <= floor(line))
        p_push = 0.0
        p_over = 1.0 - p_under

    p_over = max(min(p_over, 0.999), 0.001)
    p_under = max(min(p_under, 0.999), 0.001)

    return {
        "over": round(p_over, 6),
        "under": round(p_under, 6),
        "push": round(p_push, 6),
        "fair_over": round(1.0 / p_over, 4) if p_over > 0 else None,
        "fair_under": round(1.0 / p_under, 4) if p_under > 0 else None,
    }


def find_fair_ah_line(
    lambda_h: float,
    lambda_a: float,
    step: float = 0.25,
) -> float:
    """Find the AH line where home effective_prob is closest to 50%.

    Searches lines from -5 to +5 in increments of `step`.
    """
    best_line = 0.0
    best_diff = float("inf")

    line = -5.0
    while line <= 5.0:
        r = ah_probability(lambda_h, lambda_a, line, side="home")
        diff = abs(r["effective_prob"] - 0.5)
        if diff < best_diff:
            best_diff = diff
            best_line = line
        line += step

    return best_line


def margin_distribution(
    lambda_h: float,
    lambda_a: float,
    max_diff: int = 8,
) -> dict[int, float]:
    """P(home_goals - away_goals = k) for k in [-max_diff, +max_diff].

    Uses Skellam PMF directly.
    """
    dist: dict[int, float] = {}
    for k in range(-max_diff, max_diff + 1):
        dist[k] = round(float(skellam.pmf(k, lambda_h, lambda_a)), 6)
    return dist


def compute_extended_markets(
    lambda_h: float,
    lambda_a: float,
    ah_lines: list[float],
    totals_lines: list[float],
) -> dict:
    """All-in-one: margin dist + fair AH line + all AH/totals line probs.

    Args:
        lambda_h: expected home goals.
        lambda_a: expected away goals.
        ah_lines: list of AH lines from odds data.
        totals_lines: list of totals lines from odds data.

    Returns:
        {
            margin_distribution: {k: prob},
            fair_ah_line: float,
            ah: {line: {home: {...}, away: {...}}},
            totals: {line: {over, under, ...}},
        }
    """
    fair_line = find_fair_ah_line(lambda_h, lambda_a)

    # Include fair line in AH lines if not already present
    all_ah_lines = sorted(set(ah_lines) | {fair_line})

    ah_results: dict[float, dict] = {}
    for line in all_ah_lines:
        ah_results[line] = {
            "home": ah_probability(lambda_h, lambda_a, line, "home"),
            "away": ah_probability(lambda_h, lambda_a, line, "away"),
        }

    totals_results: dict[float, dict] = {}
    for line in sorted(set(totals_lines)):
        totals_results[line] = totals_probability(lambda_h, lambda_a, line)

    return {
        "margin_distribution": margin_distribution(lambda_h, lambda_a),
        "fair_ah_line": fair_line,
        "ah": ah_results,
        "totals": totals_results,
    }
