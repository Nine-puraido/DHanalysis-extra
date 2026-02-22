"""Shared score-grid and market-probability utilities with Dixon-Coles correction.

All model classes (Poisson, XGBoost, Ensemble) delegate to these functions
so that the Dixon-Coles tau adjustment and market derivation logic lives in
one place.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

# Pre-compute log factorials for score grid (0! through 10!)
_LOG_FACTORIAL = np.array([math.lgamma(k + 1) for k in range(11)])


# ------------------------------------------------------------------
# Dixon-Coles tau correction
# ------------------------------------------------------------------


def _tau(x: int, y: int, lh: float, la: float, rho: float) -> float:
    """Dixon-Coles correction factor for low-scoring outcomes.

    Adjusts P(x,y) for (0,0), (1,0), (0,1), (1,1) to fix the independence
    assumption's under/over-prediction of draws and low scores.

    Returns a multiplicative factor (>0).
    """
    if x == 0 and y == 0:
        return 1.0 - lh * la * rho
    if x == 1 and y == 0:
        return 1.0 + la * rho
    if x == 0 and y == 1:
        return 1.0 + lh * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


# ------------------------------------------------------------------
# Score grid
# ------------------------------------------------------------------


def compute_score_grid(
    lh: float, la: float, max_goals: int = 10, rho: float = 0.0
) -> np.ndarray:
    """Compute (max_goals+1) x (max_goals+1) joint probability matrix.

    Applies Dixon-Coles tau correction for scores (0,0), (1,0), (0,1), (1,1)
    and re-normalizes.
    """
    mg = max_goals + 1
    log_lam_h = math.log(max(lh, 1e-10))
    log_lam_a = math.log(max(la, 1e-10))

    log_pmf_h = np.array([-lh + k * log_lam_h - _LOG_FACTORIAL[k] for k in range(mg)])
    log_pmf_a = np.array([-la + k * log_lam_a - _LOG_FACTORIAL[k] for k in range(mg)])

    # Outer sum in log-space, then exponentiate
    log_grid = log_pmf_h[:, None] + log_pmf_a[None, :]
    grid = np.exp(log_grid)

    # Apply Dixon-Coles tau for low scores
    if rho != 0.0:
        for i in range(min(2, mg)):
            for j in range(min(2, mg)):
                grid[i, j] *= _tau(i, j, lh, la, rho)

    # Re-normalize
    grid /= grid.sum()
    return grid


# ------------------------------------------------------------------
# Market probabilities
# ------------------------------------------------------------------


def _clip_prob(p: float) -> float:
    """Clip probability to (0.001, 0.999) for DB constraint."""
    return max(0.001, min(0.999, p))


def compute_market_probabilities(
    lh: float, la: float, max_goals: int = 10, rho: float = 0.0
) -> dict:
    """Derive 1x2, totals o/u 2.5, and BTTS probabilities from score grid."""
    grid = compute_score_grid(lh, la, max_goals, rho)
    mg = max_goals + 1

    p_home = 0.0
    p_draw = 0.0
    p_away = 0.0
    p_over25 = 0.0
    p_btts_yes = 0.0

    for i in range(mg):
        for j in range(mg):
            p = grid[i, j]
            if i > j:
                p_home += p
            elif i == j:
                p_draw += p
            else:
                p_away += p
            if i + j >= 3:
                p_over25 += p
            if i >= 1 and j >= 1:
                p_btts_yes += p

    return {
        "1x2": {
            "home": _clip_prob(p_home),
            "draw": _clip_prob(p_draw),
            "away": _clip_prob(p_away),
        },
        "totals": {
            "over": _clip_prob(p_over25),
            "under": _clip_prob(1.0 - p_over25),
        },
        "btts": {
            "yes": _clip_prob(p_btts_yes),
            "no": _clip_prob(1.0 - p_btts_yes),
        },
    }


# ------------------------------------------------------------------
# Rho fitting via profile likelihood
# ------------------------------------------------------------------


def fit_rho(
    predict_fn,
    df: pl.DataFrame,
    y_col_home: str = "label_home_score",
    y_col_away: str = "label_away_score",
    max_goals: int = 10,
) -> float:
    """Estimate Dixon-Coles rho via profile likelihood.

    Given a function that predicts lambdas (predict_fn: DataFrame -> DataFrame
    with lambda_home, lambda_away columns), search rho in [-0.15, 0.15] that
    maximizes the log-likelihood of observed scores.

    Returns the fitted rho value.
    """
    from scipy.optimize import minimize_scalar

    lambdas_df = predict_fn(df)
    lh_arr = lambdas_df["lambda_home"].to_numpy()
    la_arr = lambdas_df["lambda_away"].to_numpy()
    yh = df[y_col_home].to_numpy().astype(int)
    ya = df[y_col_away].to_numpy().astype(int)

    def neg_log_likelihood(rho: float) -> float:
        ll = 0.0
        for i in range(len(yh)):
            grid = compute_score_grid(float(lh_arr[i]), float(la_arr[i]), max_goals, rho)
            hi, ai = int(yh[i]), int(ya[i])
            if hi <= max_goals and ai <= max_goals:
                p = grid[hi, ai]
            else:
                p = 1e-10
            ll += math.log(max(p, 1e-10))
        return -ll

    result = minimize_scalar(neg_log_likelihood, bounds=(-0.15, 0.15), method="bounded")
    rho = float(result.x)
    logger.info(f"Dixon-Coles rho={rho:.6f} (neg_ll={result.fun:.4f})")
    return rho
