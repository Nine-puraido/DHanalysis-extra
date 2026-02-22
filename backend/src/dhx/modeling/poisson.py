"""Poisson GLM fitted via IRLS for home/away goal prediction.

Includes Dixon-Coles rho correction for low-scoring outcomes.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import numpy as np
import polars as pl

from dhx.modeling.base import compute_market_probabilities, compute_score_grid, fit_rho

logger = logging.getLogger(__name__)

# Features selected to predict home goals (lambda_home)
FEATURES_HOME = [
    "home_form_xg_for_r5",
    "home_form_goals_for_r5",
    "away_form_goals_against_r5",
    "home_venue_form_goals_for_r5",
    "delta_form_xg_for_r5",
    "home_ppg",
    "mkt_1x2_home_prob",
    "home_days_since_last",
]

# Features selected to predict away goals (lambda_away)
FEATURES_AWAY = [
    "away_form_xg_for_r5",
    "away_form_goals_for_r5",
    "home_form_goals_against_r5",
    "away_venue_form_goals_for_r5",
    "delta_form_xg_against_r5",
    "away_ppg",
    "mkt_1x2_away_prob",
    "away_days_since_last",
]


class PoissonModel:
    """Independent Poisson GLM for home and away goals.

    Fitted via Iteratively Reweighted Least Squares (IRLS).
    Produces market probabilities for 1x2, totals o/u 2.5, and BTTS.
    Includes Dixon-Coles rho correction for score correlation.
    """

    def __init__(self, max_goals: int = 10) -> None:
        self.feature_columns_home: list[str] = list(FEATURES_HOME)
        self.feature_columns_away: list[str] = list(FEATURES_AWAY)
        self.beta_home: list[float] = []
        self.beta_away: list[float] = []
        self.impute_medians_home: dict[str, float] = {}
        self.impute_medians_away: dict[str, float] = {}
        self.max_goals = max_goals
        self.rho: float = 0.0
        self.training_window: str = ""
        self.training_samples: int = 0
        self.fit_timestamp: str = ""
        self.convergence_info: dict = {}

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pl.DataFrame,
        y_col_home: str = "label_home_score",
        y_col_away: str = "label_away_score",
    ) -> None:
        """Fit home and away Poisson GLMs on the training data."""
        # Drop rows where labels are null
        df_clean = df.drop_nulls(subset=[y_col_home, y_col_away])
        n = len(df_clean)
        if n < 200:
            raise ValueError(f"Need >= 200 training fixtures, got {n}")
        logger.info(f"Fitting on {n} fixtures")

        # Compute and store medians for imputation (from training set only)
        self.impute_medians_home = _compute_medians(df_clean, self.feature_columns_home)
        self.impute_medians_away = _compute_medians(df_clean, self.feature_columns_away)

        # Extract numpy arrays
        xmat_home = _extract_features(df_clean, self.feature_columns_home, self.impute_medians_home)
        y_home = df_clean[y_col_home].to_numpy().astype(np.float64)

        xmat_away = _extract_features(df_clean, self.feature_columns_away, self.impute_medians_away)
        y_away = df_clean[y_col_away].to_numpy().astype(np.float64)

        # Fit via IRLS
        self.beta_home, conv_home = _irls_fit(xmat_home, y_home)
        self.beta_away, conv_away = _irls_fit(xmat_away, y_away)

        self.beta_home = self.beta_home.tolist()
        self.beta_away = self.beta_away.tolist()
        self.training_samples = n
        self.fit_timestamp = datetime.now(UTC).isoformat()
        self.convergence_info = {"home": conv_home, "away": conv_away}

        logger.info(
            f"Home model: {conv_home['iterations']} iters, deviance={conv_home['deviance']:.4f}"
        )
        logger.info(
            f"Away model: {conv_away['iterations']} iters, deviance={conv_away['deviance']:.4f}"
        )

        # Fit Dixon-Coles rho
        logger.info("Fitting Dixon-Coles rho...")
        self.rho = fit_rho(
            self.predict_lambdas, df_clean, y_col_home, y_col_away, self.max_goals
        )
        self.convergence_info["rho"] = round(self.rho, 6)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_lambdas(self, df: pl.DataFrame) -> pl.DataFrame:
        """Predict lambda_home and lambda_away for each fixture.

        Returns a DataFrame with columns: fixture_id, lambda_home, lambda_away.
        """
        xmat_home = _extract_features(df, self.feature_columns_home, self.impute_medians_home)
        xmat_away = _extract_features(df, self.feature_columns_away, self.impute_medians_away)

        beta_h = np.array(self.beta_home)
        beta_a = np.array(self.beta_away)

        eta_h = xmat_home @ beta_h
        eta_a = xmat_away @ beta_a
        lambda_h = np.exp(np.clip(eta_h, -20.0, 20.0))
        lambda_a = np.exp(np.clip(eta_a, -20.0, 20.0))

        return pl.DataFrame(
            {
                "fixture_id": df["fixture_id"],
                "lambda_home": lambda_h,
                "lambda_away": lambda_a,
            }
        )

    def score_grid(self, lambda_h: float, lambda_a: float) -> np.ndarray:
        """Compute joint probability matrix with Dixon-Coles correction."""
        return compute_score_grid(lambda_h, lambda_a, self.max_goals, self.rho)

    def market_probabilities(self, lambda_h: float, lambda_a: float) -> dict:
        """Derive 1x2, totals o/u 2.5, and BTTS probabilities from score grid."""
        return compute_market_probabilities(lambda_h, lambda_a, self.max_goals, self.rho)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "model_type": "poisson",
            "feature_columns_home": self.feature_columns_home,
            "feature_columns_away": self.feature_columns_away,
            "beta_home": self.beta_home,
            "beta_away": self.beta_away,
            "impute_medians_home": self.impute_medians_home,
            "impute_medians_away": self.impute_medians_away,
            "max_goals": self.max_goals,
            "rho": self.rho,
            "training_window": self.training_window,
            "training_samples": self.training_samples,
            "fit_timestamp": self.fit_timestamp,
            "convergence_info": self.convergence_info,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PoissonModel:
        m = cls(max_goals=d.get("max_goals", 10))
        m.feature_columns_home = d["feature_columns_home"]
        m.feature_columns_away = d["feature_columns_away"]
        m.beta_home = d["beta_home"]
        m.beta_away = d["beta_away"]
        m.impute_medians_home = d["impute_medians_home"]
        m.impute_medians_away = d["impute_medians_away"]
        m.rho = d.get("rho", 0.0)  # backward compat with old artifacts
        m.training_window = d.get("training_window", "")
        m.training_samples = d.get("training_samples", 0)
        m.fit_timestamp = d.get("fit_timestamp", "")
        m.convergence_info = d.get("convergence_info", {})
        return m


# ======================================================================
# Internal helpers
# ======================================================================


def _compute_medians(df: pl.DataFrame, columns: list[str]) -> dict[str, float]:
    """Compute column medians for imputation, handling all-null gracefully."""
    medians: dict[str, float] = {}
    for c in columns:
        if c in df.columns:
            med = df[c].median()
            medians[c] = float(med) if med is not None else 0.0
        else:
            medians[c] = 0.0
    return medians


def _extract_features(
    df: pl.DataFrame,
    columns: list[str],
    medians: dict[str, float],
) -> np.ndarray:
    """Extract feature matrix with intercept column prepended. Median-impute nulls."""
    n = len(df)
    p = len(columns)
    xmat = np.ones((n, p + 1), dtype=np.float64)  # col-0 = intercept

    for i, c in enumerate(columns):
        if c in df.columns:
            vals = df[c].to_numpy().astype(np.float64)
            mask = np.isnan(vals)
            vals[mask] = medians.get(c, 0.0)
            xmat[:, i + 1] = vals
        else:
            xmat[:, i + 1] = medians.get(c, 0.0)

    return xmat


def _poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    """Compute Poisson deviance: 2 * sum(y*log(y/mu) - (y - mu))."""
    safe_mu = np.maximum(mu, 1e-10)
    # Handle y=0 case: 0*log(0/mu) = 0
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / safe_mu), 0.0) - (y - safe_mu)
    return float(2.0 * np.sum(term))


def _irls_fit(
    xmat: np.ndarray,
    y: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-6,
    reg: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """Fit Poisson GLM via IRLS.

    Returns (beta, convergence_info).
    """
    _n, p = xmat.shape
    beta = np.zeros(p, dtype=np.float64)
    prev_deviance = float("inf")

    for iteration in range(1, max_iter + 1):
        eta = xmat @ beta
        eta = np.clip(eta, -20.0, 20.0)
        mu = np.exp(eta)
        mu = np.clip(mu, 1e-10, 1e6)

        # Working weights and response
        w = mu
        z = eta + (y - mu) / mu

        # Weighted least squares: (X'WX + reg*I) beta = X'Wz
        wmat = np.diag(w)
        xtwx = xmat.T @ wmat @ xmat + reg * np.eye(p)
        xtwz = xmat.T @ (w * z)
        beta = np.linalg.solve(xtwx, xtwz)

        deviance = _poisson_deviance(
            y, np.exp(np.clip(xmat @ beta, -20.0, 20.0))
        )
        delta = abs(prev_deviance - deviance)

        if delta < tol:
            logger.debug(f"IRLS converged at iter {iteration}, dev={deviance:.4f}")
            return beta, {
                "iterations": iteration,
                "deviance": round(deviance, 6),
                "converged": True,
            }

        prev_deviance = deviance

    logger.warning(f"IRLS did not converge after {max_iter} iters, dev={deviance:.4f}")
    return beta, {
        "iterations": max_iter,
        "deviance": round(deviance, 6),
        "converged": False,
    }
