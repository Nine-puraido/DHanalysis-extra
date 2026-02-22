"""Ensemble model blending Poisson GLM and XGBoost at the lambda level.

lambda_ensemble = w * lambda_poisson + (1-w) * lambda_xgboost

Weight w is optimized on training data to minimize Poisson deviance.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import numpy as np
import polars as pl

from dhx.modeling.base import compute_market_probabilities, compute_score_grid, fit_rho
from dhx.modeling.poisson import PoissonModel
from dhx.modeling.xgboost_model import XGBoostModel

logger = logging.getLogger(__name__)


class EnsembleModel:
    """Weighted lambda average of Poisson GLM and XGBoost.

    Same interface as PoissonModel for predict_lambdas / market_probabilities.
    """

    def __init__(self, max_goals: int = 10) -> None:
        self.poisson = PoissonModel(max_goals=max_goals)
        self.xgboost = XGBoostModel(max_goals=max_goals)
        self.weight_poisson: float = 0.5
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
        """Fit both sub-models, then optimize blending weight."""
        from scipy.optimize import minimize_scalar

        df_clean = df.drop_nulls(subset=[y_col_home, y_col_away])
        n = len(df_clean)
        if n < 200:
            raise ValueError(f"Need >= 200 training fixtures, got {n}")

        # Fit both sub-models
        logger.info("Fitting Poisson sub-model...")
        self.poisson.fit(df_clean, y_col_home, y_col_away)

        logger.info("Fitting XGBoost sub-model...")
        self.xgboost.fit(df_clean, y_col_home, y_col_away)

        # Get lambdas from both models on training data
        lam_p = self.poisson.predict_lambdas(df_clean)
        lam_x = self.xgboost.predict_lambdas(df_clean)

        lh_p = lam_p["lambda_home"].to_numpy()
        la_p = lam_p["lambda_away"].to_numpy()
        lh_x = lam_x["lambda_home"].to_numpy()
        la_x = lam_x["lambda_away"].to_numpy()

        yh = df_clean[y_col_home].to_numpy().astype(np.float64)
        ya = df_clean[y_col_away].to_numpy().astype(np.float64)

        # Optimize weight via Poisson deviance on training data
        def neg_deviance(w: float) -> float:
            lh = w * lh_p + (1 - w) * lh_x
            la = w * la_p + (1 - w) * la_x
            # Poisson deviance: 2 * sum(y*log(y/mu) - (y - mu))
            dev = 0.0
            for arr_y, arr_mu in [(yh, lh), (ya, la)]:
                mu = np.maximum(arr_mu, 1e-10)
                with np.errstate(divide="ignore", invalid="ignore"):
                    term = np.where(arr_y > 0, arr_y * np.log(arr_y / mu), 0.0) - (
                        arr_y - mu
                    )
                dev += float(2.0 * np.sum(term))
            return dev

        result = minimize_scalar(neg_deviance, bounds=(0.0, 1.0), method="bounded")
        self.weight_poisson = float(result.x)
        logger.info(
            f"Ensemble weight: poisson={self.weight_poisson:.4f}, "
            f"xgboost={1 - self.weight_poisson:.4f}"
        )

        self.training_samples = n
        self.fit_timestamp = datetime.now(UTC).isoformat()
        self.convergence_info = {
            "weight_poisson": round(self.weight_poisson, 6),
            "weight_xgboost": round(1 - self.weight_poisson, 6),
            "blend_deviance": round(result.fun, 6),
            "poisson": self.poisson.convergence_info,
            "xgboost": self.xgboost.convergence_info,
        }

        # Fit rho on blended lambdas
        logger.info("Fitting Dixon-Coles rho on blended lambdas...")
        self.rho = fit_rho(
            self.predict_lambdas, df_clean, y_col_home, y_col_away, self.max_goals
        )
        self.convergence_info["rho"] = round(self.rho, 6)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_lambdas(self, df: pl.DataFrame) -> pl.DataFrame:
        """Predict blended lambda_home and lambda_away."""
        lam_p = self.poisson.predict_lambdas(df)
        lam_x = self.xgboost.predict_lambdas(df)

        w = self.weight_poisson
        lambda_h = w * lam_p["lambda_home"].to_numpy() + (1 - w) * lam_x["lambda_home"].to_numpy()
        lambda_a = w * lam_p["lambda_away"].to_numpy() + (1 - w) * lam_x["lambda_away"].to_numpy()

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
            "model_type": "ensemble",
            "weight_poisson": self.weight_poisson,
            "max_goals": self.max_goals,
            "rho": self.rho,
            "training_window": self.training_window,
            "training_samples": self.training_samples,
            "fit_timestamp": self.fit_timestamp,
            "convergence_info": self.convergence_info,
            "poisson": self.poisson.to_dict(),
            "xgboost": self.xgboost.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> EnsembleModel:
        m = cls(max_goals=d.get("max_goals", 10))
        m.weight_poisson = d.get("weight_poisson", 0.5)
        m.rho = d.get("rho", 0.0)
        m.training_window = d.get("training_window", "")
        m.training_samples = d.get("training_samples", 0)
        m.fit_timestamp = d.get("fit_timestamp", "")
        m.convergence_info = d.get("convergence_info", {})
        m.poisson = PoissonModel.from_dict(d["poisson"])
        m.xgboost = XGBoostModel.from_dict(d["xgboost"])
        return m
