"""XGBoost Poisson regressor for home/away goal prediction.

Uses all available features (auto-discovered from DataFrame) and fits
two XGBoost models with count:poisson objective. Includes Dixon-Coles
rho correction post-hoc.
"""

from __future__ import annotations

import base64
import logging
from datetime import UTC, datetime

import numpy as np
import polars as pl

from dhx.modeling.base import compute_market_probabilities, compute_score_grid, fit_rho

logger = logging.getLogger(__name__)

# Columns to exclude from features (metadata + labels)
_EXCLUDE_COLS = {
    "fixture_id",
    "home_team_id",
    "away_team_id",
    "kickoff_at",
    "league_id",
    "league_key",
    "label_home_score",
    "label_away_score",
    "label_total_goals",
    "label_btts",
    "label_result",
    "label_home_win",
    "label_draw",
    "label_away_win",
}

# Conservative hyperparameters for ~900 training samples
_DEFAULT_PARAMS = {
    "objective": "count:poisson",
    "max_depth": 4,
    "learning_rate": 0.05,
    "n_estimators": 300,
    "early_stopping_rounds": 20,
    "colsample_bytree": 0.6,
    "subsample": 0.8,
    "reg_alpha": 1.0,
    "reg_lambda": 5.0,
    "min_child_weight": 10,
    "random_state": 42,
    "verbosity": 0,
}


class XGBoostModel:
    """XGBoost Poisson regressor for home and away goals.

    Uses all ~145 features with early stopping. Same interface as PoissonModel.
    """

    def __init__(self, max_goals: int = 10) -> None:
        self.feature_columns: list[str] = []
        self.impute_medians: dict[str, float] = {}
        self.max_goals = max_goals
        self.rho: float = 0.0
        self.training_window: str = ""
        self.training_samples: int = 0
        self.fit_timestamp: str = ""
        self.convergence_info: dict = {}
        # Serialized XGBoost model bytes (set after fit or from_dict)
        self._xgb_home_raw: bytes | None = None
        self._xgb_away_raw: bytes | None = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pl.DataFrame,
        y_col_home: str = "label_home_score",
        y_col_away: str = "label_away_score",
    ) -> None:
        """Fit two XGBoost Poisson regressors (home + away goals)."""
        import xgboost as xgb

        df_clean = df.drop_nulls(subset=[y_col_home, y_col_away])
        n = len(df_clean)
        if n < 200:
            raise ValueError(f"Need >= 200 training fixtures, got {n}")

        # Auto-discover feature columns
        self.feature_columns = sorted(
            c for c in df_clean.columns if c not in _EXCLUDE_COLS
        )
        logger.info(f"Using {len(self.feature_columns)} features")

        # Compute medians for imputation
        self.impute_medians = _compute_medians(df_clean, self.feature_columns)

        # Extract feature matrix (no intercept — XGBoost handles bias)
        xmat = _extract_features_no_intercept(df_clean, self.feature_columns, self.impute_medians)
        y_home = df_clean[y_col_home].to_numpy().astype(np.float64)
        y_away = df_clean[y_col_away].to_numpy().astype(np.float64)

        # Chronological 80/20 split for early stopping
        split_idx = int(n * 0.8)
        x_train, x_val = xmat[:split_idx], xmat[split_idx:]
        yh_train, yh_val = y_home[:split_idx], y_home[split_idx:]
        ya_train, ya_val = y_away[:split_idx], y_away[split_idx:]

        logger.info(f"Early-stopping split: train={split_idx}, val={n - split_idx}")

        # Fit home model
        model_home = xgb.XGBRegressor(**_DEFAULT_PARAMS)
        model_home.fit(
            x_train,
            yh_train,
            eval_set=[(x_val, yh_val)],
            verbose=False,
        )
        best_home = model_home.best_iteration
        logger.info(f"Home XGBoost: best_iteration={best_home}")

        # Fit away model
        model_away = xgb.XGBRegressor(**_DEFAULT_PARAMS)
        model_away.fit(
            x_train,
            ya_train,
            eval_set=[(x_val, ya_val)],
            verbose=False,
        )
        best_away = model_away.best_iteration
        logger.info(f"Away XGBoost: best_iteration={best_away}")

        # Store raw model bytes for serialization
        self._xgb_home_raw = model_home.get_booster().save_raw(raw_format="json")
        self._xgb_away_raw = model_away.get_booster().save_raw(raw_format="json")

        self.training_samples = n
        self.fit_timestamp = datetime.now(UTC).isoformat()
        self.convergence_info = {
            "home_best_iteration": best_home,
            "away_best_iteration": best_away,
            "n_features": len(self.feature_columns),
        }

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
        """Predict lambda_home and lambda_away for each fixture."""
        import xgboost as xgb

        xmat = _extract_features_no_intercept(df, self.feature_columns, self.impute_medians)

        booster_home = xgb.Booster()
        booster_home.load_model(bytearray(self._xgb_home_raw))
        dmat = xgb.DMatrix(xmat, feature_names=self.feature_columns)
        lambda_h = booster_home.predict(dmat)

        booster_away = xgb.Booster()
        booster_away.load_model(bytearray(self._xgb_away_raw))
        lambda_a = booster_away.predict(dmat)

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
            "model_type": "xgboost",
            "feature_columns": self.feature_columns,
            "impute_medians": self.impute_medians,
            "max_goals": self.max_goals,
            "rho": self.rho,
            "training_window": self.training_window,
            "training_samples": self.training_samples,
            "fit_timestamp": self.fit_timestamp,
            "convergence_info": self.convergence_info,
            # XGBoost raw bytes as base64 strings for JSON compat
            "xgb_home_raw_b64": base64.b64encode(self._xgb_home_raw).decode()
            if self._xgb_home_raw
            else None,
            "xgb_away_raw_b64": base64.b64encode(self._xgb_away_raw).decode()
            if self._xgb_away_raw
            else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> XGBoostModel:
        m = cls(max_goals=d.get("max_goals", 10))
        m.feature_columns = d["feature_columns"]
        m.impute_medians = d["impute_medians"]
        m.rho = d.get("rho", 0.0)
        m.training_window = d.get("training_window", "")
        m.training_samples = d.get("training_samples", 0)
        m.fit_timestamp = d.get("fit_timestamp", "")
        m.convergence_info = d.get("convergence_info", {})
        m._xgb_home_raw = (
            base64.b64decode(d["xgb_home_raw_b64"]) if d.get("xgb_home_raw_b64") else None
        )
        m._xgb_away_raw = (
            base64.b64decode(d["xgb_away_raw_b64"]) if d.get("xgb_away_raw_b64") else None
        )
        return m


# ======================================================================
# Internal helpers
# ======================================================================


def _compute_medians(df: pl.DataFrame, columns: list[str]) -> dict[str, float]:
    """Compute column medians for imputation."""
    medians: dict[str, float] = {}
    for c in columns:
        if c in df.columns:
            med = df[c].median()
            medians[c] = float(med) if med is not None else 0.0
        else:
            medians[c] = 0.0
    return medians


def _extract_features_no_intercept(
    df: pl.DataFrame,
    columns: list[str],
    medians: dict[str, float],
) -> np.ndarray:
    """Extract feature matrix WITHOUT intercept. Median-impute nulls."""
    n = len(df)
    p = len(columns)
    xmat = np.zeros((n, p), dtype=np.float64)

    for i, c in enumerate(columns):
        if c in df.columns:
            vals = df[c].to_numpy().astype(np.float64)
            mask = np.isnan(vals)
            vals[mask] = medians.get(c, 0.0)
            xmat[:, i] = vals
        else:
            xmat[:, i] = medians.get(c, 0.0)

    return xmat
