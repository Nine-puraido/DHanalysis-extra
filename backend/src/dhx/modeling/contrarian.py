"""Contrarian model: XGBoost base + ridge regression reversion correction.

Wraps an XGBoost model and applies a learned residual correction based on
mean-reversion signals (xG overperformance, form spikes, trap indicators).
When reversion signals are strong, the correction pulls lambdas back toward
the league average.

Training flow:
    1. Fit base XGBoost on training data
    2. Predict base lambdas on training data
    3. Compute residuals (actual - base_lambda)
    4. Compute 19 reversion features from training data
    5. Fit ridge regression: reversion_features -> residual
    6. Fit Dixon-Coles rho on adjusted lambdas

Prediction flow:
    1. Get base lambdas from XGBoost
    2. Compute reversion features
    3. correction = X @ beta
    4. lambda_adj = max(0.15, base + correction)
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import numpy as np
import polars as pl

from dhx.modeling.base import compute_market_probabilities, compute_score_grid, fit_rho
from dhx.modeling.xgboost_model import XGBoostModel

logger = logging.getLogger(__name__)

# 19 reversion features: 7 per-team x 2 sides + 5 match-level
_REVERSION_FEATURES: list[tuple[str, str]] = [
    # Per-team mean reversion (home side)
    ("home_xg_overperf", "home"),
    ("home_goals_spike", "home"),
    ("home_win_spike", "home"),
    ("home_points_spike", "home"),
    ("home_shot_acc_spike", "home"),
    ("home_cs_fragility", "home"),
    ("home_scoring_streak", "home"),
    # Per-team mean reversion (away side)
    ("away_xg_overperf", "away"),
    ("away_goals_spike", "away"),
    ("away_win_spike", "away"),
    ("away_points_spike", "away"),
    ("away_shot_acc_spike", "away"),
    ("away_cs_fragility", "away"),
    ("away_scoring_streak", "away"),
    # Match-level trap signals
    ("delta_xg_extreme", "match"),
    ("consensus_home", "match"),
    ("combined_scoring", "match"),
    ("ppg_gap", "match"),
    ("h2h_dominance", "match"),
]

_REVERSION_NAMES = [name for name, _ in _REVERSION_FEATURES]

# Source columns used to derive reversion features
_HOME_COLS = {
    "form_goals_for_r5": "home_form_goals_for_r5",
    "form_goals_for_r3": "home_form_goals_for_r3",
    "form_goals_for_r10": "home_form_goals_for_r10",
    "form_xg_for_r5": "home_form_xg_for_r5",
    "form_xg_against_r5": "home_form_xg_against_r5",
    "form_win_r3": "home_form_win_r3",
    "form_win_r10": "home_form_win_r10",
    "form_points_r5": "home_form_points_r5",
    "form_points_r10": "home_form_points_r10",
    "form_shot_accuracy_r3": "home_form_shot_accuracy_r3",
    "form_shot_accuracy_r10": "home_form_shot_accuracy_r10",
    "form_clean_sheet_r3": "home_form_clean_sheet_r3",
}
_AWAY_COLS = {k: v.replace("home_", "away_") for k, v in _HOME_COLS.items()}

# Ridge regularization strength
_RIDGE_ALPHA = 10.0


class ContrarianModel:
    """XGBoost base model with ridge regression reversion correction.

    Same interface as PoissonModel/XGBoostModel for predict_lambdas,
    market_probabilities, to_dict, from_dict.
    """

    def __init__(self, max_goals: int = 10) -> None:
        self.base_model = XGBoostModel(max_goals=max_goals)
        self.reversion_feature_names: list[str] = list(_REVERSION_NAMES)
        self.reversion_beta_home: list[float] = []  # length 20: intercept + 19
        self.reversion_beta_away: list[float] = []
        self.reversion_medians: dict[str, float] = {}
        self.league_avg_home: float = 0.0
        self.league_avg_away: float = 0.0
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
        """Fit base XGBoost + reversion correction layer."""
        df_clean = df.drop_nulls(subset=[y_col_home, y_col_away])
        n = len(df_clean)
        if n < 200:
            raise ValueError(f"Need >= 200 training fixtures, got {n}")

        # League averages (used for context, not directly in correction)
        self.league_avg_home = float(df_clean[y_col_home].mean())
        self.league_avg_away = float(df_clean[y_col_away].mean())

        # 1. Fit base XGBoost model
        logger.info("Fitting base XGBoost model...")
        self.base_model.fit(df_clean, y_col_home, y_col_away)

        # 2. Get base lambdas on training data
        base_lambdas = self.base_model.predict_lambdas(df_clean)
        base_lh = base_lambdas["lambda_home"].to_numpy().astype(np.float64)
        base_la = base_lambdas["lambda_away"].to_numpy().astype(np.float64)

        # 3. Compute residuals: actual - base_lambda
        y_home = df_clean[y_col_home].to_numpy().astype(np.float64)
        y_away = df_clean[y_col_away].to_numpy().astype(np.float64)
        residual_home = y_home - base_lh
        residual_away = y_away - base_la

        # 4. Compute reversion features
        rev_matrix, medians = self._compute_reversion_features(df_clean, fit=True)
        self.reversion_medians = medians

        # 5. Ridge regression: features -> residuals
        # X = [1 | reversion_features], shape (n, 20)
        ones = np.ones((n, 1), dtype=np.float64)
        X = np.column_stack([ones, rev_matrix])  # (n, 20)

        alpha = _RIDGE_ALPHA
        I = np.eye(X.shape[1], dtype=np.float64)
        I[0, 0] = 0.0  # don't regularize intercept

        XtX = X.T @ X + alpha * I

        self.reversion_beta_home = (np.linalg.solve(XtX, X.T @ residual_home)).tolist()
        self.reversion_beta_away = (np.linalg.solve(XtX, X.T @ residual_away)).tolist()

        # Log correction magnitude
        correction_home = X @ np.array(self.reversion_beta_home)
        correction_away = X @ np.array(self.reversion_beta_away)
        logger.info(
            f"Reversion correction: home mean={correction_home.mean():.4f} "
            f"std={correction_home.std():.4f}, "
            f"away mean={correction_away.mean():.4f} "
            f"std={correction_away.std():.4f}"
        )

        self.training_samples = n
        self.fit_timestamp = datetime.now(UTC).isoformat()

        # 6. Fit Dixon-Coles rho on adjusted lambdas
        logger.info("Fitting Dixon-Coles rho on adjusted lambdas...")
        self.rho = fit_rho(
            self.predict_lambdas, df_clean, y_col_home, y_col_away, self.max_goals
        )

        self.convergence_info = {
            "base_xgboost": self.base_model.convergence_info,
            "n_reversion_features": len(self.reversion_feature_names),
            "ridge_alpha": _RIDGE_ALPHA,
            "correction_home_std": round(float(correction_home.std()), 6),
            "correction_away_std": round(float(correction_away.std()), 6),
            "league_avg_home": round(self.league_avg_home, 4),
            "league_avg_away": round(self.league_avg_away, 4),
            "rho": round(self.rho, 6),
        }

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_lambdas(self, df: pl.DataFrame) -> pl.DataFrame:
        """Predict adjusted lambda_home and lambda_away."""
        # 1. Base lambdas from XGBoost
        base_lambdas = self.base_model.predict_lambdas(df)
        base_lh = base_lambdas["lambda_home"].to_numpy().astype(np.float64)
        base_la = base_lambdas["lambda_away"].to_numpy().astype(np.float64)

        # 2. Compute reversion features
        rev_matrix, _ = self._compute_reversion_features(df, fit=False)

        # 3. correction = X @ beta
        n = len(df)
        ones = np.ones((n, 1), dtype=np.float64)
        X = np.column_stack([ones, rev_matrix])

        beta_h = np.array(self.reversion_beta_home)
        beta_a = np.array(self.reversion_beta_away)
        correction_h = X @ beta_h
        correction_a = X @ beta_a

        # 4. Adjusted lambdas, floor at 0.15
        lambda_h = np.maximum(0.15, base_lh + correction_h)
        lambda_a = np.maximum(0.15, base_la + correction_a)

        return pl.DataFrame(
            {
                "fixture_id": df["fixture_id"],
                "lambda_home": lambda_h,
                "lambda_away": lambda_a,
            }
        )

    def score_grid(self, lambda_h: float, lambda_a: float) -> np.ndarray:
        return compute_score_grid(lambda_h, lambda_a, self.max_goals, self.rho)

    def market_probabilities(self, lambda_h: float, lambda_a: float) -> dict:
        return compute_market_probabilities(lambda_h, lambda_a, self.max_goals, self.rho)

    # ------------------------------------------------------------------
    # Reversion feature computation
    # ------------------------------------------------------------------

    def _compute_reversion_features(
        self, df: pl.DataFrame, fit: bool = False
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Extract 19 reversion features from the DataFrame.

        Args:
            df: input DataFrame with existing feature columns
            fit: if True, compute and store medians; if False, use stored medians

        Returns:
            (feature_matrix of shape (n, 19), medians dict)
        """
        n = len(df)
        result = np.zeros((n, 19), dtype=np.float64)

        def _safe_col(col_name: str) -> np.ndarray:
            """Extract column as numpy, filling missing with 0."""
            if col_name in df.columns:
                vals = df[col_name].to_numpy().astype(np.float64)
                vals = np.nan_to_num(vals, nan=0.0)
                return vals
            return np.zeros(n, dtype=np.float64)

        # ---- Home team reversion features (0-6) ----
        h_goals_r5 = _safe_col("home_form_goals_for_r5")
        h_goals_r3 = _safe_col("home_form_goals_for_r3")
        h_goals_r10 = _safe_col("home_form_goals_for_r10")
        h_xg_r5 = _safe_col("home_form_xg_for_r5")
        h_xg_against_r5 = _safe_col("home_form_xg_against_r5")
        h_win_r3 = _safe_col("home_form_win_r3")
        h_win_r10 = _safe_col("home_form_win_r10")
        h_pts_r5 = _safe_col("home_form_points_r5")
        h_pts_r10 = _safe_col("home_form_points_r10")
        h_shotacc_r3 = _safe_col("home_form_shot_accuracy_r3")
        h_shotacc_r10 = _safe_col("home_form_shot_accuracy_r10")
        h_cs_r3 = _safe_col("home_form_clean_sheet_r3")

        result[:, 0] = h_goals_r5 - h_xg_r5              # xG overperformance
        result[:, 1] = h_goals_r3 - h_goals_r10           # goals spike
        result[:, 2] = h_win_r3 - h_win_r10               # win rate spike
        result[:, 3] = h_pts_r5 - h_pts_r10               # points spike
        result[:, 4] = h_shotacc_r3 - h_shotacc_r10       # shot accuracy spike
        result[:, 5] = h_cs_r3 * h_xg_against_r5          # clean sheet fragility
        result[:, 6] = h_goals_r3                          # scoring streak

        # ---- Away team reversion features (7-13) ----
        a_goals_r5 = _safe_col("away_form_goals_for_r5")
        a_goals_r3 = _safe_col("away_form_goals_for_r3")
        a_goals_r10 = _safe_col("away_form_goals_for_r10")
        a_xg_r5 = _safe_col("away_form_xg_for_r5")
        a_xg_against_r5 = _safe_col("away_form_xg_against_r5")
        a_win_r3 = _safe_col("away_form_win_r3")
        a_win_r10 = _safe_col("away_form_win_r10")
        a_pts_r5 = _safe_col("away_form_points_r5")
        a_pts_r10 = _safe_col("away_form_points_r10")
        a_shotacc_r3 = _safe_col("away_form_shot_accuracy_r3")
        a_shotacc_r10 = _safe_col("away_form_shot_accuracy_r10")
        a_cs_r3 = _safe_col("away_form_clean_sheet_r3")

        result[:, 7] = a_goals_r5 - a_xg_r5               # xG overperformance
        result[:, 8] = a_goals_r3 - a_goals_r10            # goals spike
        result[:, 9] = a_win_r3 - a_win_r10                # win rate spike
        result[:, 10] = a_pts_r5 - a_pts_r10               # points spike
        result[:, 11] = a_shotacc_r3 - a_shotacc_r10       # shot accuracy spike
        result[:, 12] = a_cs_r3 * a_xg_against_r5          # clean sheet fragility
        result[:, 13] = a_goals_r3                          # scoring streak

        # ---- Match-level trap signals (14-18) ----
        delta_xg_r5 = _safe_col("delta_form_xg_for_r5")
        mkt_home = _safe_col("mkt_1x2_home_prob")
        delta_ppg = _safe_col("delta_ppg")
        h2h_win_rate = _safe_col("home_h2h_win_rate")

        result[:, 14] = np.abs(delta_xg_r5)                # delta extreme
        result[:, 15] = mkt_home                            # consensus home
        result[:, 16] = h_goals_r3 + a_goals_r3            # combined scoring
        result[:, 17] = np.abs(delta_ppg)                   # PPG gap
        result[:, 18] = h2h_win_rate                        # H2H dominance

        # Impute NaN with medians
        medians: dict[str, float]
        if fit:
            medians = {}
            for i, name in enumerate(_REVERSION_NAMES):
                col = result[:, i]
                valid = col[~np.isnan(col)]
                medians[name] = float(np.median(valid)) if len(valid) > 0 else 0.0
        else:
            medians = self.reversion_medians

        for i, name in enumerate(_REVERSION_NAMES):
            mask = np.isnan(result[:, i])
            if mask.any():
                result[mask, i] = medians.get(name, 0.0)

        return result, medians

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "model_type": "contrarian",
            "base_model": self.base_model.to_dict(),
            "reversion_feature_names": self.reversion_feature_names,
            "reversion_beta_home": self.reversion_beta_home,
            "reversion_beta_away": self.reversion_beta_away,
            "reversion_medians": self.reversion_medians,
            "league_avg_home": self.league_avg_home,
            "league_avg_away": self.league_avg_away,
            "max_goals": self.max_goals,
            "rho": self.rho,
            "training_window": self.training_window,
            "training_samples": self.training_samples,
            "fit_timestamp": self.fit_timestamp,
            "convergence_info": self.convergence_info,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ContrarianModel:
        m = cls(max_goals=d.get("max_goals", 10))
        m.base_model = XGBoostModel.from_dict(d["base_model"])
        m.reversion_feature_names = d.get("reversion_feature_names", list(_REVERSION_NAMES))
        m.reversion_beta_home = d["reversion_beta_home"]
        m.reversion_beta_away = d["reversion_beta_away"]
        m.reversion_medians = d.get("reversion_medians", {})
        m.league_avg_home = d.get("league_avg_home", 0.0)
        m.league_avg_away = d.get("league_avg_away", 0.0)
        m.rho = d.get("rho", 0.0)
        m.training_window = d.get("training_window", "")
        m.training_samples = d.get("training_samples", 0)
        m.fit_timestamp = d.get("fit_timestamp", "")
        m.convergence_info = d.get("convergence_info", {})
        return m
