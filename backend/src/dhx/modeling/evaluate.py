"""Evaluation metrics for model predictions.

Works with any model that exposes predict_lambdas() and market_probabilities().
"""

from __future__ import annotations

import logging
import math

import polars as pl

from dhx.modeling.markets import ah_probability, find_fair_ah_line

logger = logging.getLogger(__name__)


def compute_1x2_metrics(
    probs: list[dict],
    actuals: list[str],
) -> dict:
    """Compute multiclass metrics for 1x2 market.

    Args:
        probs: list of {"home": p, "draw": p, "away": p} dicts.
        actuals: list of actual outcomes ("H", "D", "A").
    """
    n = len(actuals)
    if n == 0:
        return {"n": 0}

    brier_sum = 0.0
    logloss_sum = 0.0
    correct = 0
    # Calibration bins: track predicted vs actual for each class
    cal_bins: dict[str, list[tuple[float, int]]] = {"home": [], "draw": [], "away": []}
    label_map = {"H": "home", "D": "draw", "A": "away"}

    for prob, actual in zip(probs, actuals):
        actual_key = label_map.get(actual, "home")
        # One-hot encoding
        for cls in ("home", "draw", "away"):
            indicator = 1.0 if cls == actual_key else 0.0
            p = prob[cls]
            brier_sum += (p - indicator) ** 2
            cal_bins[cls].append((p, int(indicator)))

        # Log-loss: -log(p_actual)
        p_actual = max(prob[actual_key], 1e-10)
        logloss_sum += -math.log(p_actual)

        # Accuracy: predicted class = highest probability
        pred_cls = max(prob, key=prob.get)
        if pred_cls == actual_key:
            correct += 1

    brier = brier_sum / n
    logloss = logloss_sum / n
    accuracy = correct / n
    cal_error = _calibration_error(cal_bins)

    return {
        "n": n,
        "brier": round(brier, 6),
        "log_loss": round(logloss, 6),
        "accuracy": round(accuracy, 4),
        "calibration_error": round(cal_error, 6),
    }


def compute_binary_metrics(
    probs_positive: list[float],
    actuals: list[int],
    label: str = "positive",
) -> dict:
    """Compute binary metrics (used for totals and BTTS).

    Args:
        probs_positive: predicted probability of positive class (over / yes).
        actuals: actual outcomes (1 = positive, 0 = negative).
    """
    n = len(actuals)
    if n == 0:
        return {"n": 0}

    brier_sum = 0.0
    logloss_sum = 0.0
    correct = 0

    pred_probs: list[tuple[float, int]] = []

    for p, y in zip(probs_positive, actuals):
        brier_sum += (p - y) ** 2
        p_clamped = max(min(p, 1 - 1e-10), 1e-10)
        logloss_sum += -(y * math.log(p_clamped) + (1 - y) * math.log(1 - p_clamped))
        pred = 1 if p >= 0.5 else 0
        if pred == y:
            correct += 1
        pred_probs.append((p, y))

    brier = brier_sum / n
    logloss = logloss_sum / n
    accuracy = correct / n

    # Calibration error (5 bins)
    cal_error = _binary_calibration_error(pred_probs)

    return {
        "n": n,
        "brier": round(brier, 6),
        "log_loss": round(logloss, 6),
        "accuracy": round(accuracy, 4),
        "calibration_error": round(cal_error, 6),
    }


def compute_ah_metrics(
    lambdas_and_scores: list[tuple[float, float, int, int]],
) -> dict:
    """Compute AH evaluation metrics.

    Args:
        lambdas_and_scores: list of (lambda_h, lambda_a, home_score, away_score).

    For each fixture, computes the model fair AH line, gets effective_prob at that
    line, and evaluates against the actual goal difference.
    """
    n = len(lambdas_and_scores)
    if n == 0:
        return {"n": 0}

    brier_sum = 0.0
    logloss_sum = 0.0
    correct = 0
    pred_probs: list[tuple[float, int]] = []

    for lh, la, hs, as_ in lambdas_and_scores:
        fair_line = find_fair_ah_line(lh, la)
        r = ah_probability(lh, la, fair_line, side="home")
        eff = r["effective_prob"]

        # Actual outcome: did home "win" on the fair AH line?
        actual_diff = hs - as_
        threshold = -fair_line
        if actual_diff > threshold:
            y = 1  # home wins AH
        elif actual_diff == threshold:
            y = 0  # push — treat as 0.5 for Brier
            brier_sum += (eff - 0.5) ** 2
            p_clamped = max(min(eff, 1 - 1e-10), 1e-10)
            logloss_sum += -(0.5 * math.log(p_clamped) + 0.5 * math.log(1 - p_clamped))
            pred_probs.append((eff, 0))  # count push as neither for calibration
            continue
        else:
            y = 0  # home loses AH

        brier_sum += (eff - y) ** 2
        p_clamped = max(min(eff, 1 - 1e-10), 1e-10)
        logloss_sum += -(y * math.log(p_clamped) + (1 - y) * math.log(1 - p_clamped))
        if (eff >= 0.5 and y == 1) or (eff < 0.5 and y == 0):
            correct += 1
        pred_probs.append((eff, y))

    brier = brier_sum / n
    logloss = logloss_sum / n
    accuracy = correct / n
    cal_error = _binary_calibration_error(pred_probs)

    return {
        "n": n,
        "brier": round(brier, 6),
        "log_loss": round(logloss, 6),
        "accuracy": round(accuracy, 4),
        "calibration_error": round(cal_error, 6),
    }


def evaluate_all(model, test_df: pl.DataFrame) -> dict[str, dict]:
    """Run full evaluation on test set. Returns metrics per market."""
    # Filter to only fixtures with actual results (exclude unplayed matches)
    test_df = test_df.filter(pl.col("label_result").is_not_null())
    lambdas = model.predict_lambdas(test_df)

    # Collect predictions and actuals
    probs_1x2: list[dict] = []
    probs_over25: list[float] = []
    probs_btts_yes: list[float] = []
    ah_data: list[tuple[float, float, int, int]] = []

    for row in lambdas.iter_rows(named=True):
        mp = model.market_probabilities(row["lambda_home"], row["lambda_away"])
        probs_1x2.append(mp["1x2"])
        probs_over25.append(mp["totals"]["over"])
        probs_btts_yes.append(mp["btts"]["yes"])

    # Actuals
    actuals_result = test_df["label_result"].to_list()
    actuals_total_goals = test_df["label_total_goals"].to_numpy()
    actuals_over25 = [1 if g >= 3 else 0 for g in actuals_total_goals]
    actuals_btts = test_df["label_btts"].to_list()

    # AH data: lambdas + actual scores
    home_scores = test_df["label_home_score"].to_list()
    away_scores = test_df["label_away_score"].to_list()
    for row, hs, as_ in zip(lambdas.iter_rows(named=True), home_scores, away_scores):
        if hs is not None and as_ is not None:
            ah_data.append((row["lambda_home"], row["lambda_away"], int(hs), int(as_)))

    metrics = {
        "1x2": compute_1x2_metrics(probs_1x2, actuals_result),
        "totals": compute_binary_metrics(probs_over25, actuals_over25, "over2.5"),
        "btts": compute_binary_metrics(probs_btts_yes, actuals_btts, "btts_yes"),
        "ah": compute_ah_metrics(ah_data),
    }

    for market, m in metrics.items():
        logger.info(
            f"{market}: brier={m.get('brier')}, "
            f"log_loss={m.get('log_loss')}, acc={m.get('accuracy')}"
        )

    return metrics


# ======================================================================
# Internal helpers
# ======================================================================


def _calibration_error(
    bins: dict[str, list[tuple[float, int]]],
    n_bins: int = 5,
) -> float:
    """Expected calibration error across all classes (multiclass)."""
    total_error = 0.0
    total_samples = 0

    for cls, pairs in bins.items():
        if not pairs:
            continue
        sorted_pairs = sorted(pairs, key=lambda x: x[0])
        bin_size = max(1, len(sorted_pairs) // n_bins)

        for i in range(0, len(sorted_pairs), bin_size):
            chunk = sorted_pairs[i : i + bin_size]
            avg_pred = sum(p for p, _ in chunk) / len(chunk)
            avg_actual = sum(y for _, y in chunk) / len(chunk)
            total_error += abs(avg_pred - avg_actual) * len(chunk)
            total_samples += len(chunk)

    return total_error / max(total_samples, 1)


def _binary_calibration_error(
    pairs: list[tuple[float, int]],
    n_bins: int = 5,
) -> float:
    """Expected calibration error for binary predictions."""
    if not pairs:
        return 0.0

    sorted_pairs = sorted(pairs, key=lambda x: x[0])
    bin_size = max(1, len(sorted_pairs) // n_bins)
    total_error = 0.0
    total_samples = 0

    for i in range(0, len(sorted_pairs), bin_size):
        chunk = sorted_pairs[i : i + bin_size]
        avg_pred = sum(p for p, _ in chunk) / len(chunk)
        avg_actual = sum(y for _, y in chunk) / len(chunk)
        total_error += abs(avg_pred - avg_actual) * len(chunk)
        total_samples += len(chunk)

    return total_error / max(total_samples, 1)
