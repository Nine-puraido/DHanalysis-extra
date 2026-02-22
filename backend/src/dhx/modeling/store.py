"""Save / load model artifacts and register model_versions in Supabase.

Supports all model types via the registry module.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

from dhx.config import get_settings
from dhx.db import get_client, insert_rows, select_rows, update_rows
from dhx.modeling.registry import model_from_dict
from dhx.storage import upload_json

logger = logging.getLogger(__name__)

MARKETS = ("1x2", "totals", "btts", "ah")

# Model type -> model_name used in model_versions rows
_MODEL_NAMES = {
    "poisson": "poisson_glm",
    "xgboost": "xgboost_poisson",
    "ensemble": "ensemble_poisson",
    "contrarian": "contrarian_xgboost",
}


def save_model(
    model,
    metrics: dict[str, dict],
    feature_set_version_id: int,
    model_type: str = "poisson",
) -> dict[str, int]:
    """Save model artifact to Storage and register model_versions rows.

    Returns {"1x2": id, "totals": id, "btts": id, "ah": id}.
    """
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = f"{model_type}/{ts}.json"
    bucket = get_settings().supabase_bucket_models

    # Upload artifact
    artifact_path = upload_json(bucket, path, model.to_dict())
    logger.info(f"Saved model artifact to {artifact_path}")

    model_name = _MODEL_NAMES.get(model_type, model_type)

    # Build params_json based on model type
    params_json: dict = {
        "max_goals": model.max_goals,
        "training_samples": model.training_samples,
    }
    if model_type == "poisson":
        params_json["features_home"] = model.feature_columns_home
        params_json["features_away"] = model.feature_columns_away
    elif model_type == "xgboost":
        params_json["n_features"] = len(model.feature_columns)
    elif model_type == "ensemble":
        params_json["weight_poisson"] = model.weight_poisson
    elif model_type == "contrarian":
        params_json["base_type"] = "xgboost"
        params_json["n_reversion_features"] = len(model.reversion_feature_names)

    # Register one model_version per market
    version_ids: dict[str, int] = {}
    for market in MARKETS:
        row = {
            "model_name": model_name,
            "model_type": model_type,
            "market": market,
            "feature_set_version_id": feature_set_version_id,
            "artifact_path": artifact_path,
            "training_window": model.training_window,
            "metrics_json": metrics.get(market, {}),
            "params_json": params_json,
            "is_active": False,
        }
        inserted = insert_rows("model_versions", [row])
        vid = inserted[0]["id"]
        version_ids[market] = vid
        logger.info(f"Registered model_versions id={vid} for market={market}")

    return version_ids


def load_model(artifact_path: str):
    """Download a model artifact from Storage and deserialize.

    Uses registry.model_from_dict for type dispatch.
    """
    parts = artifact_path.split("/", 1)
    bucket = parts[0]
    path = parts[1]

    data = get_client().storage.from_(bucket).download(path)
    payload = json.loads(data)
    return model_from_dict(payload)


def activate_model(market: str, model_version_id: int) -> None:
    """Deactivate current active model for a market, then activate the given one."""
    # Deactivate current
    current = select_rows(
        "model_versions", "id", filters={"market": market, "is_active": True}
    )
    if current:
        old_id = current[0]["id"]
        update_rows("model_versions", {"is_active": False}, {"id": old_id})
        logger.info(f"Deactivated model_versions id={old_id} for market={market}")

    # Activate new
    now = datetime.now(UTC).isoformat()
    update_rows(
        "model_versions",
        {"is_active": True, "activated_at": now},
        {"id": model_version_id},
    )
    logger.info(f"Activated model_versions id={model_version_id} for market={market}")


def load_active_model(market: str):
    """Load the currently active model for a market.

    Returns (model, model_version_id).
    Raises ValueError if no active model exists.
    """
    rows = select_rows(
        "model_versions",
        "id,artifact_path",
        filters={"market": market, "is_active": True},
    )
    if not rows:
        raise ValueError(f"No active model for market={market}")

    row = rows[0]
    model = load_model(row["artifact_path"])
    return model, row["id"]
