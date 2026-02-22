"""Model dispatch: create, serialize, and deserialize by model type.

Supports: poisson, xgboost, ensemble, contrarian.
Old artifacts without a ``model_type`` key default to "poisson".
"""

from __future__ import annotations

from dhx.modeling.poisson import PoissonModel


def create_model(model_type: str, **kwargs):
    """Instantiate a model by type name."""
    if model_type == "poisson":
        return PoissonModel(**kwargs)
    if model_type == "xgboost":
        from dhx.modeling.xgboost_model import XGBoostModel

        return XGBoostModel(**kwargs)
    if model_type == "ensemble":
        from dhx.modeling.ensemble import EnsembleModel

        return EnsembleModel(**kwargs)
    if model_type == "contrarian":
        from dhx.modeling.contrarian import ContrarianModel

        return ContrarianModel(**kwargs)
    raise ValueError(f"Unknown model_type: {model_type!r}")


def model_from_dict(d: dict):
    """Deserialize a model from its artifact dict.

    Reads ``d["model_type"]`` to pick the right class. Defaults to "poisson"
    for backward compatibility with artifacts that lack this key.
    """
    model_type = d.get("model_type", "poisson")
    if model_type == "poisson":
        return PoissonModel.from_dict(d)
    if model_type == "xgboost":
        from dhx.modeling.xgboost_model import XGBoostModel

        return XGBoostModel.from_dict(d)
    if model_type == "ensemble":
        from dhx.modeling.ensemble import EnsembleModel

        return EnsembleModel.from_dict(d)
    if model_type == "contrarian":
        from dhx.modeling.contrarian import ContrarianModel

        return ContrarianModel.from_dict(d)
    raise ValueError(f"Unknown model_type in artifact: {model_type!r}")
