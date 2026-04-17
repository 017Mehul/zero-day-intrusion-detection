from __future__ import annotations

from typing import Any

import numpy as np


def compute_shap_importance(model: Any, X: np.ndarray, top_k: int | None = None) -> list[tuple[str, float]]:
    """Compute SHAP feature importance for a fitted model.

    Requires `shap` to be installed. Returns list of (feature_idx, mean_abs_shap).
    """
    try:
        import shap
    except Exception as e:
        raise RuntimeError("shap is required for compute_shap_importance") from e

    explainer = shap.Explainer(model)
    shap_vals = explainer(X)
    # shap_vals.values shape (n_samples, n_features) for tabular
    mean_abs = np.mean(np.abs(shap_vals.values), axis=0)
    idxs = np.argsort(mean_abs)[::-1]
    if top_k is None:
        top_k = len(mean_abs)
    return [(int(i), float(mean_abs[i])) for i in idxs[:top_k]]
