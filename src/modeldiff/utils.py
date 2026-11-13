"""Input normalization and small shared helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

VALID_TASKS = {"binary_classification", "regression"}
VALID_METRICS = {
    "binary_classification": {"log_loss", "brier", "accuracy"},
    "regression": {"squared_error", "absolute_error", "mse", "mae", "rmse"},
}


def normalize_X(X: Any) -> pd.DataFrame:
    """Convert supported feature inputs to a defensive DataFrame copy."""
    if isinstance(X, pd.DataFrame):
        if X.columns.duplicated().any():
            duplicated = list(X.columns[X.columns.duplicated()])
            raise ValueError(f"X contains duplicate columns: {duplicated}")
        result = X.reset_index(drop=True).copy()
        result.columns = [str(column) for column in result.columns]
        if pd.Index(result.columns).duplicated().any():
            raise ValueError("X column names are duplicated after string conversion")
        return result
    array = np.asarray(X)
    if array.ndim != 2:
        raise ValueError("X must be a two-dimensional DataFrame or array")
    return pd.DataFrame(array, columns=[f"x{i}" for i in range(array.shape[1])])


def normalize_1d(values: Any, name: str, *, numeric: bool = True) -> np.ndarray:
    """Convert a Series, list, or array to a one-dimensional NumPy array."""
    array = np.asarray(values)
    if array.ndim == 2 and 1 in array.shape:
        array = array.reshape(-1)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if numeric:
        try:
            array = array.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain numeric values") from exc
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")
    return array


def validate_inputs(
    X: Any,
    y: Any,
    baseline_predictions: Any,
    candidate_predictions: Any,
    task: str,
    metric: str,
    *,
    min_samples: int = 4,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Normalize and validate all inputs for a comparison."""
    if task not in VALID_TASKS:
        raise ValueError(f"task must be one of {sorted(VALID_TASKS)}")
    if metric not in VALID_METRICS[task]:
        raise ValueError(f"metric {metric!r} is not supported for task {task!r}")
    frame = normalize_X(X)
    target = normalize_1d(y, "y")
    baseline = normalize_1d(baseline_predictions, "baseline_predictions")
    candidate = normalize_1d(candidate_predictions, "candidate_predictions")
    lengths = {len(frame), len(target), len(baseline), len(candidate)}
    if len(lengths) != 1:
        raise ValueError("X, y, and predictions must have equal sample lengths")
    if len(frame) < min_samples:
        noun = "observation" if min_samples == 1 else "observations"
        raise ValueError(f"at least {min_samples} {noun} are required")
    if task == "binary_classification":
        unique = np.unique(target)
        if not np.all(np.isin(unique, [0.0, 1.0])):
            raise ValueError("binary classification targets must contain only 0 and 1")
        if np.any((baseline < 0) | (baseline > 1)) or np.any((candidate < 0) | (candidate > 1)):
            raise ValueError("binary classification predictions must be probabilities in [0, 1]")
    return frame, target, baseline, candidate


def python_value(value: Any) -> Any:
    """Convert NumPy scalar values to JSON-serializable Python values."""
    return value.item() if isinstance(value, np.generic) else value
