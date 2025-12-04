"""Drift metrics ingestion."""

from __future__ import annotations

from collections.abc import Iterable


def summarize_drift(statistics: Iterable[float]) -> dict[str, float]:
    """Compute basic statistics for drift signals.

    Calculates max and mean of drift statistics from feature pipelines.

    Args:
        statistics: An iterable of numeric drift values (floats or ints).

    Returns:
        A dictionary with 'max' and 'mean' keys. Returns {'max': 0.0, 'mean': 0.0}
        for empty input.

    Raises:
        TypeError: If any element in statistics is not numeric (int or float).

    Examples:
        >>> summarize_drift([0.1, 0.2, 0.3])
        {'max': 0.3, 'mean': 0.2}
        >>> summarize_drift([])
        {'max': 0.0, 'mean': 0.0}
    """
    stats = list(statistics)

    if not stats:
        return {"max": 0.0, "mean": 0.0}

    # Validate all elements are numeric
    for i, val in enumerate(stats):
        if not isinstance(val, (int, float)):
            raise TypeError(
                f"All statistics must be numeric (int or float), "
                f"got {type(val).__name__} at index {i}"
            )

    return {"max": float(max(stats)), "mean": sum(stats) / len(stats)}
