"""Drift metrics ingestion."""

from __future__ import annotations

from typing import Iterable


def summarize_drift(statistics: Iterable[float]) -> dict[str, float]:
    stats = list(statistics)
    if not stats:
        return {"max": 0.0, "mean": 0.0}
    return {"max": max(stats), "mean": sum(stats) / len(stats)}
