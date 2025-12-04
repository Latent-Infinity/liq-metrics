"""QA metrics ingestion."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any


def summarize_qa(qa_result: Any) -> dict[str, float | int]:
    """Convert QAResult-like object to a flat dict."""
    if qa_result is None:
        return {}
    try:
        return asdict(qa_result)
    except Exception:
        # fallback for simple namespaces/dicts
        if isinstance(qa_result, dict):
            return qa_result
        return {
            "missing_ratio": getattr(qa_result, "missing_ratio", 0),
            "zero_volume_ratio": getattr(qa_result, "zero_volume_ratio", 0),
            "ohlc_inconsistencies": getattr(qa_result, "ohlc_inconsistencies", 0),
            "extreme_moves": getattr(qa_result, "extreme_moves", 0),
            "negative_volume": getattr(qa_result, "negative_volume", 0),
            "non_monotonic_ts": getattr(qa_result, "non_monotonic_ts", 0),
        }
