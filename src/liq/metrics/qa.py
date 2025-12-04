"""QA metrics ingestion."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class QAResultLike(Protocol):
    """Protocol for QA result objects."""

    missing_ratio: float
    zero_volume_ratio: float
    ohlc_inconsistencies: int
    extreme_moves: int
    negative_volume: int
    non_monotonic_ts: int


def summarize_qa(qa_result: QAResultLike | dict[str, Any] | None) -> dict[str, float | int]:
    """Convert QAResult-like object to a flat dict.

    Accepts dataclasses, SimpleNamespace objects with QA attributes, or dicts.

    Args:
        qa_result: A QA result object (dataclass, namespace, or dict) containing
            QA metrics like missing_ratio, zero_volume_ratio, etc.

    Returns:
        A flat dictionary with QA metrics. Returns empty dict if input is None.

    Raises:
        TypeError: If qa_result is not a supported type (dict, dataclass, or
            object with QA attributes).

    Examples:
        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class QAResult:
        ...     missing_ratio: float
        ...     zero_volume_ratio: float
        ...     ohlc_inconsistencies: int
        ...     extreme_moves: int
        ...     negative_volume: int
        ...     non_monotonic_ts: int
        >>> result = QAResult(0.1, 0.0, 1, 2, 0, 0)
        >>> summarize_qa(result)
        {'missing_ratio': 0.1, 'zero_volume_ratio': 0.0, 'ohlc_inconsistencies': 1, ...}
    """
    if qa_result is None:
        return {}

    # Handle dict input directly
    if isinstance(qa_result, dict):
        return qa_result

    # Handle dataclass objects
    if is_dataclass(qa_result) and not isinstance(qa_result, type):
        return asdict(qa_result)

    # Handle objects with QA attributes (SimpleNamespace, etc.)
    if isinstance(qa_result, QAResultLike):
        return {
            "missing_ratio": qa_result.missing_ratio,
            "zero_volume_ratio": qa_result.zero_volume_ratio,
            "ohlc_inconsistencies": qa_result.ohlc_inconsistencies,
            "extreme_moves": qa_result.extreme_moves,
            "negative_volume": qa_result.negative_volume,
            "non_monotonic_ts": qa_result.non_monotonic_ts,
        }

    raise TypeError(
        f"qa_result must be a dict, dataclass, or object with QA attributes, "
        f"got {type(qa_result).__name__}"
    )
