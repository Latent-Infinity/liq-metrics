"""Label metrics (triple-barrier/meta-label summary)."""

from __future__ import annotations

from collections.abc import Iterable


def summarize_labels(labels: Iterable[int]) -> dict[str, int]:
    """Count triple-barrier or meta-label outcomes.

    Categorizes labels into positive (>0), negative (<0), and neutral (0).

    Args:
        labels: An iterable of integer labels representing trade outcomes.
            Typically 1 for profit, -1 for loss, 0 for neutral/timeout.

    Returns:
        A dictionary with counts for 'positive', 'negative', and 'neutral'.

    Raises:
        TypeError: If any element in labels is not an integer.

    Examples:
        >>> summarize_labels([1, -1, 0, 1, 0])
        {'positive': 2, 'negative': 1, 'neutral': 2}
        >>> summarize_labels([])
        {'positive': 0, 'negative': 0, 'neutral': 0}
    """
    counts: dict[str, int] = {"positive": 0, "negative": 0, "neutral": 0}

    for i, lbl in enumerate(labels):
        if not isinstance(lbl, int) or isinstance(lbl, bool):
            raise TypeError(
                f"All labels must be integers, got {type(lbl).__name__} at index {i}"
            )

        if lbl > 0:
            counts["positive"] += 1
        elif lbl < 0:
            counts["negative"] += 1
        else:
            counts["neutral"] += 1

    return counts
