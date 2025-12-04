"""Label metrics (triple-barrier/meta-label summary)."""

from __future__ import annotations

from typing import Iterable


def summarize_labels(labels: Iterable[int]) -> dict[str, int]:
    counts = {"positive": 0, "negative": 0, "neutral": 0}
    for lbl in labels:
        if lbl > 0:
            counts["positive"] += 1
        elif lbl < 0:
            counts["negative"] += 1
        else:
            counts["neutral"] += 1
    return counts
