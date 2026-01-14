"""Metrics utilities for QA, drift, and labels.

This module provides utilities to summarize and evaluate performance metrics
from the LIQ Stack ecosystem.

Functions:
    summarize_qa: Convert QA results to flat dictionaries for reporting.
    summarize_drift: Compute statistics for drift signals from feature pipelines.
    summarize_labels: Count triple-barrier/meta-label outcomes.
"""

from liq.metrics.drift import summarize_drift
from liq.metrics.labels import summarize_labels
from liq.metrics.prediction import summarize_classification, summarize_regression
from liq.metrics.qa import QAResultLike, summarize_qa

__all__ = [
    "summarize_qa",
    "summarize_drift",
    "summarize_labels",
    "summarize_classification",
    "summarize_regression",
    "QAResultLike",
]
