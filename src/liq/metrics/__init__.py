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
from liq.metrics.performance import (
    ComparisonResult,
    PerformanceAnalyzer,
    PerformanceReport,
    RegimeMetrics,
)
from liq.metrics.prediction import summarize_classification, summarize_regression
from liq.metrics.qa import QAResultLike, summarize_qa
from liq.metrics.six_curves import SixCurveInputs, SixCurveResult, compute_six_curves
from liq.metrics.tax_curves import (
    CurveFPeriod,
    CurveFResult,
    OpenTaxPosition,
    RealizedTaxEvent,
    TaxPolicy,
    TaxRates,
    compute_curve_f,
    compute_curve_f_periods,
    event_tax,
    mark_to_market_tax,
    realized_tax,
    terminal_open_position_tax,
)

__all__ = [
    "summarize_qa",
    "summarize_drift",
    "summarize_labels",
    "summarize_classification",
    "summarize_regression",
    "QAResultLike",
    "ComparisonResult",
    "PerformanceAnalyzer",
    "PerformanceReport",
    "RegimeMetrics",
    "SixCurveInputs",
    "SixCurveResult",
    "compute_six_curves",
    "TaxRates",
    "TaxPolicy",
    "RealizedTaxEvent",
    "OpenTaxPosition",
    "CurveFResult",
    "CurveFPeriod",
    "event_tax",
    "realized_tax",
    "mark_to_market_tax",
    "terminal_open_position_tax",
    "compute_curve_f",
    "compute_curve_f_periods",
]
