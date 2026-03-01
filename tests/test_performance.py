"""Tests for regime-stratified performance analysis."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from liq.metrics.performance import (
    ComparisonResult,
    PerformanceAnalyzer,
    PerformanceReport,
    RegimeMetrics,
)


@pytest.fixture
def analyzer() -> PerformanceAnalyzer:
    return PerformanceAnalyzer()


def _make_equity_curve(
    start: datetime,
    values: list[float],
    interval_hours: float = 1.0,
) -> list[tuple[datetime, Decimal]]:
    """Create equity curve from values."""
    return [
        (start + timedelta(hours=i * interval_hours), Decimal(str(v))) for i, v in enumerate(values)
    ]


def _make_regime_labels(
    start: datetime,
    labels: list[str],
    interval_hours: float = 1.0,
) -> list[tuple[datetime, str]]:
    """Create regime labels aligned to equity curve timestamps."""
    return [(start + timedelta(hours=i * interval_hours), label) for i, label in enumerate(labels)]


class TestRegimeMetrics:
    """Tests for RegimeMetrics model."""

    def test_creation(self) -> None:
        m = RegimeMetrics(
            regime="bull",
            total_return=0.10,
            sharpe_ratio=1.5,
            max_drawdown=-0.05,
            num_bars=100,
            win_rate=0.55,
        )
        assert m.regime == "bull"
        assert m.total_return == 0.10

    def test_sharpe_none_for_few_observations(self) -> None:
        m = RegimeMetrics(
            regime="crisis",
            total_return=-0.02,
            sharpe_ratio=None,
            max_drawdown=-0.02,
            num_bars=1,
            win_rate=None,
        )
        assert m.sharpe_ratio is None
        assert m.win_rate is None


class TestPerformanceAnalyzer:
    """Tests for PerformanceAnalyzer."""

    def test_single_regime_aggregate_matches(self, analyzer: PerformanceAnalyzer) -> None:
        """With one regime, aggregate matches regime metrics."""
        start = datetime(2024, 1, 1, tzinfo=UTC)
        equity = _make_equity_curve(start, [100, 102, 101, 103, 105])
        labels = _make_regime_labels(start, ["bull", "bull", "bull", "bull", "bull"])

        report = analyzer.analyze(equity, labels)
        assert isinstance(report, PerformanceReport)
        assert "bull" in report.by_regime
        assert report.aggregate.num_bars == 5
        assert report.aggregate.regime == "aggregate"
        # With single regime, aggregate total_return should match
        assert report.aggregate.total_return == pytest.approx(report.by_regime["bull"].total_return)

    def test_multiple_regimes_stratification(self, analyzer: PerformanceAnalyzer) -> None:
        """Multiple regimes produce correct stratification."""
        start = datetime(2024, 1, 1, tzinfo=UTC)
        equity = _make_equity_curve(start, [100, 102, 104, 103, 101, 100])
        labels = _make_regime_labels(start, ["bull", "bull", "bull", "bear", "bear", "bear"])

        report = analyzer.analyze(equity, labels)
        assert "bull" in report.by_regime
        assert "bear" in report.by_regime
        assert report.by_regime["bull"].num_bars == 3
        assert report.by_regime["bear"].num_bars == 3

    def test_sharpe_ratio_calculation(self, analyzer: PerformanceAnalyzer) -> None:
        """Sharpe ratio calculated when sufficient observations."""
        start = datetime(2024, 1, 1, tzinfo=UTC)
        # Steady upward → positive Sharpe
        values = [100 + i * 0.5 for i in range(20)]
        equity = _make_equity_curve(start, values)
        labels = _make_regime_labels(start, ["trend"] * 20)

        report = analyzer.analyze(equity, labels)
        assert report.aggregate.sharpe_ratio is not None
        assert report.aggregate.sharpe_ratio > 0

    def test_max_drawdown_calculation(self, analyzer: PerformanceAnalyzer) -> None:
        """Max drawdown correctly computed."""
        start = datetime(2024, 1, 1, tzinfo=UTC)
        # Peak at 110, trough at 95 → drawdown = (110-95)/110 ≈ 13.6%
        equity = _make_equity_curve(start, [100, 105, 110, 100, 95, 98])
        labels = _make_regime_labels(start, ["mixed"] * 6)

        report = analyzer.analyze(equity, labels)
        assert report.aggregate.max_drawdown < 0  # Negative = drawdown
        assert report.aggregate.max_drawdown == pytest.approx(-15 / 110, abs=0.001)

    def test_win_rate_calculation(self, analyzer: PerformanceAnalyzer) -> None:
        """Win rate = fraction of positive bar returns."""
        start = datetime(2024, 1, 1, tzinfo=UTC)
        # Returns: +2, -1, +2, +2 → 3/4 = 75% win rate
        equity = _make_equity_curve(start, [100, 102, 101, 103, 105])
        labels = _make_regime_labels(start, ["trend"] * 5)

        report = analyzer.analyze(equity, labels)
        assert report.aggregate.win_rate == pytest.approx(0.75, abs=0.01)

    def test_empty_equity_curve(self, analyzer: PerformanceAnalyzer) -> None:
        """Empty equity curve produces safe defaults."""
        report = analyzer.analyze([], [])
        assert report.aggregate.num_bars == 0
        assert report.aggregate.total_return == 0.0
        assert report.aggregate.sharpe_ratio is None
        assert report.by_regime == {}

    def test_single_bar(self, analyzer: PerformanceAnalyzer) -> None:
        """Single bar produces valid report with None sharpe."""
        start = datetime(2024, 1, 1, tzinfo=UTC)
        equity = _make_equity_curve(start, [100])
        labels = _make_regime_labels(start, ["flat"])

        report = analyzer.analyze(equity, labels)
        assert report.aggregate.num_bars == 1
        assert report.aggregate.sharpe_ratio is None
        assert report.aggregate.total_return == 0.0

    def test_regime_labels_partial_coverage(self, analyzer: PerformanceAnalyzer) -> None:
        """Bars without regime labels go to 'unknown' bucket."""
        start = datetime(2024, 1, 1, tzinfo=UTC)
        equity = _make_equity_curve(start, [100, 102, 104, 103])
        # Only label first 2 bars
        labels = _make_regime_labels(start, ["bull", "bull"])

        report = analyzer.analyze(equity, labels)
        assert "bull" in report.by_regime
        assert "unknown" in report.by_regime


class TestPerformanceComparison:
    """Tests for candidate vs baseline comparison."""

    def test_candidate_outperforms(self, analyzer: PerformanceAnalyzer) -> None:
        """Candidate outperforms baseline in all regimes."""
        start = datetime(2024, 1, 1, tzinfo=UTC)
        # Candidate: strong positive
        candidate_equity = _make_equity_curve(start, [100, 105, 110, 115, 120])
        # Baseline: weaker positive
        baseline_equity = _make_equity_curve(start, [100, 101, 102, 103, 104])
        labels = _make_regime_labels(start, ["bull"] * 5)

        candidate_report = analyzer.analyze(candidate_equity, labels)
        baseline_report = analyzer.analyze(baseline_equity, labels)

        comparison = analyzer.compare(candidate_report, baseline_report)
        assert isinstance(comparison, ComparisonResult)
        assert comparison.outperforms_aggregate is True
        assert comparison.outperforms_per_regime["bull"] is True

    def test_candidate_underperforms_one_regime(self, analyzer: PerformanceAnalyzer) -> None:
        """Candidate underperforms in one regime."""
        start = datetime(2024, 1, 1, tzinfo=UTC)
        # Candidate: good in bull, bad in bear, still net positive
        candidate_equity = _make_equity_curve(start, [100, 105, 110, 108, 105, 104])
        # Baseline: steady in both
        baseline_equity = _make_equity_curve(start, [100, 101, 102, 102.5, 103, 103])
        labels = _make_regime_labels(start, ["bull", "bull", "bull", "bear", "bear", "bear"])

        candidate_report = analyzer.analyze(candidate_equity, labels)
        baseline_report = analyzer.analyze(baseline_equity, labels)

        comparison = analyzer.compare(candidate_report, baseline_report)
        # Aggregate candidate still outperforms
        assert comparison.outperforms_aggregate is True
        # Per-regime: bull outperforms, bear underperforms
        assert comparison.outperforms_per_regime["bull"] is True
        assert comparison.outperforms_per_regime["bear"] is False

    def test_analyze_by_regime(self, analyzer: PerformanceAnalyzer) -> None:
        """analyze_by_regime returns dict of regime → RegimeMetrics."""
        start = datetime(2024, 1, 1, tzinfo=UTC)
        equity = _make_equity_curve(start, [100, 102, 101, 99, 100])
        labels = _make_regime_labels(start, ["up", "up", "down", "down", "flat"])

        by_regime = analyzer.analyze_by_regime(equity, labels)
        assert "up" in by_regime
        assert "down" in by_regime
        assert "flat" in by_regime
        assert all(isinstance(v, RegimeMetrics) for v in by_regime.values())
