"""Regime-stratified performance analysis.

Provides PerformanceAnalyzer for computing aggregate and per-regime metrics
from equity curves with regime labels.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal


@dataclass(frozen=True)
class RegimeMetrics:
    """Performance metrics for a single regime."""

    regime: str
    total_return: float
    sharpe_ratio: float | None  # None if < 2 observations
    max_drawdown: float  # Negative value (e.g., -0.10 for 10% drawdown)
    num_bars: int
    win_rate: float | None  # None if < 2 observations


@dataclass(frozen=True)
class PerformanceReport:
    """Aggregate + regime-stratified performance."""

    aggregate: RegimeMetrics
    by_regime: dict[str, RegimeMetrics] = field(default_factory=dict)


@dataclass(frozen=True)
class ComparisonResult:
    """Candidate vs baseline comparison."""

    candidate: PerformanceReport
    baseline: PerformanceReport
    outperforms_aggregate: bool
    outperforms_per_regime: dict[str, bool] = field(default_factory=dict)


class PerformanceAnalyzer:
    """Stateless regime-stratified performance analysis.

    Bins equity curve segments by regime labels and computes per-regime
    and aggregate metrics.
    """

    def analyze(
        self,
        equity_curve: list[tuple[datetime, Decimal]],
        regime_labels: list[tuple[datetime, str]],
    ) -> PerformanceReport:
        """Produce aggregate + regime-stratified performance report."""
        if not equity_curve:
            return PerformanceReport(
                aggregate=RegimeMetrics(
                    regime="aggregate",
                    total_return=0.0,
                    sharpe_ratio=None,
                    max_drawdown=0.0,
                    num_bars=0,
                    win_rate=None,
                ),
                by_regime={},
            )

        by_regime = self.analyze_by_regime(equity_curve, regime_labels)
        aggregate = self._compute_aggregate(equity_curve)

        return PerformanceReport(aggregate=aggregate, by_regime=by_regime)

    def analyze_by_regime(
        self,
        equity_curve: list[tuple[datetime, Decimal]],
        regime_labels: list[tuple[datetime, str]],
    ) -> dict[str, RegimeMetrics]:
        """Compute per-regime metrics."""
        if not equity_curve:
            return {}

        # Build timestamp → regime lookup
        label_map: dict[datetime, str] = dict(regime_labels)

        # Bin equity values by regime
        regime_values: dict[str, list[Decimal]] = defaultdict(list)
        for ts, value in equity_curve:
            regime = label_map.get(ts, "unknown")
            regime_values[regime].append(value)

        result: dict[str, RegimeMetrics] = {}
        for regime, values in regime_values.items():
            result[regime] = self._compute_regime_metrics(regime, values)

        return result

    def compare(
        self,
        candidate: PerformanceReport,
        baseline: PerformanceReport,
    ) -> ComparisonResult:
        """Compare candidate vs baseline performance."""
        outperforms_aggregate = candidate.aggregate.total_return > baseline.aggregate.total_return

        outperforms_per_regime: dict[str, bool] = {}
        all_regimes = set(candidate.by_regime.keys()) | set(baseline.by_regime.keys())

        for regime in all_regimes:
            cand = candidate.by_regime.get(regime)
            base = baseline.by_regime.get(regime)

            if cand is not None and base is not None:
                outperforms_per_regime[regime] = cand.total_return > base.total_return
            elif cand is not None:
                # Candidate has data, baseline doesn't → candidate wins
                outperforms_per_regime[regime] = True
            else:
                # Baseline has data, candidate doesn't → baseline wins
                outperforms_per_regime[regime] = False

        return ComparisonResult(
            candidate=candidate,
            baseline=baseline,
            outperforms_aggregate=outperforms_aggregate,
            outperforms_per_regime=outperforms_per_regime,
        )

    def _compute_aggregate(self, equity_curve: list[tuple[datetime, Decimal]]) -> RegimeMetrics:
        """Compute aggregate metrics over entire equity curve."""
        values = [float(v) for _, v in equity_curve]
        return self._compute_regime_metrics("aggregate", [Decimal(str(v)) for v in values])

    def _compute_regime_metrics(self, regime: str, values: list[Decimal]) -> RegimeMetrics:
        """Compute metrics for a single regime's equity values."""
        n = len(values)

        if n == 0:
            return RegimeMetrics(
                regime=regime,
                total_return=0.0,
                sharpe_ratio=None,
                max_drawdown=0.0,
                num_bars=0,
                win_rate=None,
            )

        if n == 1:
            return RegimeMetrics(
                regime=regime,
                total_return=0.0,
                sharpe_ratio=None,
                max_drawdown=0.0,
                num_bars=1,
                win_rate=None,
            )

        float_values = [float(v) for v in values]

        # Total return
        total_return = (
            (float_values[-1] - float_values[0]) / float_values[0] if float_values[0] != 0 else 0.0
        )

        # Per-bar returns
        returns = [
            (float_values[i] - float_values[i - 1]) / float_values[i - 1]
            if float_values[i - 1] != 0
            else 0.0
            for i in range(1, n)
        ]

        # Sharpe ratio (annualized not applied here — raw per-bar Sharpe)
        sharpe: float | None = None
        if len(returns) >= 2:
            mean_ret = sum(returns) / len(returns)
            var = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1)
            std = math.sqrt(var) if var > 0 else 0.0
            sharpe = mean_ret / std if std > 0 else None

        # Max drawdown
        max_dd = self._max_drawdown(float_values)

        # Win rate
        win_rate: float | None = None
        if returns:
            wins = sum(1 for r in returns if r > 0)
            win_rate = wins / len(returns)

        return RegimeMetrics(
            regime=regime,
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            num_bars=n,
            win_rate=win_rate,
        )

    @staticmethod
    def _max_drawdown(values: list[float]) -> float:
        """Compute maximum drawdown as a negative fraction."""
        if len(values) < 2:
            return 0.0

        peak = values[0]
        max_dd = 0.0

        for v in values[1:]:
            if v > peak:
                peak = v
            dd = (v - peak) / peak if peak > 0 else 0.0
            if dd < max_dd:
                max_dd = dd

        return max_dd
