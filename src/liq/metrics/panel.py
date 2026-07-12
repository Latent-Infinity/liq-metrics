"""Shared research metrics panel.

One function computes the full panel every research run reports, and one
function writes it to ``metrics_panel.csv`` (long format: ``field,value``)
with every field present even when a value is unavailable.

Inference statistics (bootstrap CI, clustered t-stats, deflated Sharpe,
PBO / null percentile) are computed by the caller — the statistics module
lives in ``liq-validation``, which depends on this package — and passed in
via :class:`InferenceInputs`.

Conventions: returns are fractional per-trade / per-day values; Sharpe is
per-period (not annualized); ``tail_loss_95``/``tail_loss_99`` are the 5th /
1st percentiles of daily returns; skew and excess kurtosis are moment
estimators of daily returns; contributions are the largest single day /
event P&L divided by the total P&L of the corresponding series.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path

METRICS_PANEL_FIELDS = (
    "n_trades",
    "n_days",
    "gross_return",
    "net_return",
    "net_bps_per_trade",
    "profit_factor",
    "win_rate",
    "mean_trade_return",
    "mean_trade_return_ci_low",
    "mean_trade_return_ci_high",
    "day_clustered_tstat",
    "day_clustered_pvalue",
    "event_clustered_tstat",
    "event_clustered_pvalue",
    "skew",
    "excess_kurtosis",
    "max_drawdown",
    "tail_loss_95",
    "tail_loss_99",
    "max_single_day_contribution",
    "max_single_event_contribution",
    "sharpe",
    "deflated_sharpe",
    "pbo",
    "null_percentile",
    "benchmark_alpha_per_period",
    "benchmark_beta",
    "portfolio_incremental_sharpe",
)


@dataclass(frozen=True)
class InferenceInputs:
    """Caller-computed inference statistics carried into the panel."""

    mean_trade_return_ci_low: float | None = None
    mean_trade_return_ci_high: float | None = None
    day_clustered_tstat: float | None = None
    day_clustered_pvalue: float | None = None
    event_clustered_tstat: float | None = None
    event_clustered_pvalue: float | None = None
    deflated_sharpe: float | None = None
    pbo: float | None = None
    null_percentile: float | None = None


@dataclass(frozen=True)
class MetricsPanel:
    """The full per-run metrics panel."""

    n_trades: int
    n_days: int
    gross_return: float
    net_return: float
    net_bps_per_trade: float
    profit_factor: float
    win_rate: float
    mean_trade_return: float
    skew: float
    excess_kurtosis: float
    max_drawdown: float
    tail_loss_95: float
    tail_loss_99: float
    max_single_day_contribution: float | None
    max_single_event_contribution: float | None
    sharpe: float | None
    mean_trade_return_ci_low: float | None = None
    mean_trade_return_ci_high: float | None = None
    day_clustered_tstat: float | None = None
    day_clustered_pvalue: float | None = None
    event_clustered_tstat: float | None = None
    event_clustered_pvalue: float | None = None
    deflated_sharpe: float | None = None
    pbo: float | None = None
    null_percentile: float | None = None
    benchmark_alpha_per_period: float | None = None
    benchmark_beta: float | None = None
    portfolio_incremental_sharpe: float | None = None
    cost_stress: Mapping[str, float] | None = None


def _quantile(sorted_values: list[float], p: float) -> float:
    """Linear-interpolation quantile of pre-sorted values."""
    index = p * (len(sorted_values) - 1)
    lo = math.floor(index)
    hi = math.ceil(index)
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (index - lo)


def _max_drawdown(daily_returns: Sequence[float]) -> float:
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in daily_returns:
        equity *= 1.0 + r
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak)
    return max_dd


def _contribution(largest: float, total: float) -> float | None:
    if total == 0.0:
        return None
    return largest / total


def compute_metrics_panel(
    *,
    trade_returns_net: Sequence[float],
    trade_returns_gross: Sequence[float],
    daily_returns: Sequence[float],
    trade_events: Sequence[str] | None = None,
    benchmark_daily_returns: Sequence[float] | None = None,
    inference: InferenceInputs | None = None,
    cost_stress: Mapping[str, float] | None = None,
    portfolio_incremental_sharpe: float | None = None,
) -> MetricsPanel:
    """Compute the shared metrics panel for one run.

    Args:
        trade_returns_net: Fractional net return per trade.
        trade_returns_gross: Fractional gross return per trade (same order).
        daily_returns: Fractional net daily portfolio returns.
        trade_events: Optional event id per trade (event-anchored arms).
        benchmark_daily_returns: Optional matched benchmark (SPY or beta
            twin) daily returns for alpha/beta attribution.
        inference: Caller-computed inference statistics.
        cost_stress: Net total return per cost-stress scenario id.
        portfolio_incremental_sharpe: Incremental Sharpe vs the plan of
            record's portfolio.
    """
    if len(trade_returns_net) == 0:
        raise ValueError("trade returns must not be empty")
    if len(trade_returns_net) != len(trade_returns_gross):
        raise ValueError(
            "net and gross trade returns must have equal length, got "
            f"{len(trade_returns_net)} and {len(trade_returns_gross)}"
        )
    if len(daily_returns) == 0:
        raise ValueError("daily returns must not be empty")
    if trade_events is not None and len(trade_events) != len(trade_returns_net):
        raise ValueError(
            "trade_events must align with trade returns, got "
            f"{len(trade_events)} events for {len(trade_returns_net)} trades"
        )
    if benchmark_daily_returns is not None and len(benchmark_daily_returns) != len(daily_returns):
        raise ValueError(
            "benchmark daily returns must align with daily returns, got "
            f"{len(benchmark_daily_returns)} and {len(daily_returns)}"
        )

    n_trades = len(trade_returns_net)
    n_days = len(daily_returns)
    wins = sum(1 for r in trade_returns_net if r > 0)
    gross_profit = sum(r for r in trade_returns_net if r > 0)
    gross_loss = -sum(r for r in trade_returns_net if r < 0)
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else math.inf
    mean_trade = sum(trade_returns_net) / n_trades

    mean_daily = sum(daily_returns) / n_days
    m2 = sum((r - mean_daily) ** 2 for r in daily_returns) / n_days
    m3 = sum((r - mean_daily) ** 3 for r in daily_returns) / n_days
    m4 = sum((r - mean_daily) ** 4 for r in daily_returns) / n_days
    skew = m3 / m2**1.5 if m2 > 0 else 0.0
    excess_kurtosis = m4 / m2**2 - 3.0 if m2 > 0 else 0.0

    sharpe: float | None = None
    if n_days >= 2:
        sample_var = sum((r - mean_daily) ** 2 for r in daily_returns) / (n_days - 1)
        if sample_var > 0:
            sharpe = mean_daily / math.sqrt(sample_var)

    sorted_daily = sorted(daily_returns)
    day_contribution = _contribution(max(daily_returns), sum(daily_returns))

    event_contribution: float | None = None
    if trade_events is not None:
        event_pnl: dict[str, float] = {}
        for event, r in zip(trade_events, trade_returns_net, strict=True):
            event_pnl[event] = event_pnl.get(event, 0.0) + r
        event_contribution = _contribution(max(event_pnl.values()), sum(trade_returns_net))

    alpha: float | None = None
    beta: float | None = None
    if benchmark_daily_returns is not None and n_days >= 2:
        mean_bench = sum(benchmark_daily_returns) / n_days
        cov = sum(
            (a - mean_daily) * (b - mean_bench)
            for a, b in zip(daily_returns, benchmark_daily_returns, strict=True)
        ) / (n_days - 1)
        var_bench = sum((b - mean_bench) ** 2 for b in benchmark_daily_returns) / (n_days - 1)
        if var_bench > 0:
            beta = cov / var_bench
            alpha = mean_daily - beta * mean_bench

    inf = inference or InferenceInputs()
    return MetricsPanel(
        n_trades=n_trades,
        n_days=n_days,
        gross_return=sum(trade_returns_gross),
        net_return=sum(trade_returns_net),
        net_bps_per_trade=mean_trade * 10_000,
        profit_factor=profit_factor,
        win_rate=wins / n_trades,
        mean_trade_return=mean_trade,
        skew=skew,
        excess_kurtosis=excess_kurtosis,
        max_drawdown=_max_drawdown(daily_returns),
        tail_loss_95=_quantile(sorted_daily, 0.05),
        tail_loss_99=_quantile(sorted_daily, 0.01),
        max_single_day_contribution=day_contribution,
        max_single_event_contribution=event_contribution,
        sharpe=sharpe,
        mean_trade_return_ci_low=inf.mean_trade_return_ci_low,
        mean_trade_return_ci_high=inf.mean_trade_return_ci_high,
        day_clustered_tstat=inf.day_clustered_tstat,
        day_clustered_pvalue=inf.day_clustered_pvalue,
        event_clustered_tstat=inf.event_clustered_tstat,
        event_clustered_pvalue=inf.event_clustered_pvalue,
        deflated_sharpe=inf.deflated_sharpe,
        pbo=inf.pbo,
        null_percentile=inf.null_percentile,
        benchmark_alpha_per_period=alpha,
        benchmark_beta=beta,
        portfolio_incremental_sharpe=portfolio_incremental_sharpe,
        cost_stress=dict(cost_stress) if cost_stress is not None else None,
    )


def write_metrics_panel_csv(panel: MetricsPanel, path: Path) -> None:
    """Write the panel as ``field,value`` rows; None values emit empty."""
    path.parent.mkdir(parents=True, exist_ok=True)
    panel_fields = {f.name for f in fields(panel)}
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["field", "value"])
        for name in METRICS_PANEL_FIELDS:
            if name not in panel_fields:
                raise ValueError(f"panel is missing declared field '{name}'")
            value = getattr(panel, name)
            writer.writerow([name, "" if value is None else value])
        for scenario_id, value in sorted((panel.cost_stress or {}).items()):
            writer.writerow([f"cost_stress.{scenario_id}", value])
