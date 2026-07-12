"""Tests for the shared research metrics panel."""

import csv
import math
from pathlib import Path

import pytest

from liq.metrics.panel import (
    METRICS_PANEL_FIELDS,
    InferenceInputs,
    MetricsPanel,
    compute_metrics_panel,
    write_metrics_panel_csv,
)

TRADES_NET = [0.01, -0.005, 0.02, -0.01]
TRADES_GROSS = [0.012, -0.003, 0.022, -0.008]
DAILY = [0.01, -0.005, 0.02, -0.01, 0.004]
BENCH = [0.008, -0.004, 0.015, -0.008, 0.003]


def _panel(**overrides: object) -> MetricsPanel:
    kwargs: dict = {
        "trade_returns_net": TRADES_NET,
        "trade_returns_gross": TRADES_GROSS,
        "daily_returns": DAILY,
    }
    kwargs.update(overrides)
    return compute_metrics_panel(**kwargs)


class TestTradeMetrics:
    def test_returns_and_bps(self) -> None:
        panel = _panel()
        assert panel.gross_return == pytest.approx(0.023)
        assert panel.net_return == pytest.approx(0.015)
        assert panel.net_bps_per_trade == pytest.approx(37.5)
        assert panel.n_trades == 4

    def test_profit_factor_and_win_rate(self) -> None:
        panel = _panel()
        assert panel.profit_factor == pytest.approx(2.0)
        assert panel.win_rate == pytest.approx(0.5)

    def test_mean_trade_return(self) -> None:
        assert _panel().mean_trade_return == pytest.approx(0.00375)

    def test_profit_factor_with_no_losers_is_infinite(self) -> None:
        panel = _panel(
            trade_returns_net=[0.01, 0.02],
            trade_returns_gross=[0.011, 0.021],
        )
        assert math.isinf(panel.profit_factor)


class TestDailyMetrics:
    def test_sharpe(self) -> None:
        assert _panel().sharpe == pytest.approx(0.3186645834368642, abs=1e-12)

    def test_skew_and_excess_kurtosis(self) -> None:
        panel = _panel()
        assert panel.skew == pytest.approx(0.19455633954831855, abs=1e-12)
        assert panel.excess_kurtosis == pytest.approx(-1.2595812043416594, abs=1e-12)

    def test_max_drawdown_on_compounded_equity(self) -> None:
        assert _panel().max_drawdown == pytest.approx(0.01, abs=1e-12)

    def test_tail_losses_are_low_quantiles(self) -> None:
        panel = _panel()
        assert panel.tail_loss_95 == pytest.approx(-0.009, abs=1e-12)
        assert panel.tail_loss_99 == pytest.approx(-0.0098, abs=1e-12)

    def test_max_single_day_contribution(self) -> None:
        # Largest daily return 0.02 against total daily P&L 0.019.
        panel = _panel()
        assert panel.max_single_day_contribution == pytest.approx(0.02 / 0.019)

    def test_n_days(self) -> None:
        assert _panel().n_days == 5


class TestEventContribution:
    def test_max_single_event_contribution(self) -> None:
        panel = _panel(trade_events=["e1", "e1", "e2", "e3"])
        # Event sums: e1 = 0.005, e2 = 0.02, e3 = -0.01; largest / total net.
        assert panel.max_single_event_contribution == pytest.approx(0.02 / 0.015)

    def test_without_events_is_none(self) -> None:
        assert _panel().max_single_event_contribution is None


class TestBenchmark:
    def test_alpha_beta_vs_benchmark(self) -> None:
        panel = _panel(benchmark_daily_returns=BENCH)
        assert panel.benchmark_beta == pytest.approx(1.295159386068477, abs=1e-12)
        assert panel.benchmark_alpha_per_period == pytest.approx(0.0001735537190082656, abs=1e-15)

    def test_without_benchmark_is_none(self) -> None:
        panel = _panel()
        assert panel.benchmark_beta is None
        assert panel.benchmark_alpha_per_period is None

    def test_benchmark_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="benchmark"):
            _panel(benchmark_daily_returns=[0.01])


class TestInferencePassthrough:
    def test_inference_fields_carried(self) -> None:
        inference = InferenceInputs(
            mean_trade_return_ci_low=-0.001,
            mean_trade_return_ci_high=0.008,
            day_clustered_tstat=1.9,
            day_clustered_pvalue=0.07,
            deflated_sharpe=0.62,
            pbo=0.3,
            null_percentile=0.97,
        )
        panel = _panel(inference=inference)
        assert panel.mean_trade_return_ci_low == pytest.approx(-0.001)
        assert panel.deflated_sharpe == pytest.approx(0.62)
        assert panel.pbo == pytest.approx(0.3)
        assert panel.null_percentile == pytest.approx(0.97)

    def test_defaults_are_none(self) -> None:
        panel = _panel()
        assert panel.deflated_sharpe is None
        assert panel.day_clustered_tstat is None


class TestValidation:
    def test_trade_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="gross"):
            _panel(trade_returns_gross=[0.01])

    def test_empty_daily_raises(self) -> None:
        with pytest.raises(ValueError, match="daily"):
            _panel(daily_returns=[])

    def test_empty_trades_raise(self) -> None:
        with pytest.raises(ValueError, match="trade"):
            _panel(trade_returns_net=[], trade_returns_gross=[])

    def test_event_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="events"):
            _panel(trade_events=["e1"])


class TestCsvEmission:
    def test_emits_every_field(self, tmp_path: Path) -> None:
        """Acceptance: the panel emits completely on committed fixtures."""
        panel = _panel(
            benchmark_daily_returns=BENCH,
            trade_events=["e1", "e1", "e2", "e3"],
            cost_stress={"spy_qqq_stress_3x_v1": 0.004},
            portfolio_incremental_sharpe=0.12,
        )
        path = tmp_path / "metrics_panel.csv"
        write_metrics_panel_csv(panel, path)
        with path.open() as fh:
            rows = list(csv.reader(fh))
        assert rows[0] == ["field", "value"]
        emitted = {row[0] for row in rows[1:]}
        assert set(METRICS_PANEL_FIELDS) <= emitted
        assert "cost_stress.spy_qqq_stress_3x_v1" in emitted

    def test_none_fields_emit_empty_values(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics_panel.csv"
        write_metrics_panel_csv(_panel(), path)
        with path.open() as fh:
            values = {row["field"]: row["value"] for row in csv.DictReader(fh)}
        assert set(METRICS_PANEL_FIELDS) <= set(values)
        assert values["deflated_sharpe"] == ""
        assert values["sharpe"] != ""

    def test_infinite_profit_factor_serializes(self, tmp_path: Path) -> None:
        panel = _panel(
            trade_returns_net=[0.01, 0.02],
            trade_returns_gross=[0.011, 0.021],
        )
        path = tmp_path / "metrics_panel.csv"
        write_metrics_panel_csv(panel, path)
        with path.open() as fh:
            values = {row["field"]: row["value"] for row in csv.DictReader(fh)}
        assert values["profit_factor"] == "inf"
