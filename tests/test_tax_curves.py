"""Tests for Curve-F after-tax views."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from liq.metrics.tax_curves import (
    CurveFPeriod,
    OpenTaxPosition,
    RealizedTaxEvent,
    TaxPolicy,
    TaxRates,
    TerminalAssumption,
    compute_curve_f,
    compute_curve_f_periods,
    event_tax,
    mark_to_market_tax,
)


def _policy(terminal: TerminalAssumption = "step_up") -> TaxPolicy:
    return TaxPolicy(
        rates=TaxRates(
            short_term=Decimal("0.37"),
            long_term=Decimal("0.20"),
            qualified_dividend=Decimal("0.15"),
            nonqualified_dividend=Decimal("0.37"),
        ),
        terminal_assumption=terminal,
    )


def test_f1_matches_hand_calculated_two_year_synthetic_case() -> None:
    events = (
        RealizedTaxEvent(date(2025, 3, 3), Decimal("1000.00"), "short_term"),
        RealizedTaxEvent(date(2025, 9, 9), Decimal("2000.00"), "long_term"),
        RealizedTaxEvent(date(2026, 1, 15), Decimal("100.00"), "qualified_dividend"),
        RealizedTaxEvent(date(2026, 2, 20), Decimal("50.00"), "nonqualified_dividend"),
        RealizedTaxEvent(date(2026, 5, 2), Decimal("-300.00"), "short_term"),
        RealizedTaxEvent(date(2026, 6, 1), Decimal("200.00"), "wash_sale_disallowed_loss"),
    )
    open_positions = (
        OpenTaxPosition("AAPL", Decimal("10000.00"), Decimal("15000.00"), "long_term"),
    )

    result = compute_curve_f(
        pre_tax_nav=Decimal("100000.00"),
        realized_events=events,
        open_positions=open_positions,
        policy=_policy("step_up"),
        period_start=date(2025, 1, 1),
        period_end=date(2026, 12, 31),
    )

    # ST gain 370 + LT gain 400 + qualified dividend 15
    # + nonqualified dividend 18.50 - ST loss benefit 111.00.
    # The disallowed wash-sale loss has no current-period tax benefit.
    assert result.f1_realized_tax == Decimal("692.50")
    assert result.f1_nav == Decimal("99307.50")
    assert result.f2_liquidation_tax == Decimal("1692.50")
    assert result.f2_nav == Decimal("98307.50")
    assert result.f3_terminal_tax == Decimal("692.50")
    assert result.f3_nav == Decimal("99307.50")


def test_f2_nav_is_below_f1_nav_in_pure_gain_scenario() -> None:
    result = compute_curve_f(
        pre_tax_nav=Decimal("50000.00"),
        realized_events=(RealizedTaxEvent(date(2026, 1, 1), Decimal("100.00"), "long_term"),),
        open_positions=(
            OpenTaxPosition("MSFT", Decimal("10000.00"), Decimal("12000.00"), "long_term"),
            OpenTaxPosition("NVDA", Decimal("2000.00"), Decimal("2500.00"), "short_term"),
        ),
        policy=_policy("hold_forever"),
    )

    assert result.f2_liquidation_tax > result.f1_realized_tax
    assert result.f2_nav <= result.f1_nav


def test_f2_equals_f1_when_no_open_positions_remain() -> None:
    result = compute_curve_f(
        pre_tax_nav=Decimal("50000.00"),
        realized_events=(RealizedTaxEvent(date(2026, 1, 1), Decimal("500.00"), "short_term"),),
        policy=_policy("liquidate"),
    )

    assert result.f2_liquidation_tax == result.f1_realized_tax
    assert result.f2_nav == result.f1_nav


def test_f3_step_up_ignores_unrealized_gain_tax() -> None:
    open_positions = (
        OpenTaxPosition("GOOG", Decimal("10000.00"), Decimal("20000.00"), "long_term"),
    )
    step_up = compute_curve_f(
        pre_tax_nav=Decimal("60000.00"),
        realized_events=(),
        open_positions=open_positions,
        policy=_policy("step_up"),
    )
    liquidate = compute_curve_f(
        pre_tax_nav=Decimal("60000.00"),
        realized_events=(),
        open_positions=open_positions,
        policy=_policy("liquidate"),
    )

    assert step_up.f3_terminal_tax == Decimal("0.00")
    assert step_up.f3_nav == Decimal("60000.00")
    assert liquidate.f3_terminal_tax == Decimal("2000.00")


def test_no_tax_smoke_mode_zeroes_all_curve_f_tax_views() -> None:
    policy = TaxPolicy(rates=_policy().rates, terminal_assumption="liquidate", no_tax_smoke=True)
    result = compute_curve_f(
        pre_tax_nav=Decimal("50000.00"),
        realized_events=(RealizedTaxEvent(date(2026, 1, 1), Decimal("1000.00"), "short_term"),),
        open_positions=(
            OpenTaxPosition("AMD", Decimal("1000.00"), Decimal("5000.00"), "short_term"),
        ),
        policy=policy,
    )

    assert result.f1_realized_tax == Decimal("0.00")
    assert result.f2_liquidation_tax == Decimal("0.00")
    assert result.f3_terminal_tax == Decimal("0.00")
    assert result.f1_nav == Decimal("50000.00")
    assert result.f2_nav == Decimal("50000.00")
    assert result.f3_nav == Decimal("50000.00")


def test_period_batch_filters_events_by_inclusive_bounds() -> None:
    events = (
        RealizedTaxEvent(date(2025, 12, 31), Decimal("1000.00"), "short_term"),
        RealizedTaxEvent(date(2026, 1, 1), Decimal("1000.00"), "long_term"),
        RealizedTaxEvent(date(2026, 12, 31), Decimal("100.00"), "qualified_dividend"),
        RealizedTaxEvent(date(2027, 1, 1), Decimal("1000.00"), "short_term"),
    )
    periods = (
        CurveFPeriod("2025", date(2025, 1, 1), date(2025, 12, 31), Decimal("10000.00")),
        CurveFPeriod("2026", date(2026, 1, 1), date(2026, 12, 31), Decimal("20000.00")),
    )

    result = compute_curve_f_periods(periods, events, _policy())

    assert result["2025"].f1_realized_tax == Decimal("370.00")
    assert result["2026"].f1_realized_tax == Decimal("215.00")


def test_wash_sale_disallowed_loss_has_no_current_tax_benefit() -> None:
    policy = _policy()
    event = RealizedTaxEvent(date(2026, 1, 1), Decimal("1000.00"), "wash_sale_disallowed_loss")

    assert event_tax(event, policy) == Decimal("0")


def test_unrealized_losses_create_liquidation_tax_benefit() -> None:
    positions = (OpenTaxPosition("INTC", Decimal("10000.00"), Decimal("9000.00"), "short_term"),)

    assert mark_to_market_tax(positions, _policy()) == Decimal("-370.00")
