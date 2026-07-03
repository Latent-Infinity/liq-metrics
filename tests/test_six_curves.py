"""TDD pins for the six-curve NAV set (A/B/C/D/E + F views).

Every experiment reports all curves on identical dates and identical capital:

* A  policy baseline            * D  active-sleeve NAV on displaced capital
* B  strategy standalone        * E  implementation-shortfall NAV (measured costs)
* C  core + overlay             * F1/F2/F3 after-tax views (via tax_curves)
* A3 leverage/beta-matched baseline net of financing (comparator for levered books)

Identity pins: a null overlay makes C equal A on curve E; a zero-rate policy
makes F1 equal the pre-tax NAV; leverage 1 with zero financing makes A3 equal A.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from liq.metrics.six_curves import SixCurveInputs, SixCurveResult, compute_six_curves
from liq.metrics.tax_curves import RealizedTaxEvent, TaxPolicy, TaxRates

D = Decimal
DATES = (date(2026, 1, 31), date(2026, 2, 28), date(2026, 3, 31))
ZERO_POLICY = TaxPolicy(rates=TaxRates.zero())


def _inputs(**overrides) -> SixCurveInputs:
    base = {
        "dates": DATES,
        "starting_capital": D("50000"),
        "baseline_returns": (D("0.01"), D("-0.02"), D("0.03")),
        "overlay_returns": (D("0.02"), D("0.01"), D("-0.01")),
        "sleeve_weight": D("0.2"),
        "measured_costs": (D("0.0005"), D("0.0005"), D("0.0005")),
        "tax_policy": ZERO_POLICY,
    }
    base.update(overrides)
    return SixCurveInputs(**base)


class TestCurveShapes:
    def test_all_curves_share_dates_and_start_capital(self) -> None:
        result = compute_six_curves(_inputs())
        assert isinstance(result, SixCurveResult)
        for curve in (result.a, result.b, result.c, result.d, result.e):
            assert len(curve) == len(DATES)
        # curve A first point: 50000 * 1.01
        assert result.a[0] == D("50500.00")

    def test_curve_a_compounds_baseline(self) -> None:
        result = compute_six_curves(_inputs())
        expected = D("50000") * D("1.01") * D("0.98") * D("1.03")
        assert result.a[-1] == expected.quantize(D("0.01"))

    def test_curve_b_is_standalone_overlay(self) -> None:
        result = compute_six_curves(_inputs())
        expected = D("50000") * D("1.02") * D("1.01") * D("0.99")
        assert result.b[-1] == expected.quantize(D("0.01"))

    def test_curve_c_blends_by_sleeve_weight(self) -> None:
        result = compute_six_curves(_inputs())
        # period-1 blended return: 0.8*0.01 + 0.2*0.02 = 0.012
        assert result.c[0] == (D("50000") * D("1.012")).quantize(D("0.01"))

    def test_curve_d_runs_overlay_on_displaced_capital(self) -> None:
        result = compute_six_curves(_inputs())
        displaced = D("50000") * D("0.2")
        expected = displaced * D("1.02") * D("1.01") * D("0.99")
        assert result.d[-1] == expected.quantize(D("0.01"))

    def test_curve_e_nets_measured_costs_from_c(self) -> None:
        result = compute_six_curves(_inputs())
        # period-1: blended 0.012 minus 5 bps cost = 0.0115
        assert result.e[0] == (D("50000") * D("1.0115")).quantize(D("0.01"))
        assert result.e[-1] < result.c[-1]


class TestIdentities:
    def test_null_overlay_makes_c_equal_a_on_curve_e(self) -> None:
        """Plan acceptance: with a null strategy, C = A on curve E."""
        inputs = _inputs(sleeve_weight=D("0"), measured_costs=(D("0"),) * 3)
        result = compute_six_curves(inputs)
        assert result.c == result.a
        assert result.e == result.a

    def test_f1_equals_pre_tax_nav_when_rates_zero(self) -> None:
        """Plan acceptance: F1 = pre-tax NAV when the policy zeroes all rates."""
        events = (
            RealizedTaxEvent(
                event_date=date(2026, 2, 10), amount=D("1000"), character="short_term"
            ),
        )
        result = compute_six_curves(_inputs(tax_events=events))
        assert result.f.f1_nav == result.e[-1]
        assert result.f.f1_realized_tax == D("0")

    def test_f1_debits_tax_under_nonzero_rates(self) -> None:
        policy = TaxPolicy(
            rates=TaxRates(
                short_term=D("0.24"),
                long_term=D("0.15"),
                qualified_dividend=D("0.15"),
                nonqualified_dividend=D("0.24"),
            )
        )
        events = (
            RealizedTaxEvent(
                event_date=date(2026, 2, 10), amount=D("1000"), character="short_term"
            ),
        )
        result = compute_six_curves(_inputs(tax_policy=policy, tax_events=events))
        assert result.f.f1_realized_tax == D("240.00")
        assert result.f.f1_nav == result.e[-1] - D("240.00")


class TestLeverageMatchedComparator:
    def test_a3_equals_a_at_unit_leverage_and_zero_financing(self) -> None:
        result = compute_six_curves(_inputs())
        assert result.a3 == result.a

    def test_a3_levers_baseline_net_of_financing(self) -> None:
        inputs = _inputs(
            leverage=D("2"),
            financing_rates=(D("0.001"), D("0.001"), D("0.001")),
        )
        result = compute_six_curves(inputs)
        # period-1: 2*0.01 - (2-1)*0.001 = 0.019
        assert result.a3[0] == (D("50000") * D("1.019")).quantize(D("0.01"))

    def test_levered_book_must_not_be_judged_only_vs_plain_a(self) -> None:
        """A levered C that beats A but not A3 is not excess (leverage-as-beta)."""
        inputs = _inputs(
            leverage=D("2"),
            financing_rates=(D("0"),) * 3,
            sleeve_weight=D("0"),
            measured_costs=(D("0"),) * 3,
            baseline_returns=(D("0.01"), D("0.01"), D("0.01")),
        )
        result = compute_six_curves(inputs)
        assert result.a3[-1] > result.a[-1]  # the comparator is strictly harder


class TestValidation:
    def test_mismatched_series_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="identical dates"):
            compute_six_curves(_inputs(measured_costs=(D("0"),)))

    def test_empty_dates_raise_domain_error(self) -> None:
        with pytest.raises(ValueError, match="at least one date"):
            compute_six_curves(
                _inputs(
                    dates=(),
                    baseline_returns=(),
                    overlay_returns=(),
                    measured_costs=(),
                )
            )

    def test_negative_sleeve_weight_raises(self) -> None:
        with pytest.raises(ValueError, match="sleeve_weight"):
            compute_six_curves(_inputs(sleeve_weight=D("-0.1")))

    def test_weight_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="sleeve_weight"):
            compute_six_curves(_inputs(sleeve_weight=D("1.5")))
