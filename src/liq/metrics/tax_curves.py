"""After-tax Curve-F views for portfolio evaluation.

The module intentionally consumes plain records instead of tax-lot ledgers.
Lot construction, wash-sale detection, and broker reconciliation remain
upstream concerns; Curve-F only applies a supplied account policy to realized
tax events and open-position marks.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from typing import Literal

DOLLAR = Decimal("0.01")

EventCharacter = Literal[
    "short_term",
    "long_term",
    "qualified_dividend",
    "nonqualified_dividend",
    "wash_sale_disallowed_loss",
]
HoldingPeriod = Literal["short_term", "long_term"]
TerminalAssumption = Literal["step_up", "donation", "hold_forever", "liquidate"]


@dataclass(frozen=True)
class TaxRates:
    """Account-level marginal tax rates used by Curve-F."""

    short_term: Decimal
    long_term: Decimal
    qualified_dividend: Decimal
    nonqualified_dividend: Decimal

    @classmethod
    def zero(cls) -> TaxRates:
        """Return a no-tax smoke policy rate set."""
        return cls(
            short_term=Decimal("0"),
            long_term=Decimal("0"),
            qualified_dividend=Decimal("0"),
            nonqualified_dividend=Decimal("0"),
        )


@dataclass(frozen=True)
class TaxPolicy:
    """Tax policy inputs needed by Curve-F.

    ``no_tax_smoke`` is a documented pre-promotion mode. It is useful for
    plumbing identity tests, but not for capital-facing after-tax claims.
    """

    rates: TaxRates
    terminal_assumption: TerminalAssumption = "hold_forever"
    no_tax_smoke: bool = False


@dataclass(frozen=True)
class RealizedTaxEvent:
    """Taxable event already produced by an upstream lot/accounting process.

    ``amount`` is positive for gains/income and negative for realized loss
    deductions. A ``wash_sale_disallowed_loss`` row is informational for
    Curve-F and carries no current-period tax benefit; its basis adjustment
    should appear in a later realized event if applicable.
    """

    event_date: date
    amount: Decimal
    character: EventCharacter


@dataclass(frozen=True)
class OpenTaxPosition:
    """Open-position mark used for liquidation-adjusted and terminal views."""

    symbol: str
    cost_basis: Decimal
    market_value: Decimal
    holding_period: HoldingPeriod


@dataclass(frozen=True)
class CurveFResult:
    """F1/F2/F3 after-tax NAV and tax debit for one measurement period."""

    pre_tax_nav: Decimal
    f1_realized_tax: Decimal
    f2_liquidation_tax: Decimal
    f3_terminal_tax: Decimal
    f1_nav: Decimal
    f2_nav: Decimal
    f3_nav: Decimal


@dataclass(frozen=True)
class CurveFPeriod:
    """Period input for batch Curve-F calculations."""

    label: str
    start: date
    end: date
    pre_tax_nav: Decimal
    open_positions: tuple[OpenTaxPosition, ...] = ()


def event_tax(event: RealizedTaxEvent, policy: TaxPolicy) -> Decimal:
    """Return the current-period tax debit for one realized event."""
    if policy.no_tax_smoke:
        return Decimal("0")
    if event.character == "wash_sale_disallowed_loss":
        return Decimal("0")
    return _money(event.amount * _rate_for_event(event.character, policy.rates))


def realized_tax(
    events: list[RealizedTaxEvent] | tuple[RealizedTaxEvent, ...],
    policy: TaxPolicy,
    *,
    period_start: date | None = None,
    period_end: date | None = None,
) -> Decimal:
    """Apply policy rates to realized events inside optional inclusive bounds."""
    total = Decimal("0")
    for event in events:
        if period_start is not None and event.event_date < period_start:
            continue
        if period_end is not None and event.event_date > period_end:
            continue
        total += event_tax(event, policy)
    return _money(total)


def mark_to_market_tax(
    open_positions: list[OpenTaxPosition] | tuple[OpenTaxPosition, ...],
    policy: TaxPolicy,
) -> Decimal:
    """Tax debit from liquidating all open positions at current marks."""
    if policy.no_tax_smoke:
        return Decimal("0")
    total = Decimal("0")
    for position in open_positions:
        unrealized = position.market_value - position.cost_basis
        rate = (
            policy.rates.short_term
            if position.holding_period == "short_term"
            else policy.rates.long_term
        )
        total += unrealized * rate
    return _money(total)


def terminal_open_position_tax(
    open_positions: list[OpenTaxPosition] | tuple[OpenTaxPosition, ...],
    policy: TaxPolicy,
) -> Decimal:
    """Tax debit on open positions under the policy terminal assumption."""
    if policy.terminal_assumption in {"step_up", "donation", "hold_forever"}:
        return Decimal("0")
    if policy.terminal_assumption == "liquidate":
        return mark_to_market_tax(open_positions, policy)
    raise ValueError(f"unknown terminal assumption: {policy.terminal_assumption}")


def compute_curve_f(
    *,
    pre_tax_nav: Decimal,
    realized_events: list[RealizedTaxEvent] | tuple[RealizedTaxEvent, ...],
    open_positions: list[OpenTaxPosition] | tuple[OpenTaxPosition, ...] = (),
    policy: TaxPolicy,
    period_start: date | None = None,
    period_end: date | None = None,
) -> CurveFResult:
    """Compute F1 realized, F2 liquidation-adjusted, and F3 terminal-policy NAV."""
    f1_tax = realized_tax(
        realized_events,
        policy,
        period_start=period_start,
        period_end=period_end,
    )
    mtm_tax = mark_to_market_tax(open_positions, policy)
    f2_tax = _money(f1_tax + mtm_tax)
    f3_tax = _money(f1_tax + terminal_open_position_tax(open_positions, policy))

    nav = _money(pre_tax_nav)
    return CurveFResult(
        pre_tax_nav=nav,
        f1_realized_tax=f1_tax,
        f2_liquidation_tax=f2_tax,
        f3_terminal_tax=f3_tax,
        f1_nav=_money(nav - f1_tax),
        f2_nav=_money(nav - f2_tax),
        f3_nav=_money(nav - f3_tax),
    )


def compute_curve_f_periods(
    periods: list[CurveFPeriod] | tuple[CurveFPeriod, ...],
    realized_events: list[RealizedTaxEvent] | tuple[RealizedTaxEvent, ...],
    policy: TaxPolicy,
) -> dict[str, CurveFResult]:
    """Compute Curve-F results for multiple named periods."""
    return {
        period.label: compute_curve_f(
            pre_tax_nav=period.pre_tax_nav,
            realized_events=realized_events,
            open_positions=period.open_positions,
            policy=policy,
            period_start=period.start,
            period_end=period.end,
        )
        for period in periods
    }


def _rate_for_event(character: EventCharacter, rates: TaxRates) -> Decimal:
    if character == "short_term":
        return rates.short_term
    if character == "long_term":
        return rates.long_term
    if character == "qualified_dividend":
        return rates.qualified_dividend
    if character == "nonqualified_dividend":
        return rates.nonqualified_dividend
    raise ValueError(f"unsupported realized tax event character: {character}")


def _money(value: Decimal) -> Decimal:
    return value.quantize(DOLLAR, rounding=ROUND_HALF_UP)
