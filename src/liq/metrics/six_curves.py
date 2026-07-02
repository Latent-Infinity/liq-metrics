"""Six-curve NAV set: the evaluation contract every experiment reports.

All curves are computed on identical dates and identical starting capital so
comparisons are honest by construction:

* ``a``  policy baseline
* ``b``  strategy standalone (diagnostic only)
* ``c``  core + overlay (deployable portfolio, pre-cost)
* ``d``  active-sleeve NAV on displaced capital
* ``e``  implementation-shortfall NAV (curve C net of measured costs)
* ``f``  after-tax views F1/F2/F3 computed on curve E via :mod:`tax_curves`
* ``a3`` leverage/beta-matched baseline net of financing — the required
  comparator for any levered book (leverage is beta plus financing cost,
  never alpha; a levered curve that beats ``a`` but not ``a3`` shows no excess)

Pure Decimal arithmetic; orchestration and IO live in ``liq-runner``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

from liq.metrics.tax_curves import (
    CurveFResult,
    OpenTaxPosition,
    RealizedTaxEvent,
    TaxPolicy,
    compute_curve_f,
)

_CENT = Decimal("0.01")
_ONE = Decimal("1")


@dataclass(frozen=True)
class SixCurveInputs:
    """Aligned per-period inputs for one experiment's curve set.

    Returns and costs are fractional per-period values aligned to ``dates``.
    ``measured_costs`` is the deployable book's realized cost drag per period
    (commissions + slippage + financing) as a fraction of NAV.
    """

    dates: tuple[date, ...]
    starting_capital: Decimal
    baseline_returns: tuple[Decimal, ...]
    overlay_returns: tuple[Decimal, ...]
    sleeve_weight: Decimal
    measured_costs: tuple[Decimal, ...]
    tax_policy: TaxPolicy
    tax_events: tuple[RealizedTaxEvent, ...] = ()
    open_positions: tuple[OpenTaxPosition, ...] = ()
    leverage: Decimal = _ONE
    financing_rates: tuple[Decimal, ...] = field(default=())


@dataclass(frozen=True)
class SixCurveResult:
    """NAV series per curve (aligned to the input dates) plus the F views."""

    dates: tuple[date, ...]
    a: tuple[Decimal, ...]
    b: tuple[Decimal, ...]
    c: tuple[Decimal, ...]
    d: tuple[Decimal, ...]
    e: tuple[Decimal, ...]
    a3: tuple[Decimal, ...]
    f: CurveFResult


def _compound(start: Decimal, returns: tuple[Decimal, ...]) -> tuple[Decimal, ...]:
    nav = start
    series: list[Decimal] = []
    for r in returns:
        nav = nav * (_ONE + r)
        series.append(nav.quantize(_CENT))
    return tuple(series)


def _validate(inputs: SixCurveInputs) -> tuple[Decimal, ...]:
    n = len(inputs.dates)
    financing = inputs.financing_rates or (Decimal("0"),) * n
    series = {
        "baseline_returns": inputs.baseline_returns,
        "overlay_returns": inputs.overlay_returns,
        "measured_costs": inputs.measured_costs,
        "financing_rates": financing,
    }
    for name, values in series.items():
        if len(values) != n:
            raise ValueError(
                f"all curves require identical dates and capital: {name} has "
                f"{len(values)} periods for {n} dates"
            )
    if not (Decimal("0") <= inputs.sleeve_weight <= _ONE):
        raise ValueError(f"sleeve_weight must be in [0, 1], got {inputs.sleeve_weight}")
    if inputs.leverage < _ONE:
        raise ValueError(f"leverage must be >= 1, got {inputs.leverage}")
    return financing


def compute_six_curves(inputs: SixCurveInputs) -> SixCurveResult:
    """Compute the full curve set from aligned per-period inputs."""
    financing = _validate(inputs)
    w = inputs.sleeve_weight
    core_w = _ONE - w

    blended = tuple(
        core_w * a + w * b
        for a, b in zip(inputs.baseline_returns, inputs.overlay_returns, strict=True)
    )
    shortfall = tuple(c - cost for c, cost in zip(blended, inputs.measured_costs, strict=True))
    levered = tuple(
        inputs.leverage * a - (inputs.leverage - _ONE) * fin
        for a, fin in zip(inputs.baseline_returns, financing, strict=True)
    )

    curve_e = _compound(inputs.starting_capital, shortfall)
    curve_f = compute_curve_f(
        pre_tax_nav=curve_e[-1],
        realized_events=inputs.tax_events,
        open_positions=inputs.open_positions,
        policy=inputs.tax_policy,
        period_start=inputs.dates[0],
        period_end=inputs.dates[-1],
    )

    return SixCurveResult(
        dates=inputs.dates,
        a=_compound(inputs.starting_capital, inputs.baseline_returns),
        b=_compound(inputs.starting_capital, inputs.overlay_returns),
        c=_compound(inputs.starting_capital, blended),
        d=_compound(inputs.starting_capital * w, inputs.overlay_returns),
        e=curve_e,
        a3=_compound(inputs.starting_capital, levered),
        f=curve_f,
    )
