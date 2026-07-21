"""Economic accounting for fixed no-trade selector decisions."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class SelectorEconomics:
    """Metrics comparing one candidate selector with its reference policy."""

    reference_trade_count: int
    candidate_trade_count: int
    equal_trade_count: bool
    avoided_loss: float
    missed_profit: float
    avoided_loss_minus_missed_profit: float
    added_net_pnl: float
    net_pnl_delta: float
    base_gross_profit_pool: float
    missed_profit_fraction: float | None
    reference_sharpe: float | None
    candidate_sharpe: float | None
    sharpe_delta: float | None

    def as_dict(self) -> dict[str, float | int | bool | None]:
        """Return a serialization-safe metrics mapping."""

        return asdict(self)


def compute_selector_economics(
    *,
    reference_decision: Sequence[int],
    candidate_decision: Sequence[int],
    net_outcomes: Sequence[float],
    gross_outcomes: Sequence[float],
) -> SelectorEconomics:
    """Compare fixed candidate/reference decisions over aligned event outcomes."""

    reference = _binary_decisions(reference_decision, name="reference_decision")
    candidate = _binary_decisions(candidate_decision, name="candidate_decision")
    net = _finite_outcomes(net_outcomes, name="net_outcomes")
    gross = _finite_outcomes(gross_outcomes, name="gross_outcomes")
    lengths = {len(reference), len(candidate), len(net), len(gross)}
    if len(lengths) != 1 or not reference:
        raise ValueError("selector decisions and outcomes must be non-empty and align")

    removed = tuple(r == 1 and c == 0 for r, c in zip(reference, candidate, strict=True))
    added = tuple(r == 0 and c == 1 for r, c in zip(reference, candidate, strict=True))
    avoided_loss = -sum(
        value for value, flag in zip(net, removed, strict=True) if flag and value < 0
    )
    missed_profit = sum(
        value for value, flag in zip(net, removed, strict=True) if flag and value > 0
    )
    added_net = sum(value for value, flag in zip(net, added, strict=True) if flag)
    reference_returns = tuple(
        value * decision for value, decision in zip(net, reference, strict=True)
    )
    candidate_returns = tuple(
        value * decision for value, decision in zip(net, candidate, strict=True)
    )
    reference_sharpe = _sharpe(reference_returns)
    candidate_sharpe = _sharpe(candidate_returns)
    sharpe_delta = (
        candidate_sharpe - reference_sharpe
        if candidate_sharpe is not None and reference_sharpe is not None
        else None
    )
    profit_pool = sum(
        value for value, decision in zip(gross, reference, strict=True) if decision and value > 0
    )
    return SelectorEconomics(
        reference_trade_count=sum(reference),
        candidate_trade_count=sum(candidate),
        equal_trade_count=sum(reference) == sum(candidate),
        avoided_loss=avoided_loss,
        missed_profit=missed_profit,
        avoided_loss_minus_missed_profit=avoided_loss - missed_profit,
        added_net_pnl=added_net,
        net_pnl_delta=sum(candidate_returns) - sum(reference_returns),
        base_gross_profit_pool=profit_pool,
        missed_profit_fraction=missed_profit / profit_pool if profit_pool > 0 else None,
        reference_sharpe=reference_sharpe,
        candidate_sharpe=candidate_sharpe,
        sharpe_delta=sharpe_delta,
    )


def _binary_decisions(values: Sequence[int], *, name: str) -> tuple[int, ...]:
    normalized = tuple(values)
    if any(isinstance(value, bool) or value not in (0, 1) for value in normalized):
        raise ValueError(f"{name} must be binary")
    return normalized


def _finite_outcomes(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    normalized = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _sharpe(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean / math.sqrt(variance) if variance > 0 else None


__all__ = ["SelectorEconomics", "compute_selector_economics"]
