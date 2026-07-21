"""Economic accounting for count-matched no-trade selectors."""

from __future__ import annotations

import pytest

from liq.metrics.selector import compute_selector_economics


def test_selector_economics_accounts_for_removed_and_added_events() -> None:
    result = compute_selector_economics(
        reference_decision=(1, 1, 0, 0),
        candidate_decision=(0, 0, 1, 1),
        net_outcomes=(-0.04, 0.03, 0.02, -0.01),
        gross_outcomes=(-0.03, 0.04, 0.03, 0.00),
    )

    assert result.reference_trade_count == 2
    assert result.candidate_trade_count == 2
    assert result.equal_trade_count is True
    assert result.avoided_loss == pytest.approx(0.04)
    assert result.missed_profit == pytest.approx(0.03)
    assert result.avoided_loss_minus_missed_profit == pytest.approx(0.01)
    assert result.added_net_pnl == pytest.approx(0.01)
    assert result.net_pnl_delta == pytest.approx(0.02)
    assert result.base_gross_profit_pool == pytest.approx(0.04)
    assert result.missed_profit_fraction == pytest.approx(0.75)
    assert result.sharpe_delta is not None


def test_selector_economics_reports_undefined_fraction_for_empty_profit_pool() -> None:
    result = compute_selector_economics(
        reference_decision=(1, 0),
        candidate_decision=(0, 1),
        net_outcomes=(-0.02, -0.01),
        gross_outcomes=(-0.01, 0.0),
    )

    assert result.missed_profit_fraction is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"candidate_decision": (1,)}, "align"),
        ({"reference_decision": (2, 0)}, "binary"),
        ({"net_outcomes": (float("nan"), 0.0)}, "finite"),
        ({"gross_outcomes": (0.0,)}, "align"),
    ],
)
def test_selector_economics_rejects_invalid_inputs(
    kwargs: dict[str, tuple[float, ...] | tuple[int, ...]], message: str
) -> None:
    inputs: dict[str, object] = {
        "reference_decision": (1, 0),
        "candidate_decision": (0, 1),
        "net_outcomes": (-0.01, 0.02),
        "gross_outcomes": (0.0, 0.03),
    }
    inputs.update(kwargs)

    with pytest.raises(ValueError, match=message):
        compute_selector_economics(**inputs)  # type: ignore[arg-type]
