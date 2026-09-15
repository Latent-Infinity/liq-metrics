from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal, localcontext

import pytest

from liq.metrics.entitlements import (
    CashRates,
    Entitlement,
    EntitlementBook,
    EntitlementError,
    IncompleteEntitlements,
)

EFFECTIVE = datetime(2024, 12, 12, 14, 30, tzinfo=UTC)
PAYMENT = datetime(2025, 1, 9, 14, 30, tzinfo=UTC)
SOURCE = "https://investor.tsmc.com/english/dividends/2q24"


def entitlement() -> Entitlement:
    return Entitlement(
        action_id="tsm-2024-12-dividend",
        security_id="US8740391003",
        unit_basis="legal-ADR",
        eligible_quantity=Decimal("1"),
        legal_units_per_held_unit=Decimal("1"),
        rates=CashRates(
            gross=Decimal("0.608106"),
            net=Decimal("0.480404"),
            withholding=None,
            depositary_fee=None,
        ),
        currency="USD",
        reporting_basis="net",
        effective_at=EFFECTIVE,
        payment_at=PAYMENT,
        available_at=EFFECTIVE,
        provenance=(SOURCE, "Software mechanics fixture; availability is not research evidence"),
    )


def test_receivable_and_settlement_conserve_value() -> None:
    book = EntitlementBook(currency="USD", reporting_basis="net")
    item = entitlement()
    book.accrue(item, at=EFFECTIVE, unit_basis="legal-ADR")
    before = book.value(at=EFFECTIVE)
    assert before == Decimal("0.480404")
    cash_delta = book.settle(item.action_id, at=PAYMENT)
    assert cash_delta == before
    assert book.value(at=PAYMENT) == 0
    assert cash_delta + book.value(at=PAYMENT) == before


def test_unpaid_entitlement_survives_terminal_valuation() -> None:
    book = EntitlementBook(currency="USD", reporting_basis="net")
    book.accrue(entitlement(), at=EFFECTIVE, unit_basis="legal-ADR")
    assert book.value(at=datetime(2024, 12, 31, 21, tzinfo=UTC)) == Decimal("0.480404")


@pytest.mark.parametrize("field", ["available_at", "legal_units_per_held_unit", "rates"])
def test_missing_or_future_facts_never_become_zero(field: str) -> None:
    item = entitlement()
    match field:
        case "available_at":
            item = replace(item, available_at=PAYMENT)
        case "legal_units_per_held_unit":
            item = replace(item, legal_units_per_held_unit=None)
        case "rates":
            item = replace(item, rates=replace(item.rates, net=None))
    book = EntitlementBook(currency="USD", reporting_basis="net")
    book.accrue(item, at=EFFECTIVE, unit_basis="legal-ADR")
    with pytest.raises(IncompleteEntitlements) as failure:
        book.value(at=EFFECTIVE)
    assert failure.value.action_ids == (item.action_id,)


def test_settlement_rejects_early_duplicate_and_missing_actions() -> None:
    book = EntitlementBook(currency="USD", reporting_basis="net")
    item = entitlement()
    book.accrue(item, at=EFFECTIVE, unit_basis="legal-ADR")
    with pytest.raises(EntitlementError):
        book.settle(item.action_id, at=EFFECTIVE)
    assert book.value(at=EFFECTIVE) == Decimal("0.480404")
    book.settle(item.action_id, at=PAYMENT)
    with pytest.raises(EntitlementError):
        book.settle(item.action_id, at=PAYMENT)
    with pytest.raises(EntitlementError):
        book.settle("missing", at=PAYMENT)


def test_duplicates_and_mismatched_units_cannot_mutate_book() -> None:
    book = EntitlementBook(currency="USD", reporting_basis="net")
    item = entitlement()
    with pytest.raises(EntitlementError):
        book.accrue(item, at=EFFECTIVE, unit_basis="unverified-adjusted-unit")
    assert book.value(at=EFFECTIVE) == 0
    book.accrue(item, at=EFFECTIVE, unit_basis="legal-ADR")
    with pytest.raises(EntitlementError):
        book.accrue(item, at=EFFECTIVE, unit_basis="legal-ADR")
    with pytest.raises(EntitlementError):
        book.accrue(
            replace(item, eligible_quantity=Decimal("2")), at=EFFECTIVE, unit_basis="legal-ADR"
        )
    assert book.value(at=EFFECTIVE) == Decimal("0.480404")


def test_currency_and_basis_mismatches_are_incomplete() -> None:
    for book in (
        EntitlementBook(currency="EUR", reporting_basis="net"),
        EntitlementBook(currency="USD", reporting_basis="gross"),
    ):
        book.accrue(entitlement(), at=EFFECTIVE, unit_basis="legal-ADR")
        with pytest.raises(IncompleteEntitlements):
            book.value(at=EFFECTIVE)


def test_gross_rate_is_explicit_and_never_inferred_from_net() -> None:
    book = EntitlementBook(currency="USD", reporting_basis="gross")
    book.accrue(
        replace(entitlement(), reporting_basis="gross"), at=EFFECTIVE, unit_basis="legal-ADR"
    )
    assert book.value(at=EFFECTIVE) == Decimal("0.608106")


def test_unknown_required_amount_cannot_settle() -> None:
    item = entitlement()
    book = EntitlementBook(currency="USD", reporting_basis="net")
    book.accrue(
        replace(item, rates=replace(item.rates, net=None)), at=EFFECTIVE, unit_basis="legal-ADR"
    )
    with pytest.raises(IncompleteEntitlements):
        book.settle(item.action_id, at=PAYMENT)


def test_settlement_preserves_historical_receivables() -> None:
    book = EntitlementBook(currency="USD", reporting_basis="net")
    with pytest.raises(EntitlementError):
        book.accrue(entitlement(), at=EFFECTIVE.replace(day=11), unit_basis="legal-ADR")
    book.accrue(entitlement(), at=EFFECTIVE, unit_basis="legal-ADR")
    book.settle(entitlement().action_id, at=PAYMENT)
    assert book.value(at=EFFECTIVE) == Decimal("0.480404")
    assert book.value(at=PAYMENT) == 0


def test_payment_date_alone_does_not_settle_and_accrual_credits_no_cash() -> None:
    book = EntitlementBook(currency="USD", reporting_basis="net")
    assert book.accrue(entitlement(), at=EFFECTIVE, unit_basis="legal-ADR") == 0
    assert book.value(at=PAYMENT) == Decimal("0.480404")
    assert book.value(at=EFFECTIVE.replace(day=11)) == 0


def test_future_facts_never_rewrite_historical_values() -> None:
    book = EntitlementBook(currency="USD", reporting_basis="net")
    item = replace(entitlement(), available_at=PAYMENT)
    book.accrue(item, at=EFFECTIVE, unit_basis="legal-ADR")
    assert book.value(at=PAYMENT) == Decimal("0.480404")
    book.settle(item.action_id, at=PAYMENT)
    with pytest.raises(IncompleteEntitlements):
        book.value(at=EFFECTIVE)


def test_exact_conversion_is_independent_of_caller_precision() -> None:
    book = EntitlementBook(currency="USD", reporting_basis="net")
    item = replace(
        entitlement(), eligible_quantity=Decimal("0.5"), legal_units_per_held_unit=Decimal("2")
    )
    book.accrue(item, at=EFFECTIVE, unit_basis="legal-ADR")
    with localcontext(prec=2):
        assert book.value(at=EFFECTIVE) == Decimal("0.480404")


@pytest.mark.parametrize("value", [Decimal("NaN"), Decimal("Infinity"), Decimal("-1")])
def test_invalid_rates_rejected(value: Decimal) -> None:
    with pytest.raises(EntitlementError):
        replace(entitlement().rates, net=value)


def test_invalid_identity_units_and_clocks_rejected() -> None:
    item = entitlement()
    for invalid in ("", " "):
        with pytest.raises(EntitlementError):
            replace(item, action_id=invalid)
    with pytest.raises(EntitlementError):
        replace(item, provenance=("",))
    with pytest.raises(EntitlementError):
        replace(item, eligible_quantity=Decimal(0))
    with pytest.raises(EntitlementError):
        replace(item, effective_at=EFFECTIVE.replace(tzinfo=None))
    with pytest.raises(EntitlementError):
        replace(item, payment_at=EFFECTIVE.replace(day=11))
    with pytest.raises(EntitlementError):
        EntitlementBook(currency="", reporting_basis="net")


def test_unknown_payment_time_prevents_settlement() -> None:
    book = EntitlementBook(currency="USD", reporting_basis="net")
    item = replace(entitlement(), payment_at=None)
    book.accrue(item, at=EFFECTIVE, unit_basis="legal-ADR")
    assert book.value(at=PAYMENT) == Decimal("0.480404")
    with pytest.raises(IncompleteEntitlements):
        book.settle(item.action_id, at=PAYMENT)


def test_backdated_mutations_rejected_and_unresolved_ids_sorted() -> None:
    book = EntitlementBook(currency="USD", reporting_basis="net")
    item = replace(entitlement(), effective_at=PAYMENT, available_at=None)
    book.accrue(item, at=PAYMENT, unit_basis="legal-ADR")
    with pytest.raises(EntitlementError):
        book.accrue(replace(item, action_id="other"), at=EFFECTIVE, unit_basis="legal-ADR")
    book.accrue(replace(item, action_id="another"), at=PAYMENT, unit_basis="legal-ADR")
    with pytest.raises(IncompleteEntitlements) as failure:
        book.value(at=PAYMENT)
    assert failure.value.action_ids == ("another", item.action_id)


def test_late_accrual_cannot_omit_an_earned_receivable_from_history() -> None:
    book = EntitlementBook(currency="USD", reporting_basis="net")
    with pytest.raises(EntitlementError):
        book.accrue(entitlement(), at=PAYMENT, unit_basis="legal-ADR")
    assert book.value(at=PAYMENT) == 0


def test_settled_copy_leaves_original_receivable_and_histories_independent() -> None:
    book = EntitlementBook(currency="USD", reporting_basis="net")
    item = entitlement()
    book.accrue(item, at=EFFECTIVE, unit_basis="legal-ADR")
    candidate, amount = book.settled_copy(item.action_id, at=PAYMENT)
    assert amount == Decimal("0.480404")
    assert book.value(at=PAYMENT) == amount
    assert candidate.value(at=PAYMENT) == 0
    later = replace(item, action_id="later", effective_at=PAYMENT)
    candidate.accrue(later, at=PAYMENT, unit_basis="legal-ADR")
    assert book.value(at=PAYMENT) == amount
    assert candidate.value(at=PAYMENT) == amount
    assert book.settle(item.action_id, at=PAYMENT) == amount
