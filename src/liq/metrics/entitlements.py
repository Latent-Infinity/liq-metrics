"""Explicit same-currency receivables, separate from cash and position ledgers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, localcontext
from typing import Literal


class EntitlementError(ValueError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)


class IncompleteEntitlements(EntitlementError):
    def __init__(self, action_ids: tuple[str, ...]) -> None:
        self.action_ids = action_ids
        super().__init__(f"Incomplete entitlement valuation: {action_ids}")


def _utc(value: datetime) -> None:
    if value.utcoffset() != timedelta(0):
        raise EntitlementError("Timestamps must be timezone-aware UTC")


def _amount(value: Decimal | None, *, positive: bool = False) -> None:
    if value is not None and (not value.is_finite() or value < 0 or (positive and value == 0)):
        raise EntitlementError("Amounts must be finite and nonnegative; unit factors positive")


@dataclass(frozen=True, slots=True)
class CashRates:
    gross: Decimal | None
    net: Decimal | None
    withholding: Decimal | None
    depositary_fee: Decimal | None

    def __post_init__(self) -> None:
        for value in (self.gross, self.net, self.withholding, self.depositary_fee):
            _amount(value)


@dataclass(frozen=True, slots=True)
class Entitlement:
    action_id: str
    security_id: str
    unit_basis: str
    eligible_quantity: Decimal
    legal_units_per_held_unit: Decimal | None
    rates: CashRates
    currency: str
    reporting_basis: Literal["gross", "net"]
    effective_at: datetime
    payment_at: datetime | None
    available_at: datetime | None
    provenance: tuple[str, ...]

    def __post_init__(self) -> None:
        if not all(
            (
                self.action_id.strip(),
                self.security_id.strip(),
                self.unit_basis.strip(),
                self.currency.strip(),
                self.provenance,
            )
        ) or any(not source.strip() for source in self.provenance):
            raise EntitlementError(
                "Explicit identity, unit basis, currency and provenance required"
            )
        if self.reporting_basis not in ("gross", "net"):
            raise EntitlementError("Reporting basis must be explicit gross or net")
        _amount(self.eligible_quantity, positive=True)
        _amount(self.legal_units_per_held_unit, positive=True)
        for value in (self.effective_at, self.payment_at, self.available_at):
            if value is not None:
                _utc(value)
        if self.payment_at is not None and self.payment_at < self.effective_at:
            raise EntitlementError("Payment before entitlement is unsupported")


class EntitlementBook:
    """Track accrual and settlement history; returned cash deltas need one-time application.

    Eligibility and source admission belong upstream. No holdings or cash are owned
    here. A sale cannot erase an earned receivable, and payment dates do not settle
    automatically. Historical queries use recorded accrual and settlement times.
    """

    def __init__(self, *, currency: str, reporting_basis: Literal["gross", "net"]) -> None:
        if not currency.strip() or reporting_basis not in ("gross", "net"):
            raise EntitlementError("Explicit currency and gross/net reporting basis required")
        self._currency = currency
        self._basis: Literal["gross", "net"] = reporting_basis
        self._entries: dict[str, tuple[Entitlement, datetime]] = {}
        self._settlements: dict[str, datetime] = {}
        self._last_mutation: datetime | None = None

    def _mutation_time(self, at: datetime) -> None:
        _utc(at)
        if self._last_mutation is not None and at < self._last_mutation:
            raise EntitlementError("Backdated mutations are unsupported")

    def accrue(self, item: Entitlement, *, at: datetime, unit_basis: str) -> Decimal:
        self._mutation_time(at)
        if at != item.effective_at or item.unit_basis != unit_basis:
            raise EntitlementError(
                "Accrual requires the effective timestamp and matching held units"
            )
        if item.action_id in self._entries:
            raise EntitlementError("Duplicate entitlement identity")
        self._entries[item.action_id] = (item, at)
        self._last_mutation = at
        return Decimal(0)

    def _value(self, item: Entitlement, at: datetime) -> Decimal:
        rate = item.rates.net if self._basis == "net" else item.rates.gross
        factor = item.legal_units_per_held_unit
        if (
            item.currency != self._currency
            or item.reporting_basis != self._basis
            or item.available_at is None
            or item.available_at > at
            or rate is None
            or factor is None
        ):
            raise IncompleteEntitlements((item.action_id,))
        precision = sum(
            len(value.as_tuple().digits) for value in (item.eligible_quantity, factor, rate)
        )
        with localcontext(prec=max(1, precision)):
            return item.eligible_quantity * factor * rate

    def value(self, *, at: datetime) -> Decimal:
        _utc(at)
        unresolved: list[str] = []
        amounts: list[Decimal] = []
        for action_id, (item, accrued_at) in self._entries.items():
            settled_at = self._settlements.get(action_id)
            if accrued_at > at or (settled_at is not None and settled_at <= at):
                continue
            try:
                amounts.append(self._value(item, at))
            except IncompleteEntitlements:
                unresolved.append(action_id)
        if unresolved:
            raise IncompleteEntitlements(tuple(sorted(unresolved)))
        precision = 1 + sum(
            len(value.as_tuple().digits) + abs(int(value.as_tuple().exponent)) for value in amounts
        )
        with localcontext(prec=precision):
            return sum(amounts, Decimal(0))

    def settle(self, action_id: str, *, at: datetime) -> Decimal:
        self._mutation_time(at)
        if action_id not in self._entries or action_id in self._settlements:
            raise EntitlementError("Settlement requires an unsettled entitlement identity")
        item, _ = self._entries[action_id]
        if item.payment_at is None:
            raise IncompleteEntitlements((action_id,))
        if at < item.payment_at:
            raise EntitlementError("Settlement precedes payment")
        amount = self._value(item, at)
        self._settlements[action_id] = at
        self._last_mutation = at
        return amount

    def settled_copy(self, action_id: str, *, at: datetime) -> tuple[EntitlementBook, Decimal]:
        candidate = EntitlementBook(currency=self._currency, reporting_basis=self._basis)
        candidate._entries = self._entries.copy()
        candidate._settlements = self._settlements.copy()
        candidate._last_mutation = self._last_mutation
        return candidate, candidate.settle(action_id, at=at)
