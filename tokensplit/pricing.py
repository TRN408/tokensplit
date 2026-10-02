"""Versioned provider/model/TTL price-table resolution.

Price data is deliberately kept outside the arithmetic implementation. A
catalog can be loaded from a checked-in JSON file or from an application-owned
file, then resolved as of a date without making network calls.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
import json
from pathlib import Path
from typing import Any

from .cache import CachePricing
from .cost import ProviderRates


class PriceTableError(ValueError):
    """Raised when a price table is malformed or cannot be resolved."""


def _as_date(value: date | datetime | str, *, field: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise PriceTableError(f"{field} must be an ISO date") from exc
    raise PriceTableError(f"{field} must be an ISO date")


def _required(data: Mapping[str, Any], name: str) -> Any:
    if name not in data:
        raise PriceTableError(f"price row is missing {name}")
    return data[name]


@dataclass(frozen=True)
class PriceRecord:
    """One price version for one provider/model/TTL combination."""

    provider: str
    model: str
    ttl_seconds: int
    effective_from: date
    input_usd_per_million: float
    cache_write_usd_per_million: float
    cache_read_usd_per_million: float
    output_usd_per_million: float | None = None
    source_url: str | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "PriceRecord":
        try:
            record = cls(
                provider=str(_required(data, "provider")),
                model=str(_required(data, "model")),
                ttl_seconds=_required(data, "ttl_seconds"),
                effective_from=_as_date(_required(data, "effective_from"), field="effective_from"),
                input_usd_per_million=_required(data, "input_usd_per_million"),
                cache_write_usd_per_million=_required(data, "cache_write_usd_per_million"),
                cache_read_usd_per_million=_required(data, "cache_read_usd_per_million"),
                output_usd_per_million=data.get("output_usd_per_million"),
                source_url=data.get("source_url"),
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, PriceTableError):
                raise
            raise PriceTableError(f"invalid price row: {exc}") from exc
        try:
            CachePricing(
                provider=record.provider,
                model=record.model,
                input_usd_per_million=record.input_usd_per_million,
                cache_write_usd_per_million=record.cache_write_usd_per_million,
                cache_read_usd_per_million=record.cache_read_usd_per_million,
                ttl_seconds=record.ttl_seconds,
            )
        except (TypeError, ValueError) as exc:
            raise PriceTableError(f"invalid price row: {exc}") from exc
        return record

    def to_pricing(self, *, catalog_version: int | None = None) -> CachePricing:
        """Create the existing arithmetic price object for this version."""

        return CachePricing(
            provider=self.provider,
            model=self.model,
            input_usd_per_million=self.input_usd_per_million,
            cache_write_usd_per_million=self.cache_write_usd_per_million,
            cache_read_usd_per_million=self.cache_read_usd_per_million,
            ttl_seconds=self.ttl_seconds,
            effective_from=self.effective_from,
            catalog_version=catalog_version,
        )

    def to_provider_rates(self, *, catalog_version: int) -> ProviderRates:
        """Create the dated provider/model cost rates for this version.

        The legacy cache-only table did not contain output prices. Such rows
        remain valid for :meth:`to_pricing`, but cannot be used for a complete
        provider cost conversion until an output rate is supplied.
        """

        if self.output_usd_per_million is None:
            raise PriceTableError(
                "price row is missing output_usd_per_million; "
                "required to generate ProviderRates"
            )
        try:
            return ProviderRates.from_per_million(
                provider=self.provider,
                model=self.model,
                input_usd=self.input_usd_per_million,
                cached_input_usd=self.cache_read_usd_per_million,
                cache_write_usd=self.cache_write_usd_per_million,
                output_usd=self.output_usd_per_million,
                ttl_seconds=self.ttl_seconds,
                catalog_version=catalog_version,
                effective_from=self.effective_from,
            )
        except (TypeError, ValueError) as exc:
            raise PriceTableError(f"invalid provider rates: {exc}") from exc

    def as_mapping(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "ttl_seconds": self.ttl_seconds,
            "effective_from": self.effective_from.isoformat(),
            "input_usd_per_million": self.input_usd_per_million,
            "cache_write_usd_per_million": self.cache_write_usd_per_million,
            "cache_read_usd_per_million": self.cache_read_usd_per_million,
            "output_usd_per_million": self.output_usd_per_million,
            "source_url": self.source_url,
        }


@dataclass(frozen=True)
class PriceCatalog:
    """Immutable collection of dated price records."""

    records: tuple[PriceRecord, ...]
    version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version < 1:
            raise PriceTableError("price table version must be a positive integer")
        seen: set[tuple[str, str, int, date]] = set()
        for record in self.records:
            if not isinstance(record, PriceRecord):
                raise PriceTableError("records must contain PriceRecord values")
            key = (record.provider.casefold(), record.model, record.ttl_seconds, record.effective_from)
            if key in seen:
                raise PriceTableError(f"duplicate price record: {key}")
            seen.add(key)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "PriceCatalog":
        version = data.get("version", 1)
        rows = data.get("prices")
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise PriceTableError("price table prices must be an array")
        records = tuple(PriceRecord.from_mapping(row) for row in rows if isinstance(row, Mapping))
        if len(records) != len(rows):
            raise PriceTableError("each price table entry must be an object")
        return cls(records=records, version=version)

    @classmethod
    def from_json(cls, path: str | Path) -> "PriceCatalog":
        source = Path(path)
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PriceTableError(f"cannot read price table {source}: {exc}") from exc
        if not isinstance(data, Mapping):
            raise PriceTableError("price table root must be an object")
        return cls.from_mapping(data)

    def resolve(
        self,
        *,
        provider: str,
        model: str,
        ttl_seconds: int,
        as_of: date | datetime | str | None = None,
    ) -> CachePricing:
        """Return the latest record effective on or before ``as_of``."""

        effective_date = date.today() if as_of is None else _as_date(as_of, field="as_of")
        candidates = [
            record
            for record in self.records
            if record.provider.casefold() == provider.casefold()
            and record.model == model
            and record.ttl_seconds == ttl_seconds
            and record.effective_from <= effective_date
        ]
        if not candidates:
            raise PriceTableError(
                f"no price for provider={provider!r}, model={model!r}, "
                f"ttl_seconds={ttl_seconds}, as_of={effective_date.isoformat()}"
            )
        return max(candidates, key=lambda record: record.effective_from).to_pricing(
            catalog_version=self.version
        )

    def resolve_provider_rates(
        self,
        *,
        provider: str,
        model: str,
        ttl_seconds: int,
        as_of: date | datetime | str | None = None,
    ) -> ProviderRates:
        """Return dated :class:`ProviderRates` for a provider/model/TTL key."""

        effective_date = date.today() if as_of is None else _as_date(as_of, field="as_of")
        candidates = [
            record
            for record in self.records
            if record.provider.casefold() == provider.casefold()
            and record.model == model
            and record.ttl_seconds == ttl_seconds
            and record.effective_from <= effective_date
        ]
        if not candidates:
            raise PriceTableError(
                f"no price for provider={provider!r}, model={model!r}, "
                f"ttl_seconds={ttl_seconds}, as_of={effective_date.isoformat()}"
            )
        return max(candidates, key=lambda record: record.effective_from).to_provider_rates(
            catalog_version=self.version
        )

    def as_mapping(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "prices": [record.as_mapping() for record in self.records],
        }


def load_price_catalog(path: str | Path) -> PriceCatalog:
    """Load a local versioned price table."""

    return PriceCatalog.from_json(path)


def pricing_for(
    path: str | Path,
    *,
    provider: str,
    model: str,
    ttl_seconds: int,
    as_of: date | datetime | str | None = None,
) -> CachePricing:
    """Load a table and generate the matching :class:`CachePricing`."""

    return load_price_catalog(path).resolve(
        provider=provider,
        model=model,
        ttl_seconds=ttl_seconds,
        as_of=as_of,
    )


def provider_rates_for(
    path: str | Path,
    *,
    provider: str,
    model: str,
    ttl_seconds: int,
    as_of: date | datetime | str | None = None,
) -> ProviderRates:
    """Load a versioned table and generate dated :class:`ProviderRates`."""

    return load_price_catalog(path).resolve_provider_rates(
        provider=provider,
        model=model,
        ttl_seconds=ttl_seconds,
        as_of=as_of,
    )
