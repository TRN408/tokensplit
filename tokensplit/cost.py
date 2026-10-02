"""Provider/model-specific token cost conversion.

Rates are supplied by the caller because model prices and cache TTL prices can
change. The result explicitly records whether cache-write usage was known; an
unknown write count is never silently treated as a verified bill.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Any

from .usage import NormalizedUsage


class CostConversionError(ValueError):
    """Raised when a usage/rate pair cannot produce a meaningful estimate."""


@dataclass(frozen=True)
class ProviderRates:
    provider: str
    model: str
    input_usd_per_million: Decimal
    cached_input_usd_per_million: Decimal
    cache_write_usd_per_million: Decimal
    output_usd_per_million: Decimal
    ttl_seconds: int | None = None
    catalog_version: int | None = None
    effective_from: date | None = None

    @classmethod
    def from_per_million(
        cls,
        *,
        provider: str,
        model: str,
        input_usd: Any,
        cached_input_usd: Any,
        cache_write_usd: Any,
        output_usd: Any,
        ttl_seconds: int | None = None,
        catalog_version: int | None = None,
        effective_from: date | None = None,
    ) -> "ProviderRates":
        values = (input_usd, cached_input_usd, cache_write_usd, output_usd)
        rates = tuple(Decimal(str(value)) for value in values)
        if any(rate < 0 for rate in rates):
            raise CostConversionError("token rates cannot be negative")
        return cls(
            provider,
            model,
            *rates,
            ttl_seconds=ttl_seconds,
            catalog_version=catalog_version,
            effective_from=effective_from,
        )


@dataclass(frozen=True)
class CostBreakdown:
    provider: str
    model: str
    input_tokens: int
    uncached_input_tokens: int
    cached_input_tokens: int
    cache_write_tokens: int
    output_tokens: int
    uncached_input_usd: Decimal
    cached_input_usd: Decimal
    cache_write_usd: Decimal
    output_usd: Decimal
    total_usd: Decimal
    cache_write_tokens_known: bool

    @property
    def complete(self) -> bool:
        return self.cache_write_tokens_known


def _token_cost(tokens: int, usd_per_million: Decimal) -> Decimal:
    return Decimal(tokens) * usd_per_million / Decimal(1_000_000)


def calculate_cost(usage: NormalizedUsage, rates: ProviderRates) -> CostBreakdown:
    """Convert normalized usage to a USD estimate without rounding early."""

    if usage.provider != rates.provider:
        raise CostConversionError(
            f"provider mismatch: usage={usage.provider!r}, rates={rates.provider!r}"
        )
    if usage.input_tokens < usage.cached_input_tokens:
        raise CostConversionError("cached input tokens cannot exceed total input tokens")
    if usage.cache_write_tokens > usage.input_tokens - usage.cached_input_tokens:
        raise CostConversionError("cache-write tokens exceed uncached input tokens")

    cached = usage.cached_input_tokens
    writes = usage.cache_write_tokens if usage.cache_write_tokens_known else 0
    uncached = usage.input_tokens - cached - writes
    uncached_usd = _token_cost(uncached, rates.input_usd_per_million)
    cached_usd = _token_cost(cached, rates.cached_input_usd_per_million)
    write_usd = _token_cost(writes, rates.cache_write_usd_per_million)
    output_usd = _token_cost(usage.output_tokens, rates.output_usd_per_million)
    return CostBreakdown(
        provider=usage.provider,
        model=rates.model,
        input_tokens=usage.input_tokens,
        uncached_input_tokens=uncached,
        cached_input_tokens=cached,
        cache_write_tokens=writes,
        output_tokens=usage.output_tokens,
        uncached_input_usd=uncached_usd,
        cached_input_usd=cached_usd,
        cache_write_usd=write_usd,
        output_usd=output_usd,
        total_usd=uncached_usd + cached_usd + write_usd + output_usd,
        cache_write_tokens_known=usage.cache_write_tokens_known,
    )
