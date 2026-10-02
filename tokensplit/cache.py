"""Prompt-cache cost accounting and break-even analysis.

The module only performs local arithmetic. It accepts provider-reported token
volumes through :class:`~tokensplit.context.CacheMetrics` and caller-supplied
price models, so it never makes network calls or silently embeds a price.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor

from .context import CacheMetrics


_TOKENS_PER_MILLION = 1_000_000


def _validate_non_negative_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _validate_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_price(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"{name} must be a non-negative number")


@dataclass(frozen=True)
class CachePricing:
    """Prices for one model and one cache TTL, in USD per million tokens."""

    model: str
    input_usd_per_million: float
    cache_write_usd_per_million: float
    cache_read_usd_per_million: float
    ttl_seconds: int

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model must not be empty")
        for value, name in (
            (self.input_usd_per_million, "input_usd_per_million"),
            (self.cache_write_usd_per_million, "cache_write_usd_per_million"),
            (self.cache_read_usd_per_million, "cache_read_usd_per_million"),
        ):
            _validate_price(value, name)
        _validate_positive_int(self.ttl_seconds, "ttl_seconds")

    @classmethod
    def from_multipliers(
        cls,
        model: str,
        *,
        input_usd_per_million: float,
        cache_write_multiplier: float,
        cache_read_multiplier: float,
        ttl_seconds: int,
    ) -> "CachePricing":
        """Build prices from provider multipliers such as ``1.25x`` and ``0.1x``."""

        _validate_price(cache_write_multiplier, "cache_write_multiplier")
        _validate_price(cache_read_multiplier, "cache_read_multiplier")
        return cls(
            model=model,
            input_usd_per_million=input_usd_per_million,
            cache_write_usd_per_million=input_usd_per_million * cache_write_multiplier,
            cache_read_usd_per_million=input_usd_per_million * cache_read_multiplier,
            ttl_seconds=ttl_seconds,
        )


@dataclass(frozen=True)
class CacheCostEstimate:
    """Volumes, costs, and recommendations for one observed or projected run."""

    model: str
    ttl_seconds: int
    requests: int
    cache_writes: int
    cache_reads: int
    cache_hit_rate: float
    normal_input_tokens: int
    cache_write_tokens: int
    cache_read_tokens: int
    baseline_cost_usd: float
    cached_cost_usd: float
    savings_usd: float
    break_even_repetitions: int | None
    cache_enabled_but_expensive: bool
    short_one_off: bool
    cache_write_tokens_known: bool = True

    @property
    def estimated_cost_usd(self) -> float:
        """Return the cost with cache accounting enabled."""

        return self.cached_cost_usd

    @property
    def cache_is_worthwhile(self) -> bool:
        return self.complete and self.savings_usd >= 0

    @property
    def complete(self) -> bool:
        """Whether all observed cache-write fields were reported by the provider."""

        return self.cache_write_tokens_known


def _cost(tokens: int, price_per_million: float) -> float:
    return tokens / _TOKENS_PER_MILLION * price_per_million


def _write_count(requests: int, *, ttl_seconds: int, request_interval_seconds: int) -> int:
    if requests == 0:
        return 0
    if request_interval_seconds == 0:
        return 1
    # Requests at exactly the TTL boundary need a fresh cache entry.
    elapsed = (requests - 1) * request_interval_seconds
    return 1 + floor(elapsed / ttl_seconds)


def _break_even_repetitions(
    *,
    prefix_tokens: int,
    normal_input_tokens_per_request: int,
    pricing: CachePricing,
    request_interval_seconds: int,
) -> int | None:
    """Find the first request count where cached input is no more expensive."""

    if prefix_tokens == 0:
        return None
    # A million requests is enough for practical price models and prevents a
    # pathological model from making this arithmetic unbounded.
    for repetitions in range(1, 1_000_001):
        writes = _write_count(
            repetitions,
            ttl_seconds=pricing.ttl_seconds,
            request_interval_seconds=request_interval_seconds,
        )
        reads = repetitions - writes
        baseline = _cost(
            (prefix_tokens + normal_input_tokens_per_request) * repetitions,
            pricing.input_usd_per_million,
        )
        cached = (
            _cost(normal_input_tokens_per_request * repetitions, pricing.input_usd_per_million)
            + _cost(prefix_tokens * writes, pricing.cache_write_usd_per_million)
            + _cost(prefix_tokens * reads, pricing.cache_read_usd_per_million)
        )
        if cached <= baseline:
            return repetitions
    return None


def estimate_cache_economics(
    *,
    prefix_tokens: int,
    repetitions: int,
    pricing: CachePricing,
    normal_input_tokens_per_request: int = 0,
    request_interval_seconds: int = 0,
) -> CacheCostEstimate:
    """Estimate cache volumes and cost for repeated requests.

    ``repetitions`` includes the first request. A zero interval means all
    repetitions fit inside one TTL window. A positive interval causes a fresh
    cache write whenever elapsed time reaches the TTL.
    """

    _validate_positive_int(prefix_tokens, "prefix_tokens")
    _validate_positive_int(repetitions, "repetitions")
    _validate_non_negative_int(normal_input_tokens_per_request, "normal_input_tokens_per_request")
    _validate_non_negative_int(request_interval_seconds, "request_interval_seconds")
    writes = _write_count(
        repetitions,
        ttl_seconds=pricing.ttl_seconds,
        request_interval_seconds=request_interval_seconds,
    )
    reads = repetitions - writes
    normal_tokens = normal_input_tokens_per_request * repetitions
    write_tokens = prefix_tokens * writes
    read_tokens = prefix_tokens * reads
    baseline_cost = _cost(
        (prefix_tokens + normal_input_tokens_per_request) * repetitions,
        pricing.input_usd_per_million,
    )
    cached_cost = (
        _cost(normal_tokens, pricing.input_usd_per_million)
        + _cost(write_tokens, pricing.cache_write_usd_per_million)
        + _cost(read_tokens, pricing.cache_read_usd_per_million)
    )
    break_even = _break_even_repetitions(
        prefix_tokens=prefix_tokens,
        normal_input_tokens_per_request=normal_input_tokens_per_request,
        pricing=pricing,
        request_interval_seconds=request_interval_seconds,
    )
    savings = baseline_cost - cached_cost
    return CacheCostEstimate(
        model=pricing.model,
        ttl_seconds=pricing.ttl_seconds,
        requests=repetitions,
        cache_writes=writes,
        cache_reads=reads,
        cache_hit_rate=reads / repetitions,
        normal_input_tokens=normal_tokens,
        cache_write_tokens=write_tokens,
        cache_read_tokens=read_tokens,
        baseline_cost_usd=baseline_cost,
        cached_cost_usd=cached_cost,
        savings_usd=savings,
        break_even_repetitions=break_even,
        cache_enabled_but_expensive=savings < 0,
        short_one_off=repetitions == 1 and savings < 0,
        cache_write_tokens_known=True,
    )


def summarize_cache_metrics(
    metrics: CacheMetrics,
    pricing: CachePricing,
    *,
    request_interval_seconds: int = 0,
) -> CacheCostEstimate:
    """Price observed normal, write, and read volumes.

    The break-even request count uses the observed average cacheable prefix.
    It is a planning estimate when a run contains multiple prefix sizes; the
    returned token volumes remain the exact provider-reported totals.
    """

    _validate_non_negative_int(request_interval_seconds, "request_interval_seconds")
    requests = metrics.requests
    reads = metrics.cache_read_tokens
    writes = metrics.cache_write_tokens
    baseline_tokens = metrics.normal_input_tokens + writes + reads
    baseline_cost = _cost(baseline_tokens, pricing.input_usd_per_million)
    cached_cost = (
        _cost(metrics.normal_input_tokens, pricing.input_usd_per_million)
        + _cost(writes, pricing.cache_write_usd_per_million)
        + _cost(reads, pricing.cache_read_usd_per_million)
    )
    savings = baseline_cost - cached_cost
    write_tokens_known = metrics.unknown_cache_write_observations == 0
    average_prefix_tokens = (writes + reads) // requests if requests else 0
    average_normal_tokens = metrics.normal_input_tokens // requests if requests else 0
    break_even = (
        _break_even_repetitions(
            prefix_tokens=average_prefix_tokens,
            normal_input_tokens_per_request=average_normal_tokens,
            pricing=pricing,
            request_interval_seconds=request_interval_seconds,
        )
        if average_prefix_tokens
        else None
    )
    return CacheCostEstimate(
        model=pricing.model,
        ttl_seconds=pricing.ttl_seconds,
        requests=requests,
        cache_writes=0 if writes == 0 else 1,
        cache_reads=metrics.cache_hits,
        cache_hit_rate=metrics.hit_rate,
        normal_input_tokens=metrics.normal_input_tokens,
        cache_write_tokens=writes,
        cache_read_tokens=reads,
        baseline_cost_usd=baseline_cost,
        cached_cost_usd=cached_cost,
        savings_usd=savings,
        break_even_repetitions=break_even,
        cache_enabled_but_expensive=write_tokens_known and savings < 0,
        short_one_off=write_tokens_known and requests == 1 and savings < 0,
        cache_write_tokens_known=write_tokens_known,
    )
