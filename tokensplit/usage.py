"""Normalize provider usage payloads into :class:`CacheMetrics` observations.

The adapters accept response dictionaries and SDK response objects exposing
``model_dump()``, ``to_dict()``, or ``dict()``. They never make network calls
and reject missing or malformed usage fields instead of estimating values.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .context import CacheMetrics


class UsageAdapterError(ValueError):
    """Raised when a provider payload cannot be mapped without guessing."""


@dataclass(frozen=True)
class NormalizedUsage:
    provider: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    cache_write_tokens: int
    cache_write_tokens_known: bool = True

    @property
    def uncached_input_tokens(self) -> int:
        return max(0, self.input_tokens - self.cached_input_tokens - self.cache_write_tokens)

    def record(self, metrics: CacheMetrics, *, prefix_fingerprint: str) -> None:
        metrics.observe(
            prefix_fingerprint=prefix_fingerprint,
            cached_input_tokens=self.cached_input_tokens,
            cache_write_tokens=self.cache_write_tokens,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cache_write_tokens_known=self.cache_write_tokens_known,
        )


def _as_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    for method_name in ("model_dump", "to_dict", "dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            candidate = method()
            if isinstance(candidate, Mapping):
                return candidate
    raise UsageAdapterError(f"{field} must be a mapping or SDK object with a mapping export")


def _payload_usage(payload: Any) -> Mapping[str, Any]:
    data = _as_mapping(payload, field="payload")
    usage = data.get("usage", data)
    if usage is None:
        raise UsageAdapterError("payload has no usage field")
    return _as_mapping(usage, field="usage")


def _int_field(data: Mapping[str, Any], name: str, *, default: int | None = None) -> int:
    value = data.get(name, default)
    if value is None:
        raise UsageAdapterError(f"usage is missing {name}")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise UsageAdapterError(f"usage.{name} must be a non-negative integer")
    return value


def _nested_int(data: Mapping[str, Any], parent: str, name: str, *, default: int = 0) -> int:
    child = data.get(parent)
    if child is None:
        return default
    return _int_field(_as_mapping(child, field=parent), name, default=default)


def normalize_openai_usage(payload: Any) -> NormalizedUsage:
    """Normalize Responses or Chat Completions usage."""

    usage = _payload_usage(payload)
    if "input_tokens" in usage:
        input_tokens = _int_field(usage, "input_tokens")
        output_tokens = _int_field(usage, "output_tokens", default=0)
        cached = _nested_int(usage, "input_tokens_details", "cached_tokens")
        details = usage.get("input_tokens_details")
        if isinstance(details, Mapping) and "cached_tokens" not in details:
            cached = _int_field(usage, "cached_tokens", default=cached)
    elif "prompt_tokens" in usage:
        input_tokens = _int_field(usage, "prompt_tokens")
        output_tokens = _int_field(usage, "completion_tokens", default=0)
        cached = _nested_int(usage, "prompt_tokens_details", "cached_tokens")
    else:
        raise UsageAdapterError("OpenAI usage requires input_tokens or prompt_tokens")

    if cached > input_tokens:
        raise UsageAdapterError("OpenAI cached tokens cannot exceed input tokens")
    cache_write_known = "cache_write_tokens" in usage
    cache_write = _int_field(usage, "cache_write_tokens", default=0)
    return NormalizedUsage("openai", input_tokens, output_tokens, cached, cache_write, cache_write_known)


def normalize_anthropic_usage(payload: Any) -> NormalizedUsage:
    """Normalize Messages API or organization usage-report fields."""

    usage = _payload_usage(payload)
    uncached = _int_field(usage, "input_tokens")
    cached = _int_field(usage, "cache_read_input_tokens", default=0)
    if "cache_creation_input_tokens" in usage:
        cache_write = _int_field(usage, "cache_creation_input_tokens")
    else:
        creation = usage.get("cache_creation")
        cache_write = 0
        if creation is not None:
            creation_map = _as_mapping(creation, field="cache_creation")
            for name, value in creation_map.items():
                if name.endswith("_input_tokens"):
                    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                        raise UsageAdapterError(f"usage.cache_creation.{name} must be a non-negative integer")
                    cache_write += value
    output_tokens = _int_field(usage, "output_tokens", default=0)
    return NormalizedUsage(
        "anthropic",
        uncached + cached + cache_write,
        output_tokens,
        cached,
        cache_write,
    )


def record_openai_usage(payload: Any, metrics: CacheMetrics, *, prefix_fingerprint: str) -> NormalizedUsage:
    normalized = normalize_openai_usage(payload)
    normalized.record(metrics, prefix_fingerprint=prefix_fingerprint)
    return normalized


def record_anthropic_usage(payload: Any, metrics: CacheMetrics, *, prefix_fingerprint: str) -> NormalizedUsage:
    normalized = normalize_anthropic_usage(payload)
    normalized.record(metrics, prefix_fingerprint=prefix_fingerprint)
    return normalized


def record_provider_usage(
    provider: str,
    payload: Any,
    metrics: CacheMetrics,
    *,
    prefix_fingerprint: str,
) -> NormalizedUsage:
    """Dispatch a supported provider payload to the matching adapter."""

    normalized_provider = provider.casefold().replace("-", "_")
    if normalized_provider in {"openai", "open_ai"}:
        return record_openai_usage(payload, metrics, prefix_fingerprint=prefix_fingerprint)
    if normalized_provider in {"anthropic", "claude"}:
        return record_anthropic_usage(payload, metrics, prefix_fingerprint=prefix_fingerprint)
    raise UsageAdapterError(f"unsupported usage provider: {provider}")
