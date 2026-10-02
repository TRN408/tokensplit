"""Streaming usage aggregators for OpenAI and Anthropic event streams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .context import CacheMetrics
from .usage import (
    NormalizedUsage,
    UsageAdapterError,
    _as_mapping,
    _int_field,
    normalize_anthropic_usage,
    normalize_openai_usage,
)


class StreamingUsageError(UsageAdapterError):
    """Raised when a stream ends without enough usage data to account for it."""


@dataclass
class StreamingUsageAdapter:
    """Collect usage-bearing events and emit one normalized usage record.

    OpenAI streams replace the stored usage with the last usage-bearing chunk;
    this handles a final usage chunk after ordinary content chunks. Anthropic
    input/cache usage is read from ``message_start`` and output usage from the
    last ``message_delta`` usage object.
    """

    provider: str
    _openai_usage: NormalizedUsage | None = None
    _anthropic_input_tokens: int | None = None
    _anthropic_cached_tokens: int = 0
    _anthropic_write_tokens: int = 0
    _anthropic_output_tokens: int = 0

    def __post_init__(self) -> None:
        normalized = self.provider.casefold().replace("-", "_")
        if normalized in {"openai", "open_ai"}:
            self.provider = "openai"
        elif normalized in {"anthropic", "claude"}:
            self.provider = "anthropic"
        else:
            raise StreamingUsageError(f"unsupported streaming provider: {self.provider}")

    def add(self, chunk: Any) -> bool:
        """Consume one event; return whether it contained usage information."""

        data = _as_mapping(chunk, field="stream chunk")
        if self.provider == "openai":
            usage = data.get("usage")
            if usage is None and data.get("response") is not None:
                response = _as_mapping(data["response"], field="response")
                usage = response.get("usage")
            if usage is None:
                return False
            self._openai_usage = normalize_openai_usage({"usage": usage})
            return True

        event_type = data.get("type")
        if event_type == "message_start":
            message = data.get("message")
            message_map = _as_mapping(message, field="message") if message is not None else data
            usage = message_map.get("usage") or data.get("usage")
            if usage is None:
                return False
            normalized = normalize_anthropic_usage({"usage": usage})
            self._anthropic_input_tokens = normalized.input_tokens
            self._anthropic_cached_tokens = normalized.cached_input_tokens
            self._anthropic_write_tokens = normalized.cache_write_tokens
            return True

        if event_type == "message_delta":
            delta = data.get("delta")
            delta_map = _as_mapping(delta, field="delta") if delta is not None else data
            usage = data.get("usage") or delta_map.get("usage")
            if usage is None:
                return False
            usage_map = _as_mapping(usage, field="usage")
            self._anthropic_output_tokens = _int_field(
                usage_map,
                "output_tokens",
                default=self._anthropic_output_tokens,
            )
            return True

        return False

    def finalize(self) -> NormalizedUsage:
        if self.provider == "openai":
            if self._openai_usage is None:
                raise StreamingUsageError("OpenAI stream contained no usage-bearing chunk")
            return self._openai_usage
        if self._anthropic_input_tokens is None:
            raise StreamingUsageError("Anthropic stream contained no message_start usage")
        return NormalizedUsage(
            "anthropic",
            self._anthropic_input_tokens,
            self._anthropic_output_tokens,
            self._anthropic_cached_tokens,
            self._anthropic_write_tokens,
        )

    def record(self, metrics: CacheMetrics, *, prefix_fingerprint: str) -> NormalizedUsage:
        normalized = self.finalize()
        normalized.record(metrics, prefix_fingerprint=prefix_fingerprint)
        return normalized
