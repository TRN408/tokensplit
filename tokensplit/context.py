"""Provider-neutral context layout that keeps a cacheable prefix stable.

The module deliberately does not depend on an SDK or tokenizer.  Applications
can provide their real tokenizer through ``ContextPolicy.token_counter`` and
can report provider usage to ``CacheMetrics.observe`` after a request.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
import json
from typing import Any, Protocol

from .output_gate import Extractor, GateLimits, GateResult, gate_tool_output


TokenCounter = Callable[[str], int]


def _estimate_tokens(text: str) -> int:
    """Return a deterministic, dependency-free token estimate.

    This is only a fallback for planning.  Production callers should pass the
    tokenizer used by their provider so budget decisions match billing data.
    """

    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class StaticContext:
    """The only content allowed in the cacheable prefix.

    ``tool_definitions`` are sorted by their stable name and serialized with
    canonical JSON, so connection order or dictionary key order cannot cause a
    needless prefix change.
    """

    system_instructions: str
    tool_definitions: Sequence[Mapping[str, Any]] = ()
    fixed_context: str = ""
    purpose: str = ""
    constraints: Sequence[str] = ()

    def render(self) -> str:
        tools = sorted(
            (dict(tool) for tool in self.tool_definitions),
            key=lambda tool: (
                str(tool.get("name", "")),
                _canonical_json(tool),
            ),
        )
        parts = [
            "<static-context version=\"1\">",
            "<system-instructions>",
            self.system_instructions,
            "</system-instructions>",
            "<tool-definitions>",
            _canonical_json(tools),
            "</tool-definitions>",
            "<fixed-context>",
            self.fixed_context,
            "</fixed-context>",
        ]
        if self.purpose:
            parts.extend(["<purpose>", self.purpose, "</purpose>"])
        if self.constraints:
            parts.extend(
                ["<constraints>", _canonical_json(list(self.constraints)), "</constraints>"]
            )
        parts.append("</static-context>")
        return "\n".join(parts)


@dataclass(frozen=True)
class ToolOutput:
    """A tool result rendered through the bounded output gate.

    ``content`` may be text, a sequence of search results, or a JSON-like
    value.  The gate warning is part of the rendered message, so omitted data
    is visible to the agent instead of disappearing silently.
    """

    content: Any
    query: str | None = None
    limits: GateLimits = field(default_factory=GateLimits)
    extractor: Extractor | None = None

    def gate(self) -> GateResult:
        return gate_tool_output(
            self.content,
            query=self.query,
            limits=self.limits,
            extractor=self.extractor,
        )

    def render(self) -> str:
        return self.gate().text


@dataclass(frozen=True)
class Message:
    role: str
    content: str | ToolOutput

    def render(self) -> str:
        content = self.content.render() if isinstance(self.content, ToolOutput) else self.content
        return f"<{self.role}>{content}</{self.role}>"

    @property
    def is_tool_output(self) -> bool:
        """Whether this message contains tool output that can be archived."""

        return isinstance(self.content, ToolOutput) or self.role.casefold() in {
            "tool",
            "function",
        }


class ExternalMemoryStore(Protocol):
    """Storage boundary for history removed from the active context."""

    def save(self, messages: Sequence[Message]) -> str:
        """Persist messages and return a reference that can be resolved later."""


class InMemoryExternalMemory:
    """Dependency-free memory store suitable for tests and short-lived sessions.

    Production callers can provide an adapter implementing
    :class:`ExternalMemoryStore` to persist archived history in a database,
    object store, or retrieval system. The builder never silently discards a
    batch when the store raises an error.
    """

    def __init__(self) -> None:
        self._entries: dict[str, tuple[Message, ...]] = {}

    def save(self, messages: Sequence[Message]) -> str:
        frozen = tuple(messages)
        payload = "\n".join(message.render() for message in frozen)
        reference = f"memory://{sha256(payload.encode('utf-8')).hexdigest()[:16]}"
        self._entries[reference] = frozen
        return reference

    def load(self, reference: str) -> tuple[Message, ...]:
        return self._entries[reference]

    @property
    def entries(self) -> Mapping[str, tuple[Message, ...]]:
        return dict(self._entries)


@dataclass(frozen=True)
class DynamicTurn:
    """Per-turn data, always rendered after ``StaticContext``.

    ``history`` is expected to be the complete history for this context.  The
    builder remembers how much of that history it has already folded, so a
    later full-history render does not duplicate the compressed messages.
    """

    user_input: str
    timestamp: str = ""
    state: Mapping[str, Any] = field(default_factory=dict)
    history: Sequence[Message] = ()


@dataclass(frozen=True)
class ContextPolicy:
    """Budget and compression policy for the dynamic suffix."""

    max_input_tokens: int = 16_000
    compression_threshold: float = 0.85
    recent_messages: int = 6
    summary_max_tokens: int = 2_048
    token_counter: TokenCounter = _estimate_tokens

    def __post_init__(self) -> None:
        if self.max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be positive")
        if not 0 < self.compression_threshold <= 1:
            raise ValueError("compression_threshold must be in (0, 1]")
        if self.recent_messages < 1:
            raise ValueError("recent_messages must be at least 1")
        if self.summary_max_tokens < 1:
            raise ValueError("summary_max_tokens must be positive")


@dataclass
class CacheMetrics:
    """Observable cache metrics independent of a provider SDK.

    ``observe`` accepts usage reported by a provider.  A request is a hit when
    it reports at least one cached input token; this avoids pretending that a
    stable local fingerprint alone proves a remote cache hit.
    """

    requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    normal_input_tokens: int = 0
    prefix_changes: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    unknown_cache_write_observations: int = 0
    _last_prefix_fingerprint: str | None = field(default=None, repr=False)

    def observe(
        self,
        *,
        prefix_fingerprint: str,
        cached_input_tokens: int = 0,
        cache_write_tokens: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_write_tokens_known: bool = True,
    ) -> None:
        counts = (cached_input_tokens, cache_write_tokens, input_tokens, output_tokens)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts):
            raise ValueError("cache token counts cannot be negative")
        if not isinstance(cache_write_tokens_known, bool):
            raise ValueError("cache_write_tokens_known must be a boolean")
        self.requests += 1
        if self._last_prefix_fingerprint is not None and self._last_prefix_fingerprint != prefix_fingerprint:
            self.prefix_changes += 1
        self._last_prefix_fingerprint = prefix_fingerprint
        self.cached_input_tokens += cached_input_tokens
        self.cache_write_tokens += cache_write_tokens
        if input_tokens:
            normal_input_tokens = input_tokens - cached_input_tokens - cache_write_tokens
            if normal_input_tokens < 0:
                raise ValueError("cached and cache-write tokens cannot exceed input tokens")
            self.normal_input_tokens += normal_input_tokens
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        if not cache_write_tokens_known:
            self.unknown_cache_write_observations += 1
        if cached_input_tokens > 0:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

    @property
    def hit_rate(self) -> float:
        return self.cache_hits / self.requests if self.requests else 0.0

    @property
    def prefix_stable(self) -> bool:
        return self.prefix_changes == 0

    @property
    def cache_read_tokens(self) -> int:
        """Return cache-read volume using billing terminology."""

        return self.cached_input_tokens

    @property
    def regular_input_tokens(self) -> int:
        """Return input tokens billed at the normal input rate."""

        return self.normal_input_tokens

    def cost_summary(self, pricing: Any, *, request_interval_seconds: int = 0):
        """Summarize observed volumes using a :class:`CachePricing` model."""

        from .cache import summarize_cache_metrics

        return summarize_cache_metrics(
            self,
            pricing,
            request_interval_seconds=request_interval_seconds,
        )


@dataclass(frozen=True)
class RenderedContext:
    static_prefix: str
    dynamic_suffix: str
    full_prompt: str
    prefix_fingerprint: str
    static_prefix_tokens: int
    dynamic_tokens: int
    total_tokens: int
    compression_applied: bool
    compression_count: int
    remaining_tokens: int = 0
    compression_trigger_tokens: int = 0
    archived_memory_refs: tuple[str, ...] = ()
    structured_state: Mapping[str, Any] = field(default_factory=dict)


class ContextBuilder:
    """Build prompts while preserving a stable cacheable prefix."""

    def __init__(
        self,
        static: StaticContext,
        policy: ContextPolicy | None = None,
        memory_store: ExternalMemoryStore | None = None,
    ) -> None:
        self.static = static
        self.policy = policy or ContextPolicy()
        self.memory_store = memory_store or InMemoryExternalMemory()
        self._static_prefix = static.render()
        self._prefix_fingerprint = sha256(self._static_prefix.encode("utf-8")).hexdigest()
        self._summary = ""
        self._compression_count = 0
        self._compressed_message_count = 0
        self._compressed_history_snapshot: tuple[Message, ...] = ()
        self._archive_records: list[dict[str, Any]] = []

    @property
    def static_prefix(self) -> str:
        return self._static_prefix

    @property
    def prefix_fingerprint(self) -> str:
        return self._prefix_fingerprint

    @property
    def compression_count(self) -> int:
        return self._compression_count

    @property
    def archived_memory_refs(self) -> tuple[str, ...]:
        """References for all history batches removed from the active prompt."""

        return tuple(record["reference"] for record in self._archive_records)

    def build(self, turn: DynamicTurn) -> RenderedContext:
        """Render a turn and compress only when the configured threshold is met."""

        history = list(turn.history)
        self._reset_summary_if_history_changed(history)
        visible_history = history[self._compressed_message_count :]
        suffix = self._render_dynamic(turn, visible_history, self._summary)
        total = self._token_count(self._static_prefix + "\n" + suffix)
        trigger = int(self.policy.max_input_tokens * self.policy.compression_threshold)
        compressed = False

        # Compression is event driven. Below the trigger, the builder only
        # appends the new dynamic suffix; it does not rewrite a summary on
        # every turn.
        if total > trigger and len(visible_history) > self.policy.recent_messages:
            old = visible_history[:-self.policy.recent_messages]
            visible_history = visible_history[-self.policy.recent_messages :]
            self._archive_history(old)
            base_suffix = self._render_dynamic(turn, visible_history, "")
            summary_budget = self.policy.max_input_tokens - self._token_count(
                self._static_prefix + "\n" + base_suffix
            )
            self._compressed_message_count = len(history) - len(visible_history)
            self._compressed_history_snapshot = tuple(history[: self._compressed_message_count])
            while True:
                self._summary = self._render_archive_summary(summary_budget)
                suffix = self._render_dynamic(turn, visible_history, self._summary)
                total = self._token_count(self._static_prefix + "\n" + suffix)
                if total <= self.policy.max_input_tokens or summary_budget <= 1:
                    break
                summary_budget -= 1
            self._compression_count += 1
            compressed = True

        # If the fixed prefix plus the current turn alone is too large, fail
        # explicitly instead of silently dropping user input or static rules.
        if total > self.policy.max_input_tokens and not compressed:
            raise ValueError("context exceeds max_input_tokens and cannot be compressed safely")
        if total > self.policy.max_input_tokens:
            raise ValueError("context still exceeds max_input_tokens after compression")

        static_tokens = self._token_count(self._static_prefix)
        dynamic_tokens = self._token_count(suffix)
        return RenderedContext(
            static_prefix=self._static_prefix,
            dynamic_suffix=suffix,
            full_prompt=self._static_prefix + "\n" + suffix,
            prefix_fingerprint=self._prefix_fingerprint,
            static_prefix_tokens=static_tokens,
            dynamic_tokens=dynamic_tokens,
            total_tokens=total,
            compression_applied=compressed,
            compression_count=self._compression_count,
            remaining_tokens=self.policy.max_input_tokens - total,
            compression_trigger_tokens=trigger,
            archived_memory_refs=self.archived_memory_refs,
            structured_state=dict(turn.state),
        )

    def _render_dynamic(
        self,
        turn: DynamicTurn,
        history: Sequence[Message],
        summary: str,
    ) -> str:
        state = _canonical_json(dict(turn.state))
        parts = [
            "<dynamic-context version=\"1\">",
            "<timestamp>",
            turn.timestamp,
            "</timestamp>",
            "<state>",
            state,
            "</state>",
        ]
        if summary:
            parts.extend(["<compressed-history>", summary, "</compressed-history>"])
        parts.append("<history>")
        parts.extend(message.render() for message in history)
        parts.extend(["</history>", "<user-input>", turn.user_input, "</user-input>", "</dynamic-context>"])
        return "\n".join(parts)

    def _token_count(self, text: str) -> int:
        return self.policy.token_counter(text)

    def _reset_summary_if_history_changed(self, history: Sequence[Message]) -> None:
        """Reset the cursor when callers replace or trim the full history."""

        if self._compressed_message_count and (
            len(history) < self._compressed_message_count
            or tuple(history[: self._compressed_message_count]) != self._compressed_history_snapshot
        ):
            self._summary = ""
            self._compressed_message_count = 0
            self._compressed_history_snapshot = ()
            self._archive_records = []

    def _archive_history(self, old: Sequence[Message]) -> None:
        """Persist old history before removing it from the active context."""

        reference = self.memory_store.save(tuple(old))
        self._archive_records.append(
            {
                "reference": reference,
                "message_count": len(old),
                "tool_output_count": sum(message.is_tool_output for message in old),
            }
        )

    def _render_archive_summary(self, available_tokens: int | None = None) -> str:
        """Render a bounded structured index instead of replaying tool output."""

        default_budget = min(self.policy.summary_max_tokens, max(1, self.policy.max_input_tokens // 4))
        budget = min(default_budget, available_tokens) if available_tokens is not None else default_budget
        budget = max(1, budget)
        records = [dict(record) for record in self._archive_records]
        candidate = _canonical_json(
            {
                "strategy": "external-memory",
                "archives": records,
            }
        )
        if self._token_count(candidate) <= budget:
            return candidate

        # Preserve every reference and count first. The full messages remain
        # in memory and are never reinserted into the active prompt.
        compact_records = records
        candidate = _canonical_json(
            {
                "strategy": "external-memory",
                "archives": compact_records,
            }
        )
        if self._token_count(candidate) <= budget:
            return candidate

        # A very long-running session can produce more references than the
        # summary budget can display. Keep the newest references in the prompt
        # and expose the total so omission is explicit and recoverable.
        for start in range(1, len(compact_records)):
            candidate = _canonical_json(
                {
                    "strategy": "external-memory",
                    "omitted_archive_count": start,
                    "archives": compact_records[start:],
                }
            )
            if self._token_count(candidate) <= budget:
                return candidate

        return _canonical_json(
            {
                "strategy": "external-memory",
                "omitted_archive_count": len(compact_records),
            }
        )
