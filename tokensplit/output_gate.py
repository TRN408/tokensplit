"""Bound and selectively extract tool output before it reaches an agent."""
from __future__ import annotations

import json
import re
import base64
from hashlib import sha256
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


TokenCounter = Callable[[str], int]


def _estimate_tokens(text: str) -> int:
    """Return a deterministic estimate when a provider tokenizer is unavailable."""

    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


@dataclass(frozen=True)
class GateLimits:
    """Limits applied to one tool result."""

    max_chars: int = 12_000
    max_items: int = 100
    context_lines: int = 2
    purpose: str = "generic"
    fields: tuple[str, ...] = ()
    sample_items: int | None = None
    max_output_tokens: int | None = None

    def __post_init__(self) -> None:
        for name in ("max_chars", "max_items"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.context_lines, bool)
            or not isinstance(self.context_lines, int)
            or self.context_lines < 0
        ):
            raise ValueError("context_lines must be a non-negative integer")
        if not isinstance(self.purpose, str) or not self.purpose.strip():
            raise ValueError("purpose must be a non-empty string")
        if self.sample_items is not None and (
            isinstance(self.sample_items, bool)
            or not isinstance(self.sample_items, int)
            or self.sample_items <= 0
        ):
            raise ValueError("sample_items must be a positive integer when provided")
        if self.max_output_tokens is not None and (
            isinstance(self.max_output_tokens, bool)
            or not isinstance(self.max_output_tokens, int)
            or self.max_output_tokens <= 0
        ):
            raise ValueError("max_output_tokens must be a positive integer when provided")
        if any(not isinstance(path, str) or not path.strip() for path in self.fields):
            raise ValueError("fields must contain non-empty strings")

    @classmethod
    def for_purpose(cls, purpose: str, **overrides: Any) -> "GateLimits":
        """Return conservative defaults for a search, log, or file inspection.

        The limits are display policies, not scan limits: callers may still scan
        every record internally and pass only the bounded result to the agent.
        """

        if not isinstance(purpose, str) or not purpose.strip():
            raise ValueError("purpose must be a non-empty string")
        profiles = {
            "generic": dict(max_chars=12_000, max_items=100, context_lines=2, max_output_tokens=4_096),
            "search": dict(max_chars=8_000, max_items=20, context_lines=1, max_output_tokens=2_048),
            "log": dict(max_chars=10_000, max_items=60, context_lines=3, max_output_tokens=3_072),
            "file": dict(max_chars=12_000, max_items=40, context_lines=2, max_output_tokens=4_096),
            "service_guide": dict(max_chars=6_000, max_items=32, context_lines=1, max_output_tokens=1_200),
            "subagent": dict(max_chars=12_000, max_items=80, context_lines=2, max_output_tokens=4_096),
        }
        aliases = {"search_results": "search", "logs": "log", "files": "file"}
        key = aliases.get(purpose.casefold().strip(), purpose.casefold().strip())
        if key not in profiles:
            raise ValueError(f"unsupported output purpose: {purpose}")
        values = {**profiles[key], "purpose": key, **overrides}
        return cls(**values)


@dataclass(frozen=True)
class GateResult:
    """Bounded output plus an audit-friendly description of any omissions."""

    text: str
    warnings: tuple[str, ...] = ()
    truncated: bool = False
    used_fallback: bool = False
    original_chars: int = 0
    original_items: int | None = None
    returned_items: int | None = None
    next_cursor: str | None = None
    original_tokens: int = 0
    rendered_tokens: int = 0
    max_output_tokens: int | None = None
    retained_categories: tuple[str, ...] = ()

    @property
    def tokens_saved(self) -> int:
        """Estimated tokens removed before the result enters agent context."""

        return max(0, self.original_tokens - self.rendered_tokens)

    @property
    def output_tokens_saved(self) -> int:
        """Alias emphasizing that the measured source is tool output."""

        return self.tokens_saved

    @property
    def output_tokens(self) -> int:
        """The rendered output token count, using the configured counter."""

        return self.rendered_tokens

    @property
    def input_tokens_saved(self) -> int:
        """Alias for the same local estimate when the gated text is prompt input."""

        return self.tokens_saved

    @property
    def token_savings_ratio(self) -> float:
        return self.tokens_saved / self.original_tokens if self.original_tokens else 0.0

    @property
    def has_more(self) -> bool:
        return self.next_cursor is not None

    @property
    def continuation_cursor(self) -> str | None:
        """Readable alias for integrations that call the value a cursor."""

        return self.next_cursor


Extractor = Callable[[str], str]


_PRIORITY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "error",
        re.compile(
            r"\b(?:error|err|fail(?:ed|ure)?|exception|traceback|fatal|critical|ng)\b"
            r"|(?:^|\s)(?:✗|❌)",
            re.IGNORECASE,
        ),
    ),
    (
        "reference",
        re.compile(
            r"https?://\S+|(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+(?::\d+)?"
            r"|\b(?:issue|pr|commit|ref(?:erence)?)\b\s*[#:=]?\s*[A-Za-z0-9_.-]+",
            re.IGNORECASE,
        ),
    ),
    (
        "evidence",
        re.compile(
            r"\b(?:because|evidence|reason|observed|expected|actual|assert|verified|proof|根拠|理由|観測|期待|実際|確認)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "value",
        re.compile(
            r"\b(?:total|count|tokens?|tokens_saved|cost|latency|status|result|rate|success|failure|changed|added|removed)\b\s*[:=]"
            r"|\b\d+(?:\.\d+)?\s*(?:%|ms|s|sec|seconds?|tokens?|items?|files?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "conclusion",
        re.compile(
            r"\b(?:verdict|conclusion|summary|recommendation|next\s+steps?|action(?:s)?)\b\s*[:#]?"
            r"|(?:^|\s)(?:結論|判定|要約|推奨|次の手順)\s*[:：]",
            re.IGNORECASE,
        ),
    ),
)


def _line_categories(line: str) -> tuple[str, ...]:
    return tuple(name for name, pattern in _PRIORITY_PATTERNS if pattern.search(line))


def _line_priority(line: str, index: int, total: int) -> int:
    categories = _line_categories(line)
    score = sum(
        {"error": 100, "reference": 70, "evidence": 65, "value": 60, "conclusion": 55}[category]
        for category in categories
    )
    if line.lstrip().startswith(("#", "- ", "* ")):
        score += 10
    if index in {0, total - 1}:
        score += 5
    return score


def _validate_token_count(token_counter: TokenCounter, text: str) -> int:
    value = token_counter(text)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("token_counter must return a non-negative integer")
    return value


def _fit_to_token_budget(text: str, max_tokens: int, token_counter: TokenCounter) -> str:
    """Clip text to a token budget without assuming a particular tokenizer."""

    if _validate_token_count(token_counter, text) <= max_tokens:
        return text
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if _validate_token_count(token_counter, text[:middle]) <= max_tokens:
            low = middle
        else:
            high = middle - 1
    return text[:low]


def _encode_cursor(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii").rstrip("=")


def _decode_cursor(cursor: str) -> Mapping[str, Any]:
    if not isinstance(cursor, str) or not cursor:
        raise ValueError("cursor must be a non-empty string")
    try:
        padding = "=" * (-len(cursor) % 4)
        value = json.loads(base64.urlsafe_b64decode((cursor + padding).encode("ascii")))
    except (ValueError, TypeError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid output continuation cursor") from exc
    if not isinstance(value, Mapping) or value.get("v") != 1:
        raise ValueError("unsupported output continuation cursor")
    return value


def _field_value(item: Mapping[str, Any], path: str) -> Any:
    value: Any = item
    for component in path.split("."):
        if not isinstance(value, Mapping) or component not in value:
            return None
        value = value[component]
    return value


def _project_fields(item: Any, fields: Sequence[str]) -> Any:
    if not fields or not isinstance(item, Mapping):
        return item
    return {path: _field_value(item, path) for path in fields}


def _selected_indices(
    length: int,
    page_size: int,
    query: str | None,
    cursor: str | None,
    *,
    purpose: str,
    values: Sequence[Any] = (),
) -> tuple[list[int], str | None, bool]:
    """Choose a representative first page or a deterministic continuation page."""

    if length <= page_size and cursor is None:
        return list(range(length)), None, False

    if cursor is not None:
        state = _decode_cursor(cursor)
        if (
            state.get("kind") != "sequence"
            or state.get("purpose") != purpose
            or state.get("total") != length
        ):
            raise ValueError("cursor does not match this output")
        mode = state.get("mode")
        offset = state.get("offset")
        end = state.get("end")
        if mode not in {"ranked", "middle"} or not all(
            isinstance(value, int) for value in (offset, end)
        ):
            raise ValueError("invalid output continuation cursor")
        if offset < 0 or end < offset or end > length:
            raise ValueError("invalid output continuation range")
        if mode == "ranked":
            # The caller's query is intentionally required for ranked pages.
            if not query or state.get("query_hash") != _query_hash(query):
                raise ValueError("cursor requires the original query")
            order = _ranked_indices(range(length), query, values)
            page = order[offset : min(offset + page_size, end)]
        else:
            page = list(range(offset, min(offset + page_size, end)))
        next_offset = offset + len(page)
        next_cursor = None
        if next_offset < end:
            next_cursor = _encode_cursor(
                {
                    "v": 1,
                    "kind": "sequence",
                    "purpose": purpose,
                    "total": length,
                    "mode": mode,
                    "offset": next_offset,
                    "end": end,
                    **({"query_hash": _query_hash(query)} if mode == "ranked" else {}),
                }
            )
        return page, next_cursor, True

    if query:
        order = _ranked_indices(range(length), query, values)
        page = order[:page_size]
        end = len(order)
        mode = "ranked"
        query_hash = {"query_hash": _query_hash(query)}
    else:
        head_count = (page_size + 1) // 2
        tail_count = page_size - head_count
        page = list(range(head_count))
        if tail_count:
            page.extend(range(length - tail_count, length))
        end = length - tail_count
        mode = "middle"
        query_hash = {}

    next_cursor = None
    offset = page_size if mode == "ranked" else head_count
    if offset < end:
        next_cursor = _encode_cursor(
            {
                "v": 1,
                "kind": "sequence",
                "purpose": purpose,
                "total": length,
                "mode": mode,
                "offset": offset,
                "end": end,
                **query_hash,
            }
        )
    return page, next_cursor, True


def _query_hash(query: str) -> str:
    return sha256(query.casefold().encode("utf-8")).hexdigest()[:16]


def _ranked_indices(indices: Sequence[int], query: str, values: Sequence[Any]) -> list[int]:
    terms = _terms(query)
    if not terms:
        return list(indices)
    scored = []
    for index in indices:
        candidate = str(values[index]).casefold() if values else str(index)
        scored.append((sum(candidate.count(term) for term in terms), index))
    return [index for _, index in sorted(scored, key=lambda pair: (-pair[0], pair[1]))]


def _serialize(value: Any) -> tuple[str, int | None]:
    if isinstance(value, str):
        return value, None
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="replace"), None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return json.dumps(value, ensure_ascii=False, indent=2, default=str), len(value)
    if isinstance(value, Mapping):
        return json.dumps(value, ensure_ascii=False, indent=2, default=str), None
    return str(value), None


def _terms(query: str) -> list[str]:
    return [term.casefold() for term in re.findall(r"\S+", query) if term.strip()]


def _extract_relevant(text: str, query: str | None, limits: GateLimits) -> str:
    if not query:
        raise LookupError("no query")
    terms = _terms(query)
    if not terms:
        raise LookupError("empty query")

    lines = text.splitlines()
    matched = [
        index
        for index, line in enumerate(lines)
        if any(term in line.casefold() for term in terms)
    ]
    if not matched:
        raise LookupError("no matching line")

    selected: set[int] = set()
    for index in matched:
        selected.update(
            range(
                max(0, index - limits.context_lines),
                min(len(lines), index + limits.context_lines + 1),
            )
        )
    return "\n".join(lines[index] for index in sorted(selected))


def _head_tail(text: str, budget: int) -> str:
    if len(text) <= budget:
        return text
    marker = "\n...[middle omitted by output gate]...\n"
    if budget <= len(marker):
        return text[:budget]
    remaining = budget - len(marker)
    head = (remaining + 1) // 2
    tail = remaining - head
    return text[:head] + marker + text[-tail:]


def _warning_header(warnings: Sequence[str]) -> str:
    if not warnings:
        return ""
    return "[output gate warnings]\n" + "\n".join(f"- {warning}" for warning in warnings) + "\n\n"


def _compact_warning(warning: str) -> str:
    compact = warning
    compact = re.sub(r"character count \d+ exceeded max_chars=\d+", "size limit exceeded", compact)
    compact = re.sub(r"item count \d+ exceeded max_items=\d+", "item limit exceeded", compact)
    compact = compact.replace("; unselected items are omitted", "; items omitted")
    compact = compact.replace("; bounded head/tail fallback used", "; fallback")
    return compact


def _render_with_warnings(content: str, warnings: Sequence[str], limits: GateLimits) -> str:
    header = _warning_header(warnings)
    if len(header) >= limits.max_chars:
        header = "[output gate: " + "; ".join(_compact_warning(warning) for warning in warnings) + "]\n"
    if len(header) >= limits.max_chars:
        header = "[output gate warning]\n"
        if len(header) >= limits.max_chars:
            return _head_tail(content, limits.max_chars)
    return header + _head_tail(content, limits.max_chars - len(header))


def gate_tool_output(
    output: Any,
    *,
    query: str | None = None,
    limits: GateLimits | None = None,
    extractor: Extractor | None = None,
    purpose: str = "generic",
    fields: Sequence[str] | None = None,
    cursor: str | None = None,
    token_counter: TokenCounter = _estimate_tokens,
) -> GateResult:
    """Return bounded tool output with explicit warnings and safe fallback.

    ``purpose`` selects conservative defaults for common investigation tools;
    pass ``limits`` to override them. For sequence results, ``max_items`` is a
    page size. The first page is a representative sample and ``next_cursor``
    can be passed back as ``cursor`` to retrieve omitted records without
    putting them in the current context. ``fields`` projects mapping results
    before serialization, so sensitive or irrelevant fields never enter the
    agent prompt.

    ``token_counter`` is used only for local measurement. It should be replaced
    with the provider tokenizer when comparing actual prompt budgets.
    """

    if limits is None:
        limits = GateLimits.for_purpose(purpose)
    elif purpose == "generic":
        purpose = limits.purpose
    selected_fields = tuple(fields) if fields is not None else limits.fields
    if any(not isinstance(path, str) or not path.strip() for path in selected_fields):
        raise ValueError("fields must contain non-empty strings")
    page_size = min(limits.max_items, limits.sample_items or limits.max_items)
    raw_text, original_items = _serialize(output)
    line_mode = isinstance(output, str) and purpose.casefold() in {"log", "file"}
    source_items: Sequence[Any] | None = None
    if line_mode:
        source_items = raw_text.splitlines()
        original_items = len(source_items)
    elif isinstance(output, Sequence) and not isinstance(output, (str, bytes, bytearray)):
        source_items = output
    warnings: list[str] = []
    content = raw_text
    used_fallback = False
    truncated = False
    next_cursor: str | None = None
    returned_items = original_items

    if source_items is not None and original_items is not None and original_items > page_size:
        indices, next_cursor, _ = _selected_indices(
            original_items,
            page_size,
            query,
            cursor,
            purpose=purpose,
            values=source_items,
        )
        selected = [
            _project_fields(source_items[index], selected_fields)
            for index in indices
        ]
        item_label = "line count" if line_mode else "item count"
        warnings.append(
            f"{item_label} {original_items} exceeded max_items={page_size}; "
            "representative sample returned; unselected items are omitted"
        )
        if next_cursor:
            warnings.append("more items available through next_cursor")
        if selected_fields:
            warnings.append("fields projected: " + ", ".join(selected_fields))
        if line_mode:
            content = "\n".join(selected)
            returned_items = len(selected)
        elif isinstance(output, Sequence) and not isinstance(output, (str, bytes, bytearray)):
            content, _ = _serialize(selected)
            returned_items = len(selected)
        truncated = True
    elif selected_fields and isinstance(output, Mapping):
        content, _ = _serialize(_project_fields(output, selected_fields))
        returned_items = 1
        warnings.append("fields projected: " + ", ".join(selected_fields))
    elif selected_fields and isinstance(output, Sequence) and not isinstance(output, (str, bytes, bytearray)):
        projected = [_project_fields(item, selected_fields) for item in output]
        content, _ = _serialize(projected)
        returned_items = len(projected)
        warnings.append("fields projected: " + ", ".join(selected_fields))

    if line_mode and source_items is not None and original_items is not None and original_items <= page_size:
        returned_items = original_items

    if cursor is not None and original_items is not None and original_items <= page_size:
        raise ValueError("cursor is not needed for an output within the item limit")

    if len(raw_text) > limits.max_chars:
        warnings.append(
            f"character count {len(raw_text)} exceeded max_chars={limits.max_chars}"
        )

    if len(content) > limits.max_chars:
        try:
            extracted = (
                extractor(content)
                if extractor is not None
                else _extract_relevant(content, query, limits)
            )
            if not isinstance(extracted, str) or not extracted.strip():
                raise ValueError("empty extraction")
            content = extracted
        except Exception as exc:  # extraction is a safety boundary
            reason = "custom extractor" if extractor is not None else "relevance extractor"
            warnings.append(
                f"{reason} failed ({type(exc).__name__}); bounded head/tail fallback used"
            )
            content = _head_tail(content, limits.max_chars)
            used_fallback = True
        truncated = True

    rendered = _render_with_warnings(content, warnings, limits)
    if len(rendered) > limits.max_chars:
        rendered = _render_with_warnings(_head_tail(content, limits.max_chars), warnings, limits)
        truncated = True

    original_tokens = _validate_token_count(token_counter, raw_text)
    rendered_tokens = _validate_token_count(token_counter, rendered)
    if limits.max_output_tokens is not None and rendered_tokens > limits.max_output_tokens:
        warnings.append(
            f"output tokens {rendered_tokens} exceeded max_output_tokens={limits.max_output_tokens}"
        )
        # Keep the enforcement reason visible even when the budget is so small
        # that the full audit header itself would consume the remaining room.
        rendered = f"[output gate: max_output_tokens={limits.max_output_tokens}]\n{content}"
        if len(rendered) > limits.max_chars:
            rendered = rendered[: limits.max_chars]
        rendered = _fit_to_token_budget(
            rendered,
            limits.max_output_tokens,
            token_counter,
        )
        truncated = True
        rendered_tokens = _validate_token_count(token_counter, rendered)

    return GateResult(
        text=rendered,
        warnings=tuple(warnings),
        truncated=truncated,
        used_fallback=used_fallback,
        original_chars=len(raw_text),
        original_items=original_items,
        returned_items=returned_items,
        next_cursor=next_cursor,
        original_tokens=original_tokens,
        rendered_tokens=rendered_tokens,
        max_output_tokens=limits.max_output_tokens,
    )


def format_subagent_output(
    output: Any,
    *,
    limits: GateLimits | None = None,
    token_counter: TokenCounter = _estimate_tokens,
    required_markers: Sequence[str] = (),
) -> GateResult:
    """Format a sub-agent report with bounded, priority-preserving output.

    The formatter keeps the original lines verbatim. When a report exceeds a
    line, character, or token budget, it selects the highest-value lines first
    and then restores their original order. Error lines outrank references,
    evidence, measured values, and conclusions; caller-provided
    ``required_markers`` outrank all heuristic categories, and unclassified
    narrative fills the remaining space. This is intentionally deterministic
    and does not pretend that a generic summarizer can infer task semantics
    safely.
    """

    marker_values = tuple(required_markers)
    if any(not isinstance(marker, str) or not marker for marker in marker_values):
        raise ValueError("required_markers must contain non-empty strings")
    if limits is None:
        limits = GateLimits.for_purpose("subagent")
    elif limits.purpose.casefold().strip() != "subagent":
        limits = GateLimits(
            max_chars=limits.max_chars,
            max_items=limits.max_items,
            context_lines=limits.context_lines,
            purpose="subagent",
            fields=limits.fields,
            sample_items=limits.sample_items,
            max_output_tokens=limits.max_output_tokens,
        )

    raw_text, _ = _serialize(output)
    lines = raw_text.splitlines() or ([raw_text] if raw_text else [])
    original_items = len(lines)
    original_tokens = _validate_token_count(token_counter, raw_text)
    priorities = [_line_priority(line, index, original_items) for index, line in enumerate(lines)]
    for index, line in enumerate(lines):
        if any(marker in line for marker in marker_values):
            priorities[index] += 10_000
    categories = [_line_categories(line) for line in lines]
    needs_bounding = (
        original_items > limits.max_items
        or len(raw_text) > limits.max_chars
        or (
            limits.max_output_tokens is not None
            and original_tokens > limits.max_output_tokens
        )
    )

    warnings: list[str] = []
    if original_items > limits.max_items:
        warnings.append(
            f"line count {original_items} exceeded max_items={limits.max_items}; "
            "priority lines retained; other lines omitted"
        )
    if len(raw_text) > limits.max_chars:
        warnings.append(
            f"character count {len(raw_text)} exceeded max_chars={limits.max_chars}"
        )
    if limits.max_output_tokens is not None and original_tokens > limits.max_output_tokens:
        warnings.append(
            f"output tokens {original_tokens} exceeded max_output_tokens={limits.max_output_tokens}"
        )
    if needs_bounding:
        warnings.append(
            "priority order: errors, references, evidence, values, and conclusions"
        )

    if not needs_bounding:
        return GateResult(
            text=raw_text,
            warnings=(),
            truncated=False,
            used_fallback=False,
            original_chars=len(raw_text),
            original_items=original_items,
            returned_items=original_items,
            original_tokens=original_tokens,
            rendered_tokens=original_tokens,
            max_output_tokens=limits.max_output_tokens,
            retained_categories=tuple(
                category
                for category in ("error", "reference", "evidence", "value", "conclusion")
                if any(category in line_categories for line_categories in categories)
            ),
        )

    header = _warning_header(warnings)
    if len(header) >= limits.max_chars:
        header = "[output gate: " + "; ".join(_compact_warning(warning) for warning in warnings) + "]\n"
    if len(header) >= limits.max_chars:
        header = "[output gate warning]\n"

    candidate_indices = sorted(
        range(original_items),
        key=lambda index: (-priorities[index], index),
    )
    if not any(categories):
        head_count = (limits.max_items + 1) // 2
        tail_count = max(0, limits.max_items - head_count)
        candidate_indices = list(range(min(head_count, original_items)))
        if tail_count:
            candidate_indices.extend(range(max(0, original_items - tail_count), original_items))
        candidate_indices = list(dict.fromkeys(candidate_indices))

    selected: list[int] = []
    for index in candidate_indices:
        if len(selected) >= limits.max_items:
            break
        proposed = selected + [index]
        body = "\n".join(lines[item] for item in sorted(proposed))
        rendered = header + body
        if len(rendered) > limits.max_chars:
            continue
        if (
            limits.max_output_tokens is not None
            and _validate_token_count(token_counter, rendered) > limits.max_output_tokens
        ):
            continue
        selected.append(index)

    selected.sort()
    body = "\n".join(lines[index] for index in selected)
    rendered = header + body
    if not selected and lines:
        available = max(0, limits.max_chars - len(header))
        body = lines[candidate_indices[0]][:available]
        rendered = header + body
    if len(rendered) > limits.max_chars:
        rendered = rendered[: limits.max_chars]
    if limits.max_output_tokens is not None:
        rendered = _fit_to_token_budget(rendered, limits.max_output_tokens, token_counter)

    rendered_tokens = _validate_token_count(token_counter, rendered)
    rendered_categories = [_line_categories(line) for line in rendered.splitlines()]
    retained_categories = tuple(
        category
        for category in ("error", "reference", "evidence", "value", "conclusion")
        if any(category in line_categories for line_categories in rendered_categories)
    )
    returned_items = len(selected)
    truncated = needs_bounding or returned_items != original_items or rendered != raw_text
    return GateResult(
        text=rendered,
        warnings=tuple(warnings),
        truncated=truncated,
        used_fallback=False,
        original_chars=len(raw_text),
        original_items=original_items,
        returned_items=returned_items,
        original_tokens=original_tokens,
        rendered_tokens=rendered_tokens,
        max_output_tokens=limits.max_output_tokens,
        retained_categories=retained_categories,
    )


SubagentOutputResult = GateResult
