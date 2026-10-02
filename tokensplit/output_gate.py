"""Bound and selectively extract tool output before it reaches an agent."""
from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GateLimits:
    """Limits applied to one tool result."""

    max_chars: int = 12_000
    max_items: int = 100
    context_lines: int = 2

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


@dataclass(frozen=True)
class GateResult:
    """Bounded output plus an audit-friendly description of any omissions."""

    text: str
    warnings: tuple[str, ...] = ()
    truncated: bool = False
    used_fallback: bool = False
    original_chars: int = 0
    original_items: int | None = None


Extractor = Callable[[str], str]


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


def _select_items(value: Sequence[Any], limits: GateLimits, query: str | None) -> list[Any]:
    if len(value) <= limits.max_items:
        return list(value)

    if query:
        terms = _terms(query)
        scored = []
        for index, item in enumerate(value):
            candidate = str(item).casefold()
            score = sum(candidate.count(term) for term in terms)
            scored.append((score, index))
        selected_indices = sorted(
            index
            for score, index in sorted(scored, key=lambda pair: (-pair[0], pair[1]))[
                : limits.max_items
            ]
        )
    else:
        head_count = (limits.max_items + 1) // 2
        tail_count = limits.max_items - head_count
        selected_indices = list(range(head_count))
        if tail_count:
            selected_indices.extend(range(len(value) - tail_count, len(value)))

    return [value[index] for index in selected_indices]


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
) -> GateResult:
    """Return bounded tool output with explicit warnings and safe fallback."""

    limits = limits or GateLimits()
    raw_text, original_items = _serialize(output)
    warnings: list[str] = []
    content = raw_text
    used_fallback = False
    truncated = False

    if original_items is not None and original_items > limits.max_items:
        warnings.append(
            f"item count {original_items} exceeded max_items={limits.max_items}; "
            "unselected items are omitted"
        )
        if isinstance(output, Sequence) and not isinstance(output, (str, bytes, bytearray)):
            content, _ = _serialize(_select_items(output, limits, query))
        truncated = True

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

    return GateResult(
        text=rendered,
        warnings=tuple(warnings),
        truncated=truncated,
        used_fallback=used_fallback,
        original_chars=len(raw_text),
        original_items=original_items,
    )
