"""Qwen API provider adapter for bounded orchestration integration tests.

The adapter uses Qwen's OpenAI-compatible chat-completions endpoint.  It
normalizes the response into the same privacy-safe run result consumed by the
existing orchestration and comparison-log adapter.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .claude_cli import (
    ClaudeCliError,
    ClaudeCliRunResult,
    ClaudeCliRunner,
    ClaudeCodeCliAdapter,
    StderrDiagnostic,
    _as_cost,
    _usage_value,
    classify_stderr,
)
from .orchestration import AgentController, ComparisonLog
from .output_gate import GateLimits, TokenCounter
from .routing import ModelRouter, RoutingLog


DEFAULT_QWEN_API_URL = (
    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
)


@dataclass(frozen=True)
class QwenApiConfig:
    """HTTP settings for one Qwen API invocation."""

    api_key: str | None = None
    base_url: str | None = None
    timeout_seconds: float = 120.0
    max_retries: int = 2
    retry_initial_delay_seconds: float = 1.0
    retry_backoff_factor: float = 2.0
    retry_max_delay_seconds: float = 8.0
    retry_sleep: Callable[[float], None] | None = None
    output_limits: GateLimits | None = None
    token_counter: TokenCounter | None = None
    working_directory: str | Path | None = None

    def __post_init__(self) -> None:
        if self.api_key is not None and (not isinstance(self.api_key, str) or not self.api_key.strip()):
            raise ClaudeCliError("api_key must be a non-empty string when provided")
        if self.base_url is not None and (not isinstance(self.base_url, str) or not self.base_url.strip()):
            raise ClaudeCliError("base_url must be a non-empty string when provided")
        if self.working_directory is not None and not Path(self.working_directory).is_dir():
            raise ClaudeCliError("working_directory must be an existing directory")
        if not isinstance(self.timeout_seconds, (int, float)) or isinstance(self.timeout_seconds, bool) or self.timeout_seconds <= 0:
            raise ClaudeCliError("timeout_seconds must be positive")
        if not isinstance(self.max_retries, int) or isinstance(self.max_retries, bool) or self.max_retries < 0:
            raise ClaudeCliError("max_retries must be a non-negative integer")
        if self.retry_initial_delay_seconds < 0 or self.retry_max_delay_seconds < 0:
            raise ClaudeCliError("retry delays must be non-negative")
        if self.retry_backoff_factor < 1:
            raise ClaudeCliError("retry_backoff_factor must be at least 1")
        if self.retry_sleep is not None and not callable(self.retry_sleep):
            raise ClaudeCliError("retry_sleep must be callable or None")


class QwenApiError(ClaudeCliError):
    """Safe Qwen API failure carrying only a diagnostic category."""


class _QwenResponseFailure(Exception):
    def __init__(self, diagnostic: StderrDiagnostic) -> None:
        self.diagnostic = diagnostic


class QwenApiRunner:
    """Run prompts through Qwen's OpenAI-compatible JSON API."""

    def __init__(self, config: QwenApiConfig | None = None) -> None:
        self.config = config or QwenApiConfig()

    def _url(self) -> str:
        url = (
            self.config.base_url
            or os.environ.get("QWEN_API_BASE_URL")
            or DEFAULT_QWEN_API_URL
        )
        normalized = url.rstrip("/")
        return normalized + "/chat/completions" if normalized.endswith("/v1") else normalized

    def _api_key(self) -> str:
        key = self.config.api_key or os.environ.get("QWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
        if not key:
            diagnostic = StderrDiagnostic("auth", False)
            raise QwenApiError(
                "Qwen API key is not configured",
                diagnostic=diagnostic,
                failure_categories=(diagnostic.category,),
                permanent_failures=1,
            )
        return key

    @staticmethod
    def _content(value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            parts = []
            for item in value:
                if isinstance(item, Mapping) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            return "".join(parts)
        return ""

    @staticmethod
    def _http_diagnostic(status: int) -> StderrDiagnostic:
        if status in {401, 403}:
            return StderrDiagnostic("auth", False, status)
        if status in {408, 504}:
            return StderrDiagnostic("timeout", True, status)
        if status in {409, 425, 429, 500, 502, 503, 529}:
            return StderrDiagnostic("rate_limit", True, status)
        if 400 <= status < 500:
            return StderrDiagnostic("invalid_request", False, status)
        return StderrDiagnostic("process", False, status)

    @classmethod
    def _normalize_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        request_id: str,
        model: str,
        batch_index: int | None,
        output_limits: GateLimits | None,
        token_counter: TokenCounter | None,
        required_markers: Sequence[str],
    ) -> ClaudeCliRunResult:
        choices = payload.get("choices")
        if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes, bytearray)) or not choices:
            raise ValueError("Qwen API response did not contain choices")
        choice = choices[0] if isinstance(choices[0], Mapping) else {}
        message = choice.get("message") if isinstance(choice, Mapping) else {}
        message = message if isinstance(message, Mapping) else {}
        content = cls._content(message.get("content"))
        if not content:
            raise ValueError("Qwen API response did not contain message content")
        usage = payload.get("usage")
        usage = usage if isinstance(usage, Mapping) else {}
        normalized = {
            "result": content,
            "model": payload.get("model") or model,
            "usage": {
                "input_tokens": _usage_value(usage, "prompt_tokens", "input_tokens"),
                "output_tokens": _usage_value(usage, "completion_tokens", "output_tokens"),
            },
            "total_cost_usd": _as_cost(payload.get("cost_usd")),
        }
        return ClaudeCliRunner.parse_payload(
            normalized,
            request_id=request_id,
            model=model,
            batch_index=batch_index,
            output_limits=output_limits,
            token_counter=token_counter,
            required_markers=required_markers,
        )

    def _post_messages(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        model: str,
        tools: Sequence[Mapping[str, Any]] = (),
    ) -> Mapping[str, Any]:
        request_body: dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "stream": False,
        }
        if tools:
            request_body["tools"] = list(tools)
        body = json.dumps(
            request_body
        ).encode("utf-8")
        request = Request(
            self._url(),
            data=body,
            headers={
                "Authorization": f"Bearer {self._api_key()}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urlopen(request, timeout=self.config.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("Qwen API response was not an object")
        error = payload.get("error")
        if isinstance(error, Mapping):
            hint = " ".join(str(error.get(key, "")) for key in ("code", "type", "message"))
            raise _QwenResponseFailure(classify_stderr(hint))
        return payload

    def _request(self, *, prompt: str, model: str) -> Mapping[str, Any]:
        return self._post_messages(
            messages=[{"role": "user", "content": prompt}],
            model=model,
        )

    def run(
        self,
        *,
        request_id: str,
        prompt: str,
        model: str | None,
        batch_index: int | None = None,
        required_markers: Sequence[str] = (),
    ) -> ClaudeCliRunResult:
        if not isinstance(request_id, str) or not request_id.strip():
            raise QwenApiError("request_id must be a non-empty string")
        if not isinstance(prompt, str) or not prompt.strip():
            raise QwenApiError("prompt must be a non-empty string")
        if not isinstance(model, str) or not model.strip():
            raise QwenApiError("Qwen model must be supplied")

        attempt = 0
        failure_categories: list[str] = []
        retry_wait_seconds = 0.0
        transient_failures = 0
        permanent_failures = 0
        last_category: str | None = None
        while True:
            attempt += 1
            try:
                payload = self._request(prompt=prompt, model=model)
                parsed = self._normalize_payload(
                    payload,
                    request_id=request_id,
                    model=model,
                    batch_index=batch_index,
                    output_limits=self.config.output_limits,
                    token_counter=self.config.token_counter,
                    required_markers=required_markers,
                )
                return replace(
                    parsed,
                    attempts=attempt,
                    retry_count=attempt - 1,
                    stderr_category=last_category,
                    failure_categories=tuple(failure_categories),
                    retry_wait_seconds=retry_wait_seconds,
                    transient_failures=transient_failures,
                    permanent_failures=permanent_failures,
                )
            except HTTPError as exc:
                category = self._http_diagnostic(exc.code)
            except (URLError, TimeoutError):
                category = classify_stderr("network error", returncode=None)
            except _QwenResponseFailure as exc:
                category = exc.diagnostic
            except (json.JSONDecodeError, ValueError, TypeError):
                category = StderrDiagnostic("invalid_json", True)
            except QwenApiError:
                raise

            failure_categories.append(category.category)
            if category.retryable:
                transient_failures += 1
            else:
                permanent_failures += 1
            if category.retryable and attempt <= self.config.max_retries:
                last_category = category.category
                delay = min(
                    self.config.retry_initial_delay_seconds
                    * (self.config.retry_backoff_factor ** (attempt - 1)),
                    self.config.retry_max_delay_seconds,
                )
                retry_wait_seconds += delay
                (self.config.retry_sleep or time.sleep)(delay)
                continue
            raise QwenApiError(
                f"Qwen API failed: category={category.category}, attempts={attempt}",
                diagnostic=category,
                attempts=attempt,
                failure_categories=failure_categories,
                retry_wait_seconds=retry_wait_seconds,
                transient_failures=transient_failures,
                permanent_failures=permanent_failures,
            )


class QwenApiAdapter(ClaudeCodeCliAdapter):
    """Reuse orchestration planning and comparison logging with Qwen API."""

    def __init__(
        self,
        runner: QwenApiRunner | None = None,
        controller: AgentController | None = None,
        *,
        default_model: str | None = None,
        comparison_log: ComparisonLog | None = None,
        router: ModelRouter | None = None,
        routing_log: RoutingLog | None = None,
        routing_log_path: str | None = None,
    ) -> None:
        super().__init__(
            runner=runner or QwenApiRunner(),
            controller=controller,
            default_model=default_model,
            comparison_log=comparison_log,
            router=router,
            routing_log=routing_log,
            routing_log_path=routing_log_path,
        )


class QwenToolCallingRunner(QwenApiRunner):
    """Qwen runner with a bounded, read-only repository inspection toolset."""

    _TOOLS: tuple[Mapping[str, Any], ...] = (
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a UTF-8 text file within the repository.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "start_line": {"type": "integer", "minimum": 1},
                        "end_line": {"type": "integer", "minimum": 1},
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List repository files matching an optional glob.",
                "parameters": {
                    "type": "object",
                    "properties": {"glob": {"type": "string"}},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_text",
                "description": "Search text files in the repository for a literal string.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "glob": {"type": "string"},
                    },
                    "required": ["pattern"],
                },
            },
        },
    )

    def __init__(self, config: QwenApiConfig | None = None, *, max_tool_rounds: int = 8) -> None:
        super().__init__(config)
        if isinstance(max_tool_rounds, bool) or not isinstance(max_tool_rounds, int) or max_tool_rounds < 1:
            raise ClaudeCliError("max_tool_rounds must be a positive integer")
        self.max_tool_rounds = max_tool_rounds

    @property
    def _root(self) -> Path:
        return Path(self.config.working_directory or os.getcwd()).resolve()

    def _safe_path(self, raw_path: Any) -> Path | None:
        if not isinstance(raw_path, str) or not raw_path.strip():
            return None
        candidate = (self._root / raw_path).resolve()
        if candidate != self._root and self._root not in candidate.parents:
            return None
        blocked = {".git", ".env", ".env.local", "id_rsa", "credentials"}
        if any(part.lower() in blocked for part in candidate.relative_to(self._root).parts):
            return None
        return candidate

    def _read_file(self, arguments: Mapping[str, Any]) -> str:
        path = self._safe_path(arguments.get("path"))
        if path is None or not path.is_file():
            return "ERROR: file is unavailable"
        start = arguments.get("start_line", 1)
        end = arguments.get("end_line", start + 199 if isinstance(start, int) else 200)
        if not isinstance(start, int) or isinstance(start, bool) or not isinstance(end, int) or isinstance(end, bool):
            return "ERROR: line bounds must be integers"
        if start < 1 or end < start or end - start > 199:
            return "ERROR: line range is limited to 200 lines"
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            return "ERROR: file is unavailable"
        return "\n".join(lines[start - 1:end])[:16000] or "(empty file)"

    def _list_files(self, arguments: Mapping[str, Any]) -> str:
        pattern = arguments.get("glob", "**/*")
        if not isinstance(pattern, str) or not pattern:
            pattern = "**/*"
        values = []
        for path in self._root.glob(pattern):
            if not path.is_file() or ".git" in path.parts:
                continue
            if self._safe_path(str(path.relative_to(self._root))) is None:
                continue
            values.append(str(path.relative_to(self._root)))
            if len(values) >= 100:
                break
        return "\n".join(sorted(values)) or "(no matching files)"

    def _search_text(self, arguments: Mapping[str, Any]) -> str:
        pattern = arguments.get("pattern")
        glob = arguments.get("glob", "**/*")
        if not isinstance(pattern, str) or not pattern or not isinstance(glob, str):
            return "ERROR: pattern must be a non-empty string"
        matches = []
        for path in self._root.glob(glob):
            if not path.is_file() or ".git" in path.parts or self._safe_path(str(path.relative_to(self._root))) is None:
                continue
            try:
                for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                    if pattern in line:
                        matches.append(f"{path.relative_to(self._root)}:{line_number}:{line[:240]}")
                        if len(matches) >= 40:
                            return "\n".join(matches)
            except (OSError, UnicodeError):
                continue
        return "\n".join(matches) or "(no matches)"

    def _execute_tool(self, name: Any, arguments: Any) -> str:
        if not isinstance(arguments, Mapping):
            arguments = {}
        if name == "read_file":
            return self._read_file(arguments)
        if name == "list_files":
            return self._list_files(arguments)
        if name == "search_text":
            return self._search_text(arguments)
        return "ERROR: tool is not available"

    def _request(self, *, prompt: str, model: str) -> Mapping[str, Any]:
        messages: list[Mapping[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are a read-only repository reviewer. Use the provided tools only when needed, "
                    "prefer the minimum number of calls, never invent file contents, and return a final "
                    "answer after you have enough evidence. Do not request shell commands or file changes."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        for _ in range(self.max_tool_rounds):
            payload = self._post_messages(messages=messages, model=model, tools=self._TOOLS)
            choices = payload.get("choices")
            choice = choices[0] if isinstance(choices, Sequence) and choices and isinstance(choices[0], Mapping) else {}
            message = choice.get("message") if isinstance(choice, Mapping) else {}
            message = message if isinstance(message, Mapping) else {}
            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, Sequence) or isinstance(tool_calls, (str, bytes, bytearray)) or not tool_calls:
                return payload

            assistant_message = {
                "role": "assistant",
                "content": message.get("content"),
                "tool_calls": list(tool_calls),
            }
            messages.append(assistant_message)
            for call in tool_calls:
                if not isinstance(call, Mapping):
                    messages.append({"role": "tool", "tool_call_id": "invalid", "content": "ERROR: invalid tool call"})
                    continue
                function = call.get("function")
                function = function if isinstance(function, Mapping) else {}
                name = function.get("name")
                raw_arguments = function.get("arguments", "{}")
                try:
                    arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
                except (json.JSONDecodeError, TypeError):
                    arguments = {}
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(call.get("id", "invalid")),
                        "content": self._execute_tool(name, arguments),
                    }
                )
        raise ValueError("Qwen tool-call loop exceeded the configured limit")


__all__ = [
    "DEFAULT_QWEN_API_URL",
    "QwenApiConfig",
    "QwenApiError",
    "QwenApiRunner",
    "QwenApiAdapter",
    "QwenToolCallingRunner",
]
