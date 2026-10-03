"""Qwen API provider adapter for bounded orchestration integration tests.

The adapter uses Qwen's OpenAI-compatible chat-completions endpoint.  It
normalizes the response into the same privacy-safe run result consumed by the
existing orchestration and comparison-log adapter.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
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

    def __post_init__(self) -> None:
        if self.api_key is not None and (not isinstance(self.api_key, str) or not self.api_key.strip()):
            raise ClaudeCliError("api_key must be a non-empty string when provided")
        if self.base_url is not None and (not isinstance(self.base_url, str) or not self.base_url.strip()):
            raise ClaudeCliError("base_url must be a non-empty string when provided")
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

    def _request(self, *, prompt: str, model: str) -> Mapping[str, Any]:
        body = json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            }
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
                (self.config.retry_sleep or __import__("time").sleep)(delay)
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


__all__ = ["DEFAULT_QWEN_API_URL", "QwenApiConfig", "QwenApiError", "QwenApiRunner", "QwenApiAdapter"]
