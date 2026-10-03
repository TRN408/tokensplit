"""Claude Code CLI adapter for bounded orchestration integration tests.

The adapter uses Claude Code's documented non-interactive interface:
``claude --print --output-format json --model MODEL PROMPT``.  It is kept
separate from the planner so tests can inject a local CLI-compatible command
without requiring credentials or spending tokens.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
import subprocess
import time
from typing import Any, Callable, Literal

from .orchestration import (
    AgentController,
    AgentRequest,
    ComparisonLog,
    ComparisonMeasurement,
    DispatchPlan,
    ExecutionDiagnostic,
    OrchestrationError,
)
from .output_gate import GateLimits, TokenCounter, format_subagent_output
from .routing import ModelRouter, RoutingLog


StderrCategory = Literal[
    "auth",
    "invalid_request",
    "network",
    "rate_limit",
    "timeout",
    "unknown_exit",
    "invalid_json",
    "process",
]


@dataclass(frozen=True)
class StderrDiagnostic:
    """A safe classification of stderr without retaining its contents."""

    category: StderrCategory
    retryable: bool
    exit_code: int | None = None


_STDERR_PATTERNS: tuple[tuple[StderrCategory, re.Pattern[str], bool], ...] = (
    ("auth", re.compile(r"unauthori[sz]|not logged in|invalid (?:api )?key|forbidden|\b401\b|\b403\b", re.I), False),
    ("invalid_request", re.compile(r"invalid request|unknown option|usage:\s*claude|bad argument", re.I), False),
    ("rate_limit", re.compile(r"rate limit|too many requests|overloaded|capacity|\b429\b|\b503\b|\b529\b|retry[- ]after", re.I), True),
    ("network", re.compile(r"connection|econn|dns|network|socket|service unavailable|temporar(?:y|ily) unavailable", re.I), True),
)


def classify_stderr(
    stderr: str | None,
    *,
    returncode: int | None = None,
    timed_out: bool = False,
) -> StderrDiagnostic:
    """Classify CLI failure text without exposing or storing stderr.

    Empty stderr with exit code 1 is treated as a bounded transient failure;
    this covers intermittent Claude CLI exits that provide no diagnostic text.
    Authentication, permission, and malformed-request failures are never
    retried automatically.
    """

    if timed_out:
        return StderrDiagnostic("timeout", True, returncode)
    text = stderr if isinstance(stderr, str) else ""
    for category, pattern, retryable in _STDERR_PATTERNS:
        if pattern.search(text):
            return StderrDiagnostic(category, retryable, returncode)
    if returncode == 1:
        return StderrDiagnostic("unknown_exit", True, returncode)
    return StderrDiagnostic("process", False, returncode)


class ClaudeCliError(RuntimeError):
    """Raised when the Claude Code CLI cannot produce a valid result."""

    def __init__(
        self,
        message: str,
        *,
        diagnostic: StderrDiagnostic | None = None,
        attempts: int = 1,
        failure_categories: Sequence[str] = (),
        retry_wait_seconds: float = 0.0,
        transient_failures: int = 0,
        permanent_failures: int = 0,
    ) -> None:
        super().__init__(message)
        self.diagnostic = diagnostic
        self.attempts = attempts
        self.failure_categories = tuple(failure_categories)
        self.retry_wait_seconds = retry_wait_seconds
        self.transient_failures = transient_failures
        self.permanent_failures = permanent_failures

    def execution_diagnostic(self, execution_id: str) -> ExecutionDiagnostic:
        return ExecutionDiagnostic(
            execution_id=execution_id,
            attempts=self.attempts,
            retry_count=max(0, self.attempts - 1),
            failure_categories=self.failure_categories,
            retry_wait_seconds=self.retry_wait_seconds,
            transient_failures=self.transient_failures,
            permanent_failures=self.permanent_failures,
            outcome="failure",
        )


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _as_cost(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        cost = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return cost if cost.is_finite() and cost >= 0 else None


def _usage_value(usage: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = _as_int(usage.get(key))
        if value is not None:
            return value
    return None


@dataclass(frozen=True)
class ClaudeCliConfig:
    """Process settings for one Claude Code CLI invocation."""

    executable: str | Sequence[str] = "claude"
    timeout_seconds: float = 120.0
    working_directory: str | Path | None = None
    extra_args: tuple[str, ...] = ()
    tools: tuple[str, ...] | None = ()
    no_session_persistence: bool = True
    output_limits: GateLimits | None = None
    token_counter: TokenCounter | None = None
    max_retries: int = 2
    retry_initial_delay_seconds: float = 1.0
    retry_backoff_factor: float = 2.0
    retry_max_delay_seconds: float = 8.0
    retry_sleep: Callable[[float], None] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.executable, str):
            if not self.executable.strip():
                raise ClaudeCliError("executable must be a non-empty string")
        elif not self.executable or any(not isinstance(value, str) or not value for value in self.executable):
            raise ClaudeCliError("executable must be a command or non-empty command sequence")
        if isinstance(self.timeout_seconds, bool) or not isinstance(self.timeout_seconds, (int, float)) or self.timeout_seconds <= 0:
            raise ClaudeCliError("timeout_seconds must be positive")
        if self.working_directory is not None and not Path(self.working_directory).is_dir():
            raise ClaudeCliError("working_directory must be an existing directory")
        if any(not isinstance(value, str) for value in self.extra_args):
            raise ClaudeCliError("extra_args must contain strings")
        if self.tools is not None and any(not isinstance(value, str) for value in self.tools):
            raise ClaudeCliError("tools must contain strings")
        if self.output_limits is not None and not isinstance(self.output_limits, GateLimits):
            raise ClaudeCliError("output_limits must be GateLimits or None")
        if isinstance(self.max_retries, bool) or not isinstance(self.max_retries, int) or self.max_retries < 0:
            raise ClaudeCliError("max_retries must be a non-negative integer")
        for name in (
            "retry_initial_delay_seconds",
            "retry_max_delay_seconds",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ClaudeCliError(f"{name} must be a non-negative number")
        if (
            isinstance(self.retry_backoff_factor, bool)
            or not isinstance(self.retry_backoff_factor, (int, float))
            or self.retry_backoff_factor < 1
        ):
            raise ClaudeCliError("retry_backoff_factor must be at least 1")
        if self.retry_sleep is not None and not callable(self.retry_sleep):
            raise ClaudeCliError("retry_sleep must be callable or None")


@dataclass(frozen=True)
class ClaudeCliRunResult:
    """Sanitized result from one CLI process."""

    request_id: str
    model: str
    text: str
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: Decimal | None
    batch_index: int | None
    baseline_task_success: bool | None = None
    task_success: bool | None = None
    important_information_retention: float | None = None
    required_markers: int = 0
    baseline_retained_markers: int | None = None
    retained_markers: int | None = None
    output_gate_original_tokens: int = 0
    output_gate_rendered_tokens: int = 0
    output_gate_tokens_saved: int = 0
    output_gate_truncated: bool = False
    output_gate_retained_categories: tuple[str, ...] = ()
    attempts: int = 1
    retry_count: int = 0
    stderr_category: StderrCategory | None = None
    failure_categories: tuple[str, ...] = ()
    retry_wait_seconds: float = 0.0
    transient_failures: int = 0
    permanent_failures: int = 0

    def execution_diagnostic(self) -> ExecutionDiagnostic:
        return ExecutionDiagnostic(
            execution_id=self.request_id,
            attempts=self.attempts,
            retry_count=self.retry_count,
            failure_categories=self.failure_categories,
            retry_wait_seconds=self.retry_wait_seconds,
            transient_failures=self.transient_failures,
            permanent_failures=self.permanent_failures,
            outcome="success",
        )


@dataclass(frozen=True)
class ClaudeCliDispatchResult:
    plan: DispatchPlan
    runs: tuple[ClaudeCliRunResult, ...]
    measurement: ComparisonMeasurement | None


class ClaudeCliRunner:
    """Run one prompt through Claude Code's non-interactive JSON interface."""

    def __init__(self, config: ClaudeCliConfig | None = None) -> None:
        self.config = config or ClaudeCliConfig()

    def _command(self, *, prompt: str, model: str | None) -> list[str]:
        executable = [self.config.executable] if isinstance(self.config.executable, str) else list(self.config.executable)
        command = [*executable, "--print", "--output-format", "json"]
        if self.config.no_session_persistence:
            command.append("--no-session-persistence")
        if model:
            command.extend(("--model", model))
        if self.config.tools is not None:
            command.append("--tools")
            command.extend(self.config.tools or ("",))
        command.extend(self.config.extra_args)
        if self.config.tools is not None:
            # ``--tools`` accepts a variadic value, so terminate its values
            # before appending the positional prompt.
            command.append("--")
        command.append(prompt)
        return command

    @staticmethod
    def parse_payload(
        payload: Mapping[str, Any],
        *,
        request_id: str,
        model: str | None,
        batch_index: int | None = None,
        output_limits: GateLimits | None = None,
        token_counter: TokenCounter | None = None,
        required_markers: Sequence[str] = (),
    ) -> ClaudeCliRunResult:
        """Parse one Claude JSON result; safe for recorded response fixtures."""

        if not isinstance(payload, Mapping):
            raise ClaudeCliError("Claude CLI JSON result must be an object")
        if payload.get("is_error") is True:
            raise ClaudeCliError("Claude CLI returned an error result")
        result = payload.get("result")
        if not isinstance(result, str):
            result = json.dumps(result, ensure_ascii=False, sort_keys=True) if result is not None else ""
        marker_values = tuple(required_markers)
        if any(not isinstance(marker, str) or not marker for marker in marker_values):
            raise ClaudeCliError("required_markers must contain non-empty strings")
        if token_counter is None:
            gated = format_subagent_output(
                result,
                limits=output_limits,
                required_markers=marker_values,
            )
        else:
            gated = format_subagent_output(
                result,
                limits=output_limits,
                token_counter=token_counter,
                required_markers=marker_values,
            )
        baseline_retained = sum(marker in result for marker in marker_values)
        retained = sum(marker in gated.text for marker in marker_values)
        baseline_success = all(marker in result for marker in marker_values) if marker_values else None
        task_success = all(marker in gated.text for marker in marker_values) if marker_values else None
        retention = retained / baseline_retained if baseline_retained else None
        usage = payload.get("usage")
        usage_mapping = usage if isinstance(usage, Mapping) else {}
        reported_model = payload.get("model")
        effective_model = reported_model if isinstance(reported_model, str) and reported_model else model
        if not isinstance(effective_model, str) or not effective_model:
            effective_model = "unspecified"
        return ClaudeCliRunResult(
            request_id=request_id,
            model=effective_model,
            text=gated.text,
            input_tokens=_usage_value(usage_mapping, "input_tokens", "prompt_tokens"),
            output_tokens=_usage_value(usage_mapping, "output_tokens", "completion_tokens"),
            cost_usd=_as_cost(payload.get("total_cost_usd", payload.get("cost_usd"))),
            batch_index=batch_index,
            baseline_task_success=baseline_success,
            task_success=task_success,
            important_information_retention=retention,
            required_markers=len(marker_values),
            baseline_retained_markers=baseline_retained if marker_values else None,
            retained_markers=retained if marker_values else None,
            output_gate_original_tokens=gated.original_tokens,
            output_gate_rendered_tokens=gated.rendered_tokens,
            output_gate_tokens_saved=gated.tokens_saved,
            output_gate_truncated=gated.truncated,
            output_gate_retained_categories=gated.retained_categories,
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
            raise ClaudeCliError("request_id must be a non-empty string")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ClaudeCliError("prompt must be a non-empty string")
        marker_values = tuple(required_markers)
        if any(not isinstance(marker, str) or not marker for marker in marker_values):
            raise ClaudeCliError("required_markers must contain non-empty strings")
        command = self._command(prompt=prompt, model=model)
        attempt = 0
        last_category: StderrCategory | None = None
        failure_categories: list[str] = []
        retry_wait_seconds = 0.0
        transient_failures = 0
        permanent_failures = 0
        while True:
            attempt += 1
            try:
                completed = subprocess.run(
                    command,
                    cwd=self.config.working_directory,
                    env=dict(os.environ),
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout_seconds,
                    check=False,
                )
            except FileNotFoundError as exc:
                diagnostic = StderrDiagnostic("process", False)
                failure_categories.append(diagnostic.category)
                permanent_failures += 1
                raise ClaudeCliError(
                    "Claude CLI executable was not found",
                    diagnostic=diagnostic,
                    attempts=attempt,
                    failure_categories=failure_categories,
                    retry_wait_seconds=retry_wait_seconds,
                    transient_failures=transient_failures,
                    permanent_failures=permanent_failures,
                ) from exc
            except subprocess.TimeoutExpired as exc:
                diagnostic = classify_stderr("", timed_out=True)
                failure_categories.append(diagnostic.category)
                transient_failures += 1
                if self._should_retry(diagnostic, attempt):
                    last_category = diagnostic.category
                    retry_wait_seconds += self._sleep(attempt)
                    continue
                raise ClaudeCliError(
                    f"Claude CLI failed: category={diagnostic.category}, attempts={attempt}",
                    diagnostic=diagnostic,
                    attempts=attempt,
                    failure_categories=failure_categories,
                    retry_wait_seconds=retry_wait_seconds,
                    transient_failures=transient_failures,
                    permanent_failures=permanent_failures,
                ) from exc
            except OSError as exc:
                diagnostic = StderrDiagnostic("process", False)
                failure_categories.append(diagnostic.category)
                permanent_failures += 1
                raise ClaudeCliError(
                    f"Claude CLI process failed ({type(exc).__name__})",
                    diagnostic=diagnostic,
                    attempts=attempt,
                    failure_categories=failure_categories,
                    retry_wait_seconds=retry_wait_seconds,
                    transient_failures=transient_failures,
                    permanent_failures=permanent_failures,
                ) from exc
            if completed.returncode != 0:
                diagnostic = classify_stderr(
                    completed.stderr,
                    returncode=completed.returncode,
                )
                failure_categories.append(diagnostic.category)
                if diagnostic.retryable:
                    transient_failures += 1
                else:
                    permanent_failures += 1
                if self._should_retry(diagnostic, attempt):
                    last_category = diagnostic.category
                    retry_wait_seconds += self._sleep(attempt)
                    continue
                raise ClaudeCliError(
                    f"Claude CLI failed: category={diagnostic.category}, "
                    f"exit_code={completed.returncode}, attempts={attempt}",
                    diagnostic=diagnostic,
                    attempts=attempt,
                    failure_categories=failure_categories,
                    retry_wait_seconds=retry_wait_seconds,
                    transient_failures=transient_failures,
                    permanent_failures=permanent_failures,
                )
            try:
                payload = json.loads(completed.stdout)
            except (json.JSONDecodeError, TypeError) as exc:
                diagnostic = StderrDiagnostic("invalid_json", True, completed.returncode)
                failure_categories.append(diagnostic.category)
                transient_failures += 1
                if self._should_retry(diagnostic, attempt):
                    last_category = diagnostic.category
                    retry_wait_seconds += self._sleep(attempt)
                    continue
                raise ClaudeCliError(
                    f"Claude CLI failed: category={diagnostic.category}, attempts={attempt}",
                    diagnostic=diagnostic,
                    attempts=attempt,
                    failure_categories=failure_categories,
                    retry_wait_seconds=retry_wait_seconds,
                    transient_failures=transient_failures,
                    permanent_failures=permanent_failures,
                ) from exc
            parsed = self.parse_payload(
                payload,
                request_id=request_id,
                model=model,
                batch_index=batch_index,
                output_limits=self.config.output_limits,
                token_counter=self.config.token_counter,
                required_markers=marker_values,
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

    def _should_retry(self, diagnostic: StderrDiagnostic, attempt: int) -> bool:
        return diagnostic.retryable and attempt <= self.config.max_retries

    def _sleep(self, attempt: int) -> float:
        delay = min(
            self.config.retry_initial_delay_seconds
            * (self.config.retry_backoff_factor ** (attempt - 1)),
            self.config.retry_max_delay_seconds,
        )
        sleeper = self.config.retry_sleep or time.sleep
        sleeper(delay)
        return delay


class ClaudeCodeCliAdapter:
    """Apply an :class:`AgentController` plan using Claude Code CLI calls."""

    def __init__(
        self,
        runner: ClaudeCliRunner | None = None,
        controller: AgentController | None = None,
        *,
        default_model: str | None = None,
        comparison_log: ComparisonLog | None = None,
        router: ModelRouter | None = None,
        routing_log: RoutingLog | None = None,
        routing_log_path: str | Path | None = None,
    ) -> None:
        if default_model is not None and (not isinstance(default_model, str) or not default_model.strip()):
            raise OrchestrationError("default_model must be a non-empty string when provided")
        self.runner = runner or ClaudeCliRunner()
        self.controller = controller or AgentController(root_model=default_model, router=router)
        self.default_model = default_model
        self.comparison_log = comparison_log or ComparisonLog()
        self.router = router or self.controller.router
        self.routing_log = routing_log or RoutingLog()
        if routing_log_path is not None and (
            not isinstance(routing_log_path, (str, Path)) or not str(routing_log_path).strip()
        ):
            raise OrchestrationError("routing_log_path must be a non-empty path when provided")
        configured_path = routing_log_path or os.environ.get("TOKENSPLIT_ROUTING_LOG")
        self.routing_log_path = Path(configured_path) if configured_path else None

    @staticmethod
    def _prompt(request: AgentRequest) -> str:
        return request.prompt.strip() or request.task_id

    def _model(self, model: str | None) -> str:
        effective = model or self.default_model
        if not effective:
            raise ClaudeCliError("no model was supplied or inherited for CLI invocation")
        return effective

    def dispatch(
        self,
        requests: Sequence[AgentRequest],
        *,
        parent_model: str | None = None,
        parent_depth: int = 0,
        quality_score: float | None = None,
        quality_passed: bool | None = None,
    ) -> ClaudeCliDispatchResult:
        plan = self.controller.plan(requests, parent_model=parent_model, parent_depth=parent_depth)
        request_map = {request.task_id: request for request in requests}
        runs: list[ClaudeCliRunResult] = []
        execution_diagnostics: list[ExecutionDiagnostic] = []
        timestamp = datetime.now(timezone.utc).isoformat()

        def invoke(**kwargs: Any) -> ClaudeCliRunResult:
            execution_id = str(kwargs.get("request_id", "cli-execution"))
            try:
                run = self.runner.run(**kwargs)
            except ClaudeCliError as exc:
                execution_diagnostics.append(exc.execution_diagnostic(execution_id))
                failed_measurement = self._measurement(
                    plan,
                    tuple(runs),
                    quality_score=quality_score,
                    quality_passed=quality_passed,
                    timestamp=timestamp,
                    additional_models=(
                        str(kwargs["model"])
                        if isinstance(kwargs.get("model"), str)
                        else None,
                    ),
                    execution_diagnostics=tuple(execution_diagnostics),
                )
                if failed_measurement is not None:
                    self.comparison_log.record(failed_measurement)
                raise
            execution_diagnostics.append(run.execution_diagnostic())
            return run

        for group in plan.integrated_groups:
            group_requests = [request_map[task_id] for task_id in group]
            prompt = "\n\n".join(
                f"Integrated task {request.task_id}:\n{self._prompt(request)}" for request in group_requests
            )
            model = self._model(plan.parent_model)
            runs.append(
                invoke(
                    request_id="integrated:" + "+".join(group),
                    prompt=prompt,
                    model=model,
                    required_markers=tuple(
                        marker
                        for request in group_requests
                        for marker in request.required_markers
                    ),
                )
            )

        for batch_index, batch in enumerate(plan.execution_batches):
            for task_id in batch:
                decision = next(decision for decision in plan.decisions if decision.task_id == task_id)
                request = request_map[task_id]
                runs.append(
                    invoke(
                        request_id=task_id,
                        prompt=self._prompt(request),
                        model=self._model(decision.model),
                        batch_index=batch_index,
                        required_markers=request.required_markers,
                    )
                )

        measurement = self._measurement(
            plan,
            tuple(runs),
            quality_score=quality_score,
            quality_passed=quality_passed,
            timestamp=timestamp,
            execution_diagnostics=tuple(execution_diagnostics),
        )
        if measurement is not None:
            self.comparison_log.record(measurement)
        self._record_routing_outcomes(
            plan,
            tuple(runs),
            request_map,
            quality_score=quality_score,
            quality_passed=quality_passed,
        )
        if self.routing_log_path is not None:
            self.routing_log.append_jsonl(
                self.routing_log_path,
                self.routing_log.records()[-len(runs):],
            )
        return ClaudeCliDispatchResult(plan=plan, runs=tuple(runs), measurement=measurement)

    def _record_routing_outcomes(
        self,
        plan: DispatchPlan,
        runs: tuple[ClaudeCliRunResult, ...],
        request_map: Mapping[str, AgentRequest],
        *,
        quality_score: float | None,
        quality_passed: bool | None,
    ) -> None:
        """Persist per-invocation model, quality, and cost telemetry."""

        decisions = {decision.task_id: decision for decision in plan.decisions}
        for run in runs:
            task_ids = tuple(
                run.request_id.removeprefix("integrated:").split("+")
                if run.request_id.startswith("integrated:")
                else (run.request_id,)
            )
            selected = [decisions[task_id] for task_id in task_ids if task_id in decisions]
            if selected:
                task_types = {decision.task_type for decision in selected}
                task_type = next(iter(task_types)) if len(task_types) == 1 else "integrated"
                selection_reason = (
                    selected[0].selection_reason if len(selected) == 1 else "integrated"
                )
                fallback_used = any(decision.fallback_used for decision in selected)
            else:
                request = request_map.get(run.request_id)
                task_type = request.task_type if request and request.task_type else "unknown"
                selection_reason = "explicit_model" if request and request.model else "legacy_inheritance"
                fallback_used = False
            self.routing_log.record_outcome(
                request_id=run.request_id,
                task_type=task_type,
                selected_model=run.model,
                selection_reason=selection_reason,
                fallback_used=fallback_used,
                input_tokens=run.input_tokens,
                output_tokens=run.output_tokens,
                cost_usd=run.cost_usd,
                quality_score=quality_score,
                quality_passed=quality_passed,
                task_success=run.task_success,
                required_markers=run.required_markers,
                retained_markers=run.retained_markers,
            )

    @staticmethod
    def _measurement(
        plan: DispatchPlan,
        runs: tuple[ClaudeCliRunResult, ...],
        *,
        quality_score: float | None,
        quality_passed: bool | None,
        timestamp: str | None = None,
        additional_models: Sequence[str | None] = (),
        execution_diagnostics: Sequence[ExecutionDiagnostic] = (),
    ) -> ComparisonMeasurement | None:
        if not runs and not execution_diagnostics:
            return None
        models = {run.model for run in runs}
        models.update(model for model in additional_models if model)
        input_values = tuple(run.input_tokens for run in runs)
        output_values = tuple(run.output_tokens for run in runs)
        cost_values = tuple(run.cost_usd for run in runs)
        marked_runs = tuple(run for run in runs if run.task_success is not None)
        baseline_marked_runs = tuple(run for run in runs if run.baseline_task_success is not None)
        baseline_task_success = (
            all(run.baseline_task_success for run in baseline_marked_runs)
            if baseline_marked_runs
            else None
        )
        task_success = all(run.task_success for run in marked_runs) if marked_runs else None
        total_required_markers = sum(run.required_markers for run in runs)
        total_baseline_retained = sum(run.baseline_retained_markers or 0 for run in runs)
        total_retained = sum(run.retained_markers or 0 for run in runs)
        retention_values = tuple(
            run.important_information_retention
            for run in runs
            if run.important_information_retention is not None
        )
        execution_ids = tuple(
            diagnostic.execution_id for diagnostic in execution_diagnostics
        ) or tuple(run.request_id for run in runs)
        return ComparisonMeasurement(
            run_id="cli-dispatch:" + ",".join(execution_ids),
            mode="split" if plan.requested_child_agent_count else "single",
            model=(next(iter(models)) if len(models) == 1 else "mixed")
            if models
            else (plan.parent_model or "unknown"),
            agent_count=max(1, len(runs)),
            tool_calls=sum(
                decision.estimated_tool_calls
                for decision in plan.decisions
                if decision.action in {"integrate", "launch", "serialize"}
            ),
            input_tokens=sum(input_values) if all(value is not None for value in input_values) else None,
            output_tokens=sum(output_values) if all(value is not None for value in output_values) else None,
            cost_usd=sum(cost_values, Decimal(0)) if cost_values and all(value is not None for value in cost_values) else None,
            quality_score=quality_score,
            quality_passed=quality_passed,
            baseline_task_success=baseline_task_success,
            task_success=task_success,
            important_information_retention=(
                total_retained / total_baseline_retained
                if total_baseline_retained
                else (sum(retention_values) / len(retention_values) if retention_values else None)
            ),
            required_markers=total_required_markers,
            baseline_retained_markers=total_baseline_retained if marked_runs else None,
            retained_markers=total_retained if marked_runs else None,
            output_gate_original_tokens=sum(run.output_gate_original_tokens for run in runs),
            output_gate_rendered_tokens=sum(run.output_gate_rendered_tokens for run in runs),
            output_gate_tokens_saved=sum(run.output_gate_tokens_saved for run in runs),
            timestamp=timestamp,
            execution_diagnostics=tuple(execution_diagnostics),
        )


__all__ = [
    "ClaudeCliConfig",
    "ClaudeCliError",
    "ClaudeCliRunResult",
    "ClaudeCliRunner",
    "ClaudeCliDispatchResult",
    "ClaudeCodeCliAdapter",
    "StderrDiagnostic",
    "classify_stderr",
]
