"""Bounded sub-agent planning and single-vs-split measurement.

The module is deliberately provider agnostic.  It does not start an agent or
call a model; it returns an auditable dispatch plan that an adapter can apply.
That keeps safety limits testable even when the actual agent SDK is not
available.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Literal, Sequence

from .routing import ModelRouter, RouteDecision


class OrchestrationError(ValueError):
    """Raised when a dispatch request or measurement is invalid."""


OverflowStrategy = Literal["reject", "serialize"]
DispatchAction = Literal["integrate", "launch", "serialize", "reject"]
RunMode = Literal["single", "split"]
ExecutionOutcome = Literal["success", "failure"]


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise OrchestrationError(f"{name} must be a positive integer")
    return value


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OrchestrationError(f"{name} must be a non-negative integer")
    return value


def _non_negative_float(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise OrchestrationError(f"{name} must be a non-negative number")
    return float(value)


@dataclass(frozen=True)
class AgentControlPolicy:
    """Limits for one parent dispatch operation.

    ``max_depth=1`` allows a root agent to launch children, while rejecting a
    child that attempts to launch another child.  ``serialize`` preserves work
    beyond ``max_subagents`` as later batches rather than launching it at once.
    """

    max_subagents: int = 3
    max_depth: int = 1
    max_tool_calls_per_agent: int = 8
    min_independence: float = 0.5
    overflow_strategy: OverflowStrategy = "serialize"

    def __post_init__(self) -> None:
        _positive_int(self.max_subagents, "max_subagents")
        _non_negative_int(self.max_depth, "max_depth")
        _positive_int(self.max_tool_calls_per_agent, "max_tool_calls_per_agent")
        if (
            isinstance(self.min_independence, bool)
            or not isinstance(self.min_independence, (int, float))
            or not 0.0 <= self.min_independence <= 1.0
        ):
            raise OrchestrationError("min_independence must be a number between 0 and 1")
        if self.overflow_strategy not in {"reject", "serialize"}:
            raise OrchestrationError("overflow_strategy must be 'reject' or 'serialize'")


@dataclass(frozen=True)
class AgentRequest:
    """A provider-neutral request for one possible child agent."""

    task_id: str
    independence: float = 1.0
    estimated_tool_calls: int = 0
    model: str | None = None
    depth: int | None = None
    parent_id: str | None = None
    prompt: str = ""
    required_markers: tuple[str, ...] = ()
    task_type: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise OrchestrationError("task_id must be a non-empty string")
        if (
            isinstance(self.independence, bool)
            or not isinstance(self.independence, (int, float))
            or not 0.0 <= self.independence <= 1.0
        ):
            raise OrchestrationError("independence must be a number between 0 and 1")
        _non_negative_int(self.estimated_tool_calls, "estimated_tool_calls")
        if self.model is not None and (
            not isinstance(self.model, str) or not self.model.strip()
        ):
            raise OrchestrationError("model must be a non-empty string when provided")
        if self.depth is not None:
            _non_negative_int(self.depth, "depth")
        if self.parent_id is not None and (
            not isinstance(self.parent_id, str) or not self.parent_id.strip()
        ):
            raise OrchestrationError("parent_id must be a non-empty string when provided")
        if not isinstance(self.prompt, str):
            raise OrchestrationError("prompt must be a string")
        if self.task_type is not None and (
            not isinstance(self.task_type, str) or not self.task_type.strip()
        ):
            raise OrchestrationError("task_type must be a non-empty string when provided")
        if isinstance(self.required_markers, (str, bytes, bytearray)) or not isinstance(
            self.required_markers, Sequence
        ):
            raise OrchestrationError("required_markers must be a sequence of strings")
        if any(not isinstance(marker, str) or not marker for marker in self.required_markers):
            raise OrchestrationError("required_markers must contain non-empty strings")
        object.__setattr__(self, "required_markers", tuple(self.required_markers))


@dataclass
class AgentBudget:
    """Runtime counter for enforcing the per-agent tool-call limit."""

    max_tool_calls: int
    tool_calls: int = 0

    def __post_init__(self) -> None:
        _positive_int(self.max_tool_calls, "max_tool_calls")
        _non_negative_int(self.tool_calls, "tool_calls")
        if self.tool_calls > self.max_tool_calls:
            raise OrchestrationError("tool_calls cannot exceed max_tool_calls")

    @property
    def remaining(self) -> int:
        return self.max_tool_calls - self.tool_calls

    def record_tool_call(self, count: int = 1) -> int:
        """Record calls and reject the operation before it exceeds the cap."""

        _positive_int(count, "count")
        if self.tool_calls + count > self.max_tool_calls:
            raise OrchestrationError("max_tool_calls_exceeded")
        self.tool_calls += count
        return self.tool_calls


@dataclass
class AgentInvocation:
    """Minimal runtime handle for an admitted agent."""

    agent_id: str
    model: str | None
    depth: int
    budget: AgentBudget

    def __post_init__(self) -> None:
        if not isinstance(self.agent_id, str) or not self.agent_id.strip():
            raise OrchestrationError("agent_id must be a non-empty string")
        _non_negative_int(self.depth, "depth")
        if self.model is not None and (not isinstance(self.model, str) or not self.model.strip()):
            raise OrchestrationError("model must be a non-empty string when provided")

    def record_tool_call(self, count: int = 1) -> int:
        return self.budget.record_tool_call(count)


@dataclass(frozen=True)
class ExecutionDiagnostic:
    """Privacy-safe retry and failure data for one external execution."""

    execution_id: str
    attempts: int
    retry_count: int
    failure_categories: tuple[str, ...] = ()
    retry_wait_seconds: float = 0.0
    transient_failures: int = 0
    permanent_failures: int = 0
    outcome: ExecutionOutcome = "success"

    def __post_init__(self) -> None:
        if not isinstance(self.execution_id, str) or not self.execution_id.strip():
            raise OrchestrationError("execution_id must be a non-empty string")
        _positive_int(self.attempts, "attempts")
        _non_negative_int(self.retry_count, "retry_count")
        if self.retry_count > self.attempts - 1:
            raise OrchestrationError("retry_count cannot exceed attempts - 1")
        if isinstance(self.failure_categories, (str, bytes, bytearray)):
            raise OrchestrationError("failure_categories must be a sequence of strings")
        if any(not isinstance(category, str) or not category for category in self.failure_categories):
            raise OrchestrationError("failure_categories must contain non-empty strings")
        object.__setattr__(self, "failure_categories", tuple(self.failure_categories))
        _non_negative_float(self.retry_wait_seconds, "retry_wait_seconds")
        _non_negative_int(self.transient_failures, "transient_failures")
        _non_negative_int(self.permanent_failures, "permanent_failures")
        if self.outcome not in {"success", "failure"}:
            raise OrchestrationError("outcome must be 'success' or 'failure'")


@dataclass(frozen=True)
class DispatchDecision:
    """Decision for a request, suitable for an audit log."""

    task_id: str
    action: DispatchAction
    model: str | None
    depth: int
    estimated_tool_calls: int
    batch_index: int | None
    reason: str
    task_type: str = "unknown"
    selection_reason: str = "legacy_inheritance"
    fallback_used: bool = False


@dataclass(frozen=True)
class DispatchPlan:
    """The result of applying :class:`AgentControlPolicy` to requests."""

    decisions: tuple[DispatchDecision, ...]
    execution_batches: tuple[tuple[str, ...], ...]
    integrated_groups: tuple[tuple[str, ...], ...]
    parent_model: str | None
    parent_depth: int

    @property
    def launched_task_ids(self) -> tuple[str, ...]:
        return tuple(d.task_id for d in self.decisions if d.action == "launch")

    @property
    def serialized_task_ids(self) -> tuple[str, ...]:
        return tuple(d.task_id for d in self.decisions if d.action == "serialize")

    @property
    def rejected_task_ids(self) -> tuple[str, ...]:
        return tuple(d.task_id for d in self.decisions if d.action == "reject")

    @property
    def integrated_task_ids(self) -> tuple[str, ...]:
        return tuple(d.task_id for d in self.decisions if d.action == "integrate")

    @property
    def child_agent_count(self) -> int:
        """Maximum number of child agents active in any one batch."""

        return max((len(batch) for batch in self.execution_batches), default=0)

    @property
    def requested_child_agent_count(self) -> int:
        """Number of independent child tasks after integration/rejection."""

        return len(self.launched_task_ids) + len(self.serialized_task_ids)

    def audit_records(self) -> tuple[dict[str, Any], ...]:
        """Return non-sensitive decisions without prompts or source content."""

        return tuple(asdict(decision) for decision in self.decisions)


class AgentController:
    """Plan bounded child-agent dispatches and optionally apply model routing."""

    def __init__(
        self,
        policy: AgentControlPolicy | None = None,
        *,
        root_model: str | None = None,
        router: ModelRouter | None = None,
    ) -> None:
        if root_model is not None and (not isinstance(root_model, str) or not root_model.strip()):
            raise OrchestrationError("root_model must be a non-empty string when provided")
        self.policy = policy or AgentControlPolicy()
        self.root_model = root_model
        if router is not None and not isinstance(router, ModelRouter):
            raise OrchestrationError("router must be a ModelRouter or None")
        self.router = router

    def _route(
        self,
        request: AgentRequest,
        *,
        parent_model: str | None,
    ) -> RouteDecision | None:
        if self.router is None:
            return None
        try:
            return self.router.route(
                request_id=request.task_id,
                prompt=request.prompt,
                task_type=request.task_type,
                requested_model=request.model,
                parent_model=parent_model,
            )
        except ValueError as exc:
            raise OrchestrationError(str(exc)) from exc

    def plan(
        self,
        requests: Sequence[AgentRequest],
        *,
        parent_model: str | None = None,
        parent_depth: int = 0,
    ) -> DispatchPlan:
        """Return a bounded plan without starting any external work.

        Requests below ``min_independence`` are returned as one ``integrate``
        group.  They remain in the parent context and therefore consume no
        child-agent slot.  Independent requests are launched immediately up to
        the cap; remaining requests are either rejected or assigned to
        serialized batches according to ``overflow_strategy``.
        """

        if isinstance(requests, (str, bytes, bytearray)):
            raise OrchestrationError("requests must be a sequence of AgentRequest")
        if not isinstance(requests, Sequence):
            raise OrchestrationError("requests must be a sequence of AgentRequest")
        if isinstance(parent_depth, bool) or not isinstance(parent_depth, int) or parent_depth < 0:
            raise OrchestrationError("parent_depth must be a non-negative integer")
        effective_parent_model = parent_model if parent_model is not None else self.root_model
        if effective_parent_model is not None and (
            not isinstance(effective_parent_model, str) or not effective_parent_model.strip()
        ):
            raise OrchestrationError("parent_model must be a non-empty string when provided")

        normalized = tuple(requests)
        if len({request.task_id for request in normalized if isinstance(request, AgentRequest)}) != len(normalized):
            raise OrchestrationError("task_id values must be unique within one dispatch")
        for request in normalized:
            if not isinstance(request, AgentRequest):
                raise OrchestrationError("requests must contain only AgentRequest values")

        decisions: list[DispatchDecision] = []
        routes = {
            request.task_id: self._route(request, parent_model=effective_parent_model)
            for request in normalized
        }
        independent: list[AgentRequest] = []
        low_independence: list[AgentRequest] = []
        for request in normalized:
            depth = request.depth if request.depth is not None else parent_depth + 1
            route = routes[request.task_id]
            model = route.selected_model if route is not None else (request.model or effective_parent_model)
            if depth > self.policy.max_depth:
                decisions.append(
                    DispatchDecision(
                        request.task_id,
                        "reject",
                        model,
                        depth,
                        request.estimated_tool_calls,
                        None,
                        "max_depth_exceeded",
                        task_type=route.task_type if route is not None else (request.task_type or "unknown"),
                        selection_reason=route.selection_reason if route is not None else (
                            "explicit_model" if request.model else "legacy_inheritance"
                        ),
                        fallback_used=route.fallback_used if route is not None else False,
                    )
                )
            elif request.estimated_tool_calls > self.policy.max_tool_calls_per_agent:
                decisions.append(
                    DispatchDecision(
                        request.task_id,
                        "reject",
                        model,
                        depth,
                        request.estimated_tool_calls,
                        None,
                        "max_tool_calls_exceeded",
                        task_type=route.task_type if route is not None else (request.task_type or "unknown"),
                        selection_reason=route.selection_reason if route is not None else (
                            "explicit_model" if request.model else "legacy_inheritance"
                        ),
                        fallback_used=route.fallback_used if route is not None else False,
                    )
                )
            elif request.independence < self.policy.min_independence:
                low_independence.append(request)
            else:
                independent.append(request)

        if low_independence:
            grouped_integrated: list[list[AgentRequest]] = []
            for request in low_independence:
                route = routes[request.task_id]
                selected_model = route.selected_model if route is not None else (
                    request.model or effective_parent_model
                )
                if self.router is not None and (
                    not grouped_integrated
                    or (
                        routes[grouped_integrated[-1][0].task_id] is not None
                        and routes[grouped_integrated[-1][0].task_id].selected_model != selected_model
                    )
                ):
                    grouped_integrated.append([])
                elif not grouped_integrated:
                    grouped_integrated.append([])
                grouped_integrated[-1].append(request)
            integrated_groups = tuple(
                tuple(request.task_id for request in group) for group in grouped_integrated
            )
            for request in low_independence:
                route = routes[request.task_id]
                decisions.append(
                    DispatchDecision(
                        request.task_id,
                        "integrate",
                        route.selected_model if route is not None else (request.model or effective_parent_model),
                        parent_depth,
                        request.estimated_tool_calls,
                        None,
                        "independence_below_threshold",
                        task_type=route.task_type if route is not None else (request.task_type or "unknown"),
                        selection_reason=route.selection_reason if route is not None else (
                            "explicit_model" if request.model else "legacy_inheritance"
                        ),
                        fallback_used=route.fallback_used if route is not None else False,
                    )
                )
        else:
            integrated_groups = ()

        batches: list[list[AgentRequest]] = []
        if self.policy.overflow_strategy == "serialize":
            for offset in range(0, len(independent), self.policy.max_subagents):
                batches.append(independent[offset : offset + self.policy.max_subagents])
        else:
            batches = [independent[: self.policy.max_subagents]] if independent else []

        allowed_ids = {request.task_id for batch in batches for request in batch}
        for batch_index, batch in enumerate(batches):
            action: DispatchAction = "launch" if batch_index == 0 else "serialize"
            for request in batch:
                depth = request.depth if request.depth is not None else parent_depth + 1
                route = routes[request.task_id]
                model = route.selected_model if route is not None else (request.model or effective_parent_model)
                decisions.append(
                    DispatchDecision(
                        request.task_id,
                        action,
                        model,
                        depth,
                        request.estimated_tool_calls,
                        batch_index,
                        "within_limit" if action == "launch" else "max_subagents_serialized",
                        task_type=route.task_type if route is not None else (request.task_type or "unknown"),
                        selection_reason=route.selection_reason if route is not None else (
                            "explicit_model" if request.model else "legacy_inheritance"
                        ),
                        fallback_used=route.fallback_used if route is not None else False,
                    )
                )

        for request in independent:
            if request.task_id in allowed_ids:
                continue
            depth = request.depth if request.depth is not None else parent_depth + 1
            route = routes[request.task_id]
            model = route.selected_model if route is not None else (request.model or effective_parent_model)
            decisions.append(
                DispatchDecision(
                    request.task_id,
                    "reject",
                    model,
                    depth,
                    request.estimated_tool_calls,
                    None,
                    "max_subagents_exceeded",
                    task_type=route.task_type if route is not None else (request.task_type or "unknown"),
                    selection_reason=route.selection_reason if route is not None else (
                        "explicit_model" if request.model else "legacy_inheritance"
                    ),
                    fallback_used=route.fallback_used if route is not None else False,
                )
            )

        order = {request.task_id: index for index, request in enumerate(normalized)}
        decisions.sort(key=lambda decision: order[decision.task_id])
        return DispatchPlan(
            decisions=tuple(decisions),
            execution_batches=tuple(tuple(request.task_id for request in batch) for batch in batches),
            integrated_groups=integrated_groups,
            parent_model=effective_parent_model,
            parent_depth=parent_depth,
        )


def _decimal(value: Any, name: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise OrchestrationError(f"{name} must be a finite decimal") from exc
    if not result.is_finite() or result < 0:
        raise OrchestrationError(f"{name} must be a finite non-negative decimal")
    return result


@dataclass(frozen=True)
class ComparisonMeasurement:
    """One run's cost and quality evidence.

    Token and quality fields are optional to preserve the distinction between
    measured values and unavailable provider data.  Missing values are never
    silently converted to zero by :class:`ComparisonLog`.
    """

    run_id: str
    mode: RunMode
    model: str
    agent_count: int
    tool_calls: int
    fixed_context_tokens_per_agent: int | None = None
    work_tokens: int | None = None
    coordination_tokens: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    input_usd_per_million: Decimal | None = None
    output_usd_per_million: Decimal | None = None
    quality_score: float | None = None
    quality_passed: bool | None = None
    cost_usd: Decimal | None = None
    baseline_task_success: bool | None = None
    task_success: bool | None = None
    important_information_retention: float | None = None
    required_markers: int = 0
    baseline_retained_markers: int | None = None
    retained_markers: int | None = None
    output_gate_original_tokens: int | None = None
    output_gate_rendered_tokens: int | None = None
    output_gate_tokens_saved: int | None = None
    timestamp: str | None = None
    execution_diagnostics: tuple[ExecutionDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise OrchestrationError("run_id must be a non-empty string")
        if self.mode not in {"single", "split"}:
            raise OrchestrationError("mode must be 'single' or 'split'")
        if not isinstance(self.model, str) or not self.model.strip():
            raise OrchestrationError("model must be a non-empty string")
        _positive_int(self.agent_count, "agent_count")
        _non_negative_int(self.tool_calls, "tool_calls")
        for name in ("fixed_context_tokens_per_agent", "work_tokens", "coordination_tokens", "input_tokens", "output_tokens"):
            value = getattr(self, name)
            if value is not None:
                _non_negative_int(value, name)
        for name in ("input_usd_per_million", "output_usd_per_million"):
            value = getattr(self, name)
            if value is not None:
                _decimal(value, name)
        if self.cost_usd is not None:
            _decimal(self.cost_usd, "cost_usd")
        for name in ("baseline_task_success", "task_success", "quality_passed"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise OrchestrationError(f"{name} must be a boolean or None")
        if self.quality_score is not None and (
            isinstance(self.quality_score, bool)
            or not isinstance(self.quality_score, (int, float))
            or not 0.0 <= self.quality_score <= 1.0
        ):
            raise OrchestrationError("quality_score must be a number between 0 and 1")
        if self.important_information_retention is not None and (
            isinstance(self.important_information_retention, bool)
            or not isinstance(self.important_information_retention, (int, float))
            or not 0.0 <= self.important_information_retention <= 1.0
        ):
            raise OrchestrationError(
                "important_information_retention must be a number between 0 and 1"
            )
        _non_negative_int(self.required_markers, "required_markers")
        for name in ("baseline_retained_markers", "retained_markers"):
            value = getattr(self, name)
            if value is not None:
                _non_negative_int(value, name)
        for name in (
            "output_gate_original_tokens",
            "output_gate_rendered_tokens",
            "output_gate_tokens_saved",
        ):
            value = getattr(self, name)
            if value is not None:
                _non_negative_int(value, name)
        if self.timestamp is not None and (
            not isinstance(self.timestamp, str) or not self.timestamp.strip()
        ):
            raise OrchestrationError("timestamp must be a non-empty string or None")
        if isinstance(self.execution_diagnostics, (str, bytes, bytearray)):
            raise OrchestrationError("execution_diagnostics must be a sequence")
        if any(not isinstance(item, ExecutionDiagnostic) for item in self.execution_diagnostics):
            raise OrchestrationError(
                "execution_diagnostics must contain ExecutionDiagnostic values"
            )
        object.__setattr__(self, "execution_diagnostics", tuple(self.execution_diagnostics))

    @property
    def estimated_input_tokens(self) -> int | None:
        if self.input_tokens is not None:
            return self.input_tokens
        if self.fixed_context_tokens_per_agent is None or self.work_tokens is None or self.coordination_tokens is None:
            return None
        return (
            self.agent_count * self.fixed_context_tokens_per_agent
            + self.work_tokens
            + self.coordination_tokens
        )

    @property
    def total_tokens(self) -> int | None:
        input_tokens = self.estimated_input_tokens
        if input_tokens is None or self.output_tokens is None:
            return None
        return input_tokens + self.output_tokens

    @property
    def estimated_cost_usd(self) -> Decimal | None:
        if self.cost_usd is not None:
            return _decimal(self.cost_usd, "cost_usd")
        input_tokens = self.estimated_input_tokens
        if input_tokens is None or self.output_tokens is None:
            return None
        if self.input_usd_per_million is None or self.output_usd_per_million is None:
            return None
        return (
            Decimal(input_tokens) * _decimal(self.input_usd_per_million, "input_usd_per_million") / Decimal(1_000_000)
            + Decimal(self.output_tokens) * _decimal(self.output_usd_per_million, "output_usd_per_million") / Decimal(1_000_000)
        )


@dataclass(frozen=True)
class ComparisonSummary:
    single_runs: int
    split_runs: int
    single_tokens: int | None
    split_tokens: int | None
    single_cost_usd: Decimal | None
    split_cost_usd: Decimal | None
    cost_savings_ratio: float | None
    single_quality: float | None
    split_quality: float | None
    quality_delta: float | None
    single_task_success_rate: float | None
    split_task_success_rate: float | None
    single_information_retention: float | None
    split_information_retention: float | None
    single_output_gate_tokens_saved: int | None
    split_output_gate_tokens_saved: int | None
    single_retry_count: int
    split_retry_count: int
    single_retry_wait_seconds: float
    split_retry_wait_seconds: float
    single_transient_failures: int
    split_transient_failures: int
    single_permanent_failures: int
    split_permanent_failures: int
    single_failure_categories: tuple[str, ...]
    split_failure_categories: tuple[str, ...]
    recommendation: Literal["single", "split", "insufficient_data"]


class ComparisonLog:
    """Collect privacy-safe single-vs-split measurements and compare them."""

    def __init__(self) -> None:
        self._records: list[ComparisonMeasurement] = []

    def record(self, measurement: ComparisonMeasurement) -> None:
        if not isinstance(measurement, ComparisonMeasurement):
            raise OrchestrationError("measurement must be a ComparisonMeasurement")
        self._records.append(measurement)

    def records(self) -> tuple[ComparisonMeasurement, ...]:
        return tuple(self._records)

    @staticmethod
    def _complete_sum(values: Sequence[int | None]) -> int | None:
        if not values or any(value is None for value in values):
            return None
        return sum(value for value in values if value is not None)

    @staticmethod
    def _complete_average(values: Sequence[float | None]) -> float | None:
        if not values or any(value is None for value in values):
            return None
        return sum(value for value in values if value is not None) / len(values)

    @staticmethod
    def _quality_values(records: Sequence[ComparisonMeasurement]) -> tuple[float | None, ...]:
        scores = tuple(record.quality_score for record in records)
        if scores and all(value is not None for value in scores):
            return scores
        passed = tuple(
            None if record.quality_passed is None else (1.0 if record.quality_passed else 0.0)
            for record in records
        )
        return passed

    @staticmethod
    def _complete_rate(
        records: Sequence[ComparisonMeasurement],
        attribute: str,
    ) -> float | None:
        values = tuple(getattr(record, attribute) for record in records)
        if not values or any(value is None for value in values):
            return None
        return sum(bool(value) for value in values) / len(values)

    @staticmethod
    def _retention_average(
        records: Sequence[ComparisonMeasurement],
    ) -> float | None:
        values = tuple(record.important_information_retention for record in records)
        if not values or any(value is None for value in values):
            return None
        return sum(value for value in values if value is not None) / len(values)

    @staticmethod
    def _execution_summary(
        records: Sequence[ComparisonMeasurement],
    ) -> tuple[int, float, int, int, tuple[str, ...]]:
        diagnostics = tuple(
            diagnostic
            for record in records
            for diagnostic in record.execution_diagnostics
        )
        categories: list[str] = []
        for diagnostic in diagnostics:
            for category in diagnostic.failure_categories:
                if category not in categories:
                    categories.append(category)
        return (
            sum(diagnostic.retry_count for diagnostic in diagnostics),
            sum(diagnostic.retry_wait_seconds for diagnostic in diagnostics),
            sum(diagnostic.transient_failures for diagnostic in diagnostics),
            sum(diagnostic.permanent_failures for diagnostic in diagnostics),
            tuple(categories),
        )

    def summary(self) -> ComparisonSummary:
        single = tuple(record for record in self._records if record.mode == "single")
        split = tuple(record for record in self._records if record.mode == "split")
        single_tokens = self._complete_sum(tuple(record.total_tokens for record in single))
        split_tokens = self._complete_sum(tuple(record.total_tokens for record in split))
        single_costs = tuple(record.estimated_cost_usd for record in single)
        split_costs = tuple(record.estimated_cost_usd for record in split)
        single_cost = sum(single_costs, Decimal(0)) if single_costs and all(value is not None for value in single_costs) else None
        split_cost = sum(split_costs, Decimal(0)) if split_costs and all(value is not None for value in split_costs) else None
        if single_cost is not None and split_cost is not None and single_cost > 0:
            savings = float((single_cost - split_cost) / single_cost)
        else:
            savings = None
        single_quality = self._complete_average(self._quality_values(single))
        split_quality = self._complete_average(self._quality_values(split))
        quality_delta = (
            split_quality - single_quality
            if single_quality is not None and split_quality is not None
            else None
        )
        single_task_success_rate = self._complete_rate(single, "task_success")
        split_task_success_rate = self._complete_rate(split, "task_success")
        single_information_retention = self._retention_average(single)
        split_information_retention = self._retention_average(split)
        single_output_gate_tokens_saved = self._complete_sum(
            tuple(record.output_gate_tokens_saved for record in single)
        )
        split_output_gate_tokens_saved = self._complete_sum(
            tuple(record.output_gate_tokens_saved for record in split)
        )
        (
            single_retry_count,
            single_retry_wait_seconds,
            single_transient_failures,
            single_permanent_failures,
            single_failure_categories,
        ) = self._execution_summary(single)
        (
            split_retry_count,
            split_retry_wait_seconds,
            split_transient_failures,
            split_permanent_failures,
            split_failure_categories,
        ) = self._execution_summary(split)
        if savings is None or quality_delta is None:
            recommendation: Literal["single", "split", "insufficient_data"] = "insufficient_data"
        elif split_quality >= single_quality and split_cost <= single_cost:
            recommendation = "split"
        else:
            recommendation = "single"
        return ComparisonSummary(
            single_runs=len(single),
            split_runs=len(split),
            single_tokens=single_tokens,
            split_tokens=split_tokens,
            single_cost_usd=single_cost,
            split_cost_usd=split_cost,
            cost_savings_ratio=savings,
            single_quality=single_quality,
            split_quality=split_quality,
            quality_delta=quality_delta,
            single_task_success_rate=single_task_success_rate,
            split_task_success_rate=split_task_success_rate,
            single_information_retention=single_information_retention,
            split_information_retention=split_information_retention,
            single_output_gate_tokens_saved=single_output_gate_tokens_saved,
            split_output_gate_tokens_saved=split_output_gate_tokens_saved,
            single_retry_count=single_retry_count,
            split_retry_count=split_retry_count,
            single_retry_wait_seconds=single_retry_wait_seconds,
            split_retry_wait_seconds=split_retry_wait_seconds,
            single_transient_failures=single_transient_failures,
            split_transient_failures=split_transient_failures,
            single_permanent_failures=single_permanent_failures,
            split_permanent_failures=split_permanent_failures,
            single_failure_categories=single_failure_categories,
            split_failure_categories=split_failure_categories,
            recommendation=recommendation,
        )

    def render_jsonl(self) -> str:
        """Serialize measurements only; no prompts, source paths, or secrets."""

        lines: list[str] = []
        for record in self._records:
            payload = asdict(record)
            for key in (
                "input_usd_per_million",
                "output_usd_per_million",
                "cost_usd",
            ):
                if payload[key] is not None:
                    payload[key] = str(payload[key])
            lines.append(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return "\n".join(lines)

    @classmethod
    def from_jsonl(cls, text: str) -> "ComparisonLog":
        """Load privacy-safe operational measurements emitted by ``render_jsonl``."""

        if not isinstance(text, str):
            raise OrchestrationError("comparison JSONL must be text")
        log = cls()
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise OrchestrationError(
                    f"invalid comparison JSONL at line {line_number}"
                ) from exc
            if not isinstance(payload, dict):
                raise OrchestrationError(
                    f"comparison JSONL line {line_number} must be an object"
                )
            for key in (
                "input_usd_per_million",
                "output_usd_per_million",
                "cost_usd",
            ):
                if payload.get(key) is not None:
                    try:
                        payload[key] = Decimal(str(payload[key]))
                    except (InvalidOperation, TypeError, ValueError) as exc:
                        raise OrchestrationError(
                            f"invalid decimal field {key} at line {line_number}"
                        ) from exc
            raw_diagnostics = payload.get("execution_diagnostics", ())
            if raw_diagnostics is None:
                raw_diagnostics = ()
            if isinstance(raw_diagnostics, (str, bytes, bytearray)) or not isinstance(
                raw_diagnostics, Sequence
            ):
                raise OrchestrationError(
                    f"invalid execution diagnostics at line {line_number}"
                )
            try:
                payload["execution_diagnostics"] = tuple(
                    item if isinstance(item, ExecutionDiagnostic) else ExecutionDiagnostic(**item)
                    for item in raw_diagnostics
                )
            except (TypeError, OrchestrationError) as exc:
                raise OrchestrationError(
                    f"invalid execution diagnostics at line {line_number}"
                ) from exc
            try:
                log.record(ComparisonMeasurement(**payload))
            except TypeError as exc:
                raise OrchestrationError(
                    f"invalid comparison fields at line {line_number}"
                ) from exc
        return log


__all__ = [
    "AgentControlPolicy",
    "AgentController",
    "AgentBudget",
    "AgentInvocation",
    "AgentRequest",
    "ComparisonLog",
    "ComparisonMeasurement",
    "ComparisonSummary",
    "DispatchDecision",
    "DispatchPlan",
    "ExecutionDiagnostic",
    "OrchestrationError",
]
