"""Operational gate for orchestration failure-period reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence


THRESHOLD_EXIT_CODE = 10


def _threshold(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


@dataclass(frozen=True)
class OrchestrationFailurePolicy:
    """Thresholds applied independently to each period/model row."""

    minimum_executions: int = 1
    max_permanent_failure_rate: float | None = None
    max_retry_wait_seconds: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.minimum_executions, bool) or not isinstance(self.minimum_executions, int):
            raise ValueError("minimum_executions must be a positive integer")
        if self.minimum_executions < 1:
            raise ValueError("minimum_executions must be a positive integer")
        _threshold(self.max_permanent_failure_rate, "max_permanent_failure_rate")
        _threshold(self.max_retry_wait_seconds, "max_retry_wait_seconds")


def _row_with_rate(row: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(row)
    execution_count = int(result.get("execution_count", 0))
    permanent_failures = int(result.get("permanent_failures", 0))
    result["permanent_failure_rate"] = (
        permanent_failures / execution_count if execution_count else None
    )
    return result


@dataclass(frozen=True)
class OrchestrationGateResult:
    """A quiet report or a threshold-breach notification event."""

    report: dict[str, Any]
    policy: OrchestrationFailurePolicy
    alerts: tuple[dict[str, Any], ...]

    @property
    def notify(self) -> bool:
        return bool(self.alerts)

    @property
    def event(self) -> str | None:
        return "orchestration_failure_threshold_exceeded" if self.notify else None

    @property
    def exit_code(self) -> int:
        return THRESHOLD_EXIT_CODE if self.notify else 0

    def as_mapping(self) -> dict[str, Any]:
        return {
            "notify": self.notify,
            "event": self.event,
            "exit_code": self.exit_code,
            "policy": asdict(self.policy),
            "alerts": list(self.alerts),
            "report": self.report,
        }

    def notification_payload(self) -> dict[str, Any] | None:
        if not self.notify:
            return None
        return {
            "event": self.event,
            "message": "Orchestration failure thresholds were exceeded.",
            "policy": asdict(self.policy),
            "alerts": list(self.alerts),
        }

    def render_markdown(self) -> str:
        lines = ["# Orchestration failure report", ""]
        if not self.alerts:
            lines.append("No orchestration failure thresholds were exceeded.")
            return "\n".join(lines) + "\n"
        lines.extend(
            [
                "Thresholds exceeded:",
                "",
                "| Period | Model | Permanent failure rate | Retry wait (s) | Violations |",
                "| --- | --- | ---: | ---: | --- |",
            ]
        )
        for alert in self.alerts:
            lines.append(
                "| {period} | {model} | {rate:.2%} | {wait:.3f} | {violations} |".format(
                    period=alert["period"],
                    model=alert["model"],
                    rate=float(alert["permanent_failure_rate"]),
                    wait=float(alert["retry_wait_seconds"]),
                    violations=", ".join(alert["violations"]),
                )
            )
        return "\n".join(lines) + "\n"


def evaluate_orchestration_gate(
    report: Mapping[str, Any],
    *,
    policy: OrchestrationFailurePolicy | None = None,
) -> OrchestrationGateResult:
    """Evaluate period/model rows without emitting notifications for quiet states."""

    effective_policy = policy or OrchestrationFailurePolicy()
    raw_periods = report.get("periods", ())
    if isinstance(raw_periods, (str, bytes, bytearray)) or not isinstance(raw_periods, Sequence):
        raise ValueError("report periods must be a sequence")

    enriched_periods: list[dict[str, Any]] = []
    alerts: list[dict[str, Any]] = []
    for raw_row in raw_periods:
        if not isinstance(raw_row, Mapping):
            raise ValueError("report period rows must be objects")
        row = _row_with_rate(raw_row)
        enriched_periods.append(row)
        execution_count = int(row.get("execution_count", 0))
        if execution_count < effective_policy.minimum_executions:
            continue
        violations: list[str] = []
        rate = row["permanent_failure_rate"]
        if (
            effective_policy.max_permanent_failure_rate is not None
            and rate is not None
            and float(rate) > effective_policy.max_permanent_failure_rate
        ):
            violations.append("permanent_failure_rate")
        if (
            effective_policy.max_retry_wait_seconds is not None
            and float(row.get("retry_wait_seconds", 0.0)) > effective_policy.max_retry_wait_seconds
        ):
            violations.append("retry_wait_seconds")
        if violations:
            row["violations"] = violations
            alerts.append(row)

    enriched_report = dict(report)
    enriched_report["periods"] = enriched_periods
    return OrchestrationGateResult(
        report=enriched_report,
        policy=effective_policy,
        alerts=tuple(alerts),
    )


__all__ = [
    "THRESHOLD_EXIT_CODE",
    "OrchestrationFailurePolicy",
    "OrchestrationGateResult",
    "evaluate_orchestration_gate",
]
