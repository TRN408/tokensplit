#!/usr/bin/env python3
"""Run the orchestration failure report as a quiet CI/monitoring gate.

The gate evaluates each period/model row independently. It exits quietly by
default and emits a notification artifact only when an explicit threshold is
exceeded. Use ``--fail-on-threshold`` when the gate should fail CI or wake a
periodic monitor.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.orchestration_report import periodic_summary
from tokensplit.orchestration_gate import (
    OrchestrationFailurePolicy,
    OrchestrationGateResult,
    evaluate_orchestration_gate,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_github_output(path: Path, result: OrchestrationGateResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"orchestration_failure_alert={str(result.notify).lower()}\n")
        handle.write(f"orchestration_failure_event={result.event or ''}\n")


def _append_github_summary(path: Path, result: OrchestrationGateResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(result.render_markdown())


def run_gate(
    text: str,
    *,
    bucket: str = "day",
    start: str | None = None,
    end: str | None = None,
    policy: OrchestrationFailurePolicy | None = None,
    report_output: Path | None = None,
    notification_output: Path | None = None,
    github_output: Path | None = None,
    github_summary: Path | None = None,
) -> tuple[dict[str, Any], int]:
    """Evaluate a comparison JSONL and write quiet/alert artifacts."""

    report = periodic_summary(text, bucket=bucket, start=start, end=end)
    result = evaluate_orchestration_gate(
        report,
        policy=policy,
    )
    payload = result.as_mapping()
    if report_output is not None:
        _write_json(report_output, payload)
    if notification_output is not None:
        if result.notify:
            _write_json(notification_output, result.notification_payload() or {})
        else:
            try:
                notification_output.unlink()
            except FileNotFoundError:
                pass
    if github_output is not None:
        _write_github_output(github_output, result)
    if github_summary is not None:
        _append_github_summary(github_summary, result)
    return payload, result.exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="ComparisonLog JSONL path, or '-' for stdin")
    parser.add_argument("--period", choices=("day", "week", "month"), default="day")
    parser.add_argument("--start", help="inclusive ISO 8601 start timestamp")
    parser.add_argument("--end", help="exclusive ISO 8601 end timestamp")
    parser.add_argument("--min-executions", type=int, default=1)
    parser.add_argument("--max-permanent-failure-rate", type=float)
    parser.add_argument("--max-retry-wait-seconds", type=float)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--notification-output", type=Path)
    parser.add_argument("--github-output", type=Path, help="Defaults to GITHUB_OUTPUT when set")
    parser.add_argument("--github-summary", type=Path, help="Defaults to GITHUB_STEP_SUMMARY when set")
    parser.add_argument("--fail-on-threshold", action="store_true")
    args = parser.parse_args()
    try:
        text = sys.stdin.read() if args.path == "-" else Path(args.path).read_text(encoding="utf-8")
        policy = OrchestrationFailurePolicy(
            minimum_executions=args.min_executions,
            max_permanent_failure_rate=args.max_permanent_failure_rate,
            max_retry_wait_seconds=args.max_retry_wait_seconds,
        )
        github_output = args.github_output or (
            Path(os.environ["GITHUB_OUTPUT"])
            if os.environ.get("GITHUB_OUTPUT")
            else None
        )
        github_summary = args.github_summary or (
            Path(os.environ["GITHUB_STEP_SUMMARY"])
            if os.environ.get("GITHUB_STEP_SUMMARY")
            else None
        )
        payload, gate_exit_code = run_gate(
            text,
            bucket=args.period,
            start=args.start,
            end=args.end,
            policy=policy,
            report_output=args.report_output,
            notification_output=args.notification_output,
            github_output=github_output,
            github_summary=github_summary,
        )
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return gate_exit_code if args.fail_on_threshold else 0
    except (OSError, ValueError) as exc:
        print(f"orchestration failure gate failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
