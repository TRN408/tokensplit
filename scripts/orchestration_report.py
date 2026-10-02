#!/usr/bin/env python3
"""Summarize privacy-safe sub-agent comparison JSONL logs."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import asdict
from decimal import Decimal
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tokensplit.orchestration import ComparisonLog, ComparisonMeasurement, OrchestrationError


def _json_safe(value: object) -> object:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    return value


def summarize(text: str) -> dict[str, object]:
    """Return success, retention, cost, and token metrics from comparison JSONL."""

    summary = asdict(ComparisonLog.from_jsonl(text).summary())
    return _json_safe(summary)  # type: ignore[return-value]


def _parse_datetime(value: str | None, *, field: str) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise OrchestrationError(f"{field} must be an ISO 8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise OrchestrationError(f"{field} must be an ISO 8601 timestamp") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _period_key(timestamp: datetime, bucket: str) -> str:
    if bucket == "day":
        return timestamp.date().isoformat()
    if bucket == "week":
        start = timestamp.date() - timedelta(days=timestamp.weekday())
        return start.isoformat()
    if bucket == "month":
        return timestamp.strftime("%Y-%m")
    raise OrchestrationError(f"unsupported period bucket: {bucket}")


def _new_period_row(period: str, model: str) -> dict[str, object]:
    return {
        "period": period,
        "model": model,
        "measurement_count": 0,
        "execution_count": 0,
        "successful_executions": 0,
        "failed_executions": 0,
        "retry_count": 0,
        "retry_wait_seconds": 0.0,
        "transient_failures": 0,
        "permanent_failures": 0,
        "failure_categories": [],
    }


def _add_measurement(row: dict[str, object], measurement: ComparisonMeasurement) -> None:
    row["measurement_count"] = int(row["measurement_count"]) + 1
    categories = row["failure_categories"]
    if not isinstance(categories, list):
        raise OrchestrationError("internal report category state is invalid")
    for diagnostic in measurement.execution_diagnostics:
        row["execution_count"] = int(row["execution_count"]) + 1
        if diagnostic.outcome == "success":
            row["successful_executions"] = int(row["successful_executions"]) + 1
        else:
            row["failed_executions"] = int(row["failed_executions"]) + 1
        row["retry_count"] = int(row["retry_count"]) + diagnostic.retry_count
        row["retry_wait_seconds"] = float(row["retry_wait_seconds"]) + diagnostic.retry_wait_seconds
        row["transient_failures"] = int(row["transient_failures"]) + diagnostic.transient_failures
        row["permanent_failures"] = int(row["permanent_failures"]) + diagnostic.permanent_failures
        for category in diagnostic.failure_categories:
            if category not in categories:
                categories.append(category)


def periodic_summary(
    text: str,
    *,
    bucket: str = "day",
    start: str | None = None,
    end: str | None = None,
) -> dict[str, object]:
    """Aggregate operational failure telemetry by UTC period and model.

    ``start`` is inclusive and ``end`` is exclusive.  Measurements from old
    JSONL records without a timestamp are retained under the ``undated``
    period unless a date filter is supplied.
    """

    if bucket not in {"day", "week", "month"}:
        raise OrchestrationError("bucket must be one of: day, week, month")
    start_at = _parse_datetime(start, field="start")
    end_at = _parse_datetime(end, field="end")
    if start_at is not None and end_at is not None and start_at >= end_at:
        raise OrchestrationError("start must be earlier than end")

    log = ComparisonLog.from_jsonl(text)
    rows: OrderedDict[tuple[str, str], dict[str, object]] = OrderedDict()
    for measurement in log.records():
        timestamp = _parse_datetime(measurement.timestamp, field="measurement timestamp")
        if timestamp is None:
            if start_at is not None or end_at is not None:
                continue
            period = "undated"
        else:
            if start_at is not None and timestamp < start_at:
                continue
            if end_at is not None and timestamp >= end_at:
                continue
            period = _period_key(timestamp, bucket)
        key = (period, measurement.model)
        row = rows.setdefault(key, _new_period_row(period, measurement.model))
        _add_measurement(row, measurement)

    ordered_rows = sorted(rows.values(), key=lambda row: (str(row["period"]), str(row["model"])))
    return _json_safe(
        {
            "bucket": bucket,
            "timezone": "UTC",
            "start": start_at.isoformat() if start_at is not None else None,
            "end": end_at.isoformat() if end_at is not None else None,
            "periods": ordered_rows,
        }
    )  # type: ignore[return-value]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="ComparisonLog JSONL path, or '-' for stdin")
    parser.add_argument(
        "--period",
        choices=("day", "week", "month"),
        help="emit period/model failure telemetry instead of the lifetime summary",
    )
    parser.add_argument("--start", help="inclusive ISO 8601 start timestamp")
    parser.add_argument("--end", help="exclusive ISO 8601 end timestamp")
    args = parser.parse_args()
    try:
        text = sys.stdin.read() if args.path == "-" else Path(args.path).read_text(encoding="utf-8")
        if args.period or args.start or args.end:
            report = periodic_summary(
                text,
                bucket=args.period or "day",
                start=args.start,
                end=args.end,
            )
        else:
            report = summarize(text)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    except (OSError, OrchestrationError) as exc:
        print(f"cannot summarize orchestration log: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
