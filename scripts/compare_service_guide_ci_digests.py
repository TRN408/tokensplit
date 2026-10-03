#!/usr/bin/env python3
"""Compare service-guide matrix digests with the previous CI run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any


class DigestComparisonError(ValueError):
    """Raised when matrix digest reports cannot be compared safely."""


def _load_reports(directory: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if not directory.is_dir():
        return {}
    reports: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise DigestComparisonError(f"cannot read matrix report {path}: {exc}") from exc
        environment = payload.get("environment")
        catalog_format = payload.get("format")
        if not isinstance(environment, str) or not isinstance(catalog_format, str):
            raise DigestComparisonError(f"matrix report lacks environment/format: {path}")
        key = (environment, catalog_format)
        if key in reports:
            raise DigestComparisonError(f"duplicate matrix report for {environment}/{catalog_format}")
        reports[key] = payload
    return reports


def _digest(payload: dict[str, Any]) -> tuple[str | None, str | None]:
    sqlite_fixture = payload.get("sqlite_fixture")
    if not isinstance(sqlite_fixture, dict):
        return None, None
    source = sqlite_fixture.get("source_sha256")
    checked_in = sqlite_fixture.get("checked_in_sha256")
    return (
        source if isinstance(source, str) else None,
        checked_in if isinstance(checked_in, str) else None,
    )


def _report_valid(payload: dict[str, Any] | None) -> bool:
    if payload is None or payload.get("valid") is not True:
        return False
    source, checked_in = _digest(payload)
    sqlite_fixture = payload.get("sqlite_fixture")
    return (
        isinstance(source, str)
        and isinstance(checked_in, str)
        and isinstance(sqlite_fixture, dict)
        and sqlite_fixture.get("matches") is True
    )


def _cell_comparison(
    key: tuple[str, str],
    current: dict[str, Any] | None,
    previous: dict[str, Any] | None,
) -> dict[str, Any]:
    environment, catalog_format = key
    current_source, current_checked_in = _digest(current or {})
    previous_source, previous_checked_in = _digest(previous or {})
    current_valid = _report_valid(current)
    previous_valid = _report_valid(previous)
    if current is None:
        status = "missing_current"
    elif not current_valid:
        status = "invalid_current"
    elif previous is None:
        status = "baseline_missing"
    elif not previous_valid:
        status = "baseline_invalid"
    elif (current_source, current_checked_in) == (previous_source, previous_checked_in):
        status = "unchanged"
    else:
        status = "changed"
    return {
        "environment": environment,
        "format": catalog_format,
        "status": status,
        "current_source_sha256": current_source,
        "current_checked_in_sha256": current_checked_in,
        "previous_source_sha256": previous_source,
        "previous_checked_in_sha256": previous_checked_in,
    }


def compare_reports(
    current_dir: Path,
    history_dir: Path,
    *,
    update_history: bool = False,
) -> dict[str, Any]:
    current = _load_reports(current_dir)
    previous = _load_reports(history_dir)
    keys = sorted(set(current) | set(previous))
    cells = [_cell_comparison(key, current.get(key), previous.get(key)) for key in keys]
    if not cells:
        raise DigestComparisonError("no service-guide matrix reports were found")
    if update_history:
        history_dir.mkdir(parents=True, exist_ok=True)
        for key, payload in current.items():
            if not _report_valid(payload):
                continue
            source = current_dir / f"{key[0]}-{key[1]}.json"
            destination = history_dir / source.name
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            shutil.copyfile(source, temporary)
            temporary.replace(destination)
    return {
        "schema_version": 1,
        "baseline_available": bool(previous),
        "cells": cells,
        "valid": all(cell["status"] not in {"missing_current", "invalid_current"} for cell in cells),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Service-guide matrix digest diff",
        "",
        f"Baseline available: **{'yes' if report['baseline_available'] else 'no'}**",
        "",
        "| Environment | Format | Status | Current checked-in digest | Previous checked-in digest |",
        "| --- | --- | --- | --- | --- |",
    ]
    for cell in report["cells"]:
        lines.append(
            "| {environment} | {format} | **{status}** | `{current}` | `{previous}` |".format(
                environment=cell["environment"],
                format=cell["format"],
                status=cell["status"],
                current=cell["current_checked_in_sha256"] or "-",
                previous=cell["previous_checked_in_sha256"] or "-",
            )
        )
    lines.extend(
        [
            "",
            "`changed` means the canonical SQL-generated or checked-in SQLite digest changed.",
            "`baseline_missing` is expected on the first run for a branch.",
        ]
    )
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-dir", type=Path, required=True)
    parser.add_argument("--history-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--update-history", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = compare_reports(
            args.current_dir,
            args.history_dir,
            update_history=args.update_history,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    except (DigestComparisonError, OSError) as exc:
        print(f"service-guide digest comparison error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
