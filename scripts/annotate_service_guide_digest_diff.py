#!/usr/bin/env python3
"""Create a pull-request comment for changed service-guide matrix digests."""

# The fork PR changes this file only to trigger the cross-repository E2E run.

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


MARKER = "<!-- service-guide-digest-diff -->"


def _changed_cells(payload: dict[str, Any]) -> list[dict[str, Any]]:
    cells = payload.get("cells")
    if not isinstance(cells, list):
        raise ValueError("digest diff payload lacks a cells list")
    changed: list[dict[str, Any]] = []
    for cell in cells:
        if not isinstance(cell, dict):
            raise ValueError("digest diff cell must be an object")
        if cell.get("status") != "changed":
            continue
        for field in ("environment", "format"):
            if not isinstance(cell.get(field), str) or not cell[field]:
                raise ValueError(f"changed digest cell lacks {field}")
        changed.append(cell)
    return changed


def render_comment(payload: dict[str, Any]) -> str:
    changed = _changed_cells(payload)
    if not changed:
        return ""
    lines = [
        MARKER,
        "## Service-guide catalog digest changes",
        "",
        "The following CI matrix cells changed from the previous run:",
        "",
        "| Environment | Format | Source digest (previous → current) | Checked-in digest (previous → current) |",
        "| --- | --- | --- | --- |",
    ]
    for cell in changed:
        lines.append(
            "| {environment} | {format} | `{previous_source}` → `{current_source}` | "
            "`{previous_checked_in}` → `{current_checked_in}` |".format(
                environment=cell["environment"],
                format=cell["format"],
                previous_source=cell.get("previous_source_sha256") or "-",
                current_source=cell.get("current_source_sha256") or "-",
                previous_checked_in=cell.get("previous_checked_in_sha256") or "-",
                current_checked_in=cell.get("current_checked_in_sha256") or "-",
            )
        )
    lines.extend(
        [
            "",
            "Only cells with status `changed` are listed; the complete matrix report remains available in the job summary and artifact.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--comment-output", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("digest diff payload must be an object")
        body = render_comment(payload)
        args.comment_output.parent.mkdir(parents=True, exist_ok=True)
        args.comment_output.write_text(body, encoding="utf-8")
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        print(f"service-guide digest PR comment unavailable: {exc}", file=sys.stderr)
        return 1
    print("service-guide digest PR comment prepared" if body else "no changed service-guide digests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
