#!/usr/bin/env python3
"""Create or update one marker-owned GitHub pull-request comment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import os
import subprocess
import sys


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Markdown comment body")
    parser.add_argument("--marker", required=True, help="HTML marker owned by this report")
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY"))
    parser.add_argument("--pr-number", default=os.environ.get("PR_NUMBER"))
    parser.add_argument(
        "--delete-if-empty",
        action="store_true",
        help="delete the marker-owned comment when the input body is empty",
    )
    return parser


def _run_gh(arguments: list[str], *, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["gh", "api", *arguments],
        input=input_text,
        capture_output=True,
        text=True,
        check=False,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.repository:
        print("GITHUB_REPOSITORY is required", file=sys.stderr)
        return 2
    if not args.pr_number:
        print("PR_NUMBER is required", file=sys.stderr)
        return 2

    try:
        body = args.input.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"PR comment body unavailable: {exc}", file=sys.stderr)
        return 1

    endpoint = f"/repos/{args.repository}/issues/{args.pr_number}/comments"
    jq_filter = f".[] | select(.body | contains({json.dumps(args.marker)})) | .id"
    existing = _run_gh(["--paginate", endpoint, "--jq", jq_filter])
    if existing.returncode != 0:
        print(f"failed to list PR comments: {existing.stderr.strip()}", file=sys.stderr)
        return existing.returncode or 1
    comment_id = next((line.strip() for line in existing.stdout.splitlines() if line.strip()), None)

    if args.delete_if_empty and not body.strip():
        if not comment_id:
            print(f"PR comment unchanged: marker={args.marker}")
            return 0
        comment_endpoint = f"/repos/{args.repository}/issues/comments/{comment_id}"
        result = _run_gh(["--method", "DELETE", comment_endpoint])
        if result.returncode != 0:
            print(f"failed to delete PR comment: {result.stderr.strip()}", file=sys.stderr)
            return result.returncode or 1
        print(f"PR comment deleted: marker={args.marker}")
        return 0

    payload = json.dumps({"body": body}, ensure_ascii=False)
    if comment_id:
        comment_endpoint = f"/repos/{args.repository}/issues/comments/{comment_id}"
        result = _run_gh(
            ["--method", "PATCH", comment_endpoint, "--input", "-"],
            input_text=payload,
        )
        action = "updated"
    else:
        result = _run_gh(
            ["--method", "POST", endpoint, "--input", "-"],
            input_text=payload,
        )
        action = "created"
    if result.returncode != 0:
        print(f"failed to {action} PR comment: {result.stderr.strip()}", file=sys.stderr)
        return result.returncode or 1
    print(f"PR comment {action}: marker={args.marker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
