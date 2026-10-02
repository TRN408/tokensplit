#!/usr/bin/env python3
"""Validate the provider-agnostic agent token split policy."""
from __future__ import annotations

import json
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "policy.json"


def validate(policy: dict) -> list[str]:
    errors: list[str] = []
    if policy.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if not isinstance(policy.get("policy_version"), str):
        errors.append("policy_version must be a string")

    integer_fields = (
        "minimum_quality_passed_single_measurements",
        "minimum_single_median_tokens",
    )
    for field in integer_fields:
        value = policy.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            errors.append(f"{field} must be a positive integer")

    ratio_fields = (
        "minimum_expected_savings_ratio",
        "minimum_measured_savings_ratio_after_split",
    )
    for field in ratio_fields:
        value = policy.get(field)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 < value < 1:
            errors.append(f"{field} must be a number between 0 and 1")

    if policy.get("unknown_measurements_count_as_evidence") is not False:
        errors.append("unknown_measurements_count_as_evidence must be false")
    return errors


def main() -> int:
    try:
        policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"cannot read {POLICY_PATH}: {exc}", file=sys.stderr)
        return 2

    errors = validate(policy)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 2

    print(f"policy valid: {policy['policy_version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
