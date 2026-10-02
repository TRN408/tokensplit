#!/usr/bin/env python3
"""Validate the provider-agnostic agent token split policy."""
from __future__ import annotations

import json
import math
import pathlib
import statistics
import sys
from collections.abc import Iterable, Mapping


ROOT = pathlib.Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "policy.json"


def validate(policy: dict) -> list[str]:
    errors: list[str] = []
    if not isinstance(policy, dict):
        return ["policy must be an object"]
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
    for field in (
        "single_measurement_requires_quality_pass",
        "split_result_requires_quality_pass",
    ):
        if policy.get(field) is not True:
            errors.append(f"{field} must be true")
    return errors


def _positive_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _valid_ratio(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0 <= value < 1
    )


def should_split_before_trial(
    policy: dict,
    single_agent_measurements: Iterable[Mapping[str, object]],
    expected_savings_ratio: object,
) -> bool:
    """Return whether measured baseline evidence supports trying a split.

    A measurement is evidence only when quality_passed is explicitly true and
    tokens is a known positive integer. Missing, unknown, or malformed values
    are excluded rather than treated as successful measurements.
    """
    if validate(policy) or not _valid_ratio(expected_savings_ratio):
        return False
    if not isinstance(single_agent_measurements, Iterable):
        return False

    known_quality_passed_tokens = [
        measurement["tokens"]
        for measurement in single_agent_measurements
        if isinstance(measurement, Mapping)
        and measurement.get("quality_passed") is True
        and _positive_integer(measurement.get("tokens"))
    ]
    if len(known_quality_passed_tokens) < policy["minimum_quality_passed_single_measurements"]:
        return False
    if statistics.median(known_quality_passed_tokens) < policy["minimum_single_median_tokens"]:
        return False
    return expected_savings_ratio >= policy["minimum_expected_savings_ratio"]


def accept_split_result(
    policy: dict,
    single_agent_tokens: object,
    split_agent_tokens: object,
    quality_passed: object,
) -> bool:
    """Return whether a split trial preserved quality and met measured savings."""
    if validate(policy) or quality_passed is not True:
        return False
    if not _positive_integer(single_agent_tokens) or not _positive_integer(split_agent_tokens):
        return False

    measured_savings_ratio = (single_agent_tokens - split_agent_tokens) / single_agent_tokens
    return measured_savings_ratio >= policy["minimum_measured_savings_ratio_after_split"]


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
