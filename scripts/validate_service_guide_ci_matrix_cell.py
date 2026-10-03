#!/usr/bin/env python3
"""Validate one service-guide CI matrix cell and emit a reviewable report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.regenerate_service_guide_sqlite_fixtures import (
    FIXTURES,
    SQLiteFixtureError,
    compare_fixture,
)
from scripts.validate_service_guide_ci_config import (
    ServiceGuideCIConfigError,
    validate_config,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / ".agent-ci-policy.yml")
    parser.add_argument("--environment", choices=tuple(fixture.environment for fixture in FIXTURES), required=True)
    parser.add_argument("--format", choices=("json", "sqlite"), required=True)
    parser.add_argument("--output", type=Path)
    return parser


def _write_report(path: Path | None, report: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report: dict[str, Any] = {
        "schema_version": 1,
        "environment": args.environment,
        "format": args.format,
        "valid": False,
    }
    try:
        fixture = next(item for item in FIXTURES if item.environment == args.environment)
        report["sqlite_fixture"] = compare_fixture(fixture)
        report["catalog"] = validate_config(
            args.config,
            environment=args.environment,
            catalog_format=args.format,
        )
        report["valid"] = report["sqlite_fixture"]["matches"] is True
    except (SQLiteFixtureError, ServiceGuideCIConfigError, OSError) as exc:
        report["error"] = str(exc)
    _write_report(args.output, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
