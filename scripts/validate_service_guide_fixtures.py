#!/usr/bin/env python3
"""Validate the checked-in JSON and SQLite service-guide fixtures."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tokensplit import ServiceGuidePersistenceError, ServiceGuideStore


JSON_FIXTURE = ROOT / "tests" / "fixtures" / "service_guides.valid.json"
SQLITE_FIXTURE = ROOT / "tests" / "fixtures" / "service_guides.sqlite.sql"


def _run_cli(catalog: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "service_guides.py"),
            "validate",
            "--catalog",
            str(catalog),
            "--require-catalog",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ServiceGuidePersistenceError(
            f"fixture validation failed for {catalog.name}: {completed.stderr.strip()}"
        )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ServiceGuidePersistenceError(
            f"fixture validation returned invalid JSON for {catalog.name}"
        ) from exc
    if result.get("valid") is not True or result.get("catalog_required") is not True:
        raise ServiceGuidePersistenceError(
            f"fixture validation did not run in required mode for {catalog.name}"
        )
    return result


def _materialize_sqlite(destination: Path) -> None:
    sql = SQLITE_FIXTURE.read_text(encoding="utf-8")
    connection = sqlite3.connect(destination)
    try:
        with connection:
            connection.executescript(sql)
    finally:
        connection.close()


def main() -> int:
    try:
        json_registry = ServiceGuideStore.from_json(JSON_FIXTURE)
        json_result = _run_cli(JSON_FIXTURE)
        with tempfile.TemporaryDirectory() as directory:
            sqlite_path = Path(directory) / "service-guides.sqlite3"
            _materialize_sqlite(sqlite_path)
            sqlite_registry = ServiceGuideStore.from_sqlite(sqlite_path)
            sqlite_result = _run_cli(sqlite_path)
        if len(json_registry.guides) != 1 or len(sqlite_registry.guides) != 1:
            raise ServiceGuidePersistenceError("fixture must contain exactly one service guide")
        print(
            json.dumps(
                {
                    "json": json_result,
                    "sqlite": sqlite_result,
                    "services": [json_registry.guides[0].service_id],
                    "valid": True,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    except (OSError, sqlite3.Error, ServiceGuidePersistenceError) as exc:
        print(f"service guide fixture error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
