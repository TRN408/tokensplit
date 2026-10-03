#!/usr/bin/env python3
"""Check or regenerate the checked-in service-guide SQLite fixtures."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tokensplit import ServiceGuidePersistenceError, ServiceGuideStore


@dataclass(frozen=True)
class Fixture:
    environment: str
    sql_path: Path
    sqlite_path: Path


FIXTURES = (
    Fixture(
        "production",
        ROOT / "tests" / "fixtures" / "service_guides.sqlite.sql",
        ROOT / "tests" / "fixtures" / "service_guides.production.sqlite3",
    ),
    Fixture(
        "staging",
        ROOT / "tests" / "fixtures" / "service_guides.staging.sqlite.sql",
        ROOT / "tests" / "fixtures" / "service_guides.staging.sqlite3",
    ),
)

EXPECTED_COLUMNS = (
    ("service_id", "TEXT", 1, 0, 1),
    ("version", "INTEGER", 1, 0, 2),
    ("updated_at", "TEXT", 1, 0, 0),
    ("reviewed", "INTEGER", 1, 0, 0),
    ("key_points_json", "TEXT", 1, 0, 0),
    ("required_parameters_json", "TEXT", 1, 0, 0),
    ("authentication_json", "TEXT", 1, 0, 0),
    ("pitfalls_json", "TEXT", 1, 0, 0),
    ("source_urls_json", "TEXT", 1, 0, 0),
)


class SQLiteFixtureError(ValueError):
    """Raised when a generated or checked-in SQLite fixture is invalid."""


def _materialize(sql_path: Path, destination: Path) -> None:
    try:
        sql = sql_path.read_text(encoding="utf-8")
        connection = sqlite3.connect(destination)
        try:
            with connection:
                connection.executescript(sql)
        finally:
            connection.close()
    except (OSError, UnicodeError, sqlite3.Error) as exc:
        raise SQLiteFixtureError(f"cannot materialize {sql_path}: {type(exc).__name__}") from exc


def _canonical_snapshot(path: Path) -> dict[str, Any]:
    try:
        registry = ServiceGuideStore.from_sqlite(path)
        with sqlite3.connect(path) as connection:
            table_type = connection.execute(
                "SELECT type FROM sqlite_master WHERE name=?",
                ("service_guides",),
            ).fetchone()
            if table_type != ("table",):
                raise SQLiteFixtureError("service_guides must be a table")
            columns = tuple(connection.execute("PRAGMA table_info(service_guides)").fetchall())
            row_count = connection.execute("SELECT COUNT(*) FROM service_guides").fetchone()[0]
            integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
            if integrity != "ok":
                raise SQLiteFixtureError(f"SQLite integrity check failed: {integrity}")
    except SQLiteFixtureError:
        raise
    except (OSError, sqlite3.Error, ServiceGuidePersistenceError) as exc:
        raise SQLiteFixtureError(f"invalid SQLite fixture {path}: {exc}") from exc

    normalized_columns = tuple(
        (name, declared_type, not_null, is_generated, primary_key)
        for _, name, declared_type, not_null, _, primary_key in columns
        for is_generated in (0,)
    )
    if normalized_columns != EXPECTED_COLUMNS:
        raise SQLiteFixtureError(
            f"SQLite fixture has unexpected service_guides columns: {normalized_columns}"
        )
    if row_count != len(registry.guides):
        raise SQLiteFixtureError(
            f"SQLite fixture row count does not match reviewed guides: {row_count} vs {len(registry.guides)}"
        )
    return {
        "columns": normalized_columns,
        "row_count": row_count,
        "guides": [guide.as_dict() for guide in registry.guides],
    }


def _digest(snapshot: dict[str, Any]) -> str:
    encoded = json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def compare_fixture(fixture: Fixture) -> dict[str, Any]:
    if not fixture.sql_path.is_file():
        raise SQLiteFixtureError(f"SQL fixture does not exist: {fixture.sql_path}")
    with tempfile.TemporaryDirectory() as directory:
        generated = Path(directory) / fixture.sqlite_path.name
        _materialize(fixture.sql_path, generated)
        expected_snapshot = _canonical_snapshot(generated)
        expected_digest = _digest(expected_snapshot)
        report: dict[str, Any] = {
            "environment": fixture.environment,
            "path": str(fixture.sqlite_path.relative_to(ROOT)),
            "source_sha256": expected_digest,
            "checked_in_sha256": None,
            "services": len(expected_snapshot["guides"]),
            "matches": False,
        }
        if not fixture.sqlite_path.is_file():
            report["error"] = f"SQLite fixture does not exist: {fixture.sqlite_path}"
            return report
        try:
            actual_digest = _digest(_canonical_snapshot(fixture.sqlite_path))
        except SQLiteFixtureError as exc:
            report["error"] = str(exc)
            return report
        report["checked_in_sha256"] = actual_digest
        report["matches"] = actual_digest == expected_digest
        return report


def check_fixture(fixture: Fixture, *, write: bool = False) -> dict[str, Any]:
    if write:
        fixture.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=fixture.sqlite_path.parent) as directory:
            replacement = Path(directory) / fixture.sqlite_path.name
            _materialize(fixture.sql_path, replacement)
            replacement.replace(fixture.sqlite_path)
    comparison = compare_fixture(fixture)
    if not comparison["matches"]:
        detail = comparison.get("error") or (
            f"expected {comparison['source_sha256']}, found {comparison['checked_in_sha256']}"
        )
        action = "regenerated" if write else "checked"
        raise SQLiteFixtureError(
            f"SQLite fixture does not match SQL source ({action}): {fixture.sqlite_path}; {detail}"
        )
    return {
        "action": "written" if write else "checked",
        **comparison,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="replace SQLite fixtures from their SQL sources")
    parser.add_argument("--environment", choices=tuple(fixture.environment for fixture in FIXTURES))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    fixtures = tuple(
        fixture for fixture in FIXTURES if args.environment is None or fixture.environment == args.environment
    )
    try:
        results = [check_fixture(fixture, write=args.write) for fixture in fixtures]
    except SQLiteFixtureError as exc:
        print(f"service guide SQLite fixture error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"fixtures": results, "valid": True}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
