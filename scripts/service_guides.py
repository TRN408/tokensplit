#!/usr/bin/env python3
"""Update and migrate the reviewed service-guide catalogs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

# Allow ``python3 scripts/service_guides.py ...`` from any working directory.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tokensplit import ServiceGuide, ServiceGuidePersistenceError, ServiceGuideStore


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    update = commands.add_parser("update", help="atomically add one reviewed guide")
    update.add_argument("--format", choices=("json", "sqlite"), required=True)
    update.add_argument("--catalog", required=True, type=Path)
    update.add_argument("--guide", required=True, type=Path)
    update.add_argument("--table", default="service_guides")
    update.add_argument("--replace", action="store_true", help="replace the same service/version")

    validate = commands.add_parser("validate", help="validate one or more catalog files")
    validate.add_argument("--catalog", action="append", type=Path)
    validate.add_argument("--format", choices=("json", "sqlite"))
    validate.add_argument("--table", default="service_guides")
    validate.add_argument(
        "--require-catalog",
        action="store_true",
        help="fail when no catalog is found",
    )

    migrate = commands.add_parser("migrate", help="convert a legacy catalog to schema version 1")
    migrate.add_argument("--format", choices=("json", "sqlite"), required=True)
    migrate.add_argument("--input", required=True, type=Path)
    migrate.add_argument("--output", required=True, type=Path)
    migrate.add_argument("--version", required=True, type=int)
    migrate.add_argument("--updated-at", required=True)
    migrate.add_argument("--source-url", action="append", default=[])
    migrate.add_argument("--table", default="service_guides")
    migrate.add_argument("--mark-reviewed", action="store_true")
    migrate.add_argument("--replace-output", action="store_true")
    return parser


def _load_guide(path: Path) -> ServiceGuide:
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ServiceGuidePersistenceError(
            f"cannot read guide input: {type(exc).__name__}"
        ) from exc
    if isinstance(payload, dict) and isinstance(payload.get("guide"), dict):
        payload = payload["guide"]
    if isinstance(payload, dict) and isinstance(payload.get("guides"), list):
        if len(payload["guides"]) != 1:
            raise ServiceGuidePersistenceError("guide input must contain exactly one guide")
        payload = payload["guides"][0]
    if not isinstance(payload, dict):
        raise ServiceGuidePersistenceError("guide input must be an object")
    try:
        return ServiceGuide.from_persisted_record(payload)
    except ServiceGuidePersistenceError:
        raise
    except (TypeError, ValueError) as exc:
        raise ServiceGuidePersistenceError("guide input is invalid") from exc


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "update":
        guide = _load_guide(args.guide)
        if args.format == "json":
            registry = ServiceGuideStore.update_json(args.catalog, guide, replace=args.replace)
        else:
            registry = ServiceGuideStore.update_sqlite(
                args.catalog,
                guide,
                table=args.table,
                replace=args.replace,
            )
        return {
            "action": "updated",
            "format": args.format,
            "service_id": guide.service_id,
            "version": guide.version,
            "loaded_services": len(registry.guides),
        }

    if args.command == "validate":
        catalogs = _catalog_targets(args.catalog, args.format)
        if args.require_catalog and not catalogs:
            raise ServiceGuidePersistenceError(
                "at least one service guide catalog is required"
            )
        errors: list[str] = []
        valid_catalogs: list[dict[str, Any]] = []
        for catalog, catalog_format in catalogs:
            try:
                if catalog_format == "json":
                    registry = ServiceGuideStore.from_json(catalog)
                else:
                    registry = ServiceGuideStore.from_sqlite(catalog, table=args.table)
            except ServiceGuidePersistenceError as exc:
                errors.append(f"{catalog}: {exc}")
                continue
            valid_catalogs.append(
                {
                    "format": catalog_format,
                    "path": str(catalog),
                    "services": len(registry.guides),
                }
            )
        if errors:
            raise ServiceGuidePersistenceError("; ".join(errors))
        return {
            "action": "validated",
            "catalog_required": args.require_catalog,
            "valid": True,
            "catalogs": valid_catalogs,
        }

    if args.format == "json":
        count = ServiceGuideStore.migrate_json(
            args.input,
            args.output,
            version=args.version,
            updated_at=args.updated_at,
            source_urls=args.source_url,
            reviewed=args.mark_reviewed,
            overwrite=args.replace_output,
        )
    else:
        count = ServiceGuideStore.migrate_sqlite(
            args.input,
            args.output,
            version=args.version,
            updated_at=args.updated_at,
            source_urls=args.source_url,
            reviewed=args.mark_reviewed,
            table=args.table,
            overwrite=args.replace_output,
        )
    return {
        "action": "migrated",
        "format": args.format,
        "guides": count,
        "reviewed": args.mark_reviewed,
    }


def _catalog_targets(
    catalogs: list[Path] | None,
    catalog_format: str | None,
) -> list[tuple[Path, str]]:
    if catalogs:
        targets: list[tuple[Path, str]] = []
        for catalog in catalogs:
            detected = catalog_format or _format_for_path(catalog)
            targets.append((catalog, detected))
        return targets
    # CI can run this command in repositories that do not ship a catalog yet;
    # conventional files are validated automatically when present.
    defaults = (
        (ROOT / "service-guides.json", "json"),
        (ROOT / "service-guides.sqlite3", "sqlite"),
    )
    return [(path, detected) for path, detected in defaults if path.exists()]


def _format_for_path(path: Path) -> str:
    if path.suffix.casefold() == ".json":
        return "json"
    if path.suffix.casefold() in {".sqlite", ".sqlite3", ".db"}:
        return "sqlite"
    raise ServiceGuidePersistenceError(
        f"cannot infer catalog format from {path}; pass --format"
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        print(json.dumps(_run(args), ensure_ascii=False, sort_keys=True))
    except ServiceGuidePersistenceError as exc:
        print(f"service guide error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
