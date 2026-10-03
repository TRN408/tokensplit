#!/usr/bin/env python3
"""Validate the production CI service-guide catalog configuration."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tokensplit import ServiceGuidePersistenceError, ServiceGuideStore


class ServiceGuideCIConfigError(ValueError):
    """Raised when the production CI catalog configuration is unsafe."""


def _parse_config(path: Path) -> dict[str, dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ServiceGuideCIConfigError(f"cannot read CI policy: {path}") from exc

    in_service_guides = False
    environments_started = False
    environments: dict[str, dict[str, Any]] = {}
    current_environment: str | None = None
    current_field: str | None = None
    for line_number, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not in_service_guides:
            if stripped == "service_guides:":
                in_service_guides = True
            continue
        indent = len(raw_line) - len(raw_line.lstrip())
        if indent == 0:
            break
        if indent == 2 and stripped == "environments:":
            if environments_started:
                raise ServiceGuideCIConfigError(
                    f"duplicate service_guides.environments at line {line_number}"
                )
            environments_started = True
            continue
        if indent == 4 and stripped.endswith(":") and not stripped.startswith("-"):
            if not environments_started:
                raise ServiceGuideCIConfigError(
                    f"environment declared before service_guides.environments at line {line_number}"
                )
            environment = stripped[:-1].strip()
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", environment):
                raise ServiceGuideCIConfigError(
                    f"invalid service-guide environment at line {line_number}: {environment}"
                )
            if environment in environments:
                raise ServiceGuideCIConfigError(
                    f"duplicate service-guide environment at line {line_number}: {environment}"
                )
            environments[environment] = {"required": None, "catalogs": []}
            current_environment = environment
            current_field = None
            continue
        if indent == 6 and current_environment is not None and stripped.startswith("required:"):
            if environments[current_environment]["required"] is not None:
                raise ServiceGuideCIConfigError(
                    f"duplicate required setting at line {line_number}"
                )
            value = stripped.partition(":")[2].strip().casefold()
            if value not in {"true", "false"}:
                raise ServiceGuideCIConfigError(
                    f"service-guide required must be true or false at line {line_number}"
                )
            environments[current_environment]["required"] = value == "true"
            current_field = "required"
            continue
        if indent == 6 and current_environment is not None and stripped == "catalogs:":
            if current_field == "catalogs":
                raise ServiceGuideCIConfigError(
                    f"duplicate catalogs setting at line {line_number}"
                )
            current_field = "catalogs"
            continue
        match = re.fullmatch(r"-\s+([^#\s][^#]*)", stripped)
        if match and indent == 8 and current_environment is not None and current_field == "catalogs":
            value = match.group(1).strip().strip('"\'')
            if not value:
                raise ServiceGuideCIConfigError(
                    f"service-guide catalogs contains an empty path at line {line_number}"
                )
            environments[current_environment]["catalogs"].append(value)
            continue
        raise ServiceGuideCIConfigError(
            f"unsupported service-guide environment setting at line {line_number}: {stripped}"
        )

    if not in_service_guides:
        raise ServiceGuideCIConfigError("service_guides section is required")
    if not environments_started or not environments:
        raise ServiceGuideCIConfigError(
            "service_guides.environments must contain at least one environment"
        )
    for environment, settings in environments.items():
        if settings["required"] is not True:
            raise ServiceGuideCIConfigError(
                f"service_guides.environments.{environment}.required must be true for production CI"
            )
        if not settings["catalogs"]:
            raise ServiceGuideCIConfigError(
                f"service_guides.environments.{environment}.catalogs must contain at least one catalog"
            )
    return environments


def _format_for_path(path: Path) -> str:
    suffix = path.suffix.casefold()
    if suffix == ".json":
        return "json"
    if suffix in {".sqlite", ".sqlite3", ".db"}:
        return "sqlite"
    raise ServiceGuideCIConfigError(
        f"unsupported service-guide catalog format for {path}; use .json, .sqlite, .sqlite3, or .db"
    )


def validate_config(
    path: Path,
    *,
    environment: str = "production",
    all_environments: bool = False,
    catalog_format: str | None = None,
    root: Path = ROOT,
) -> dict[str, Any]:
    config = _parse_config(path)
    if all_environments:
        if catalog_format is not None:
            raise ServiceGuideCIConfigError(
                "catalog format selection cannot be combined with --all-environments"
            )
        selected_environments = sorted(config)
    else:
        if environment not in config:
            available = ", ".join(sorted(config))
            raise ServiceGuideCIConfigError(
                f"unknown service-guide environment {environment}; available: {available}"
            )
        selected_environments = [environment]
    repository = root.resolve()
    environment_results: dict[str, dict[str, Any]] = {}
    seen: set[Path] = set()
    for selected in selected_environments:
        validated: list[dict[str, Any]] = []
        for configured in config[selected]["catalogs"]:
            relative = Path(configured)
            if relative.is_absolute():
                raise ServiceGuideCIConfigError(
                    f"service-guide catalog paths must be relative to the repository: {configured}"
                )
            catalog = (repository / relative).resolve()
            try:
                catalog.relative_to(repository)
            except ValueError as exc:
                raise ServiceGuideCIConfigError(
                    f"service-guide catalog escapes the repository: {configured}"
                ) from exc
            detected_format = _format_for_path(catalog)
            if catalog_format is not None and detected_format != catalog_format:
                continue
            if catalog in seen:
                raise ServiceGuideCIConfigError(
                    f"duplicate service-guide catalog path across environments: {configured}"
                )
            seen.add(catalog)
            if not catalog.is_file():
                raise ServiceGuideCIConfigError(
                    f"service-guide catalog does not exist for {selected}: {configured}"
                )
            try:
                if detected_format == "json":
                    registry = ServiceGuideStore.from_json(catalog)
                else:
                    registry = ServiceGuideStore.from_sqlite(catalog)
            except (OSError, ServiceGuidePersistenceError) as exc:
                raise ServiceGuideCIConfigError(
                    f"invalid {detected_format} service-guide catalog for {selected} {configured}: {exc}"
                ) from exc
            if not registry.guides:
                raise ServiceGuideCIConfigError(
                    f"service-guide catalog has no reviewed records for {selected}: {configured}"
                )
            validated.append(
                {
                    "format": detected_format,
                    "path": configured,
                    "services": len(registry.guides),
                }
            )
        if catalog_format is not None and not validated:
            raise ServiceGuideCIConfigError(
                f"no {catalog_format} service-guide catalog configured for {selected}"
            )
        environment_results[selected] = {
            "catalog_required": True,
            "catalogs": validated,
            "valid": True,
        }
    if all_environments:
        return {"environments": environment_results, "valid": True}
    result = environment_results[environment]
    return {"environment": environment, **result}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / ".agent-ci-policy.yml")
    parser.add_argument("--environment", default="production")
    parser.add_argument("--format", choices=("json", "sqlite"))
    parser.add_argument(
        "--all-environments",
        action="store_true",
        help="validate every configured environment instead of one selected environment",
    )
    parser.add_argument(
        "--print-paths",
        action="store_true",
        help="print validated repository-relative paths separated by the platform path separator",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.all_environments and (args.print_paths or args.format):
        print("--all-environments cannot be combined with --print-paths or --format", file=sys.stderr)
        return 2
    try:
        result = validate_config(
            args.config,
            environment=args.environment,
            all_environments=args.all_environments,
            catalog_format=args.format,
        )
    except ServiceGuideCIConfigError as exc:
        print(f"service guide CI config error: {exc}", file=sys.stderr)
        return 2
    if args.print_paths:
        print(os.pathsep.join(catalog["path"] for catalog in result["catalogs"]))
    else:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
