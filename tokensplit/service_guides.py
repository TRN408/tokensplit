"""Short, structured service guides for API-oriented agents.

The registry is deliberately local and read-only at retrieval time.  A guide
is curated data, not a replacement for the provider's current documentation.
Unknown services therefore produce a bounded research plan instead of an
invented authentication scheme or guessed parameter list.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile
from typing import Any

from .output_gate import GateLimits, GateResult, TokenCounter, gate_tool_output


class ServiceGuideError(ValueError):
    """Raised when a service guide is malformed or cannot be addressed safely."""


class ServiceGuidePersistenceError(ServiceGuideError):
    """Raised when a persisted guide catalog cannot be safely loaded."""


def _service_key(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ServiceGuideError("service_id must be a non-empty string")
    if any(character in value for character in "\r\n\x00"):
        raise ServiceGuideError("service_id must be a single line")
    # ``\w`` is Unicode-aware in Python, so names such as a Japanese service
    # label remain addressable while punctuation is normalized away.
    key = re.sub(r"[^\w]+", "-", value.casefold()).strip("-_")
    if not key:
        raise ServiceGuideError("service_id must contain letters or numbers")
    return key


def _text_items(value: Any, *, field: str) -> tuple[str, ...]:
    if isinstance(value, str):
        items: Sequence[Any] = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        items = value
    else:
        raise ServiceGuideError(f"{field} must be text or a sequence of text")
    normalized: list[str] = []
    for item in items:
        if not isinstance(item, str) or not item.strip():
            raise ServiceGuideError(f"{field} must contain non-empty strings")
        if any(character in item for character in "\x00"):
            raise ServiceGuideError(f"{field} must not contain NUL characters")
        normalized.append(item.strip())
    return tuple(normalized)


def _updated_at(value: Any, *, required: bool = False) -> str | None:
    if value is None:
        if required:
            raise ServiceGuideError("updated_at is required")
        return None
    if not isinstance(value, str) or not value.strip():
        raise ServiceGuideError("updated_at must be a non-empty ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ServiceGuideError("updated_at must be a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ServiceGuideError("updated_at must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class ServiceGuide:
    """Curated facts that are useful before calling one service.

    The fields intentionally contain short lists rather than arbitrary
    provider payloads.  Secrets, access tokens, and prompt text do not belong
    in a guide.
    """

    service_id: str
    key_points: tuple[str, ...] = ()
    required_parameters: tuple[str, ...] = ()
    authentication: tuple[str, ...] = ()
    pitfalls: tuple[str, ...] = ()
    source_urls: tuple[str, ...] = ()
    version: int = 1
    updated_at: str | None = None
    reviewed: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "service_id", _service_key(self.service_id))
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version < 1:
            raise ServiceGuideError("version must be a positive integer")
        if not isinstance(self.reviewed, bool):
            raise ServiceGuideError("reviewed must be a boolean")
        object.__setattr__(self, "updated_at", _updated_at(self.updated_at))
        for field in (
            "key_points",
            "required_parameters",
            "authentication",
            "pitfalls",
            "source_urls",
        ):
            object.__setattr__(self, field, _text_items(getattr(self, field), field=field))
        for url in self.source_urls:
            if not re.match(r"^https://[^\s]+$", url):
                raise ServiceGuideError("source_urls must contain HTTPS URLs")

    def as_dict(self) -> dict[str, Any]:
        return {
            "service_id": self.service_id,
            "key_points": list(self.key_points),
            "required_parameters": list(self.required_parameters),
            "authentication": list(self.authentication),
            "pitfalls": list(self.pitfalls),
            "source_urls": list(self.source_urls),
            "version": self.version,
            "updated_at": self.updated_at,
            "reviewed": self.reviewed,
        }

    @classmethod
    def from_persisted_record(cls, record: Mapping[str, Any]) -> "ServiceGuide":
        """Validate one complete, reviewed record from a catalog."""

        return _guide_from_record(record, row="guide record")


@dataclass(frozen=True)
class ServiceGuideResult:
    """Bounded guide output plus lookup status and token accounting."""

    service_id: str
    found: bool
    guide: ServiceGuide | None
    gate: GateResult

    @property
    def text(self) -> str:
        return self.gate.text

    @property
    def used_fallback(self) -> bool:
        return not self.found

    @property
    def original_tokens(self) -> int:
        return self.gate.original_tokens

    @property
    def rendered_tokens(self) -> int:
        return self.gate.rendered_tokens

    @property
    def output_tokens(self) -> int:
        return self.gate.output_tokens

    @property
    def tokens_saved(self) -> int:
        return self.gate.tokens_saved

    @property
    def output_tokens_saved(self) -> int:
        return self.gate.output_tokens_saved

    @property
    def token_savings_ratio(self) -> float:
        return self.gate.token_savings_ratio

    def as_dict(self) -> dict[str, Any]:
        return {
            "service_id": self.service_id,
            "status": "registered" if self.found else "unknown",
            "guide": self.guide.as_dict() if self.guide is not None else None,
            "output": self.text,
            "metrics": {
                "original_tokens": self.original_tokens,
                "rendered_tokens": self.rendered_tokens,
                "tokens_saved": self.tokens_saved,
                "token_savings_ratio": self.token_savings_ratio,
            },
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False, sort_keys=True)


def _unknown_service_payload(service_id: str) -> dict[str, Any]:
    """Return a non-guessing research plan for an unregistered service."""

    return {
        "service_id": service_id,
        "status": "unknown",
        "guide": None,
        "fallback": {
            "mode": "official_docs_research",
            "queries": [
                f"{service_id} official API authentication",
                f"{service_id} official API required parameters",
                f"{service_id} official API errors rate limits",
            ],
            "required_checks": [
                "Verify the official documentation and API version before calling.",
                "Confirm authentication headers and scopes; never guess or print credentials.",
                "Confirm required path, query, and body parameters from the endpoint schema.",
                "Check rate limits, pagination, retries, and error response semantics.",
            ],
            "safety": [
                "No provider-specific facts were inferred.",
                "Do not send a request until the required checks are verified.",
            ],
        },
    }


class ServiceGuideRegistry:
    """Register curated guides and retrieve a bounded guide in one call.

    Retrieval never performs network access.  Applications can populate the
    registry from their own reviewed data and use the unknown-service result as
    a compact, explicit handoff to a separate documentation lookup step.
    """

    def __init__(self, guides: Iterable[ServiceGuide] = ()) -> None:
        self._guides: dict[str, ServiceGuide] = {}
        for guide in guides:
            self.register(guide)

    def register(self, guide: ServiceGuide, *, replace: bool = False) -> None:
        if not isinstance(guide, ServiceGuide):
            raise ServiceGuideError("guide must be a ServiceGuide")
        if not guide.reviewed:
            raise ServiceGuideError("only reviewed service guides can be registered")
        if guide.service_id in self._guides and not replace:
            raise ServiceGuideError(f"service guide already registered: {guide.service_id}")
        self._guides[guide.service_id] = guide

    def get(self, service_id: str) -> ServiceGuide | None:
        return self._guides.get(_service_key(service_id))

    @property
    def guides(self) -> tuple[ServiceGuide, ...]:
        """Return registered guides in deterministic service-id order."""

        return tuple(self._guides[key] for key in sorted(self._guides))

    def retrieve(
        self,
        service_id: str,
        *,
        limits: GateLimits | None = None,
        token_counter: TokenCounter | None = None,
    ) -> ServiceGuideResult:
        """Return a registered guide or a safe bounded fallback plan."""

        key = _service_key(service_id)
        guide = self._guides.get(key)
        payload: Mapping[str, Any]
        if guide is None:
            payload = _unknown_service_payload(key)
        else:
            payload = {
                "service_id": guide.service_id,
                "status": "registered",
                "guide": guide.as_dict(),
            }
        gate_kwargs: dict[str, Any] = {
            "limits": limits or GateLimits.for_purpose("service_guide"),
            "purpose": "service_guide",
        }
        if token_counter is not None:
            gate_kwargs["token_counter"] = token_counter
        gated = gate_tool_output(payload, **gate_kwargs)
        return ServiceGuideResult(key, guide is not None, guide, gated)


_GUIDE_LIST_FIELDS = (
    "key_points",
    "required_parameters",
    "authentication",
    "pitfalls",
    "source_urls",
)


def _required_record_value(record: Mapping[str, Any], field: str, *, row: str) -> Any:
    if field not in record or record[field] is None:
        raise ServiceGuidePersistenceError(f"{row} is missing {field}")
    return record[field]


def _guide_from_record(record: Mapping[str, Any], *, row: str, reviewed_value: Any = None) -> ServiceGuide:
    if not isinstance(record, Mapping):
        raise ServiceGuidePersistenceError(f"{row} must be an object")
    reviewed = record.get("reviewed") if reviewed_value is None else reviewed_value
    if reviewed is not True:
        raise ServiceGuidePersistenceError(f"{row} must be marked reviewed=true")
    values: dict[str, Any] = {
        "service_id": _required_record_value(record, "service_id", row=row),
        "version": _required_record_value(record, "version", row=row),
        "updated_at": _required_record_value(record, "updated_at", row=row),
        "reviewed": True,
    }
    for field in _GUIDE_LIST_FIELDS:
        values[field] = _required_record_value(record, field, row=row)
    try:
        guide = ServiceGuide(**values)
    except (ServiceGuideError, TypeError) as exc:
        raise ServiceGuidePersistenceError(f"{row} contains invalid guide metadata") from exc
    if not guide.source_urls:
        raise ServiceGuidePersistenceError(f"{row} must contain at least one source URL")
    if guide.updated_at is None:
        raise ServiceGuidePersistenceError(f"{row} must contain updated_at")
    return guide


def _select_latest_guides(guides: Iterable[ServiceGuide]) -> tuple[ServiceGuide, ...]:
    selected: dict[str, ServiceGuide] = {}
    for guide in _validate_unique_versions(guides):
        previous = selected.get(guide.service_id)
        if previous is None or guide.version > previous.version:
            selected[guide.service_id] = guide
    return tuple(selected.values())


def _validate_unique_versions(guides: Iterable[ServiceGuide]) -> tuple[ServiceGuide, ...]:
    seen: set[tuple[str, int]] = set()
    result: list[ServiceGuide] = []
    for guide in guides:
        key = (guide.service_id, guide.version)
        if key in seen:
            raise ServiceGuidePersistenceError(
                f"duplicate service guide version: {guide.service_id} v{guide.version}"
            )
        seen.add(key)
        result.append(guide)
    return tuple(result)


class ServiceGuideStore:
    """Load reviewed service guides from JSON or a local SQLite catalog.

    Loading is read-only and atomic from the caller's perspective: all rows
    are validated before a registry is returned. SQLite rows use JSON text for
    the five list fields; see :meth:`from_sqlite` for the expected columns.
    """

    JSON_SCHEMA_VERSION = 1

    @classmethod
    def update_json(
        cls,
        path: str | Path,
        guide: ServiceGuide,
        *,
        replace: bool = False,
    ) -> ServiceGuideRegistry:
        """Atomically add a reviewed guide to a JSON catalog."""

        if not isinstance(guide, ServiceGuide) or not guide.reviewed:
            raise ServiceGuidePersistenceError("update requires a reviewed ServiceGuide")
        destination = Path(path)
        if destination.exists():
            guides = list(_read_v1_guides(destination))
        else:
            guides = []
        guides = _merge_updated_guide(guides, guide, replace=replace)
        _atomic_write_json(_catalog_payload(guides), destination, overwrite=True)
        return ServiceGuideRegistry(_select_latest_guides(guides))

    @classmethod
    def update_sqlite(
        cls,
        path: str | Path,
        guide: ServiceGuide,
        *,
        table: str = "service_guides",
        replace: bool = False,
    ) -> ServiceGuideRegistry:
        """Transactionally add a reviewed guide to a SQLite catalog."""

        if not isinstance(guide, ServiceGuide) or not guide.reviewed:
            raise ServiceGuidePersistenceError("update requires a reviewed ServiceGuide")
        _validate_table_name(table)
        destination = Path(path)
        if destination.exists() and _sqlite_table_exists(destination, table):
            # Validate all existing rows before opening the write transaction;
            # a malformed catalog must never be made worse by an update.
            cls.from_sqlite(destination, table=table)
        try:
            with closing(sqlite3.connect(destination)) as connection:
                with connection:
                    connection.execute(_SQLITE_CREATE_TABLE.format(table=table))
                    rows = connection.execute(
                        f'SELECT version FROM "{table}" WHERE service_id = ?',
                        (guide.service_id,),
                    ).fetchall()
                    versions = [row[0] for row in rows]
                    if versions:
                        maximum = max(versions)
                        if guide.version < maximum:
                            raise ServiceGuidePersistenceError(
                                f"guide version {guide.version} is older than {guide.service_id} v{maximum}"
                            )
                        if guide.version == maximum and not replace:
                            raise ServiceGuidePersistenceError(
                                f"service guide already exists: {guide.service_id} v{guide.version}"
                            )
                    values = _sqlite_values(guide)
                    if guide.version in versions:
                        connection.execute(
                            f"""UPDATE \"{table}\" SET updated_at=?, reviewed=?,
                            key_points_json=?, required_parameters_json=?, authentication_json=?,
                            pitfalls_json=?, source_urls_json=?
                            WHERE service_id=? AND version=?""",
                            values[2:] + (guide.service_id, guide.version),
                        )
                    else:
                        connection.execute(
                            f"""INSERT INTO \"{table}\" (
                            service_id, version, updated_at, reviewed, key_points_json,
                            required_parameters_json, authentication_json, pitfalls_json,
                            source_urls_json
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            values,
                        )
        except ServiceGuidePersistenceError:
            raise
        except (OSError, sqlite3.Error) as exc:
            raise ServiceGuidePersistenceError(
                f"cannot update service guide SQLite catalog: {type(exc).__name__}"
            ) from exc
        return cls.from_sqlite(destination, table=table)

    @classmethod
    def migrate_json(
        cls,
        input_path: str | Path,
        output_path: str | Path,
        *,
        version: int,
        updated_at: str,
        source_urls: Sequence[str] = (),
        reviewed: bool = False,
        overwrite: bool = False,
    ) -> int:
        """Convert a legacy JSON catalog to schema version 1 atomically."""

        source = Path(input_path)
        destination = Path(output_path)
        _reject_same_path(source, destination)
        payload = _read_json_payload(source)
        guides = _legacy_guides_from_records(
            _legacy_json_records(payload),
            version=version,
            updated_at=updated_at,
            source_urls=source_urls,
            reviewed=reviewed,
        )
        _write_catalog_json(guides, destination, overwrite=overwrite)
        return len(guides)

    @classmethod
    def migrate_sqlite(
        cls,
        input_path: str | Path,
        output_path: str | Path,
        *,
        version: int,
        updated_at: str,
        source_urls: Sequence[str] = (),
        reviewed: bool = False,
        table: str = "service_guides",
        overwrite: bool = False,
    ) -> int:
        """Convert a legacy SQLite catalog to schema version 1 atomically."""

        source = Path(input_path)
        destination = Path(output_path)
        _reject_same_path(source, destination)
        records = _legacy_sqlite_records(source, table=table)
        guides = _legacy_guides_from_records(
            records,
            version=version,
            updated_at=updated_at,
            source_urls=source_urls,
            reviewed=reviewed,
        )
        _write_catalog_sqlite(guides, destination, overwrite=overwrite)
        return len(guides)

    @classmethod
    def from_json(cls, path: str | Path) -> ServiceGuideRegistry:
        source = Path(path)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ServiceGuidePersistenceError(f"cannot read service guide JSON: {type(exc).__name__}") from exc
        return ServiceGuideRegistry(_select_latest_guides(_read_v1_guides(source)))

    @classmethod
    def from_sqlite(
        cls,
        path: str | Path,
        *,
        table: str = "service_guides",
    ) -> ServiceGuideRegistry:
        source = Path(path)
        if not source.is_file():
            raise ServiceGuidePersistenceError("service guide SQLite database does not exist")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
            raise ServiceGuidePersistenceError("SQLite table must be a simple identifier")
        columns = (
            "service_id, version, updated_at, reviewed, "
            "key_points_json, required_parameters_json, authentication_json, "
            "pitfalls_json, source_urls_json"
        )
        query = f'SELECT {columns} FROM "{table}"'
        try:
            with closing(sqlite3.connect(source)) as connection:
                with connection:
                    rows = connection.execute(query).fetchall()
        except sqlite3.Error as exc:
            raise ServiceGuidePersistenceError(f"cannot read service guide SQLite catalog: {type(exc).__name__}") from exc

        guides: list[ServiceGuide] = []
        for index, row in enumerate(rows, start=1):
            raw: dict[str, Any] = {
                "service_id": row[0],
                "version": row[1],
                "updated_at": row[2],
                "reviewed": _sqlite_reviewed(row[3], row=f"SQLite guide {index}"),
            }
            values = row[4:]
            if len(values) != len(_GUIDE_LIST_FIELDS):
                raise ServiceGuidePersistenceError(
                    f"SQLite guide {index} has an invalid column count"
                )
            for field, value in zip(_GUIDE_LIST_FIELDS, values):
                try:
                    parsed = json.loads(value)
                except (TypeError, json.JSONDecodeError) as exc:
                    raise ServiceGuidePersistenceError(
                        f"SQLite guide {index} has invalid {field}_json"
                    ) from exc
                raw[field] = parsed
            guides.append(_guide_from_record(raw, row=f"SQLite guide {index}"))
        return ServiceGuideRegistry(_select_latest_guides(guides))


def _sqlite_reviewed(value: Any, *, row: str) -> bool:
    if value is True or value == 1 or (isinstance(value, str) and value.casefold() in {"1", "true"}):
        return True
    if value is False or value == 0 or (isinstance(value, str) and value.casefold() in {"0", "false"}):
        return False
    raise ServiceGuidePersistenceError(f"{row} has invalid reviewed flag")


_SQLITE_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS "{table}" (
    service_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    updated_at TEXT NOT NULL,
    reviewed INTEGER NOT NULL,
    key_points_json TEXT NOT NULL,
    required_parameters_json TEXT NOT NULL,
    authentication_json TEXT NOT NULL,
    pitfalls_json TEXT NOT NULL,
    source_urls_json TEXT NOT NULL,
    PRIMARY KEY (service_id, version)
)
"""


def _validate_table_name(table: str) -> None:
    if not isinstance(table, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
        raise ServiceGuidePersistenceError("SQLite table must be a simple identifier")


def _sqlite_table_exists(source: Path, table: str) -> bool:
    try:
        with closing(sqlite3.connect(source)) as connection:
            with connection:
                row = connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
                    (table,),
                ).fetchone()
    except sqlite3.Error as exc:
        raise ServiceGuidePersistenceError(
            f"cannot inspect service guide SQLite catalog: {type(exc).__name__}"
        ) from exc
    return row is not None


def _sqlite_values(guide: ServiceGuide) -> tuple[Any, ...]:
    return (
        guide.service_id,
        guide.version,
        guide.updated_at,
        1 if guide.reviewed else 0,
        json.dumps(list(guide.key_points), ensure_ascii=False),
        json.dumps(list(guide.required_parameters), ensure_ascii=False),
        json.dumps(list(guide.authentication), ensure_ascii=False),
        json.dumps(list(guide.pitfalls), ensure_ascii=False),
        json.dumps(list(guide.source_urls), ensure_ascii=False),
    )


def _catalog_payload(guides: Iterable[ServiceGuide]) -> dict[str, Any]:
    return {
        "schema_version": ServiceGuideStore.JSON_SCHEMA_VERSION,
        "guides": [guide.as_dict() for guide in guides],
    }


def _merge_updated_guide(
    existing: list[ServiceGuide],
    guide: ServiceGuide,
    *,
    replace: bool,
) -> list[ServiceGuide]:
    same_service = [item for item in existing if item.service_id == guide.service_id]
    if same_service:
        maximum = max(item.version for item in same_service)
        if guide.version < maximum:
            raise ServiceGuidePersistenceError(
                f"guide version {guide.version} is older than {guide.service_id} v{maximum}"
            )
        if guide.version == maximum and not replace:
            raise ServiceGuidePersistenceError(
                f"service guide already exists: {guide.service_id} v{guide.version}"
            )
    result = [
        item
        for item in existing
        if not (item.service_id == guide.service_id and item.version == guide.version)
    ]
    result.append(guide)
    return list(_validate_unique_versions(result))


def _atomic_write_json(payload: Mapping[str, Any], destination: Path, *, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        raise ServiceGuidePersistenceError("output catalog already exists")
    parent = destination.parent
    if not parent.is_dir():
        raise ServiceGuidePersistenceError("output catalog directory does not exist")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(parent)
    except (OSError, TypeError, ValueError) as exc:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        if isinstance(exc, ServiceGuidePersistenceError):
            raise
        raise ServiceGuidePersistenceError(f"cannot write service guide JSON: {type(exc).__name__}") from exc


def _fsync_directory(directory: Path) -> None:
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_json_payload(source: Path) -> Any:
    try:
        return json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ServiceGuidePersistenceError(
            f"cannot read service guide JSON: {type(exc).__name__}"
        ) from exc


def _read_v1_guides(source: Path) -> tuple[ServiceGuide, ...]:
    payload = _read_json_payload(source)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != ServiceGuideStore.JSON_SCHEMA_VERSION:
        raise ServiceGuidePersistenceError("service guide JSON requires schema_version=1")
    records = payload.get("guides")
    if not isinstance(records, list):
        raise ServiceGuidePersistenceError("service guide JSON guides must be an array")
    return tuple(
        _guide_from_record(record, row=f"JSON guide {index}")
        for index, record in enumerate(records, start=1)
    )


def _write_catalog_json(guides: Sequence[ServiceGuide], destination: Path, *, overwrite: bool) -> None:
    _atomic_write_json(_catalog_payload(guides), destination, overwrite=overwrite)


def _reject_same_path(source: Path, destination: Path) -> None:
    try:
        if source.resolve() == destination.resolve():
            raise ServiceGuidePersistenceError("migration input and output must be different paths")
    except OSError as exc:
        raise ServiceGuidePersistenceError("cannot resolve migration paths") from exc


def _write_catalog_sqlite(
    guides: Sequence[ServiceGuide],
    destination: Path,
    *,
    overwrite: bool,
) -> None:
    if destination.exists() and not overwrite:
        raise ServiceGuidePersistenceError("output catalog already exists")
    parent = destination.parent
    if not parent.is_dir():
        raise ServiceGuidePersistenceError("output catalog directory does not exist")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
        with closing(sqlite3.connect(temporary)) as connection:
            with connection:
                connection.execute(_SQLITE_CREATE_TABLE.format(table="service_guides"))
                connection.executemany(
                    """INSERT INTO service_guides (
                    service_id, version, updated_at, reviewed, key_points_json,
                    required_parameters_json, authentication_json, pitfalls_json,
                    source_urls_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (_sqlite_values(guide) for guide in guides),
                )
                connection.execute("PRAGMA user_version = 1")
        os.replace(temporary, destination)
        _fsync_directory(parent)
    except (OSError, sqlite3.Error) as exc:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise ServiceGuidePersistenceError(
            f"cannot write service guide SQLite catalog: {type(exc).__name__}"
        ) from exc


def _legacy_json_records(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, Mapping):
        records = payload.get("services", payload.get("guides", payload.get("items")))
        if not isinstance(records, list):
            raise ServiceGuidePersistenceError("legacy JSON requires a services/guides/items array")
    else:
        raise ServiceGuidePersistenceError("legacy JSON root must be an object or array")
    if any(not isinstance(record, Mapping) for record in records):
        raise ServiceGuidePersistenceError("legacy JSON records must be objects")
    return list(records)


def _legacy_value(record: Mapping[str, Any], names: Sequence[str], *, field: str) -> Any:
    for name in names:
        if name in record and record[name] is not None:
            value = record[name]
            if isinstance(value, str) and name.endswith("_json"):
                try:
                    return json.loads(value)
                except json.JSONDecodeError as exc:
                    raise ServiceGuidePersistenceError(f"legacy {field} contains invalid JSON") from exc
            return value
    return []


def _legacy_guides_from_records(
    records: Iterable[Mapping[str, Any]],
    *,
    version: int,
    updated_at: str,
    source_urls: Sequence[str],
    reviewed: bool,
) -> tuple[ServiceGuide, ...]:
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ServiceGuidePersistenceError("migration version must be a positive integer")
    if not isinstance(reviewed, bool):
        raise ServiceGuidePersistenceError("migration reviewed must be a boolean")
    try:
        canonical_updated_at = _updated_at(updated_at, required=True)
    except ServiceGuideError as exc:
        raise ServiceGuidePersistenceError("migration updated_at is invalid") from exc
    try:
        fallback_sources = _text_items(source_urls, field="source_urls") if source_urls else ()
    except ServiceGuideError as exc:
        raise ServiceGuidePersistenceError("migration source_urls are invalid") from exc

    guides: list[ServiceGuide] = []
    for index, record in enumerate(records, start=1):
        raw: dict[str, Any] = {
            "service_id": record.get("service_id"),
            "version": version,
            "updated_at": canonical_updated_at,
            "reviewed": reviewed,
            "key_points": _legacy_value(
                record,
                ("key_points", "key_points_json", "agent_tips", "agent_tips_json", "tips"),
                field="key_points",
            ),
            "required_parameters": _legacy_value(
                record,
                (
                    "required_parameters",
                    "required_parameters_json",
                    "required_params",
                    "required_params_json",
                    "required",
                ),
                field="required_parameters",
            ),
            "authentication": _legacy_value(
                record,
                ("authentication", "authentication_json", "auth", "auth_json"),
                field="authentication",
            ),
            "pitfalls": _legacy_value(
                record,
                ("pitfalls", "pitfalls_json", "known_pitfalls", "known_pitfalls_json"),
                field="pitfalls",
            ),
            "source_urls": _legacy_value(
                record,
                ("source_urls", "source_urls_json", "sources", "sources_json", "source_url"),
                field="source_urls",
            ),
        }
        if not raw["source_urls"]:
            raw["source_urls"] = fallback_sources
        try:
            guide = ServiceGuide(**raw)
        except (ServiceGuideError, TypeError) as exc:
            raise ServiceGuidePersistenceError(
                f"legacy guide {index} cannot be migrated safely"
            ) from exc
        if not guide.source_urls:
            raise ServiceGuidePersistenceError(
                f"legacy guide {index} has no source URL; provide --source-url"
            )
        guides.append(guide)
    try:
        return _select_latest_guides(guides)
    except ServiceGuidePersistenceError:
        raise


def _legacy_sqlite_records(source: Path, *, table: str) -> list[Mapping[str, Any]]:
    if not source.is_file():
        raise ServiceGuidePersistenceError("legacy service guide SQLite database does not exist")
    _validate_table_name(table)
    try:
        with closing(sqlite3.connect(source)) as connection:
            with connection:
                connection.row_factory = sqlite3.Row
                columns = [row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')]
                if not columns:
                    raise ServiceGuidePersistenceError("legacy SQLite table does not exist")
                rows = connection.execute(f'SELECT * FROM "{table}"').fetchall()
    except ServiceGuidePersistenceError:
        raise
    except sqlite3.Error as exc:
        raise ServiceGuidePersistenceError(
            f"cannot read legacy service guide SQLite catalog: {type(exc).__name__}"
        ) from exc
    return [dict(row) for row in rows]


__all__ = [
    "ServiceGuide",
    "ServiceGuideError",
    "ServiceGuideRegistry",
    "ServiceGuideResult",
    "ServiceGuidePersistenceError",
    "ServiceGuideStore",
]
