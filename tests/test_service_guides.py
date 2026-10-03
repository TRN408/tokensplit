import unittest
import json
import sqlite3
import subprocess
import sys
import tempfile
from contextlib import closing
from pathlib import Path

from tokensplit import (
    GateLimits,
    ServiceGuide,
    ServiceGuideError,
    ServiceGuideRegistry,
    ServiceGuidePersistenceError,
    ServiceGuideStore,
)


class ServiceGuideTests(unittest.TestCase):
    def setUp(self):
        self.registry = ServiceGuideRegistry(
            [
                ServiceGuide(
                    "Example API",
                    key_points=("Use v2 records endpoints.",),
                    required_parameters=("company_id", "record_id"),
                    authentication=("OAuth2 bearer token",),
                    pitfalls=("Amounts are integer minor units.",),
                    source_urls=("https://example.test/docs",),
                    version=1,
                    updated_at="2026-10-02T15:00:00+09:00",
                )
            ]
        )

    def test_registered_guide_is_structured_and_measured(self):
        result = self.registry.retrieve("example-api", token_counter=len)

        self.assertTrue(result.found)
        self.assertFalse(result.used_fallback)
        self.assertIn('"required_parameters": [', result.text)
        self.assertIn("company_id", result.text)
        self.assertGreater(result.rendered_tokens, 0)
        self.assertEqual(result.tokens_saved, result.original_tokens - result.rendered_tokens)

    def test_unknown_service_returns_non_guessing_fallback(self):
        result = self.registry.retrieve("New Service")

        self.assertFalse(result.found)
        self.assertTrue(result.used_fallback)
        self.assertIn('"status": "unknown"', result.text)
        self.assertIn("official_docs_research", result.text)
        self.assertIn("No provider-specific facts were inferred.", result.text)
        self.assertNotIn("Authorization: Bearer", result.text)

    def test_unknown_unicode_service_id_is_safe_and_addressable(self):
        result = self.registry.retrieve("経費サービス")

        self.assertFalse(result.found)
        self.assertEqual(result.service_id, "経費サービス")
        self.assertIn("経費サービス official API authentication", result.text)

    def test_fallback_respects_token_budget_and_reports_reduction(self):
        limits = GateLimits.for_purpose("service_guide", max_chars=900, max_output_tokens=90)
        result = self.registry.retrieve("unregistered", limits=limits, token_counter=lambda text: len(text.split()))

        self.assertTrue(result.gate.truncated)
        self.assertLessEqual(result.rendered_tokens, 90)
        self.assertGreaterEqual(result.original_tokens, result.rendered_tokens)

    def test_invalid_guide_data_and_duplicate_registration_are_rejected(self):
        with self.assertRaises(ServiceGuideError):
            ServiceGuide("bad\nservice")
        with self.assertRaises(ServiceGuideError):
            ServiceGuide("bad", authentication=("",))
        with self.assertRaises(ServiceGuideError):
            self.registry.register(ServiceGuide("example-api"))

    def test_json_store_loads_reviewed_latest_version_and_normalizes_timestamp(self):
        payload = {
            "schema_version": 1,
            "guides": [
                {
                    "service_id": "example-api",
                    "version": 1,
                    "updated_at": "2026-10-01T00:00:00Z",
                    "reviewed": True,
                    "key_points": ["old"],
                    "required_parameters": ["company_id"],
                    "authentication": ["OAuth2"],
                    "pitfalls": ["old pitfall"],
                    "source_urls": ["https://example.test/v1"],
                },
                {
                    "service_id": "example-api",
                    "version": 2,
                    "updated_at": "2026-10-02T15:00:00+09:00",
                    "reviewed": True,
                    "key_points": ["new"],
                    "required_parameters": ["company_id", "record_id"],
                    "authentication": ["OAuth2"],
                    "pitfalls": ["new pitfall"],
                    "source_urls": ["https://example.test/v2"],
                },
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "guides.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            registry = ServiceGuideStore.from_json(path)

        guide = registry.get("example-api")
        self.assertIsNotNone(guide)
        self.assertEqual(guide.version, 2)
        self.assertEqual(guide.updated_at, "2026-10-02T06:00:00Z")
        self.assertEqual(guide.key_points, ("new",))

    def test_json_store_rejects_unreviewed_or_incomplete_guides(self):
        record = {
            "service_id": "example-api",
            "version": 1,
            "updated_at": "2026-10-02T00:00:00Z",
            "reviewed": False,
            "key_points": [],
            "required_parameters": [],
            "authentication": [],
            "pitfalls": [],
            "source_urls": ["https://example.test/docs"],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "guides.json"
            path.write_text(json.dumps({"schema_version": 1, "guides": [record]}), encoding="utf-8")
            with self.assertRaises(ServiceGuidePersistenceError):
                ServiceGuideStore.from_json(path)

            record["reviewed"] = True
            record["updated_at"] = "2026-10-02T00:00:00"
            path.write_text(json.dumps({"schema_version": 1, "guides": [record]}), encoding="utf-8")
            with self.assertRaises(ServiceGuidePersistenceError):
                ServiceGuideStore.from_json(path)

    def test_sqlite_store_loads_latest_reviewed_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "guides.sqlite3"
            with closing(sqlite3.connect(path)) as connection:
                with connection:
                    connection.execute(
                        """
                        CREATE TABLE service_guides (
                            service_id TEXT,
                            version INTEGER,
                            updated_at TEXT,
                            reviewed INTEGER,
                            key_points_json TEXT,
                            required_parameters_json TEXT,
                            authentication_json TEXT,
                            pitfalls_json TEXT,
                            source_urls_json TEXT
                        )
                        """
                    )
                    for version, point in ((1, "old"), (2, "new")):
                        connection.execute(
                            "INSERT INTO service_guides VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (
                                "example-api",
                                version,
                                f"2026-10-0{version}T00:00:00Z",
                                1,
                                json.dumps([point]),
                                json.dumps(["company_id"]),
                                json.dumps(["OAuth2"]),
                                json.dumps([]),
                                json.dumps(["https://example.test/docs"]),
                            ),
                        )
            registry = ServiceGuideStore.from_sqlite(path)

        guide = registry.get("example-api")
        self.assertIsNotNone(guide)
        self.assertEqual(guide.version, 2)
        self.assertEqual(guide.key_points, ("new",))

    def test_sqlite_store_rejects_unreviewed_latest_and_invalid_table_name(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "guides.sqlite3"
            with closing(sqlite3.connect(path)) as connection:
                with connection:
                    connection.execute(
                        "CREATE TABLE service_guides AS SELECT 1 AS service_id, 1 AS version, '2026-10-01T00:00:00Z' AS updated_at, 0 AS reviewed, '[]' AS key_points_json, '[]' AS required_parameters_json, '[]' AS authentication_json, '[]' AS pitfalls_json, '[\"https://example.test/docs\"]' AS source_urls_json"
                    )
            with self.assertRaises(ServiceGuidePersistenceError):
                ServiceGuideStore.from_sqlite(path)
            with self.assertRaises(ServiceGuidePersistenceError):
                ServiceGuideStore.from_sqlite(path, table="service_guides; DROP TABLE service_guides")

    def test_json_update_rejects_same_version_without_replacing_catalog(self):
        guide = ServiceGuide(
            "update-api",
            key_points=("original",),
            source_urls=("https://example.test/docs",),
            updated_at="2026-10-03T00:00:00Z",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "guides.json"
            ServiceGuideStore.update_json(path, guide)
            replacement = ServiceGuide(
                "update-api",
                key_points=("replacement",),
                source_urls=("https://example.test/docs",),
                updated_at="2026-10-03T01:00:00Z",
            )
            with self.assertRaises(ServiceGuidePersistenceError):
                ServiceGuideStore.update_json(path, replacement)
            self.assertEqual(ServiceGuideStore.from_json(path).get("update-api").key_points, ("original",))
            ServiceGuideStore.update_json(path, replacement, replace=True)
            self.assertEqual(ServiceGuideStore.from_json(path).get("update-api").key_points, ("replacement",))
            newer = ServiceGuide(
                "update-api",
                key_points=("version two",),
                source_urls=("https://example.test/docs",),
                version=2,
                updated_at="2026-10-03T02:00:00Z",
            )
            ServiceGuideStore.update_json(path, newer)
            catalog = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(len(catalog["guides"]), 2)
            self.assertEqual(ServiceGuideStore.from_json(path).get("update-api").version, 2)

    def test_cli_migrates_legacy_json_and_requires_explicit_review_flag(self):
        legacy = {
            "services": [
                {
                    "service_id": "legacy-api",
                    "agent_tips": ["Use the records endpoint."],
                    "required_params": ["company_id"],
                    "auth": ["OAuth2"],
                    "pitfalls": ["Pagination is required."],
                    "sources": ["https://example.test/legacy"],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "legacy.json"
            output = Path(directory) / "v1.json"
            source.write_text(json.dumps(legacy), encoding="utf-8")
            command = [
                sys.executable,
                "scripts/service_guides.py",
                "migrate",
                "--format",
                "json",
                "--input",
                str(source),
                "--output",
                str(output),
                "--version",
                "3",
                "--updated-at",
                "2026-10-03T09:00:00+09:00",
            ]
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            self.assertIn('"reviewed": false', result.stdout)
            migrated = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(migrated["guides"][0]["reviewed"])
            with self.assertRaises(ServiceGuidePersistenceError):
                ServiceGuideStore.from_json(output)

            subprocess.run(
                [*command, "--mark-reviewed", "--replace-output"],
                check=True,
                capture_output=True,
                text=True,
            )
            guide = ServiceGuideStore.from_json(output).get("legacy-api")
            self.assertEqual(guide.version, 3)
            self.assertEqual(guide.updated_at, "2026-10-03T00:00:00Z")

            update_input = Path(directory) / "reviewed-guide.json"
            update_input.write_text(
                json.dumps(
                    {
                        "service_id": "legacy-api",
                        "version": 4,
                        "updated_at": "2026-10-03T01:00:00Z",
                        "reviewed": True,
                        "key_points": ["updated through CLI"],
                        "required_parameters": ["company_id"],
                        "authentication": ["OAuth2"],
                        "pitfalls": [],
                        "source_urls": ["https://example.test/legacy"],
                    }
                ),
                encoding="utf-8",
            )
            update_result = subprocess.run(
                [
                    sys.executable,
                    "scripts/service_guides.py",
                    "update",
                    "--format",
                    "json",
                    "--catalog",
                    str(output),
                    "--guide",
                    str(update_input),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIn('"action": "updated"', update_result.stdout)
            self.assertEqual(ServiceGuideStore.from_json(output).get("legacy-api").version, 4)

    def test_sqlite_migration_and_update_are_transactional(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "legacy.sqlite3"
            output = Path(directory) / "v1.sqlite3"
            with closing(sqlite3.connect(source)) as connection:
                with connection:
                    connection.execute(
                        """
                        CREATE TABLE service_guides (
                            service_id TEXT,
                            agent_tips_json TEXT,
                            required_params_json TEXT,
                            auth_json TEXT,
                            pitfalls_json TEXT,
                            sources_json TEXT
                        )
                        """
                    )
                    connection.execute(
                        "INSERT INTO service_guides VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            "sqlite-api",
                            json.dumps(["Use v1."]),
                            json.dumps(["tenant_id"]),
                            json.dumps(["API key header"]),
                            json.dumps(["Do not retry 4xx."]),
                            json.dumps(["https://example.test/sqlite"]),
                        ),
                    )
            count = ServiceGuideStore.migrate_sqlite(
                source,
                output,
                version=1,
                updated_at="2026-10-03T00:00:00Z",
                reviewed=True,
            )
            self.assertEqual(count, 1)
            registry = ServiceGuideStore.from_sqlite(output)
            self.assertEqual(registry.get("sqlite-api").required_parameters, ("tenant_id",))
            ServiceGuideStore.update_sqlite(
                output,
                ServiceGuide(
                    "sqlite-api",
                    key_points=("Use v2.",),
                    source_urls=("https://example.test/sqlite",),
                    version=2,
                    updated_at="2026-10-03T01:00:00Z",
                ),
            )
            self.assertEqual(ServiceGuideStore.from_sqlite(output).get("sqlite-api").version, 2)

    def test_validate_cli_checks_json_and_sqlite_catalogs(self):
        with tempfile.TemporaryDirectory() as directory:
            json_path = Path(directory) / "catalog.json"
            sqlite_path = Path(directory) / "catalog.sqlite3"
            guide = ServiceGuide(
                "validate-api",
                key_points=("safe",),
                source_urls=("https://example.test/validate",),
                updated_at="2026-10-03T00:00:00Z",
            )
            ServiceGuideStore.update_json(json_path, guide)
            ServiceGuideStore.update_sqlite(sqlite_path, guide)
            for path in (json_path, sqlite_path):
                result = subprocess.run(
                    [
                        sys.executable,
                        "scripts/service_guides.py",
                        "validate",
                        "--catalog",
                        str(path),
                        "--require-catalog",
                    ],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('"valid": true', result.stdout)

            duplicate = json.loads(json_path.read_text(encoding="utf-8"))
            duplicate["guides"].append(duplicate["guides"][0])
            json_path.write_text(json.dumps(duplicate), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, "scripts/service_guides.py", "validate", "--catalog", str(json_path)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("duplicate service guide version", result.stderr)

    def test_validate_cli_can_require_a_catalog(self):
        result = subprocess.run(
            [sys.executable, "scripts/service_guides.py", "validate", "--require-catalog"],
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("at least one service guide catalog is required", result.stderr)


if __name__ == "__main__":
    unittest.main()
