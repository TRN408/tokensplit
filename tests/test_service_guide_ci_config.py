import json
from pathlib import Path
import tempfile
import unittest

from scripts.validate_service_guide_ci_config import ServiceGuideCIConfigError, validate_config


ROOT = Path(__file__).resolve().parents[1]


class ServiceGuideCIConfigTests(unittest.TestCase):
    def test_checked_in_production_config_requires_and_loads_catalog(self):
        result = validate_config(ROOT / ".agent-ci-policy.yml")

        self.assertEqual(result["environment"], "production")
        self.assertTrue(result["catalog_required"])
        self.assertEqual(result["catalogs"][0]["format"], "json")
        self.assertEqual(result["catalogs"][0]["services"], 1)

        staging = validate_config(ROOT / ".agent-ci-policy.yml", environment="staging")
        self.assertEqual(staging["catalogs"][0]["path"], "tests/fixtures/service_guides.staging.json")

        production_sqlite = validate_config(
            ROOT / ".agent-ci-policy.yml", environment="production", catalog_format="sqlite"
        )
        self.assertEqual(production_sqlite["catalogs"][0]["format"], "sqlite")
        staging_json = validate_config(
            ROOT / ".agent-ci-policy.yml", environment="staging", catalog_format="json"
        )
        self.assertEqual(staging_json["catalogs"][0]["format"], "json")

        all_environments = validate_config(ROOT / ".agent-ci-policy.yml", all_environments=True)
        self.assertEqual(set(all_environments["environments"]), {"production", "staging"})

    def test_missing_catalog_is_rejected_before_ci(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "policy.yml"
            config.write_text(
                "service_guides:\n  environments:\n    production:\n      required: true\n      catalogs:\n        - missing.json\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ServiceGuideCIConfigError, "does not exist"):
                validate_config(config, root=root)

    def test_unsupported_format_and_invalid_catalog_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "catalog.txt").write_text("not a catalog", encoding="utf-8")
            config = root / "policy.yml"
            config.write_text(
                "service_guides:\n  environments:\n    production:\n      required: true\n      catalogs:\n        - catalog.txt\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ServiceGuideCIConfigError, "unsupported"):
                validate_config(config, root=root)

            (root / "catalog.json").write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
            config.write_text(
                "service_guides:\n  environments:\n    production:\n      required: true\n      catalogs:\n        - catalog.json\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ServiceGuideCIConfigError, "invalid json"):
                validate_config(config, root=root)

            (root / "empty.json").write_text(
                json.dumps({"schema_version": 1, "guides": []}), encoding="utf-8"
            )
            config.write_text(
                "service_guides:\n  environments:\n    production:\n      required: true\n      catalogs:\n        - empty.json\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ServiceGuideCIConfigError, "no reviewed records"):
                validate_config(config, root=root)

    def test_required_flag_and_duplicate_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "catalog.json"
            fixture.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "guides": [
                            {
                                "service_id": "example-api",
                                "version": 1,
                                "updated_at": "2026-10-03T00:00:00Z",
                                "reviewed": True,
                                "key_points": [],
                                "required_parameters": [],
                                "authentication": [],
                                "pitfalls": [],
                                "source_urls": ["https://example.test/docs"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            config = root / "policy.yml"
            config.write_text(
                "service_guides:\n  environments:\n    production:\n      required: false\n      catalogs:\n        - catalog.json\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ServiceGuideCIConfigError, "must be true"):
                validate_config(config, root=root)

            config.write_text(
                "service_guides:\n  environments:\n    production:\n      required: true\n      catalogs:\n        - catalog.json\n        - ./catalog.json\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ServiceGuideCIConfigError, "duplicate"):
                validate_config(config, root=root)

    def test_unknown_environment_is_rejected(self):
        with self.assertRaisesRegex(ServiceGuideCIConfigError, "unknown"):
            validate_config(ROOT / ".agent-ci-policy.yml", environment="qa")
