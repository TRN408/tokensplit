import json
from pathlib import Path
import tempfile
import unittest

from scripts.compare_service_guide_ci_digests import compare_reports, render_markdown


def _cell(environment: str, catalog_format: str, digest: str, *, valid: bool = True) -> dict:
    return {
        "schema_version": 1,
        "environment": environment,
        "format": catalog_format,
        "valid": valid,
        "catalog": {"valid": valid, "environment": environment, "catalogs": []},
        "sqlite_fixture": {
            "environment": environment,
            "path": f"{environment}.sqlite3",
            "source_sha256": digest,
            "checked_in_sha256": digest,
            "matches": valid,
            "services": 1,
        },
    }


def _write(directory: Path, payload: dict) -> None:
    path = directory / f"{payload['environment']}-{payload['format']}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")


class ServiceGuideCIDigestDiffTests(unittest.TestCase):
    def test_first_run_is_baseline_missing_and_updates_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current = root / "current"
            history = root / "history"
            current.mkdir()
            _write(current, _cell("production", "json", "a" * 64))

            report = compare_reports(current, history, update_history=True)

            self.assertFalse(report["baseline_available"])
            self.assertEqual(report["cells"][0]["status"], "baseline_missing")
            self.assertTrue((history / "production-json.json").is_file())

    def test_changed_digest_is_rendered_for_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current = root / "current"
            history = root / "history"
            current.mkdir()
            history.mkdir()
            _write(history, _cell("staging", "sqlite", "a" * 64))
            _write(current, _cell("staging", "sqlite", "b" * 64))

            report = compare_reports(current, history)
            markdown = render_markdown(report)

            self.assertEqual(report["cells"][0]["status"], "changed")
            self.assertIn("| staging | sqlite | **changed** |", markdown)
            self.assertIn("a" * 64, markdown)
            self.assertIn("b" * 64, markdown)

    def test_unchanged_digest_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current = root / "current"
            history = root / "history"
            current.mkdir()
            history.mkdir()
            payload = _cell("production", "sqlite", "c" * 64)
            _write(history, payload)
            _write(current, payload)

            report = compare_reports(current, history)

            self.assertEqual(report["cells"][0]["status"], "unchanged")
            self.assertTrue(report["valid"])
