import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ServiceGuideCIMatrixCellTests(unittest.TestCase):
    def test_cell_report_contains_catalog_and_sqlite_digests(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "staging-sqlite.json"
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/validate_service_guide_ci_matrix_cell.py",
                    "--config",
                    ".agent-ci-policy.yml",
                    "--environment",
                    "staging",
                    "--format",
                    "sqlite",
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(report["valid"])
            self.assertEqual(report["catalog"]["catalogs"][0]["format"], "sqlite")
            sqlite_report = report["sqlite_fixture"]
            self.assertEqual(sqlite_report["source_sha256"], sqlite_report["checked_in_sha256"])
            self.assertTrue(sqlite_report["matches"])

    def test_json_cell_also_records_sqlite_digest_for_cross_format_review(self):
        result = subprocess.run(
            [
                sys.executable,
                "scripts/validate_service_guide_ci_matrix_cell.py",
                "--environment",
                "production",
                "--format",
                "json",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertTrue(report["valid"])
        self.assertEqual(report["catalog"]["catalogs"][0]["format"], "json")
        self.assertRegex(report["sqlite_fixture"]["source_sha256"], r"^[0-9a-f]{64}$")
