from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest

from scripts.regenerate_service_guide_sqlite_fixtures import (
    FIXTURES,
    _canonical_snapshot,
    _digest,
    _materialize,
)


ROOT = Path(__file__).resolve().parents[1]


class ServiceGuideSQLiteFixtureTests(unittest.TestCase):
    def test_checked_in_fixtures_match_sql_sources(self):
        result = subprocess.run(
            [sys.executable, "scripts/regenerate_service_guide_sqlite_fixtures.py"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"environment": "production"', result.stdout)
        self.assertIn('"environment": "staging"', result.stdout)

    def test_canonical_digest_ignores_sqlite_file_metadata(self):
        fixture = FIXTURES[0]
        with tempfile.TemporaryDirectory() as directory:
            generated = Path(directory) / "generated.sqlite3"
            altered = Path(directory) / "altered.sqlite3"
            _materialize(fixture.sql_path, generated)
            shutil.copyfile(generated, altered)
            with sqlite3.connect(altered) as connection:
                connection.execute("PRAGMA user_version=42")
                connection.execute("VACUUM")

            self.assertEqual(
                _digest(_canonical_snapshot(generated)),
                _digest(_canonical_snapshot(altered)),
            )
