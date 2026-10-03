import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.annotate_service_guide_digest_diff import MARKER, render_comment


class ServiceGuideDigestAnnotationTests(unittest.TestCase):
    def _cell(self, status: str) -> dict[str, str]:
        return {
            "environment": "production",
            "format": "sqlite",
            "status": status,
            "current_source_sha256": "current-source",
            "current_checked_in_sha256": "current-checked-in",
            "previous_source_sha256": "previous-source",
            "previous_checked_in_sha256": "previous-checked-in",
        }

    def test_changed_cell_is_rendered_with_both_digest_pairs(self):
        body = render_comment({"cells": [self._cell("changed")]})

        self.assertIn(MARKER, body)
        self.assertIn("production", body)
        self.assertIn("`previous-source` → `current-source`", body)
        self.assertIn("`previous-checked-in` → `current-checked-in`", body)

    def test_non_changed_cells_produce_empty_body_to_remove_stale_comment(self):
        self.assertEqual(render_comment({"cells": [self._cell("unchanged")] }), "")

    def test_cli_writes_empty_body_for_quiet_report(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            report = directory_path / "digest-diff.json"
            comment = directory_path / "comment.md"
            report.write_text(json.dumps({"cells": [self._cell("baseline_missing")]}), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/annotate_service_guide_digest_diff.py",
                    "--input",
                    str(report),
                    "--comment-output",
                    str(comment),
                ],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
            )

            self.assertIn("no changed", result.stdout)
            self.assertEqual(comment.read_text(encoding="utf-8"), "")


if __name__ == "__main__":
    unittest.main()
