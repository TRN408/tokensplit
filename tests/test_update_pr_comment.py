from __future__ import annotations

import json
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest


FAKE_GH = """#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys

log_path = Path(os.environ['FAKE_GH_LOG'])
entry = {'args': sys.argv[1:], 'stdin': sys.stdin.read()}
with log_path.open('a', encoding='utf-8') as stream:
    stream.write(json.dumps(entry) + '\\n')
if '--paginate' in sys.argv and os.environ.get('FAKE_GH_COMMENT_ID'):
    print(os.environ['FAKE_GH_COMMENT_ID'])
"""


class UpdatePRCommentTests(unittest.TestCase):
    def _run(
        self,
        root: Path,
        comment: Path,
        gh_dir: Path,
        log: Path,
        *,
        comment_id: str | None,
        delete_if_empty: bool = False,
    ):
        environment = os.environ.copy()
        environment["PATH"] = f"{gh_dir}{os.pathsep}{environment['PATH']}"
        environment["FAKE_GH_LOG"] = str(log)
        environment["GITHUB_REPOSITORY"] = "example/tokensplit"
        environment["PR_NUMBER"] = "42"
        if comment_id is not None:
            environment["FAKE_GH_COMMENT_ID"] = comment_id
        else:
            environment.pop("FAKE_GH_COMMENT_ID", None)
        command = [
            sys.executable,
            "scripts/update_pr_comment.py",
            "--input",
            str(comment),
            "--marker",
            "<!-- pruning-regression-diff -->",
        ]
        if delete_if_empty:
            command.append("--delete-if-empty")
        return subprocess.run(
            command,
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            check=True,
        )

    def test_missing_marker_creates_comment(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            gh_dir = directory_path / "bin"
            gh_dir.mkdir()
            gh = gh_dir / "gh"
            gh.write_text(FAKE_GH, encoding="utf-8")
            gh.chmod(0o755)
            comment = directory_path / "comment.md"
            comment.write_text("<!-- pruning-regression-diff -->\nnew body\n", encoding="utf-8")
            log = directory_path / "gh.jsonl"

            result = self._run(root, comment, gh_dir, log, comment_id=None)

            self.assertIn("PR comment created", result.stdout)
            entry = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
            self.assertEqual(entry["args"][0:2], ["api", "--method"])
            self.assertIn("POST", entry["args"])
            self.assertIn("/repos/example/tokensplit/issues/42/comments", entry["args"])
            self.assertIn("new body", json.loads(entry["stdin"])["body"])

    def test_existing_marker_updates_comment_instead_of_creating_duplicate(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            gh_dir = directory_path / "bin"
            gh_dir.mkdir()
            gh = gh_dir / "gh"
            gh.write_text(FAKE_GH, encoding="utf-8")
            gh.chmod(0o755)
            comment = directory_path / "comment.md"
            comment.write_text("<!-- pruning-regression-diff -->\nupdated body\n", encoding="utf-8")
            log = directory_path / "gh.jsonl"

            result = self._run(root, comment, gh_dir, log, comment_id="comment-123")

            self.assertIn("PR comment updated", result.stdout)
            entries = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
            mutation = entries[-1]
            self.assertIn("PATCH", mutation["args"])
            self.assertIn("/repos/example/tokensplit/issues/comments/comment-123", mutation["args"])
            self.assertNotIn("POST", mutation["args"])
            self.assertIn("updated body", json.loads(mutation["stdin"])["body"])

    def test_empty_body_deletes_existing_marker_comment(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            gh_dir = directory_path / "bin"
            gh_dir.mkdir()
            gh = gh_dir / "gh"
            gh.write_text(FAKE_GH, encoding="utf-8")
            gh.chmod(0o755)
            comment = directory_path / "comment.md"
            comment.write_text("", encoding="utf-8")
            log = directory_path / "gh.jsonl"

            result = self._run(
                root,
                comment,
                gh_dir,
                log,
                comment_id="comment-123",
                delete_if_empty=True,
            )

            self.assertIn("PR comment deleted", result.stdout)
            mutation = [
                json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()
            ][-1]
            self.assertIn("DELETE", mutation["args"])
            self.assertIn(
                "/repos/example/tokensplit/issues/comments/comment-123",
                mutation["args"],
            )


if __name__ == "__main__":
    unittest.main()
