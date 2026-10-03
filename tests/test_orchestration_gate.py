import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tokensplit import (
    ComparisonLog,
    ComparisonMeasurement,
    ExecutionDiagnostic,
    OrchestrationFailurePolicy,
)
from tokensplit.orchestration_gate import THRESHOLD_EXIT_CODE
from scripts.orchestration_gate import run_gate


def _log(*, permanent_failures: int, executions: int, retry_wait_seconds: float) -> ComparisonLog:
    log = ComparisonLog()
    for index in range(executions):
        permanent = 1 if index < permanent_failures else 0
        log.record(
            ComparisonMeasurement(
                run_id=f"run-{index}",
                mode="split",
                model="claude-sonnet",
                agent_count=1,
                tool_calls=1,
                timestamp=f"2026-10-03T00:00:0{index}Z",
                execution_diagnostics=(
                    ExecutionDiagnostic(
                        execution_id=f"execution-{index}",
                        attempts=1,
                        retry_count=0,
                        retry_wait_seconds=retry_wait_seconds if index == 0 else 0.0,
                        permanent_failures=permanent,
                        failure_categories=("auth",) if permanent else (),
                        outcome="failure" if permanent else "success",
                    ),
                ),
            )
        )
    return log


class OrchestrationGateTests(unittest.TestCase):
    def test_thresholds_are_strict_and_quiet_state_removes_stale_notification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            notification = root / "notify.json"
            notification.write_text("stale alert", encoding="utf-8")
            payload, exit_code = run_gate(
                _log(permanent_failures=1, executions=2, retry_wait_seconds=2.0).render_jsonl(),
                policy=OrchestrationFailurePolicy(
                    max_permanent_failure_rate=0.5,
                    max_retry_wait_seconds=2.0,
                ),
                notification_output=notification,
            )

            self.assertEqual(exit_code, 0)
            self.assertFalse(payload["notify"])
            self.assertEqual(payload["report"]["periods"][0]["permanent_failure_rate"], 0.5)
            self.assertFalse(notification.exists())

    def test_breach_writes_alert_report_and_github_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            notification = root / "notify.json"
            report = root / "report.json"
            github_output = root / "github-output.txt"
            github_summary = root / "github-summary.md"
            payload, exit_code = run_gate(
                _log(permanent_failures=1, executions=2, retry_wait_seconds=3.0).render_jsonl(),
                policy=OrchestrationFailurePolicy(
                    max_permanent_failure_rate=0.4,
                    max_retry_wait_seconds=2.0,
                ),
                report_output=report,
                notification_output=notification,
                github_output=github_output,
                github_summary=github_summary,
            )

            self.assertEqual(exit_code, THRESHOLD_EXIT_CODE)
            self.assertTrue(payload["notify"])
            self.assertEqual(
                payload["alerts"][0]["violations"],
                ["permanent_failure_rate", "retry_wait_seconds"],
            )
            notification_payload = json.loads(notification.read_text(encoding="utf-8"))
            self.assertEqual(notification_payload["event"], "orchestration_failure_threshold_exceeded")
            self.assertEqual(json.loads(report.read_text(encoding="utf-8"))["exit_code"], THRESHOLD_EXIT_CODE)
            self.assertIn("orchestration_failure_alert=true", github_output.read_text(encoding="utf-8"))
            self.assertIn("# Orchestration failure report", github_summary.read_text(encoding="utf-8"))

    def test_cli_fail_on_threshold_returns_alert_exit_code(self):
        script = Path(__file__).parents[1] / "scripts" / "orchestration_gate.py"
        log_text = _log(permanent_failures=1, executions=1, retry_wait_seconds=0.0).render_jsonl() + "\n"
        environment = dict(os.environ)
        with tempfile.TemporaryDirectory() as directory:
            environment["GITHUB_OUTPUT"] = str(Path(directory) / "github-output.txt")
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "-",
                    "--max-permanent-failure-rate",
                    "0",
                    "--fail-on-threshold",
                ],
                input=log_text,
                text=True,
                capture_output=True,
                env=environment,
                check=False,
            )
            self.assertEqual(result.returncode, THRESHOLD_EXIT_CODE)
            self.assertIn('"notify": true', result.stdout)
            self.assertIn("orchestration_failure_alert=true", Path(environment["GITHUB_OUTPUT"]).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
