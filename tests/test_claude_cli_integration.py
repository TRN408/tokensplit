import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from decimal import Decimal
from unittest.mock import patch

from scripts.orchestration_report import summarize
from tokensplit import (
    AgentControlPolicy,
    AgentController,
    AgentRequest,
    ClaudeCliConfig,
    ClaudeCliError,
    ClaudeCliRunner,
    ClaudeCodeCliAdapter,
    ComparisonLog,
    classify_stderr,
)


FAKE_CLI = """#!/usr/bin/env python3
import json
import os
import sys

args = sys.argv[1:]
model = args[args.index('--model') + 1] if '--model' in args else 'missing-model'
prompt = args[-1]
log_path = os.environ.get('TOKENSPLIT_FAKE_CLI_LOG')
if log_path:
    with open(log_path, 'a', encoding='utf-8') as handle:
        handle.write(json.dumps({'model': model, 'prompt': prompt}) + '\\n')
if prompt == 'quality':
    result = '\\n'.join(
        ['routine progress ' + str(index) for index in range(100)]
        + [
            'VERDICT: PASS',
            'ERROR: none observed',
            'EVIDENCE: expected=12 actual=12',
            'tokens_saved=54.7%',
            'Reference: src/worker.py:42',
        ]
    )
else:
    result = 'ok:' + prompt
print(json.dumps({
    'result': result,
    'model': model,
    'usage': {'input_tokens': 10, 'output_tokens': 2},
    'total_cost_usd': '0.001',
}))
"""


RETRYING_CLI = """#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys

state = Path(os.environ['TOKENSPLIT_RETRY_STATE'])
if not state.exists():
    state.write_text('first-attempt', encoding='utf-8')
    sys.stderr.write('connection reset by peer; request id is secret-value\\n')
    raise SystemExit(1)
print(json.dumps({
    'result': 'cli-ok',
    'model': 'retry-model',
    'usage': {'input_tokens': 3, 'output_tokens': 2},
    'total_cost_usd': '0.002',
}))
"""


class ClaudeCliIntegrationTests(unittest.TestCase):
    def test_stderr_classification_is_retry_safe_and_does_not_retain_secrets(self):
        diagnostic = classify_stderr(
            "429 rate limit; sk-ant-api-secret-value",
            returncode=1,
        )
        self.assertEqual(diagnostic.category, "rate_limit")
        self.assertTrue(diagnostic.retryable)
        self.assertNotIn("sk-ant", repr(diagnostic))

        auth = classify_stderr("401 unauthorized", returncode=1)
        self.assertEqual(auth.category, "auth")
        self.assertFalse(auth.retryable)
        self.assertTrue(classify_stderr("", returncode=1).retryable)

    def test_retry_replays_only_transient_failure_and_reports_safe_diagnostics(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            script = directory / "retrying_claude.py"
            state = directory / "retry.state"
            script.write_text(RETRYING_CLI, encoding="utf-8")
            script.chmod(0o755)
            delays = []
            config = ClaudeCliConfig(
                executable=(sys.executable, str(script)),
                working_directory=directory,
                timeout_seconds=5,
                max_retries=1,
                retry_initial_delay_seconds=0,
                retry_sleep=delays.append,
            )
            runner = ClaudeCliRunner(config)
            with patch.dict(os.environ, {"TOKENSPLIT_RETRY_STATE": str(state)}, clear=False):
                result = runner.run(
                    request_id="retry",
                    prompt="Reply with exactly: cli-ok",
                    model="parent-model",
                )

            self.assertEqual(result.text, "cli-ok")
            self.assertEqual(result.attempts, 2)
            self.assertEqual(result.retry_count, 1)
            self.assertEqual(result.stderr_category, "network")
            self.assertEqual(delays, [0])

    def test_adapter_writes_retry_wait_and_failure_category_to_comparison_log(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            script = directory / "retrying_claude.py"
            state = directory / "retry.state"
            script.write_text(RETRYING_CLI, encoding="utf-8")
            script.chmod(0o755)
            config = ClaudeCliConfig(
                executable=(sys.executable, str(script)),
                working_directory=directory,
                timeout_seconds=5,
                max_retries=1,
                retry_initial_delay_seconds=1.25,
                retry_sleep=lambda _: None,
            )
            adapter = ClaudeCodeCliAdapter(
                runner=ClaudeCliRunner(config),
                controller=AgentController(root_model="parent-model"),
            )
            with patch.dict(os.environ, {"TOKENSPLIT_RETRY_STATE": str(state)}, clear=False):
                result = adapter.dispatch([AgentRequest("retry", prompt="retry")])

            diagnostic = result.measurement.execution_diagnostics[0]
            self.assertEqual(diagnostic.retry_count, 1)
            self.assertEqual(diagnostic.failure_categories, ("network",))
            self.assertEqual(diagnostic.retry_wait_seconds, 1.25)
            summary = adapter.comparison_log.summary()
            self.assertEqual(summary.split_retry_count, 1)
            self.assertEqual(summary.split_transient_failures, 1)
            self.assertEqual(summary.split_failure_categories, ("network",))
            restored = ComparisonLog.from_jsonl(adapter.comparison_log.render_jsonl())
            self.assertEqual(restored.summary().split_retry_wait_seconds, 1.25)

    def test_adapter_logs_permanent_failure_before_reraising(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            script = directory / "auth_failure.py"
            script.write_text(
                "import sys\nsys.stderr.write('401 unauthorized\\n')\nsys.exit(1)\n",
                encoding="utf-8",
            )
            adapter = ClaudeCodeCliAdapter(
                runner=ClaudeCliRunner(
                    ClaudeCliConfig(
                        executable=(sys.executable, str(script)),
                        working_directory=directory,
                        max_retries=2,
                        retry_sleep=lambda _: self.fail("auth failures must not retry"),
                    )
                ),
                controller=AgentController(root_model="parent-model"),
            )

            with self.assertRaises(ClaudeCliError):
                adapter.dispatch([AgentRequest("auth", prompt="auth", model="auth-model")])

            summary = adapter.comparison_log.summary()
            self.assertEqual(summary.split_permanent_failures, 1)
            self.assertEqual(summary.split_failure_categories, ("auth",))
            self.assertEqual(adapter.comparison_log.records()[0].model, "auth-model")

    def test_cli_command_terminates_variadic_tools_before_prompt(self):
        command = ClaudeCliRunner(ClaudeCliConfig(tools=("Bash",)))._command(
            prompt="Reply with exactly: cli-ok",
            model="sonnet",
        )

        self.assertEqual(command[-3:], ["Bash", "--", "Reply with exactly: cli-ok"])

    def test_non_retryable_auth_failure_has_no_secret_in_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            script = directory / "auth_failure.py"
            script.write_text(
                "import sys\nsys.stderr.write('401 unauthorized sk-ant-api-secret-value\\n')\nsys.exit(1)\n",
                encoding="utf-8",
            )
            config = ClaudeCliConfig(
                executable=(sys.executable, str(script)),
                working_directory=directory,
                max_retries=3,
                retry_initial_delay_seconds=0,
                retry_sleep=lambda _: self.fail("auth failures must not retry"),
            )
            with self.assertRaises(ClaudeCliError) as raised:
                ClaudeCliRunner(config).run(
                    request_id="auth",
                    prompt="noop",
                    model="parent-model",
                )

            self.assertEqual(raised.exception.diagnostic.category, "auth")
            self.assertEqual(raised.exception.attempts, 1)
            self.assertNotIn("sk-ant", str(raised.exception))

    def test_anonymized_live_response_fixture_matches_parser(self):
        fixture_path = Path(__file__).parent / "fixtures" / "claude_cli_result.json"
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))
        parsed = ClaudeCliRunner.parse_payload(
            payload,
            request_id="fixture",
            model="sonnet",
        )

        self.assertEqual(parsed.text, "cli-ok")
        self.assertEqual(parsed.model, "sonnet")
        self.assertEqual(parsed.input_tokens, 3)
        self.assertEqual(parsed.output_tokens, 6)
        self.assertEqual(parsed.cost_usd, Decimal("0.0425903"))
        self.assertNotIn("session_id", payload)
        self.assertNotIn("uuid", payload)

    def _adapter(self, directory: Path, controller: AgentController, log_path: Path):
        script = directory / "fake_claude.py"
        script.write_text(FAKE_CLI, encoding="utf-8")
        script.chmod(0o755)
        config = ClaudeCliConfig(
            executable=(sys.executable, str(script)),
            working_directory=directory,
            timeout_seconds=5,
        )
        runner = ClaudeCliRunner(config)
        adapter = ClaudeCodeCliAdapter(runner=runner, controller=controller)
        return adapter, {"TOKENSPLIT_FAKE_CLI_LOG": str(log_path)}

    def test_cli_adapter_serializes_remainder_inherits_model_and_records_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            log_path = directory / "invocations.jsonl"
            controller = AgentController(
                AgentControlPolicy(max_subagents=2, overflow_strategy="serialize"),
                root_model="parent-model",
            )
            adapter, environment = self._adapter(directory, controller, log_path)
            requests = [
                AgentRequest(f"task-{index}", prompt=f"do task {index}")
                for index in range(4)
            ]

            with patch.dict(os.environ, environment, clear=False):
                result = adapter.dispatch(requests, quality_score=1.0, quality_passed=True)

            self.assertEqual(
                result.plan.execution_batches,
                (("task-0", "task-1"), ("task-2", "task-3")),
            )
            self.assertEqual([run.request_id for run in result.runs], [f"task-{i}" for i in range(4)])
            self.assertEqual({run.model for run in result.runs}, {"parent-model"})
            self.assertEqual(result.measurement.mode, "split")
            self.assertEqual(result.measurement.input_tokens, 40)
            self.assertEqual(result.measurement.output_tokens, 8)
            self.assertEqual(result.measurement.cost_usd, Decimal("0.004"))
            self.assertEqual(adapter.comparison_log.summary().split_runs, 1)

            invocations = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([item["model"] for item in invocations], ["parent-model"] * 4)
            self.assertEqual([item["prompt"] for item in invocations], [f"do task {i}" for i in range(4)])

    def test_rejected_tasks_are_not_sent_to_the_cli(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            log_path = directory / "invocations.jsonl"
            controller = AgentController(
                AgentControlPolicy(max_subagents=1, overflow_strategy="reject"),
                root_model="parent-model",
            )
            adapter, environment = self._adapter(directory, controller, log_path)
            with patch.dict(os.environ, environment, clear=False):
                result = adapter.dispatch(
                    [AgentRequest("allowed"), AgentRequest("rejected")]
                )

            self.assertEqual(result.plan.launched_task_ids, ("allowed",))
            self.assertEqual(result.plan.rejected_task_ids, ("rejected",))
            self.assertEqual([run.request_id for run in result.runs], ["allowed"])
            self.assertEqual(len(log_path.read_text(encoding="utf-8").splitlines()), 1)

    def test_cli_path_gates_subagent_output_and_replays_quality_from_operational_log(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            log_path = directory / "invocations.jsonl"
            controller = AgentController(root_model="parent-model")
            adapter, environment = self._adapter(directory, controller, log_path)
            markers = (
                "VERDICT: PASS",
                "ERROR: none observed",
                "EVIDENCE: expected=12 actual=12",
                "tokens_saved=54.7%",
                "Reference: src/worker.py:42",
            )
            with patch.dict(os.environ, environment, clear=False):
                result = adapter.dispatch(
                    [AgentRequest("quality", prompt="quality", required_markers=markers)]
                )

            run = result.runs[0]
            self.assertTrue(run.baseline_task_success)
            self.assertTrue(run.task_success)
            self.assertEqual(run.important_information_retention, 1.0)
            self.assertEqual(run.required_markers, len(markers))
            self.assertGreater(run.output_gate_tokens_saved, 0)
            self.assertTrue(run.output_gate_truncated)
            self.assertIn("VERDICT: PASS", run.text)
            self.assertNotIn("routine progress 99", run.text)

            operational_log = adapter.comparison_log.render_jsonl()
            self.assertNotIn("routine progress", operational_log)
            replayed = ComparisonLog.from_jsonl(operational_log)
            summary = replayed.summary()
            self.assertEqual(summary.split_task_success_rate, 1.0)
            self.assertEqual(summary.split_information_retention, 1.0)
            self.assertEqual(summary.split_output_gate_tokens_saved, run.output_gate_tokens_saved)
            rendered_summary = summarize(operational_log)
            self.assertEqual(rendered_summary["split_task_success_rate"], 1.0)
            self.assertEqual(rendered_summary["split_information_retention"], 1.0)

    def test_low_independence_work_is_one_single_agent_cli_call(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            log_path = directory / "invocations.jsonl"
            controller = AgentController(
                AgentControlPolicy(min_independence=0.8),
                root_model="parent-model",
            )
            adapter, environment = self._adapter(directory, controller, log_path)
            requests = [
                AgentRequest("a", independence=0.2, prompt="first"),
                AgentRequest("b", independence=0.3, prompt="second"),
            ]
            with patch.dict(os.environ, environment, clear=False):
                result = adapter.dispatch(requests, quality_passed=True)

            self.assertEqual(result.plan.integrated_groups, (("a", "b"),))
            self.assertEqual(len(result.runs), 1)
            self.assertEqual(result.runs[0].request_id, "integrated:a+b")
            self.assertEqual(result.measurement.mode, "single")
            self.assertEqual(adapter.comparison_log.summary().single_runs, 1)
            self.assertIn("Integrated task a", result.runs[0].text)
            self.assertIn("Integrated task b", result.runs[0].text)

    @unittest.skipUnless(
        os.environ.get("TOKENSPLIT_RUN_LIVE_CLI") == "1",
        "set TOKENSPLIT_RUN_LIVE_CLI=1 to run an authenticated Claude CLI smoke test",
    )
    def test_live_claude_cli_smoke(self):
        model = os.environ.get("TOKENSPLIT_CLAUDE_MODEL", "sonnet")
        adapter = ClaudeCodeCliAdapter(
            runner=ClaudeCliRunner(ClaudeCliConfig(timeout_seconds=60)),
            controller=AgentController(root_model=model),
        )
        result = adapter.dispatch(
            [AgentRequest("live-smoke", prompt="Reply with exactly: cli-ok")],
            quality_passed=True,
        )
        self.assertEqual(len(result.runs), 1)
        self.assertTrue(result.runs[0].text)


if __name__ == "__main__":
    unittest.main()
