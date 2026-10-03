import json
from pathlib import Path
import unittest
from unittest.mock import patch
from urllib.error import URLError

from tokensplit import (
    AgentControlPolicy,
    AgentController,
    AgentRequest,
    QwenApiAdapter,
    QwenApiConfig,
    QwenApiError,
    QwenApiRunner,
)


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class QwenApiIntegrationTests(unittest.TestCase):
    def test_anonymized_qwen_fixture_is_normalized_to_provider_neutral_result(self):
        fixture = Path(__file__).parent / "fixtures" / "qwen_api_result.json"
        payload = json.loads(fixture.read_text(encoding="utf-8"))
        runner = QwenApiRunner(QwenApiConfig(api_key="test-key"))

        with patch("tokensplit.qwen_api.urlopen", return_value=_Response(payload)):
            result = runner.run(
                request_id="fixture",
                prompt="Reply with exactly: qwen-ok",
                model="qwen-plus",
            )

        self.assertEqual(result.text, "qwen-ok")
        self.assertEqual(result.model, "qwen-plus")
        self.assertEqual(result.input_tokens, 4)
        self.assertEqual(result.output_tokens, 7)
        self.assertEqual(result.retry_count, 0)
        self.assertNotIn("api_key", repr(result))

    def test_qwen_adapter_reuses_serialization_and_model_inheritance(self):
        runner = QwenApiRunner(QwenApiConfig(api_key="test-key"))

        def response(*, prompt, model):
            return {
                "model": model,
                "choices": [{"message": {"content": f"ok:{prompt}"}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 3},
            }

        controller = AgentController(
            AgentControlPolicy(max_subagents=1, overflow_strategy="serialize"),
            root_model="qwen-plus",
        )
        adapter = QwenApiAdapter(runner=runner, controller=controller)
        with patch.object(runner, "_request", side_effect=response):
            result = adapter.dispatch(
                [
                    AgentRequest("one", prompt="one"),
                    AgentRequest("two", prompt="two"),
                ]
            )

        self.assertEqual(result.plan.execution_batches, (("one",), ("two",)))
        self.assertEqual([run.model for run in result.runs], ["qwen-plus", "qwen-plus"])
        self.assertEqual(result.measurement.input_tokens, 4)
        self.assertEqual(result.measurement.output_tokens, 6)
        self.assertEqual(adapter.comparison_log.summary().split_runs, 1)

    def test_qwen_network_failure_retries_without_retaining_error_text(self):
        delays = []
        runner = QwenApiRunner(
            QwenApiConfig(
                api_key="test-key",
                max_retries=1,
                retry_initial_delay_seconds=0.25,
                retry_sleep=delays.append,
            )
        )
        payload = {
            "model": "qwen-plus",
            "choices": [{"message": {"content": "qwen-ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        with patch(
            "tokensplit.qwen_api.urlopen",
            side_effect=[URLError("secret endpoint details"), _Response(payload)],
        ):
            result = runner.run(request_id="retry", prompt="noop", model="qwen-plus")

        self.assertEqual(result.retry_count, 1)
        self.assertEqual(result.stderr_category, "network")
        self.assertEqual(result.retry_wait_seconds, 0.25)
        self.assertEqual(delays, [0.25])

    def test_missing_qwen_key_is_a_safe_permanent_auth_failure(self):
        runner = QwenApiRunner(QwenApiConfig())
        with patch.dict("os.environ", {"QWEN_API_KEY": ""}, clear=False):
            with self.assertRaises(QwenApiError) as raised:
                runner.run(request_id="auth", prompt="noop", model="qwen-plus")

        self.assertEqual(raised.exception.diagnostic.category, "auth")
        self.assertEqual(raised.exception.attempts, 1)
        self.assertEqual(raised.exception.permanent_failures, 1)


if __name__ == "__main__":
    unittest.main()
