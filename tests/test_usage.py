import unittest

from tokensplit.context import CacheMetrics
from tokensplit.usage import (
    UsageAdapterError,
    normalize_anthropic_usage,
    normalize_openai_usage,
    record_provider_usage,
)


class UsageAdapterTests(unittest.TestCase):
    def test_openai_responses_usage_is_normalized(self):
        usage = normalize_openai_usage(
            {
                "usage": {
                    "input_tokens": 2400,
                    "input_tokens_details": {"cached_tokens": 1800},
                    "output_tokens": 120,
                }
            }
        )
        self.assertEqual((usage.input_tokens, usage.cached_input_tokens, usage.output_tokens), (2400, 1800, 120))

    def test_openai_chat_completion_usage_is_normalized(self):
        usage = normalize_openai_usage(
            {
                "usage": {
                    "prompt_tokens": 1000,
                    "prompt_tokens_details": {"cached_tokens": 512},
                    "completion_tokens": 40,
                }
            }
        )
        self.assertEqual((usage.input_tokens, usage.cached_input_tokens, usage.output_tokens), (1000, 512, 40))

    def test_anthropic_usage_sums_uncached_read_and_write_input(self):
        usage = normalize_anthropic_usage(
            {
                "usage": {
                    "input_tokens": 500,
                    "cache_read_input_tokens": 1200,
                    "cache_creation": {"ephemeral_5m_input_tokens": 300},
                    "output_tokens": 80,
                }
            }
        )
        self.assertEqual(usage.input_tokens, 2000)
        self.assertEqual(usage.cached_input_tokens, 1200)
        self.assertEqual(usage.cache_write_tokens, 300)

    def test_recording_updates_metrics_without_network_access(self):
        metrics = CacheMetrics()
        normalized = record_provider_usage(
            "openai",
            {"usage": {"input_tokens": 1000, "input_tokens_details": {"cached_tokens": 800}, "output_tokens": 20}},
            metrics,
            prefix_fingerprint="stable",
        )
        self.assertEqual(metrics.total_input_tokens, 1000)
        self.assertEqual(metrics.total_output_tokens, 20)
        self.assertEqual(metrics.cached_input_tokens, 800)
        self.assertEqual(metrics.hit_rate, 1.0)
        self.assertEqual(normalized.uncached_input_tokens, 200)

    def test_unknown_or_malformed_usage_is_rejected(self):
        with self.assertRaises(UsageAdapterError):
            record_provider_usage("unknown", {}, CacheMetrics(), prefix_fingerprint="x")
        with self.assertRaises(UsageAdapterError):
            normalize_openai_usage({"usage": {"output_tokens": 20}})


if __name__ == "__main__":
    unittest.main()
