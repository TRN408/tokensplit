import unittest
from decimal import Decimal

from tokensplit import (
    CacheMetrics,
    CostConversionError,
    NormalizedUsage,
    ProviderRates,
    StreamingUsageAdapter,
    StreamingUsageError,
    calculate_cost,
)


class StreamingAndCostTests(unittest.TestCase):
    def test_openai_chat_stream_uses_final_usage_chunk(self):
        adapter = StreamingUsageAdapter("openai")
        self.assertFalse(adapter.add({"choices": [{"delta": {"content": "hello"}}]}))
        self.assertTrue(
            adapter.add(
                {
                    "usage": {
                        "input_tokens": 1000,
                        "input_tokens_details": {"cached_tokens": 800},
                        "output_tokens": 24,
                    }
                }
            )
        )
        usage = adapter.finalize()
        self.assertEqual((usage.input_tokens, usage.cached_input_tokens, usage.output_tokens), (1000, 800, 24))

    def test_openai_responses_completed_event_is_supported(self):
        adapter = StreamingUsageAdapter("openai")
        adapter.add(
            {
                "type": "response.completed",
                "response": {
                    "usage": {
                        "input_tokens": 500,
                        "input_tokens_details": {"cached_tokens": 400},
                        "output_tokens": 12,
                    }
                },
            }
        )
        self.assertEqual(adapter.finalize().cached_input_tokens, 400)

    def test_anthropic_stream_combines_message_start_and_delta(self):
        adapter = StreamingUsageAdapter("anthropic")
        adapter.add(
            {
                "type": "message_start",
                "message": {
                    "usage": {
                        "input_tokens": 200,
                        "cache_read_input_tokens": 700,
                        "cache_creation_input_tokens": 100,
                    }
                },
            }
        )
        adapter.add({"type": "content_block_delta", "delta": {"text": "hello"}})
        adapter.add({"type": "message_delta", "usage": {"output_tokens": 35}})
        usage = adapter.finalize()
        self.assertEqual((usage.input_tokens, usage.cached_input_tokens, usage.cache_write_tokens), (1000, 700, 100))
        self.assertEqual(usage.output_tokens, 35)

    def test_stream_without_usage_is_rejected(self):
        adapter = StreamingUsageAdapter("openai")
        adapter.add({"choices": []})
        with self.assertRaises(StreamingUsageError):
            adapter.finalize()

    def test_provider_cost_conversion_keeps_components_and_unknown_state(self):
        usage = NormalizedUsage("openai", 1000, 80, 700, 100, True)
        rates = ProviderRates.from_per_million(
            provider="openai",
            model="fixture-model",
            input_usd="2.00",
            cached_input_usd="0.20",
            cache_write_usd="2.50",
            output_usd="10.00",
        )
        cost = calculate_cost(usage, rates)
        self.assertEqual(cost.uncached_input_tokens, 200)
        self.assertEqual(cost.total_usd, Decimal("0.00159"))
        self.assertTrue(cost.complete)

        unknown = NormalizedUsage("openai", 1000, 80, 700, 0, False)
        unknown_cost = calculate_cost(unknown, rates)
        self.assertFalse(unknown_cost.complete)

    def test_cost_rejects_provider_mismatch(self):
        usage = NormalizedUsage("anthropic", 10, 1, 2, 0)
        rates = ProviderRates.from_per_million(
            provider="openai",
            model="fixture-model",
            input_usd=1,
            cached_input_usd=0.1,
            cache_write_usd=1.25,
            output_usd=2,
        )
        with self.assertRaises(CostConversionError):
            calculate_cost(usage, rates)

    def test_stream_record_updates_cache_metrics(self):
        adapter = StreamingUsageAdapter("openai")
        adapter.add({"usage": {"input_tokens": 10, "input_tokens_details": {"cached_tokens": 5}, "output_tokens": 2}})
        metrics = CacheMetrics()
        adapter.record(metrics, prefix_fingerprint="stable")
        self.assertEqual(metrics.cached_input_tokens, 5)
        self.assertEqual(metrics.total_output_tokens, 2)


if __name__ == "__main__":
    unittest.main()
