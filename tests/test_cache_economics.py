import unittest

from tokensplit.cache import CachePricing, estimate_cache_economics
from tokensplit.context import CacheMetrics


class CacheEconomicsTests(unittest.TestCase):
    def setUp(self):
        self.pricing = CachePricing.from_multipliers(
            "example-model",
            input_usd_per_million=10.0,
            cache_write_multiplier=1.25,
            cache_read_multiplier=0.1,
            ttl_seconds=300,
        )

    def test_projection_separates_normal_write_and_read_and_finds_break_even(self):
        estimate = estimate_cache_economics(
            prefix_tokens=1_000,
            normal_input_tokens_per_request=100,
            repetitions=3,
            pricing=self.pricing,
            request_interval_seconds=60,
        )

        self.assertEqual((estimate.cache_writes, estimate.cache_reads), (1, 2))
        self.assertEqual((estimate.normal_input_tokens, estimate.cache_write_tokens), (300, 1_000))
        self.assertEqual(estimate.cache_read_tokens, 2_000)
        self.assertAlmostEqual(estimate.cache_hit_rate, 2 / 3)
        self.assertEqual(estimate.break_even_repetitions, 2)
        self.assertGreater(estimate.savings_usd, 0)
        self.assertAlmostEqual(estimate.cached_cost_usd, 0.0175)

    def test_ttl_expiry_can_make_repetition_not_worthwhile(self):
        estimate = estimate_cache_economics(
            prefix_tokens=1_000,
            repetitions=3,
            pricing=self.pricing,
            request_interval_seconds=300,
        )

        self.assertEqual((estimate.cache_writes, estimate.cache_reads), (3, 0))
        self.assertIsNone(estimate.break_even_repetitions)
        self.assertTrue(estimate.cache_enabled_but_expensive)

    def test_single_short_request_is_flagged_when_write_cost_exceeds_baseline(self):
        estimate = estimate_cache_economics(
            prefix_tokens=100,
            repetitions=1,
            pricing=self.pricing,
        )

        self.assertTrue(estimate.short_one_off)
        self.assertTrue(estimate.cache_enabled_but_expensive)
        self.assertLess(estimate.savings_usd, 0)

    def test_observed_metrics_are_priced_without_guessing_provider_usage(self):
        metrics = CacheMetrics()
        metrics.observe(prefix_fingerprint="p", input_tokens=1_100, cache_write_tokens=1_000)
        metrics.observe(prefix_fingerprint="p", input_tokens=1_100, cached_input_tokens=1_000)
        summary = metrics.cost_summary(self.pricing)

        self.assertEqual(summary.normal_input_tokens, 200)
        self.assertEqual(summary.cache_write_tokens, 1_000)
        self.assertEqual(summary.cache_read_tokens, 1_000)
        self.assertAlmostEqual(summary.cache_hit_rate, 0.5)
        self.assertAlmostEqual(summary.baseline_cost_usd, 0.022)
        self.assertAlmostEqual(summary.cached_cost_usd, 0.0155)
        self.assertEqual(summary.break_even_repetitions, 2)

    def test_missing_write_volume_is_marked_incomplete(self):
        metrics = CacheMetrics()
        metrics.observe(
            prefix_fingerprint="p",
            input_tokens=1_000,
            cached_input_tokens=800,
            cache_write_tokens_known=False,
        )

        summary = metrics.cost_summary(self.pricing)

        self.assertFalse(summary.complete)
        self.assertFalse(summary.cache_is_worthwhile)
        self.assertFalse(summary.short_one_off)


if __name__ == "__main__":
    unittest.main()
