import unittest

from benchmarks.cache_hit_regression import run_cache_hit_regression


class CacheHitRegressionTests(unittest.TestCase):
    def test_stable_prefix_has_hits_and_mutated_prefix_does_not(self):
        result = run_cache_hit_regression(turns=20)

        self.assertGreaterEqual(result.stable_hit_rate, 0.9)
        self.assertEqual(result.stable_prefix_changes, 0)
        self.assertEqual(result.mutated_hit_rate, 0.0)
        self.assertEqual(result.mutated_prefix_changes, result.turns - 1)


if __name__ == "__main__":
    unittest.main()
