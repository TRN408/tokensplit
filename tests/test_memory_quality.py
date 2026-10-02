import unittest

from benchmarks.memory_quality import run_memory_quality_benchmark


class MemoryQualityBenchmarkTests(unittest.TestCase):
    def test_keyword_memory_retrieves_and_reinjects_all_facts(self):
        result = run_memory_quality_benchmark()
        by_backend = {item.backend: item for item in result.backends}

        keyword = by_backend["keyword"]
        recent_only = by_backend["recent_only"]
        self.assertEqual(result.cases, 3)
        self.assertEqual(keyword.search_marker_recall, 1.0)
        self.assertEqual(keyword.reinjection_marker_recall, 1.0)
        self.assertEqual(keyword.context_fit_rate, 1.0)
        self.assertLess(recent_only.search_marker_recall, keyword.search_marker_recall)
        self.assertLess(recent_only.reinjection_marker_recall, keyword.reinjection_marker_recall)


if __name__ == "__main__":
    unittest.main()
