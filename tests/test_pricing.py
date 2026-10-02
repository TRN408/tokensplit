import unittest
from datetime import date

from tokensplit import PriceCatalog, PriceTableError, pricing_for


class PricingTests(unittest.TestCase):
    def setUp(self):
        self.table = {
            "version": 2,
            "prices": [
                {
                    "provider": "openai",
                    "model": "fixture-model",
                    "ttl_seconds": 300,
                    "effective_from": "2026-01-01",
                    "input_usd_per_million": 10.0,
                    "cache_write_usd_per_million": 12.5,
                    "cache_read_usd_per_million": 1.0,
                },
                {
                    "provider": "OpenAI",
                    "model": "fixture-model",
                    "ttl_seconds": 300,
                    "effective_from": "2026-06-01",
                    "input_usd_per_million": 8.0,
                    "cache_write_usd_per_million": 10.0,
                    "cache_read_usd_per_million": 0.8,
                },
            ],
        }

    def test_resolve_uses_latest_effective_record_case_insensitively(self):
        pricing = PriceCatalog.from_mapping(self.table).resolve(
            provider="openai",
            model="fixture-model",
            ttl_seconds=300,
            as_of=date(2026, 10, 3),
        )

        self.assertEqual(pricing.provider, "OpenAI")
        self.assertEqual(pricing.input_usd_per_million, 8.0)

    def test_resolve_rejects_missing_price(self):
        with self.assertRaises(PriceTableError):
            PriceCatalog.from_mapping(self.table).resolve(
                provider="anthropic",
                model="fixture-model",
                ttl_seconds=300,
                as_of="2026-10-03",
            )

    def test_duplicate_records_and_invalid_rows_are_rejected(self):
        duplicate = {**self.table, "prices": [self.table["prices"][0], self.table["prices"][0]]}
        with self.assertRaises(PriceTableError):
            PriceCatalog.from_mapping(duplicate)
        with self.assertRaises(PriceTableError):
            PriceCatalog.from_mapping({"prices": [{"provider": "openai"}]})

    def test_pricing_for_reads_a_json_table(self):
        import json
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="utf-8") as handle:
            json.dump(self.table, handle)
            handle.flush()
            pricing = pricing_for(
                handle.name,
                provider="openai",
                model="fixture-model",
                ttl_seconds=300,
                as_of="2026-02-01",
            )

        self.assertEqual(pricing.input_usd_per_million, 10.0)


if __name__ == "__main__":
    unittest.main()
