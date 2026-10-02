import unittest
from datetime import date
from pathlib import Path

from tokensplit import CachePricing, ProviderRates
from tokensplit.pricing import PriceCatalog, PriceTableError, pricing_for, provider_rates_for


class PricingCatalogTests(unittest.TestCase):
    def make_catalog(self):
        return PriceCatalog.from_mapping(
            {
                "version": 2,
                "prices": [
                    {
                        "provider": "demo",
                        "model": "model-a",
                        "ttl_seconds": 300,
                        "effective_from": "2026-01-01",
                        "input_usd_per_million": 10,
                        "cache_write_usd_per_million": 12.5,
                        "cache_read_usd_per_million": 1,
                    },
                    {
                        "provider": "demo",
                        "model": "model-a",
                        "ttl_seconds": 300,
                        "effective_from": "2026-09-01",
                        "input_usd_per_million": 10,
                        "cache_write_usd_per_million": 12.5,
                        "cache_read_usd_per_million": 0.25,
                    },
                    {
                        "provider": "demo",
                        "model": "model-a",
                        "ttl_seconds": 3600,
                        "effective_from": "2026-09-01",
                        "input_usd_per_million": 10,
                        "cache_write_usd_per_million": 20,
                        "cache_read_usd_per_million": 0.25,
                    },
                ],
            }
        )

    def test_resolve_selects_latest_effective_version_and_ttl(self):
        catalog = self.make_catalog()

        before = catalog.resolve(
            provider="DEMO",
            model="model-a",
            ttl_seconds=300,
            as_of=date(2026, 8, 31),
        )
        after = catalog.resolve(
            provider="demo",
            model="model-a",
            ttl_seconds=300,
            as_of="2026-10-03",
        )
        long_ttl = catalog.resolve(
            provider="demo",
            model="model-a",
            ttl_seconds=3600,
            as_of="2026-10-03",
        )

        self.assertEqual((before.provider, before.model, before.cache_read_usd_per_million), ("demo", "model-a", 1))
        self.assertEqual(after.cache_read_usd_per_million, 0.25)
        self.assertEqual((long_ttl.ttl_seconds, long_ttl.cache_write_usd_per_million), (3600, 20))
        self.assertEqual(after.effective_from, date(2026, 9, 1))
        self.assertEqual(after.catalog_version, 2)

    def test_pricing_for_generates_cache_pricing_from_json_table(self):
        root = Path(__file__).resolve().parents[1]
        pricing = pricing_for(
            root / "pricing.json",
            provider="anthropic",
            model="fable-5.1",
            ttl_seconds=3600,
            as_of="2026-10-03",
        )

        self.assertIsInstance(pricing, CachePricing)
        self.assertEqual(pricing.provider, "anthropic")
        self.assertEqual(pricing.cache_write_usd_per_million, 20.0)
        self.assertEqual(pricing.cache_read_usd_per_million, 0.25)

    def test_unknown_key_and_duplicate_version_are_rejected(self):
        catalog = self.make_catalog()
        with self.assertRaises(PriceTableError):
            catalog.resolve(provider="demo", model="missing", ttl_seconds=300, as_of="2026-10-03")

        data = self.make_catalog().as_mapping()
        data["prices"].append(dict(data["prices"][0]))
        with self.assertRaises(PriceTableError):
            PriceCatalog.from_mapping(data)

    def test_resolve_provider_rates_preserves_version_and_effective_date(self):
        catalog = PriceCatalog.from_mapping(
            {
                "version": 7,
                "prices": [
                    {
                        "provider": "openai",
                        "model": "fixture-model",
                        "ttl_seconds": 300,
                        "effective_from": "2026-01-01",
                        "input_usd_per_million": 2,
                        "cache_write_usd_per_million": 2.5,
                        "cache_read_usd_per_million": 0.1,
                        "output_usd_per_million": 10,
                    }
                ],
            }
        )

        rates = catalog.resolve_provider_rates(
            provider="OPENAI",
            model="fixture-model",
            ttl_seconds=300,
            as_of="2026-10-03",
        )

        self.assertIsInstance(rates, ProviderRates)
        self.assertEqual(rates.catalog_version, 7)
        self.assertEqual(rates.effective_from, date(2026, 1, 1))
        self.assertEqual(rates.ttl_seconds, 300)
        self.assertEqual(str(rates.output_usd_per_million), "10")

    def test_provider_rates_for_requires_output_price(self):
        with self.assertRaises(PriceTableError):
            self.make_catalog().resolve_provider_rates(
                provider="demo",
                model="model-a",
                ttl_seconds=300,
                as_of="2026-10-03",
            )


if __name__ == "__main__":
    unittest.main()
