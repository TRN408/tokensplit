"""Offline cache-hit regression benchmark using provider-shaped fixtures."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import pathlib
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tokensplit.context import CacheMetrics, ContextBuilder, ContextPolicy, DynamicTurn, StaticContext
from tokensplit.usage import record_openai_usage


@dataclass(frozen=True)
class BenchmarkResult:
    turns: int
    stable_hit_rate: float
    mutated_hit_rate: float
    stable_prefix_changes: int
    mutated_prefix_changes: int


def run_cache_hit_regression(turns: int = 20) -> BenchmarkResult:
    if turns < 2:
        raise ValueError("turns must be at least 2")

    builder = ContextBuilder(
        StaticContext("Follow the fixed policy.", [{"name": "search"}], "Fixed project context"),
        ContextPolicy(max_input_tokens=2_000),
    )
    stable = CacheMetrics()
    mutated = CacheMetrics()

    for index in range(turns):
        rendered = builder.build(
            DynamicTurn(
                user_input=f"request {index}",
                timestamp=f"2026-10-03T09:{index:02d}:00+09:00",
                state={"turn": index},
            )
        )
        cached_tokens = 0 if index == 0 else rendered.static_prefix_tokens
        fixture = {
            "usage": {
                "input_tokens": rendered.total_tokens,
                "input_tokens_details": {"cached_tokens": cached_tokens},
                "output_tokens": 32,
            }
        }
        record_openai_usage(fixture, stable, prefix_fingerprint=rendered.prefix_fingerprint)
        mutated_fixture = {
            "usage": {
                "input_tokens": rendered.total_tokens,
                "input_tokens_details": {"cached_tokens": 0},
                "cache_write_tokens": rendered.static_prefix_tokens,
                "output_tokens": 32,
            }
        }
        record_openai_usage(mutated_fixture, mutated, prefix_fingerprint=f"{rendered.prefix_fingerprint}:{index}")

    return BenchmarkResult(
        turns=turns,
        stable_hit_rate=stable.hit_rate,
        mutated_hit_rate=mutated.hit_rate,
        stable_prefix_changes=stable.prefix_changes,
        mutated_prefix_changes=mutated.prefix_changes,
    )


def main() -> int:
    result = run_cache_hit_regression()
    print(json.dumps(asdict(result), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
