# Agent Token Split Policy

A small, tool-agnostic policy for deciding when splitting work across agents is
likely to reduce token usage without lowering quality.

このプロジェクトは、subagent・agent分割を採用する条件を、共有CIや特定の
コーディングエージェントから切り離して管理するためのオープンソースの
ポリシーです。

## Policy

The default policy is stored in [`policy.json`](policy.json). It requires:

- at least three quality-passed single-agent measurements;
- a median of at least 20,000 tokens for the single-agent baseline;
- an expected saving of at least 25% before splitting; and
- a measured saving of at least 15% after splitting.

Unknown or missing token measurements must not be counted as savings evidence.

The thresholds are defaults, not universal truths. Projects may create their own
policy file and explain why they changed a threshold.

## Who this helps

This project is for teams and maintainers who use multiple AI agents and want
to decide whether splitting work is worthwhile based on repeatable measurements
rather than intuition.

It is especially useful for:

- development teams using agents for research, implementation, and review;
- developers building agent orchestration or multi-agent workflows;
- platform teams that need a shared rule for evaluating agent cost;
- researchers comparing single-agent and split-agent workflows.

It is less useful for short, one-off tasks or workflows that cannot measure token
usage and quality consistently.

This is a policy and validation baseline, not an automatic token optimizer. It
does not split tasks, collect provider-specific usage data, or claim that a split
is always cheaper. Its purpose is to define the evidence required before and
after adopting an agent split.

## Validate

```bash
python3 scripts/validate_policy.py
python3 -m unittest discover -s tests -p 'test_*.py'
```

The repository does not collect provider-specific token data or decide whether
a particular task must use an agent split. The context builder below is a
provider-neutral layout and measurement component; it does not make provider
API calls.

## Stable-prefix context builder

The `tokensplit.context` module provides a provider-neutral prompt layout for
agents that need prompt caching. System instructions, canonicalized tool
definitions, and other fixed context are rendered once at the beginning.
Timestamp, state, conversation history, and the current user input are always
rendered after that prefix.

Compression is threshold-driven: the builder keeps recent messages verbatim and
folds older messages only when the remaining input budget falls below the
configured threshold. It does not rewrite a summary on every turn. The static
prefix can pin `purpose` and `constraints`, while the current `state` is kept as
structured dynamic data. Every removed history batch is saved through an
`ExternalMemoryStore`; the active prompt contains only a bounded structured
index of references, so old tool output is not silently discarded or replayed.
It never rewrites the static prefix. Use the provider's tokenizer through
`ContextPolicy(token_counter=...)` in production, then pass reported
`cached_input_tokens` and `cache_write_tokens` to `CacheMetrics.observe()` to
measure actual cache behavior.

## Prompt-cache economics

`tokensplit.cache` keeps normal input, cache write, and cache read volumes
separate and estimates costs from caller-supplied model prices. It also models
TTL expiry and reports the first repetition at which caching is no more
expensive than sending the prefix at the normal input rate.

```python
from tokensplit import CachePricing, estimate_cache_economics

pricing = CachePricing.from_multipliers(
    "example-model",
    input_usd_per_million=10.0,
    cache_write_multiplier=1.25,
    cache_read_multiplier=0.1,
    ttl_seconds=300,
)
estimate = estimate_cache_economics(
    prefix_tokens=4_500,
    normal_input_tokens_per_request=300,
    repetitions=5,
    pricing=pricing,
    request_interval_seconds=60,
)
print(estimate.cache_hit_rate, estimate.cached_cost_usd, estimate.break_even_repetitions)
```

For observed provider usage, call `metrics.cost_summary(pricing)`. A single
request whose cache write costs more than the normal input path is marked by
`short_one_off` and `cache_enabled_but_expensive`. Missing provider write
fields remain marked as incomplete by the usage adapter instead of being
treated as verified zero-volume writes.

`tokensplit.usage` converts provider usage responses without making API calls.
It supports OpenAI Responses/Chat Completions and Anthropic Messages/usage
reports, including cached input and cache-write fields. Unknown providers or
malformed payloads fail explicitly instead of being counted as evidence.

`StreamingUsageAdapter` collects OpenAI final usage chunks or Responses
completion events and Anthropic `message_start`/`message_delta` events, then
records one normalized observation. `ProviderRates` and `calculate_cost()`
convert that observation to a provider/model-specific USD breakdown. If a
provider does not report cache-write tokens, the result is marked incomplete
instead of presenting an exact bill.

```python
from tokensplit.context import ContextBuilder, DynamicTurn, StaticContext

builder = ContextBuilder(StaticContext(
    system_instructions="Follow the safety policy.",
    tool_definitions=[{"name": "search", "parameters": {"type": "object"}}],
    fixed_context="Project rules",
    purpose="Complete the migration safely.",
    constraints=("Do not delete production data.",),
))
request_text = builder.build(DynamicTurn(
    timestamp="2026-10-03T09:00:00+09:00",
    state={"phase": "research"},
    user_input="Find the relevant result.",
)).full_prompt
```

## Tool output gate

`tokensplit.context.ToolOutput` provides a dependency-free formatting boundary
for large logs, files, and search results in the existing message renderer.
`GateLimits` applies character and item limits, extracts lines relevant to an
optional query, and includes warnings in the returned text whenever anything
was omitted. If extraction fails, it falls back to a bounded head/tail excerpt
and records the failure instead of silently returning incomplete output.

## Cache-hit regression benchmark

Run the offline fixture benchmark with:

```bash
python3 benchmarks/cache_hit_regression.py
```

It compares a stable prefix with a deliberately mutated prefix. The benchmark
uses provider-shaped usage fixtures, so it validates the accounting path while
remaining deterministic and network-free.

## Local CI

The repository includes the localCI integration used before a push. The product
commands are declared in [`.agent-ci-policy.yml`](.agent-ci-policy.yml), and
the command inventory used for act coverage classification is stored in
[`.localci/product-commands.json`](.localci/product-commands.json).

The product stages are:

- `install`: create the dependency-free project virtual environment;
- `test`: validate the policy and run the unit tests;
- `typecheck`: compile-check the Python sources;
- `build`: create `dist/agent-token-split-policy.zip`;
- `full_ci`: run test, typecheck, and build together.

Enable the pre-push hook once:

```bash
scripts/install_local_hooks.sh
```

Run the local product checks directly, or run the pinned act backend:

```bash
scripts/run_ci_local_quiet.sh
AGENT_CI_BACKEND=act scripts/run_ci_local_quiet.sh
```

The act backend is external and pinned to `v0.2.89` in
[`.localci/backend.lock`](.localci/backend.lock). It requires Docker and a
locally installed `act` binary.

## Scope

This project deliberately does not define:

- a required agent provider or SDK;
- a benchmark format for any particular model.

## License

MIT. See [`LICENSE`](LICENSE).
