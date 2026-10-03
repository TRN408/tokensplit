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

This repository validates the policy configuration only. It does not collect
provider-specific token data and does not decide whether a particular task must
use an agent split.

## Authenticated orchestration CLI telemetry

`tokensplit.claude_cli.ClaudeCodeCliAdapter` and
`tokensplit.qwen_api.QwenApiAdapter` share the same orchestration and
comparison-log contract. They record prompt-free comparison measurements,
including retry counts, safe failure categories, retry wait time, and
transient/permanent failure counts. The read-only live producer writes
`reports/comparison.jsonl`:

```bash
python3 scripts/collect_live_task_markers.py reports/comparison.jsonl \
  --provider claude --model sonnet --count 20
```

Qwen can be selected with the OpenAI-compatible API adapter. The API key is
read from `QWEN_API_KEY` (with `DASHSCOPE_API_KEY` as a compatibility
fallback); `QWEN_API_BASE_URL` is optional and accepts either the full
`/chat/completions` URL or a `/v1` base URL. It defaults to the DashScope
international compatible endpoint:

```bash
QWEN_API_KEY=... python3 scripts/collect_live_task_markers.py reports/comparison.jsonl \
  --provider qwen --model qwen-plus --count 20
```

The Producer uses Qwen function calling with a bounded read-only toolset
(`read_file`, `list_files`, and `search_text`) so repository inspection tasks
can run without granting shell or mutation capabilities. The selected Qwen
model must support function calling.

The period report and quiet threshold gate consume that JSONL. A threshold
breach returns exit code `10` with `--fail-on-threshold` and creates a
notification JSON; quiet runs remove stale notification files:

```bash
python3 scripts/orchestration_gate.py reports/comparison.jsonl \
  --period day --max-permanent-failure-rate 0.05 \
  --max-retry-wait-seconds 60 --fail-on-threshold
```

GitHub Actions uses [`orchestration-cli-producer.yml`](.github/workflows/orchestration-cli-producer.yml)
to upload the `orchestration-comparison` artifact. The
[`orchestration-gate.yml`](.github/workflows/orchestration-gate.yml) workflow
downloads that artifact from the triggering run and emits a notification
artifact only when a threshold is exceeded. Configure either
`ANTHROPIC_API_KEY` or `CLAUDE_CODE_OAUTH_TOKEN` for Claude, or `QWEN_API_KEY`
for Qwen, as a repository secret before enabling the scheduled producer. The
manual producer input `provider` selects the adapter; `model` is optional and
uses the provider default when blank.

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
