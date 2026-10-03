#!/usr/bin/env python3
"""Collect task-specific acceptance telemetry through the authenticated adapter."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import os
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tokensplit.claude_cli import ClaudeCliConfig, ClaudeCliRunner, ClaudeCodeCliAdapter
from tokensplit.orchestration import AgentControlPolicy, AgentController, AgentRequest
from tokensplit.output_gate import GateLimits
from tokensplit.qwen_api import QwenApiAdapter, QwenApiConfig, QwenToolCallingRunner


@dataclass(frozen=True)
class LiveTask:
    task_id: str
    target: str
    check: str

    @property
    def markers(self) -> tuple[str, ...]:
        return (
            f"TASK_ID: {self.task_id}",
            f"TARGET: {self.target}",
            "TASK_STATUS: PASS",
        )

    @property
    def prompt(self) -> str:
        return (
            "Perform a read-only verification task in the current repository. Do "
            "not edit, create, delete, or execute commands that mutate files. "
            f"Inspect {self.target} and its related tests, then {self.check} "
            "Report concise evidence. Finish with exactly the following three "
            "lines, in this order, each on its own line. Do not use semicolons, "
            "extra punctuation, or a code fence in this three-line contract. "
            "Copy the labels and values verbatim. Set TASK_STATUS to PASS only "
            "when the check passes; otherwise use TASK_STATUS: FAIL.\n\n"
            f"TASK_ID: {self.task_id}\n"
            f"TARGET: {self.target}\n"
            "TASK_STATUS: PASS"
        )


TASKS = (
    LiveTask("task-01-output-gate", "tokensplit/output_gate.py", "verify that priority lines are retained before routine lines when a report is truncated"),
    LiveTask("task-02-cli-adapter", "tokensplit/claude_cli.py", "verify that the adapter exposes gated text and records gate token metrics"),
    LiveTask("task-03-orchestration", "tokensplit/orchestration.py", "verify that comparison logs replay without prompt or response text"),
    LiveTask("task-04-context", "tokensplit/context.py", "verify that context compression preserves required marker lines"),
    LiveTask("task-05-pruning", "tokensplit/pruning.py", "verify that pruning reports important-marker retention"),
    LiveTask("task-06-routing", "tokensplit/routing.py", "verify that routing quality evidence distinguishes unknown values from failures"),
    LiveTask("task-07-monitor", "tokensplit/monitor.py", "verify that missing cache fields are not treated as zero"),
    LiveTask("task-08-importer", "tokensplit/importer.py", "verify that malformed usage rows are diagnosed without exposing prompt text"),
    LiveTask("task-09-service-guides", "tokensplit/service_guides.py", "verify that unknown services produce a bounded research fallback"),
    LiveTask("task-10-tool-adapters", "tokensplit/tool_adapters.py", "verify that cursor expiry is surfaced as a recoverable adapter error"),
    LiveTask("task-11-opensearch", "tokensplit/opensearch_client.py", "verify that search-after pagination keeps the public snapshot stable"),
    LiveTask("task-12-langfuse", "tokensplit/langfuse.py", "verify that observation pagination deduplicates records by id"),
    LiveTask("task-13-report", "tokensplit/report.py", "verify that usage reports omit prompt contents"),
    LiveTask("task-14-pricing", "tokensplit/pricing.py", "verify that stale or missing price coverage is rejected"),
    LiveTask("task-15-usage", "tokensplit/usage.py", "verify that provider usage normalization preserves cache-write unknowns"),
    LiveTask("task-16-streaming", "tokensplit/streaming.py", "verify that streaming usage totals are normalized without guessing missing fields"),
    LiveTask("task-17-cache", "tokensplit/cache.py", "verify that cache cost estimates distinguish read and write pricing"),
    LiveTask("task-18-persistent-memory", "tokensplit/persistent_memory.py", "verify that persistent memory recovery reports partial corruption"),
    LiveTask("task-19-quality-tests", "tests/test_output_quality.py", "verify that the benchmark separates reduction from task success and retention"),
    LiveTask("task-20-documentation", "README.md", "verify that the documented calibration command matches the available script"),
    LiveTask("task-21-product-ci", "scripts/product_ci.py", "verify that the full local CI path includes tests, typecheck, and build"),
    LiveTask("task-22-policy", "policy.json", "verify that unknown measurements are excluded from evidence"),
)


def collect(
    output: Path,
    *,
    provider: str,
    model: str,
    count: int,
    max_chars: int,
    max_items: int,
    max_output_tokens: int,
) -> dict[str, Any]:
    if count < 20 or count > len(TASKS):
        raise ValueError(f"count must be between 20 and {len(TASKS)}")
    limits = GateLimits.for_purpose(
        "subagent",
        max_chars=max_chars,
        max_items=max_items,
        max_output_tokens=max_output_tokens,
    )
    controller = AgentController(
        AgentControlPolicy(max_subagents=1),
        root_model=model,
    )
    if provider == "claude":
        runner = ClaudeCliRunner(
            ClaudeCliConfig(
                tools=None,
                extra_args=("--permission-mode", "plan"),
                output_limits=limits,
                max_retries=2,
                timeout_seconds=180,
            )
        )
        adapter = ClaudeCodeCliAdapter(runner=runner, controller=controller)
    elif provider == "qwen":
        runner = QwenToolCallingRunner(
            QwenApiConfig(
                output_limits=limits,
                max_retries=1,
                timeout_seconds=60,
                working_directory=Path.cwd(),
            ),
            max_tool_rounds=4,
        )
        adapter = QwenApiAdapter(runner=runner, controller=controller)
    else:
        raise ValueError(f"unsupported provider: {provider}")
    failures: list[dict[str, str]] = []
    for task in TASKS[:count]:
        try:
            result = adapter.dispatch(
                [
                    AgentRequest(
                        task.task_id,
                        prompt=task.prompt,
                        required_markers=task.markers,
                    )
                ]
            )
            run = result.runs[0]
            if run.baseline_task_success is not True:
                failures.append({"task_id": task.task_id, "reason": "baseline_contract_failed"})
            elif run.task_success is not True:
                failures.append({"task_id": task.task_id, "reason": "formatted_contract_failed"})
        except Exception as exc:  # noqa: BLE001 - collect a safe category, not exception text
            failures.append({"task_id": task.task_id, "reason": type(exc).__name__})

    records = adapter.comparison_log.render_jsonl()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(records + ("\n" if records else ""), encoding="utf-8")
    metadata = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "source": f"authenticated_{provider}_adapter",
        "provider": provider,
        "model": model,
        "requested_count": count,
        "record_count": len(adapter.comparison_log.records()),
        "failure_count": len(failures),
        "failures": failures,
        "gate_limits": asdict(limits),
        "quality_contract": "task-specific required_markers evaluated before and after formatting",
        "privacy": "prompts, responses, session identifiers, and stderr omitted",
    }
    metadata_path = output.with_suffix(output.suffix + ".meta.json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise RuntimeError(f"{len(failures)} task contracts failed; see {metadata_path}")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--provider", choices=("claude", "qwen"), default=os.environ.get("TOKENSPLIT_PROVIDER", "claude"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument("--max-items", type=int, default=24)
    parser.add_argument("--max-output-tokens", type=int, default=300)
    args = parser.parse_args()
    model = args.model or os.environ.get(
        "TOKENSPLIT_CLAUDE_MODEL" if args.provider == "claude" else "TOKENSPLIT_QWEN_MODEL",
        "sonnet" if args.provider == "claude" else "qwen-plus",
    )
    values = vars(args)
    values["model"] = model
    try:
        print(json.dumps(collect(**values), ensure_ascii=False, sort_keys=True))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"live task-marker collection failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
