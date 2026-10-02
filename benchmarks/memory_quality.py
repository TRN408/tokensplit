"""Offline search and reinjection quality benchmark for external memory."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import pathlib
import sys
from typing import Callable

if __package__ in {None, ""}:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tokensplit.context import (
    ContextBuilder,
    ContextPolicy,
    DynamicTurn,
    InMemoryExternalMemory,
    MemoryHit,
    Message,
    SearchableExternalMemoryStore,
    StaticContext,
    ToolOutput,
)


@dataclass(frozen=True)
class MemoryQualityCase:
    name: str
    query: str
    expected_markers: tuple[str, ...]
    messages: tuple[Message, ...]


@dataclass(frozen=True)
class MemoryBackendResult:
    backend: str
    cases: int
    search_marker_recall: float
    reinjection_marker_recall: float
    context_fit_rate: float
    average_reinjected_tokens: float
    average_context_tokens: float


@dataclass(frozen=True)
class MemoryQualityBenchmarkResult:
    cases: int
    backends: tuple[MemoryBackendResult, ...]


class RecentOnlyMemory(InMemoryExternalMemory):
    """Deliberately weak baseline that ignores query relevance."""

    def search(self, query: str, *, top_k: int = 3) -> tuple[MemoryHit, ...]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        references = list(self.entries)[-top_k:]
        return tuple(
            MemoryHit(reference, self.load(reference), 1.0)
            for reference in reversed(references)
        )


MemoryFactory = Callable[[], SearchableExternalMemoryStore]


def default_cases() -> tuple[MemoryQualityCase, ...]:
    return (
        MemoryQualityCase(
            name="migration-checksum",
            query="release checksum",
            expected_markers=("release-42", "sha256:deadbeef"),
            messages=(
                Message("user", "The migration must preserve production data."),
                Message(
                    "tool",
                    ToolOutput("release=release-42 checksum=sha256:deadbeef status=verified"),
                ),
            ),
        ),
        MemoryQualityCase(
            name="budget-constraint",
            query="budget limit",
            expected_markers=("budget=50000", "currency=JPY"),
            messages=(
                Message("user", "Keep the implementation within the approved budget."),
                Message("tool", ToolOutput("budget=50000 currency=JPY approval=finance-7")),
            ),
        ),
        MemoryQualityCase(
            name="rollback-reference",
            query="rollback deployment",
            expected_markers=("rollback=deploy-17", "region=ap-northeast-1"),
            messages=(
                Message("user", "The deployment must have a tested rollback path."),
                Message(
                    "tool",
                    ToolOutput("rollback=deploy-17 region=ap-northeast-1 tested=true"),
                ),
            ),
        ),
    )


def _marker_recall(text: str, markers: tuple[str, ...]) -> float:
    if not markers:
        return 1.0
    return sum(marker in text for marker in markers) / len(markers)


def _run_backend(
    name: str,
    factory: MemoryFactory,
    cases: tuple[MemoryQualityCase, ...],
) -> MemoryBackendResult:
    store = factory()
    for case in cases:
        store.save(case.messages)

    static = StaticContext(
        system_instructions="Answer using verified project context.",
        purpose="Complete the deployment safely.",
        constraints=("Do not invent identifiers.",),
    )
    policy = ContextPolicy(max_input_tokens=600, compression_threshold=0.8)
    search_scores: list[float] = []
    reinjection_scores: list[float] = []
    fit_results: list[bool] = []
    reinjected_tokens: list[int] = []
    context_tokens: list[int] = []

    for case in cases:
        hits = store.search(case.query, top_k=1)
        retrieved = tuple(message for hit in hits for message in hit.messages)
        retrieved_text = "\n".join(message.render() for message in retrieved)
        search_scores.append(_marker_recall(retrieved_text, case.expected_markers))
        reinjected_tokens.append(policy.token_counter(retrieved_text))

        builder = ContextBuilder(static, policy, memory_store=store)
        try:
            rendered = builder.build(
                DynamicTurn(
                    user_input=f"Retrieve verified details for: {case.query}",
                    state={"memory_query": case.query},
                    retrieved_memory=retrieved,
                )
            )
        except ValueError:
            fit_results.append(False)
            reinjection_scores.append(0.0)
            context_tokens.append(policy.max_input_tokens + 1)
            continue

        fit_results.append(rendered.total_tokens <= policy.max_input_tokens)
        reinjection_scores.append(_marker_recall(rendered.full_prompt, case.expected_markers))
        context_tokens.append(rendered.total_tokens)

    return MemoryBackendResult(
        backend=name,
        cases=len(cases),
        search_marker_recall=sum(search_scores) / len(search_scores),
        reinjection_marker_recall=sum(reinjection_scores) / len(reinjection_scores),
        context_fit_rate=sum(fit_results) / len(fit_results),
        average_reinjected_tokens=sum(reinjected_tokens) / len(reinjected_tokens),
        average_context_tokens=sum(context_tokens) / len(context_tokens),
    )


def run_memory_quality_benchmark(
    backend_factories: dict[str, MemoryFactory] | None = None,
    *,
    cases: tuple[MemoryQualityCase, ...] | None = None,
) -> MemoryQualityBenchmarkResult:
    selected_cases = cases or default_cases()
    if not selected_cases:
        raise ValueError("at least one benchmark case is required")
    factories = backend_factories or {
        "keyword": InMemoryExternalMemory,
        "recent_only": RecentOnlyMemory,
    }
    results = tuple(
        _run_backend(name, factory, selected_cases)
        for name, factory in factories.items()
    )
    return MemoryQualityBenchmarkResult(cases=len(selected_cases), backends=results)


def main() -> int:
    result = run_memory_quality_benchmark()
    print(json.dumps(asdict(result), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
