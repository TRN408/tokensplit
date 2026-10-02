"""Task classification, model routing, quality evaluation, and route telemetry.

The router is intentionally provider-neutral.  It resolves a model before an
adapter starts work and never treats the parent model as an implicit default.
This prevents an expensive planning model from leaking into routine child
tasks when the caller omitted ``model``.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
from typing import Any, Literal


class RoutingError(ValueError):
    """Raised when a routing policy or route request is invalid."""


TaskType = Literal[
    "decision",
    "planning",
    "final_review",
    "routine_execution",
    "search",
    "transformation",
    "integrated",
]
ModelTier = Literal["reasoning", "execution"]

_TASK_TYPES = frozenset(
    {
        "decision",
        "planning",
        "final_review",
        "routine_execution",
        "search",
        "transformation",
        "integrated",
    }
)


def _decimal(value: Any, name: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise RoutingError(f"{name} must be a finite non-negative decimal") from exc
    if not result.is_finite() or result < 0:
        raise RoutingError(f"{name} must be a finite non-negative decimal")
    return result


def _optional_non_negative_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RoutingError(f"{name} must be a non-negative integer or None")
    return value


@dataclass(frozen=True)
class ModelProfile:
    """A model that can be selected by a :class:`ModelRouter`.

    Prices are optional: the provider may report the authoritative cost at
    runtime.  When prices are present, :class:`RoutingLog` can estimate a cost
    from reported input/output tokens without inventing missing values.
    """

    model: str
    tier: ModelTier
    input_usd_per_million: Decimal | None = None
    output_usd_per_million: Decimal | None = None
    available: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise RoutingError("model must be a non-empty string")
        if self.tier not in {"reasoning", "execution"}:
            raise RoutingError("tier must be 'reasoning' or 'execution'")
        if not isinstance(self.available, bool):
            raise RoutingError("available must be a boolean")
        for name in ("input_usd_per_million", "output_usd_per_million"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _decimal(value, name))


@dataclass(frozen=True)
class RoutingPolicy:
    """Default model assignment by task family.

    ``allow_parent_model_fallback`` is opt-in.  Keeping it disabled is the
    safety property that prevents unspecified tasks from inheriting a costly
    parent model.
    """

    decision_model: str
    execution_model: str
    fallback_model: str | None = None
    allow_parent_model_fallback: bool = False
    task_models: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        for name in ("decision_model", "execution_model", "fallback_model"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise RoutingError(f"{name} must be a non-empty string when provided")
        if not isinstance(self.allow_parent_model_fallback, bool):
            raise RoutingError("allow_parent_model_fallback must be a boolean")
        if self.task_models is not None:
            if not isinstance(self.task_models, Mapping):
                raise RoutingError("task_models must be a mapping")
            for task_type, model in self.task_models.items():
                if task_type not in _TASK_TYPES:
                    raise RoutingError(f"unsupported task type: {task_type}")
                if not isinstance(model, str) or not model.strip():
                    raise RoutingError("task_models values must be non-empty strings")

    def preferred_model(self, task_type: TaskType) -> str:
        if self.task_models and task_type in self.task_models:
            return self.task_models[task_type]
        if task_type in {"decision", "planning", "final_review"}:
            return self.decision_model
        return self.execution_model


class TaskClassifier:
    """Classify a task from an explicit hint or a conservative text heuristic."""

    _ALIASES = {
        "judgment": "decision",
        "判断": "decision",
        "plan": "planning",
        "計画": "planning",
        "review": "final_review",
        "レビュー": "final_review",
        "routine": "routine_execution",
        "execution": "routine_execution",
        "定型実行": "routine_execution",
        "search": "search",
        "research": "search",
        "調査": "search",
        "検索": "search",
        "transform": "transformation",
        "convert": "transformation",
        "変換": "transformation",
    }

    _KEYWORDS: tuple[tuple[TaskType, tuple[str, ...]], ...] = (
        ("final_review", ("final review", "code review", "監査", "最終レビュー", "レビュー")),
        ("decision", ("trade-off", "tradeoff", "decide", "判断", "方式選択", "採否")),
        ("planning", ("plan", "planning", "設計", "計画", "architecture", "adr")),
        ("search", ("search", "research", "grep", "find", "調査", "検索", "根拠")),
        ("transformation", ("transform", "convert", "format", "serialize", "変換", "整形")),
        ("routine_execution", ("routine", "batch", "mechanical", "定型", "反復", "実行")),
    )

    def classify(self, prompt: str = "", task_type: str | None = None) -> TaskType:
        if not isinstance(prompt, str):
            raise RoutingError("prompt must be a string")
        if task_type is not None:
            if not isinstance(task_type, str) or not task_type.strip():
                raise RoutingError("task_type must be a non-empty string when provided")
            normalized = self._ALIASES.get(task_type.strip().casefold(), task_type.strip().casefold())
            if normalized not in _TASK_TYPES or normalized == "integrated":
                raise RoutingError(f"unsupported task type: {task_type}")
            return normalized  # type: ignore[return-value]

        text = prompt.casefold()
        for candidate, keywords in self._KEYWORDS:
            if any(keyword.casefold() in text for keyword in keywords):
                return candidate
        # Unknown tasks default to the cheap, bounded execution route.
        return "routine_execution"


@dataclass(frozen=True)
class RouteDecision:
    request_id: str
    task_type: TaskType
    requested_model: str | None
    parent_model: str | None
    preferred_model: str
    selected_model: str
    selection_reason: str
    fallback_used: bool
    model_tier: ModelTier | None
    input_usd_per_million: Decimal | None = None
    output_usd_per_million: Decimal | None = None


class ModelRouter:
    """Resolve a task to an explicit model before execution starts."""

    def __init__(
        self,
        profiles: Sequence[ModelProfile],
        policy: RoutingPolicy,
        *,
        classifier: TaskClassifier | None = None,
    ) -> None:
        if isinstance(profiles, (str, bytes, bytearray)) or not isinstance(profiles, Sequence):
            raise RoutingError("profiles must be a sequence of ModelProfile")
        profile_map: dict[str, ModelProfile] = {}
        for profile in profiles:
            if not isinstance(profile, ModelProfile):
                raise RoutingError("profiles must contain only ModelProfile values")
            if profile.model in profile_map:
                raise RoutingError(f"duplicate model profile: {profile.model}")
            profile_map[profile.model] = profile
        if not profile_map:
            raise RoutingError("at least one model profile is required")
        if not isinstance(policy, RoutingPolicy):
            raise RoutingError("policy must be a RoutingPolicy")
        self._profiles = profile_map
        self.policy = policy
        self.classifier = classifier or TaskClassifier()

    @property
    def profiles(self) -> tuple[ModelProfile, ...]:
        return tuple(self._profiles.values())

    def _available(self, model: str | None) -> ModelProfile | None:
        if model is None:
            return None
        profile = self._profiles.get(model)
        return profile if profile and profile.available else None

    def route(
        self,
        *,
        request_id: str,
        prompt: str = "",
        task_type: str | None = None,
        requested_model: str | None = None,
        parent_model: str | None = None,
    ) -> RouteDecision:
        if not isinstance(request_id, str) or not request_id.strip():
            raise RoutingError("request_id must be a non-empty string")
        for name, value in (("requested_model", requested_model), ("parent_model", parent_model)):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise RoutingError(f"{name} must be a non-empty string when provided")
        resolved_type = self.classifier.classify(prompt, task_type)
        preferred = self.policy.preferred_model(resolved_type)

        if requested_model is not None:
            explicit = self._available(requested_model)
            if explicit is not None:
                return self._decision(
                    request_id,
                    resolved_type,
                    requested_model,
                    parent_model,
                    preferred,
                    explicit,
                    "explicit_model",
                    False,
                )

        preferred_profile = self._available(preferred)
        if preferred_profile is not None:
            return self._decision(
                request_id,
                resolved_type,
                requested_model,
                parent_model,
                preferred,
                preferred_profile,
                "task_type_default" if requested_model is None else "requested_model_unavailable",
                requested_model is not None,
            )

        fallback = self._available(self.policy.fallback_model)
        if fallback is not None:
            return self._decision(
                request_id,
                resolved_type,
                requested_model,
                parent_model,
                preferred,
                fallback,
                "fallback_model",
                True,
            )

        if self.policy.allow_parent_model_fallback and self._available(parent_model) is not None:
            parent_profile = self._profiles[parent_model]  # type: ignore[index]
            return self._decision(
                request_id,
                resolved_type,
                requested_model,
                parent_model,
                preferred,
                parent_profile,
                "parent_model_fallback",
                True,
            )
        raise RoutingError(
            f"no available model for task_type={resolved_type!r}; "
            "configure a task model or fallback_model"
        )

    @staticmethod
    def _decision(
        request_id: str,
        task_type: TaskType,
        requested_model: str | None,
        parent_model: str | None,
        preferred: str,
        profile: ModelProfile,
        reason: str,
        fallback_used: bool,
    ) -> RouteDecision:
        return RouteDecision(
            request_id=request_id,
            task_type=task_type,
            requested_model=requested_model,
            parent_model=parent_model,
            preferred_model=preferred,
            selected_model=profile.model,
            selection_reason=reason,
            fallback_used=fallback_used,
            model_tier=profile.tier,
            input_usd_per_million=profile.input_usd_per_million,
            output_usd_per_million=profile.output_usd_per_million,
        )


@dataclass(frozen=True)
class QualityAssessment:
    score: float | None
    passed: bool | None
    reason: str


class QualityEvaluator:
    """Turn available task evidence into a comparable quality signal."""

    def __init__(self, *, pass_threshold: float = 0.8) -> None:
        if isinstance(pass_threshold, bool) or not isinstance(pass_threshold, (int, float)):
            raise RoutingError("pass_threshold must be a number between 0 and 1")
        if not 0.0 <= pass_threshold <= 1.0:
            raise RoutingError("pass_threshold must be a number between 0 and 1")
        self.pass_threshold = float(pass_threshold)

    def evaluate(
        self,
        *,
        quality_score: float | None = None,
        quality_passed: bool | None = None,
        task_success: bool | None = None,
        required_markers: int = 0,
        retained_markers: int | None = None,
    ) -> QualityAssessment:
        if quality_score is not None:
            if isinstance(quality_score, bool) or not isinstance(quality_score, (int, float)):
                raise RoutingError("quality_score must be a number between 0 and 1")
            if not 0.0 <= quality_score <= 1.0:
                raise RoutingError("quality_score must be a number between 0 and 1")
            return QualityAssessment(
                float(quality_score),
                float(quality_score) >= self.pass_threshold,
                "quality_score",
            )
        if quality_passed is not None:
            if not isinstance(quality_passed, bool):
                raise RoutingError("quality_passed must be a boolean or None")
            return QualityAssessment(1.0 if quality_passed else 0.0, quality_passed, "quality_passed")
        _optional_non_negative_int(required_markers, "required_markers")
        _optional_non_negative_int(retained_markers, "retained_markers")
        if required_markers and retained_markers is not None:
            score = min(1.0, retained_markers / required_markers)
            return QualityAssessment(score, retained_markers >= required_markers, "required_markers")
        if task_success is not None:
            if not isinstance(task_success, bool):
                raise RoutingError("task_success must be a boolean or None")
            return QualityAssessment(1.0 if task_success else 0.0, task_success, "task_success")
        return QualityAssessment(None, None, "unknown")


@dataclass(frozen=True)
class RoutingRecord:
    """Privacy-safe record of one actual model execution."""

    request_id: str
    task_type: str
    selected_model: str
    selection_reason: str
    fallback_used: bool
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: Decimal | None = None
    quality_score: float | None = None
    quality_passed: bool | None = None
    quality_reason: str = "unknown"

    def __post_init__(self) -> None:
        for name in ("request_id", "task_type", "selected_model", "selection_reason"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise RoutingError(f"{name} must be a non-empty string")
        if not isinstance(self.fallback_used, bool):
            raise RoutingError("fallback_used must be a boolean")
        for name in ("input_tokens", "output_tokens"):
            _optional_non_negative_int(getattr(self, name), name)
        if self.cost_usd is not None:
            object.__setattr__(self, "cost_usd", _decimal(self.cost_usd, "cost_usd"))
        if self.quality_score is not None and (
            isinstance(self.quality_score, bool)
            or not isinstance(self.quality_score, (int, float))
            or not 0.0 <= self.quality_score <= 1.0
        ):
            raise RoutingError("quality_score must be a number between 0 and 1")
        if self.quality_passed is not None and not isinstance(self.quality_passed, bool):
            raise RoutingError("quality_passed must be a boolean or None")


@dataclass(frozen=True)
class RoutingModelSummary:
    selected_model: str
    runs: int
    known_cost_runs: int
    total_cost_usd: Decimal | None
    average_quality: float | None
    quality_pass_rate: float | None


class RoutingLog:
    """Store actual route outcomes and compare model cost/quality."""

    def __init__(self, *, quality_evaluator: QualityEvaluator | None = None) -> None:
        self._records: list[RoutingRecord] = []
        self.quality_evaluator = quality_evaluator or QualityEvaluator()

    def record(self, record: RoutingRecord) -> None:
        if not isinstance(record, RoutingRecord):
            raise RoutingError("record must be a RoutingRecord")
        self._records.append(record)

    def record_execution(
        self,
        route: RouteDecision,
        *,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cost_usd: Decimal | None = None,
        quality_score: float | None = None,
        quality_passed: bool | None = None,
        task_success: bool | None = None,
        required_markers: int = 0,
        retained_markers: int | None = None,
    ) -> QualityAssessment:
        if not isinstance(route, RouteDecision):
            raise RoutingError("route must be a RouteDecision")
        _optional_non_negative_int(input_tokens, "input_tokens")
        _optional_non_negative_int(output_tokens, "output_tokens")
        assessment = self.quality_evaluator.evaluate(
            quality_score=quality_score,
            quality_passed=quality_passed,
            task_success=task_success,
            required_markers=required_markers,
            retained_markers=retained_markers,
        )
        actual_cost = cost_usd
        if actual_cost is None and input_tokens is not None and output_tokens is not None:
            if route.input_usd_per_million is not None and route.output_usd_per_million is not None:
                actual_cost = (
                    Decimal(input_tokens) * route.input_usd_per_million / Decimal(1_000_000)
                    + Decimal(output_tokens) * route.output_usd_per_million / Decimal(1_000_000)
                )
        self.record(
            RoutingRecord(
                request_id=route.request_id,
                task_type=route.task_type,
                selected_model=route.selected_model,
                selection_reason=route.selection_reason,
                fallback_used=route.fallback_used,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=actual_cost,
                quality_score=assessment.score,
                quality_passed=assessment.passed,
                quality_reason=assessment.reason,
            )
        )
        return assessment

    def record_outcome(
        self,
        *,
        request_id: str,
        task_type: str,
        selected_model: str,
        selection_reason: str,
        fallback_used: bool = False,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cost_usd: Decimal | None = None,
        quality_score: float | None = None,
        quality_passed: bool | None = None,
        task_success: bool | None = None,
        required_markers: int = 0,
        retained_markers: int | None = None,
    ) -> QualityAssessment:
        """Record an adapter result when no :class:`RouteDecision` is retained."""

        assessment = self.quality_evaluator.evaluate(
            quality_score=quality_score,
            quality_passed=quality_passed,
            task_success=task_success,
            required_markers=required_markers,
            retained_markers=retained_markers,
        )
        self.record(
            RoutingRecord(
                request_id=request_id,
                task_type=task_type,
                selected_model=selected_model,
                selection_reason=selection_reason,
                fallback_used=fallback_used,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
                quality_score=assessment.score,
                quality_passed=assessment.passed,
                quality_reason=assessment.reason,
            )
        )
        return assessment

    def records(self) -> tuple[RoutingRecord, ...]:
        return tuple(self._records)

    def compare_models(self) -> tuple[RoutingModelSummary, ...]:
        grouped: dict[str, list[RoutingRecord]] = defaultdict(list)
        for record in self._records:
            grouped[record.selected_model].append(record)
        summaries: list[RoutingModelSummary] = []
        for model, records in sorted(grouped.items()):
            costs = tuple(record.cost_usd for record in records)
            qualities = tuple(record.quality_score for record in records)
            passed = tuple(record.quality_passed for record in records)
            summaries.append(
                RoutingModelSummary(
                    selected_model=model,
                    runs=len(records),
                    known_cost_runs=sum(value is not None for value in costs),
                    total_cost_usd=(
                        sum((value for value in costs if value is not None), Decimal(0))
                        if costs and all(value is not None for value in costs)
                        else None
                    ),
                    average_quality=(
                        sum(value for value in qualities if value is not None) / len(qualities)
                        if qualities and all(value is not None for value in qualities)
                        else None
                    ),
                    quality_pass_rate=(
                        sum(value is True for value in passed) / len(passed)
                        if passed and all(value is not None for value in passed)
                        else None
                    ),
                )
            )
        return tuple(summaries)

    def render_jsonl(self) -> str:
        lines: list[str] = []
        for record in self._records:
            payload = asdict(record)
            if payload["cost_usd"] is not None:
                payload["cost_usd"] = str(payload["cost_usd"])
            lines.append(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return "\n".join(lines)

    def append_jsonl(
        self,
        path: str | Path,
        records: Sequence[RoutingRecord] | None = None,
    ) -> int:
        """Append prompt-free records to an operational JSONL sink.

        Callers can pass only the records from the latest dispatch to avoid
        duplicating an already persisted in-memory log.
        """

        target = Path(path)
        selected = tuple(self._records if records is None else records)
        if any(not isinstance(record, RoutingRecord) for record in selected):
            raise RoutingError("records must contain only RoutingRecord values")
        if not selected:
            return 0
        target.parent.mkdir(parents=True, exist_ok=True)
        needs_separator = (
            target.exists()
            and target.stat().st_size > 0
            and not target.read_bytes().endswith(b"\n")
        )
        with target.open("a", encoding="utf-8") as handle:
            if needs_separator:
                handle.write("\n")
            for record in selected:
                payload = asdict(record)
                if payload["cost_usd"] is not None:
                    payload["cost_usd"] = str(payload["cost_usd"])
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
        return len(selected)

    @classmethod
    def from_jsonl(cls, text: str) -> "RoutingLog":
        if not isinstance(text, str):
            raise RoutingError("routing JSONL must be text")
        log = cls()
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RoutingError(f"invalid routing JSONL at line {line_number}") from exc
            if not isinstance(payload, dict):
                raise RoutingError(f"routing JSONL line {line_number} must be an object")
            if payload.get("cost_usd") is not None:
                payload["cost_usd"] = _decimal(payload["cost_usd"], "cost_usd")
            try:
                log.record(RoutingRecord(**payload))
            except TypeError as exc:
                raise RoutingError(f"invalid routing fields at line {line_number}") from exc
        return log


__all__ = [
    "ModelProfile",
    "ModelRouter",
    "QualityAssessment",
    "QualityEvaluator",
    "RouteDecision",
    "RoutingError",
    "RoutingLog",
    "RoutingModelSummary",
    "RoutingPolicy",
    "RoutingRecord",
    "TaskClassifier",
    "TaskType",
]
