import unittest
from decimal import Decimal

from tokensplit import (
    AgentBudget,
    AgentControlPolicy,
    AgentController,
    AgentRequest,
    ComparisonLog,
    ComparisonMeasurement,
    ExecutionDiagnostic,
)
from tokensplit.orchestration import OrchestrationError


class AgentControllerTests(unittest.TestCase):
    def test_runtime_budget_rejects_the_call_that_would_cross_the_limit(self):
        budget = AgentBudget(max_tool_calls=2)
        self.assertEqual(budget.record_tool_call(), 1)
        self.assertEqual(budget.remaining, 1)
        with self.assertRaises(OrchestrationError):
            budget.record_tool_call(2)
        self.assertEqual(budget.tool_calls, 1)

    def test_low_independence_tasks_are_integrated_and_model_is_inherited(self):
        controller = AgentController(
            AgentControlPolicy(min_independence=0.6),
            root_model="parent-model",
        )
        plan = controller.plan(
            [
                AgentRequest("same-file-a", independence=0.2),
                AgentRequest("independent", independence=0.9),
                AgentRequest("same-file-b", independence=0.4),
            ]
        )

        self.assertEqual(plan.integrated_groups, (("same-file-a", "same-file-b"),))
        self.assertEqual(plan.integrated_task_ids, ("same-file-a", "same-file-b"))
        self.assertEqual(plan.launched_task_ids, ("independent",))
        decisions = {decision.task_id: decision for decision in plan.decisions}
        self.assertEqual(decisions["independent"].model, "parent-model")
        self.assertEqual(decisions["same-file-a"].model, "parent-model")
        self.assertEqual(plan.child_agent_count, 1)

    def test_explicit_child_model_is_preserved(self):
        plan = AgentController(root_model="parent-model").plan(
            [AgentRequest("review", model="review-model")]
        )
        self.assertEqual(plan.decisions[0].model, "review-model")

    def test_excess_agents_are_serialized_into_bounded_batches(self):
        controller = AgentController(
            AgentControlPolicy(max_subagents=2, overflow_strategy="serialize"),
            root_model="model-a",
        )
        plan = controller.plan([AgentRequest(f"task-{index}") for index in range(5)])

        self.assertEqual(
            plan.execution_batches,
            (("task-0", "task-1"), ("task-2", "task-3"), ("task-4",)),
        )
        self.assertEqual(plan.launched_task_ids, ("task-0", "task-1"))
        self.assertEqual(
            plan.serialized_task_ids,
            ("task-2", "task-3", "task-4"),
        )
        self.assertEqual(plan.child_agent_count, 2)
        self.assertEqual(plan.requested_child_agent_count, 5)
        reasons = {decision.task_id: decision.reason for decision in plan.decisions}
        self.assertEqual(reasons["task-2"], "max_subagents_serialized")

    def test_excess_agents_can_be_rejected(self):
        plan = AgentController(
            AgentControlPolicy(max_subagents=1, overflow_strategy="reject")
        ).plan([AgentRequest("first"), AgentRequest("second")])

        self.assertEqual(plan.launched_task_ids, ("first",))
        self.assertEqual(plan.rejected_task_ids, ("second",))
        self.assertEqual(plan.execution_batches, (("first",),))

    def test_depth_and_tool_call_limits_are_rejections(self):
        controller = AgentController(
            AgentControlPolicy(max_depth=1, max_tool_calls_per_agent=2),
            root_model="model-a",
        )
        plan = controller.plan(
            [
                AgentRequest("nested", depth=2),
                AgentRequest("too-many-tools", estimated_tool_calls=3),
                AgentRequest("allowed", estimated_tool_calls=2),
            ]
        )
        decisions = {decision.task_id: decision for decision in plan.decisions}
        self.assertEqual(decisions["nested"].reason, "max_depth_exceeded")
        self.assertEqual(decisions["too-many-tools"].reason, "max_tool_calls_exceeded")
        self.assertEqual(decisions["allowed"].action, "launch")

    def test_invalid_requests_are_rejected_before_planning(self):
        with self.assertRaises(OrchestrationError):
            AgentRequest("", independence=1.0)
        with self.assertRaises(OrchestrationError):
            AgentController().plan([AgentRequest("duplicate"), AgentRequest("duplicate")])


class ComparisonLogTests(unittest.TestCase):
    def test_execution_diagnostics_round_trip_and_aggregate_failures(self):
        log = ComparisonLog()
        log.record(
            ComparisonMeasurement(
                run_id="single-retry",
                mode="single",
                model="model-a",
                agent_count=1,
                tool_calls=1,
                input_tokens=10,
                output_tokens=2,
                quality_score=1.0,
                execution_diagnostics=(
                    ExecutionDiagnostic(
                        execution_id="single-task",
                        attempts=3,
                        retry_count=2,
                        failure_categories=("network", "rate_limit"),
                        retry_wait_seconds=3.5,
                        transient_failures=2,
                    ),
                ),
            )
        )
        log.record(
            ComparisonMeasurement(
                run_id="split-auth-failure",
                mode="split",
                model="model-a",
                agent_count=1,
                tool_calls=1,
                quality_score=1.0,
                execution_diagnostics=(
                    ExecutionDiagnostic(
                        execution_id="split-task",
                        attempts=1,
                        retry_count=0,
                        failure_categories=("auth",),
                        permanent_failures=1,
                        outcome="failure",
                    ),
                ),
            )
        )

        restored = ComparisonLog.from_jsonl(log.render_jsonl())
        summary = restored.summary()

        self.assertEqual(summary.single_retry_count, 2)
        self.assertEqual(summary.single_retry_wait_seconds, 3.5)
        self.assertEqual(summary.single_transient_failures, 2)
        self.assertEqual(summary.single_permanent_failures, 0)
        self.assertEqual(summary.single_failure_categories, ("network", "rate_limit"))
        self.assertEqual(summary.split_permanent_failures, 1)
        self.assertEqual(summary.split_failure_categories, ("auth",))

    def test_summary_compares_cost_and_quality_without_counting_unknowns(self):
        log = ComparisonLog()
        log.record(
            ComparisonMeasurement(
                run_id="single-1",
                mode="single",
                model="strong-model",
                agent_count=1,
                tool_calls=2,
                fixed_context_tokens_per_agent=1000,
                work_tokens=200,
                coordination_tokens=0,
                output_tokens=100,
                input_usd_per_million=Decimal("10"),
                output_usd_per_million=Decimal("20"),
                quality_score=1.0,
            )
        )
        log.record(
            ComparisonMeasurement(
                run_id="split-1",
                mode="split",
                model="cheap-model",
                agent_count=3,
                tool_calls=6,
                fixed_context_tokens_per_agent=1000,
                work_tokens=200,
                coordination_tokens=50,
                output_tokens=100,
                input_usd_per_million=Decimal("2"),
                output_usd_per_million=Decimal("4"),
                quality_score=1.0,
            )
        )
        summary = log.summary()

        self.assertEqual(summary.single_tokens, 1300)
        self.assertEqual(summary.split_tokens, 3350)
        self.assertEqual(summary.recommendation, "split")
        self.assertIsNotNone(summary.cost_savings_ratio)
        self.assertEqual(summary.quality_delta, 0.0)
        self.assertIn('"agent_count": 3', log.render_jsonl())

    def test_missing_cost_or_quality_makes_comparison_inconclusive(self):
        log = ComparisonLog()
        log.record(
            ComparisonMeasurement(
                run_id="single-unknown",
                mode="single",
                model="model-a",
                agent_count=1,
                tool_calls=1,
                output_tokens=10,
                quality_score=1.0,
            )
        )
        log.record(
            ComparisonMeasurement(
                run_id="split-known",
                mode="split",
                model="model-a",
                agent_count=2,
                tool_calls=2,
                fixed_context_tokens_per_agent=100,
                work_tokens=10,
                coordination_tokens=5,
                output_tokens=10,
                input_usd_per_million=Decimal("1"),
                output_usd_per_million=Decimal("1"),
                quality_score=1.0,
            )
        )

        summary = log.summary()

        self.assertIsNone(summary.single_cost_usd)
        self.assertEqual(summary.recommendation, "insufficient_data")

    def test_quality_passed_is_a_quality_signal_when_score_is_unavailable(self):
        log = ComparisonLog()
        for mode, passed in (("single", True), ("split", True)):
            log.record(
                ComparisonMeasurement(
                    run_id=f"{mode}-passed",
                    mode=mode,
                    model="model-a",
                    agent_count=1,
                    tool_calls=1,
                    input_tokens=100,
                    output_tokens=10,
                    input_usd_per_million=Decimal("1"),
                    output_usd_per_million=Decimal("1"),
                    quality_passed=passed,
                )
            )
        self.assertEqual(log.summary().quality_delta, 0.0)


if __name__ == "__main__":
    unittest.main()
