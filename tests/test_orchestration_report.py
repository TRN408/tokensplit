import unittest

from scripts.orchestration_report import periodic_summary
from tokensplit import ComparisonLog, ComparisonMeasurement, ExecutionDiagnostic


class OrchestrationReportTests(unittest.TestCase):
    def test_periodic_summary_groups_by_utc_day_and_model(self):
        log = ComparisonLog()
        log.record(
            ComparisonMeasurement(
                run_id="sonnet-retry",
                mode="split",
                model="claude-sonnet",
                agent_count=1,
                tool_calls=1,
                timestamp="2026-10-03T23:30:00+09:00",
                execution_diagnostics=(
                    ExecutionDiagnostic(
                        execution_id="sonnet-1",
                        attempts=2,
                        retry_count=1,
                        failure_categories=("network",),
                        retry_wait_seconds=1.25,
                        transient_failures=1,
                    ),
                ),
            )
        )
        log.record(
            ComparisonMeasurement(
                run_id="opus-auth",
                mode="single",
                model="claude-opus",
                agent_count=1,
                tool_calls=1,
                timestamp="2026-10-04T00:15:00+09:00",
                execution_diagnostics=(
                    ExecutionDiagnostic(
                        execution_id="opus-1",
                        attempts=1,
                        retry_count=0,
                        failure_categories=("auth",),
                        permanent_failures=1,
                        outcome="failure",
                    ),
                ),
            )
        )

        report = periodic_summary(log.render_jsonl())

        self.assertEqual(report["timezone"], "UTC")
        self.assertEqual(
            report["periods"],
            [
                {
                    "period": "2026-10-03",
                    "model": "claude-opus",
                    "measurement_count": 1,
                    "execution_count": 1,
                    "successful_executions": 0,
                    "failed_executions": 1,
                    "retry_count": 0,
                    "retry_wait_seconds": 0.0,
                    "transient_failures": 0,
                    "permanent_failures": 1,
                    "failure_categories": ["auth"],
                },
                {
                    "period": "2026-10-03",
                    "model": "claude-sonnet",
                    "measurement_count": 1,
                    "execution_count": 1,
                    "successful_executions": 1,
                    "failed_executions": 0,
                    "retry_count": 1,
                    "retry_wait_seconds": 1.25,
                    "transient_failures": 1,
                    "permanent_failures": 0,
                    "failure_categories": ["network"],
                },
            ],
        )

    def test_periodic_summary_filters_range_and_keeps_legacy_undated_logs(self):
        log = ComparisonLog()
        log.record(
            ComparisonMeasurement(
                run_id="legacy",
                mode="single",
                model="legacy-model",
                agent_count=1,
                tool_calls=0,
            )
        )
        log.record(
            ComparisonMeasurement(
                run_id="in-range",
                mode="single",
                model="current-model",
                agent_count=1,
                tool_calls=0,
                timestamp="2026-10-03T12:00:00Z",
                execution_diagnostics=(
                    ExecutionDiagnostic(
                        execution_id="current-1",
                        attempts=1,
                        retry_count=0,
                    ),
                ),
            )
        )

        all_periods = periodic_summary(log.render_jsonl(), bucket="month")
        self.assertEqual([row["period"] for row in all_periods["periods"]], ["2026-10", "undated"])

        filtered = periodic_summary(
            log.render_jsonl(),
            start="2026-10-03T00:00:00+00:00",
            end="2026-10-04T00:00:00+00:00",
        )
        self.assertEqual(len(filtered["periods"]), 1)
        self.assertEqual(filtered["periods"][0]["model"], "current-model")


if __name__ == "__main__":
    unittest.main()
