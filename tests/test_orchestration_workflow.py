from pathlib import Path
import unittest


ROOT = Path(__file__).parents[1]


class OrchestrationWorkflowTests(unittest.TestCase):
    def test_producer_uploads_the_comparison_artifact(self):
        workflow = (ROOT / ".github/workflows/orchestration-cli-producer.yml").read_text(encoding="utf-8")

        self.assertIn("name: orchestration-cli-producer", workflow)
        self.assertIn("scripts/collect_live_task_markers.py reports/comparison.jsonl", workflow)
        self.assertIn('--provider "$PROVIDER"', workflow)
        self.assertIn("QWEN_API_KEY", workflow)
        self.assertIn("options:\n          - claude\n          - qwen", workflow)
        self.assertIn("name: orchestration-comparison", workflow)
        self.assertIn("path: reports/", workflow)
        self.assertIn("if: always()", workflow)

    def test_monitor_downloads_the_triggering_producer_run_artifact(self):
        workflow = (ROOT / ".github/workflows/orchestration-gate.yml").read_text(encoding="utf-8")

        self.assertIn('workflows: ["orchestration-cli-producer"]', workflow)
        self.assertIn("actions/download-artifact", workflow)
        self.assertIn("name: orchestration-comparison", workflow)
        self.assertIn("run-id: ${{ github.event.workflow_run.id }}", workflow)
        self.assertIn("scripts/orchestration_gate.py", workflow)
        self.assertIn("orchestration-failure-notification.json", workflow)


if __name__ == "__main__":
    unittest.main()
