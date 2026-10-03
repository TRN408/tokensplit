import unittest

from scripts.collect_live_task_markers import TASKS, LiveTask


class LiveTaskContractTests(unittest.TestCase):
    def test_prompt_uses_unambiguous_three_line_contract(self):
        prompt = LiveTask("task-x", "tokensplit/example.py", "verify the example").prompt
        self.assertIn(
            "TASK_ID: task-x\nTARGET: tokensplit/example.py\nTASK_STATUS: PASS",
            prompt,
        )
        self.assertIn("Do not use semicolons", prompt)
        self.assertIn("do not use FAIL merely because", prompt)
        self.assertIn("these acceptance checks are expected to pass", prompt)
        self.assertNotIn("TASK_ID: task-x; TARGET:", prompt)

    def test_task_specific_guidance_is_included_without_response_content(self):
        task = LiveTask("task-x", "tokensplit/example.py", "verify the example", "Look for the example regression test.")
        self.assertIn("Verification guidance: Look for the example regression test.", task.prompt)

    def test_observed_marker_failures_have_contract_completion_guidance(self):
        observed = {
            "task-05-pruning",
            "task-10-tool-adapters",
            "task-14-pricing",
            "task-16-streaming",
            "task-17-cache",
            "task-18-persistent-memory",
            "task-19-quality-tests",
        }
        tasks = {task.task_id: task for task in TASKS}
        for task_id in observed:
            with self.subTest(task_id=task_id):
                prompt = tasks[task_id].prompt
                self.assertIn("Verification guidance:", prompt)
                self.assertIn("three contract lines", prompt)
                self.assertIn("TASK_STATUS", prompt)


if __name__ == "__main__":
    unittest.main()
