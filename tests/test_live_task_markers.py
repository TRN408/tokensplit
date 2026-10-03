import unittest

from scripts.collect_live_task_markers import LiveTask


class LiveTaskContractTests(unittest.TestCase):
    def test_prompt_uses_unambiguous_three_line_contract(self):
        prompt = LiveTask("task-x", "tokensplit/example.py", "verify the example").prompt
        self.assertIn(
            "TASK_ID: task-x\nTARGET: tokensplit/example.py\nTASK_STATUS: PASS",
            prompt,
        )
        self.assertIn("Do not use semicolons", prompt)
        self.assertIn("do not use FAIL merely because", prompt)
        self.assertNotIn("TASK_ID: task-x; TARGET:", prompt)


if __name__ == "__main__":
    unittest.main()
