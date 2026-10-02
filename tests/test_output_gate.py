import unittest

from tokensplit import GateLimits, gate_tool_output
from tokensplit.context import Message, ToolOutput


class OutputGateTests(unittest.TestCase):
    def test_normal_input_passes_without_warning(self):
        result = gate_tool_output("small tool output")

        self.assertEqual(result.text, "small tool output")
        self.assertEqual(result.warnings, ())
        self.assertFalse(result.truncated)
        self.assertFalse(result.used_fallback)

    def test_oversized_text_extracts_query_lines_with_context(self):
        output = "\n".join(
            ["before one", "before two", "ERROR: database unavailable", "after one"]
            + [f"noise {index}" for index in range(40)]
        )
        result = gate_tool_output(
            output,
            query="database",
            limits=GateLimits(max_chars=180, context_lines=1),
        )

        self.assertTrue(result.truncated)
        self.assertIn("max_chars=180", result.text)
        self.assertIn("ERROR: database unavailable", result.text)
        self.assertNotIn("noise 39", result.text)
        self.assertLessEqual(len(result.text), 180)

    def test_item_limit_warns_and_selects_relevant_items(self):
        items = [{"id": index, "message": "keep" if index == 17 else "noise"} for index in range(20)]
        result = gate_tool_output(
            items,
            query="keep",
            limits=GateLimits(max_chars=5000, max_items=3),
        )

        self.assertEqual(result.original_items, 20)
        self.assertTrue(result.truncated)
        self.assertIn("max_items=3", result.text)
        self.assertIn('"id": 17', result.text)

    def test_extraction_failure_uses_bounded_fallback_and_warning(self):
        output = "start\n" + ("x" * 500) + "\nend"

        def failing_extractor(_: str) -> str:
            raise RuntimeError("implementation detail must not leak")

        result = gate_tool_output(
            output,
            limits=GateLimits(max_chars=140),
            extractor=failing_extractor,
        )

        self.assertTrue(result.used_fallback)
        self.assertTrue(result.truncated)
        self.assertIn("custom extractor failed (RuntimeError)", result.text)
        self.assertNotIn("implementation detail must not leak", result.text)
        self.assertIn("start", result.text)
        self.assertIn("end", result.text)
        self.assertLessEqual(len(result.text), 140)

    def test_invalid_limits_are_rejected(self):
        with self.assertRaises(ValueError):
            GateLimits(max_chars=0)

    def test_tool_output_uses_the_existing_message_rendering_layer(self):
        message = Message(
            "tool",
            ToolOutput(
                "prefix\nERROR: keep this line\n" + ("noise\n" * 50),
                query="ERROR",
                limits=GateLimits(max_chars=160),
            ),
        )

        rendered = message.render()

        self.assertTrue(rendered.startswith("<tool>"))
        self.assertIn("ERROR: keep this line", rendered)
        self.assertIn("max_chars=160", rendered)
        self.assertTrue(rendered.endswith("</tool>"))


if __name__ == "__main__":
    unittest.main()
