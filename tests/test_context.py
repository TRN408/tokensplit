import unittest

from tokensplit.context import (
    CacheMetrics,
    ContextBuilder,
    ContextPolicy,
    DynamicTurn,
    InMemoryExternalMemory,
    Message,
    StaticContext,
    ToolOutput,
)


class ContextBuilderTests(unittest.TestCase):
    def make_builder(self, **kwargs):
        static = StaticContext(
            system_instructions="Follow the safety policy.",
            tool_definitions=[
                {"name": "z_tool", "description": "Z", "parameters": {"type": "object"}},
                {"name": "a_tool", "description": "A", "parameters": {"type": "object"}},
            ],
            fixed_context="Project rules are fixed for this session.",
        )
        return ContextBuilder(static, ContextPolicy(**kwargs))

    def test_static_prefix_is_identical_when_dynamic_values_change(self):
        builder = self.make_builder(max_input_tokens=2_000)
        first = builder.build(
            DynamicTurn(user_input="first", timestamp="2026-10-03T09:00:00+09:00", state={"step": 1})
        )
        second = builder.build(
            DynamicTurn(user_input="second", timestamp="2026-10-03T09:01:00+09:00", state={"step": 2})
        )

        self.assertEqual(first.static_prefix, second.static_prefix)
        self.assertEqual(first.prefix_fingerprint, second.prefix_fingerprint)
        self.assertTrue(second.full_prompt.startswith(second.static_prefix + "\n"))
        self.assertIn("2026-10-03T09:01:00+09:00", second.dynamic_suffix)
        self.assertNotIn("2026-10-03T09:01:00+09:00", second.static_prefix)

    def test_tool_order_is_canonicalized(self):
        one = StaticContext("rules", [{"name": "b"}, {"name": "a"}]).render()
        two = StaticContext("rules", [{"name": "a"}, {"name": "b"}]).render()
        self.assertEqual(one, two)

    def test_compression_waits_for_budget_threshold(self):
        builder = self.make_builder(max_input_tokens=300, compression_threshold=0.7, recent_messages=2)
        history = [Message("user", "old " * 20), Message("assistant", "reply " * 20)]
        before = builder.build(DynamicTurn(user_input="short", history=history))
        self.assertFalse(before.compression_applied)
        self.assertEqual(before.compression_count, 0)

        history.append(Message("user", "new " * 20))
        after = builder.build(DynamicTurn(user_input="latest", history=history))
        self.assertTrue(after.compression_applied)
        self.assertEqual(after.compression_count, 1)
        self.assertEqual(before.static_prefix, after.static_prefix)
        self.assertEqual(before.prefix_fingerprint, after.prefix_fingerprint)
        self.assertIn("<compressed-history>", after.dynamic_suffix)
        self.assertIn("latest", after.dynamic_suffix)

    def test_compressed_prefix_of_full_history_is_not_summarized_twice(self):
        builder = self.make_builder(max_input_tokens=350, compression_threshold=0.7, recent_messages=2)
        history = [
            Message("user", "old " * 20),
            Message("assistant", "reply " * 20),
            Message("user", "new " * 20),
        ]
        first = builder.build(DynamicTurn(user_input="latest", history=history))
        self.assertFalse(first.compression_applied)

        history.append(Message("assistant", "new reply " * 20))
        second = builder.build(DynamicTurn(user_input="latest again", history=history))
        self.assertTrue(second.compression_applied)
        self.assertGreaterEqual(second.compression_count, 1)
        self.assertLessEqual(second.dynamic_suffix.count("<user>old "), 1)
        self.assertEqual(len(second.archived_memory_refs), 1)

    def test_metrics_report_provider_cache_usage_and_prefix_changes(self):
        metrics = CacheMetrics()
        metrics.observe(prefix_fingerprint="same", cached_input_tokens=1_000)
        metrics.observe(prefix_fingerprint="same", cached_input_tokens=1_000)
        metrics.observe(prefix_fingerprint="changed", cache_write_tokens=1_000)

        self.assertEqual(metrics.requests, 3)
        self.assertEqual(metrics.cache_hits, 2)
        self.assertEqual(metrics.cache_misses, 1)
        self.assertAlmostEqual(metrics.hit_rate, 2 / 3)
        self.assertEqual(metrics.cached_input_tokens, 2_000)
        self.assertEqual(metrics.prefix_changes, 1)
        self.assertFalse(metrics.prefix_stable)

    def test_static_purpose_constraints_and_state_survive_compression(self):
        static = StaticContext(
            system_instructions="Follow the safety policy.",
            fixed_context="Project rules are fixed for this session.",
            purpose="Ship the migration without data loss.",
            constraints=("Do not delete production data.", "Keep the API backward compatible."),
        )
        policy = ContextPolicy(max_input_tokens=360, compression_threshold=0.7, recent_messages=2)
        history = [
            Message("user", "old context " * 30),
            Message("assistant", "old response " * 30),
            Message("user", "recent context " * 10),
        ]
        baseline = ContextBuilder(static, ContextPolicy(max_input_tokens=2_000)).build(
            DynamicTurn(user_input="latest migration status", state={"phase": "verify"}, history=history)
        )
        compressed = ContextBuilder(static, policy).build(
            DynamicTurn(user_input="latest migration status", state={"phase": "verify"}, history=history)
        )

        required = (
            "Ship the migration without data loss.",
            "Do not delete production data.",
            "Keep the API backward compatible.",
            '"phase":"verify"',
            "latest migration status",
        )
        self.assertTrue(compressed.compression_applied)
        self.assertEqual(
            {marker for marker in required if marker in baseline.full_prompt},
            {marker for marker in required if marker in compressed.full_prompt},
        )
        self.assertIn('"strategy":"external-memory"', compressed.dynamic_suffix)

    def test_old_tool_output_is_archived_and_not_reinserted(self):
        memory = InMemoryExternalMemory()
        builder = ContextBuilder(
            StaticContext("Follow the fixed policy.", fixed_context="Keep the goal fixed."),
            ContextPolicy(max_input_tokens=420, compression_threshold=0.5, recent_messages=2),
            memory_store=memory,
        )
        tool_text = "sensitive search result " * 30
        history = [
            Message("user", "old request " * 20),
            Message("tool", ToolOutput(tool_text)),
            Message("user", "recent request " * 10),
            Message("assistant", "recent response " * 10),
        ]

        rendered = builder.build(DynamicTurn(user_input="continue", history=history))

        self.assertTrue(rendered.compression_applied)
        self.assertEqual(len(rendered.archived_memory_refs), 1)
        self.assertNotIn(tool_text, rendered.full_prompt)
        archived = memory.load(rendered.archived_memory_refs[0])
        self.assertEqual(archived[1].render(), history[1].render())

    def test_compression_is_not_repeated_on_each_low_pressure_turn(self):
        memory = InMemoryExternalMemory()
        builder = ContextBuilder(
            StaticContext("Follow the fixed policy.", fixed_context="Keep the goal fixed."),
            ContextPolicy(max_input_tokens=700, compression_threshold=0.7, recent_messages=2),
            memory_store=memory,
        )
        history = [
            Message("user", "large request " * 45),
            Message("assistant", "large response " * 45),
            Message("user", "latest request " * 45),
        ]
        first = builder.build(DynamicTurn(user_input="continue", history=history))
        self.assertTrue(first.compression_applied)
        history.append(Message("assistant", "ok"))
        second = builder.build(DynamicTurn(user_input="one more", history=history))

        self.assertFalse(second.compression_applied)
        self.assertEqual(second.compression_count, first.compression_count)
        self.assertEqual(len(memory.entries), 1)

    def test_cache_hit_rate_remains_measurable_after_compression(self):
        static = StaticContext(
            "Follow the fixed policy.",
            fixed_context="Fixed project context.",
            purpose="Complete the release safely.",
            constraints=("Preserve compatibility.",),
        )
        builder = ContextBuilder(
            static,
            ContextPolicy(max_input_tokens=420, compression_threshold=0.7, recent_messages=2),
        )
        metrics = CacheMetrics()
        history = []
        rendered_contexts = []
        for index in range(8):
            history.extend(
                [
                    Message("user", f"request {index} " + "detail " * 12),
                    Message("assistant", f"response {index} " + "detail " * 12),
                ]
            )
            rendered = builder.build(DynamicTurn(user_input=f"next {index}", history=history))
            rendered_contexts.append(rendered)
            metrics.observe(
                prefix_fingerprint=rendered.prefix_fingerprint,
                cached_input_tokens=0 if index == 0 else rendered.static_prefix_tokens,
                input_tokens=rendered.total_tokens,
            )

        self.assertTrue(any(rendered.compression_applied for rendered in rendered_contexts))
        self.assertGreaterEqual(metrics.hit_rate, 0.875)
        self.assertTrue(metrics.prefix_stable)


if __name__ == "__main__":
    unittest.main()
