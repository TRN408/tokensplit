import json
import unittest

from scripts.validate_policy import validate


class ValidatePolicyTests(unittest.TestCase):
    def setUp(self):
        with open("policy.json", encoding="utf-8") as handle:
            self.policy = json.load(handle)

    def test_default_policy_is_valid(self):
        self.assertEqual(validate(self.policy), [])

    def test_unknown_measurements_cannot_be_evidence(self):
        policy = {**self.policy, "unknown_measurements_count_as_evidence": True}
        self.assertIn(
            "unknown_measurements_count_as_evidence must be false",
            validate(policy),
        )

    def test_savings_ratio_must_be_between_zero_and_one(self):
        policy = {**self.policy, "minimum_expected_savings_ratio": 1}
        self.assertIn(
            "minimum_expected_savings_ratio must be a number between 0 and 1",
            validate(policy),
        )
