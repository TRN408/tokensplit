import json
import unittest

from scripts.validate_policy import accept_split_result, should_split_before_trial, validate


class ValidatePolicyTests(unittest.TestCase):
    def setUp(self):
        with open("policy.json", encoding="utf-8") as handle:
            self.policy = json.load(handle)

    def test_default_policy_is_valid(self):
        self.assertEqual(validate(self.policy), [])

    def test_default_thresholds_match_readme(self):
        self.assertEqual(self.policy["minimum_quality_passed_single_measurements"], 3)
        self.assertEqual(self.policy["minimum_single_median_tokens"], 20000)
        self.assertEqual(self.policy["minimum_expected_savings_ratio"], 0.25)
        self.assertEqual(self.policy["minimum_measured_savings_ratio_after_split"], 0.15)
        self.assertTrue(self.policy["single_measurement_requires_quality_pass"])
        self.assertTrue(self.policy["split_result_requires_quality_pass"])
        self.assertFalse(self.policy["unknown_measurements_count_as_evidence"])

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

    def test_quality_pass_requirements_cannot_be_disabled(self):
        policy = {**self.policy, "single_measurement_requires_quality_pass": False}
        self.assertIn("single_measurement_requires_quality_pass must be true", validate(policy))

    def test_pre_trial_accepts_exact_thresholds(self):
        measurements = [
            {"tokens": 20000, "quality_passed": True},
            {"tokens": 22000, "quality_passed": True},
            {"tokens": 24000, "quality_passed": True},
        ]
        self.assertTrue(should_split_before_trial(self.policy, measurements, 0.25))

    def test_pre_trial_rejects_too_few_measurements(self):
        measurements = [
            {"tokens": 25000, "quality_passed": True},
            {"tokens": 26000, "quality_passed": True},
        ]
        self.assertFalse(should_split_before_trial(self.policy, measurements, 0.5))

    def test_pre_trial_rejects_low_baseline_median(self):
        measurements = [
            {"tokens": 19999, "quality_passed": True},
            {"tokens": 19999, "quality_passed": True},
            {"tokens": 19999, "quality_passed": True},
        ]
        self.assertFalse(should_split_before_trial(self.policy, measurements, 0.5))

    def test_pre_trial_rejects_expected_savings_below_threshold(self):
        measurements = [
            {"tokens": 20000, "quality_passed": True},
            {"tokens": 20000, "quality_passed": True},
            {"tokens": 20000, "quality_passed": True},
        ]
        self.assertFalse(should_split_before_trial(self.policy, measurements, 0.2499))

    def test_unknown_or_quality_failed_measurements_do_not_count(self):
        measurements = [
            {"tokens": 30000, "quality_passed": True},
            {"tokens": None, "quality_passed": True},
            {"quality_passed": True},
            {"tokens": 40000, "quality_passed": None},
            {"tokens": 50000, "quality_passed": False},
        ]
        self.assertFalse(should_split_before_trial(self.policy, measurements, 0.5))

    def test_pre_trial_rejects_unknown_expected_savings(self):
        measurements = [
            {"tokens": 20000, "quality_passed": True},
            {"tokens": 21000, "quality_passed": True},
            {"tokens": 22000, "quality_passed": True},
        ]
        self.assertFalse(should_split_before_trial(self.policy, measurements, None))

    def test_post_trial_accepts_exact_measured_savings_threshold(self):
        self.assertTrue(accept_split_result(self.policy, 20000, 17000, True))

    def test_post_trial_rejects_savings_below_threshold(self):
        self.assertFalse(accept_split_result(self.policy, 20000, 17001, True))

    def test_post_trial_requires_explicit_quality_pass_and_known_tokens(self):
        self.assertFalse(accept_split_result(self.policy, 20000, 16000, None))
        self.assertFalse(accept_split_result(self.policy, 20000, None, True))
        self.assertFalse(accept_split_result(self.policy, 0, 0, True))
