import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("regression_compare", Path(__file__).resolve().parents[1] / "compare.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def document(times):
    return dict(schema_version=1, suite_sha256="fixture", environment=dict(device="m4-a", precision="complex128"),
                expected_records=[["qft", "native", 1, "execute"]],
                records=[dict(case="qft", backend="native", threads=1, phase="execute", correctness="passed", run_medians_ns=times)])


class ComparisonTests(unittest.TestCase):
    def test_clear_improvement_passes(self):
        self.assertEqual(module.compare(document([100]*5), document([80]*5))["status"], "pass")

    def test_regression_fails(self):
        row = module.compare(document([100]*5), document([120]*5))["records"][0]
        self.assertEqual(row["status"], "regression")

    def test_noisy_result_is_inconclusive(self):
        row = module.compare(document([100]*5), document([60, 80, 100, 120, 140]))["records"][0]
        self.assertEqual(row["status"], "inconclusive")

    def test_smoke_run_cannot_qualify(self):
        self.assertEqual(module.compare(document([100]*3), document([50]*3))["status"], "fail")

    def test_missing_case_rejected(self):
        candidate = document([100]*5)
        candidate["records"] = []
        with self.assertRaises(ValueError):
            module.compare(document([100]*5), candidate)

    def test_different_physical_device_rejected(self):
        candidate = document([100]*5)
        candidate["environment"]["device"] = "m4-b"
        with self.assertRaisesRegex(ValueError, "environment"):
            module.compare(document([100]*5), candidate)

    def test_coverage_change_rejected_even_if_each_run_complete(self):
        candidate = document([100]*5)
        candidate["records"][0]["case"] = "rx"
        candidate["expected_records"][0][0] = "rx"
        with self.assertRaisesRegex(ValueError, "coverage"):
            module.compare(document([100]*5), candidate)

    def test_correctness_failure_rejected(self):
        candidate = document([100]*5)
        candidate["records"][0]["correctness"] = "failed"
        with self.assertRaisesRegex(ValueError, "correctness"):
            module.compare(document([100]*5), candidate)

    def test_invalid_samples_rejected(self):
        for value in [float("nan"), float("inf"), 0, -1]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                module.compare(document([100]*5), document([value]*5))

    def test_duplicate_rejected(self):
        candidate = document([100]*5)
        candidate["records"].append(copy.deepcopy(candidate["records"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            module.compare(document([100]*5), candidate)


if __name__ == "__main__":
    unittest.main()
