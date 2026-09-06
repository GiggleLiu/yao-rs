import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare = load("compare")
generate = load("generate_cases")
runner = load("run_backend")


class BackendReportTests(unittest.TestCase):
    def test_tensor_memory_requires_completed_probe_and_keeps_estimates_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "memory-tensor-tenferro-32-auto.log"
            rows = [dict(dimension=32, mode="auto", backend="tenferro", labels=[0], estimate=dict(estimated_total_bytes=1000)),
                    dict(phase="matrix_inputs", retained_additional_rust_heap_bytes=500),
                    dict(phase="slice_execution", peak_additional_rust_heap_bytes=300)]
            text = "\n".join(json.dumps(r) for r in rows) + "\n4096 maximum resident set size\n"
            path.write_text(text)
            with self.assertRaisesRegex(ValueError, "Incomplete tensor memory probe"):
                compare.tensor_memory_rows(temp)
            path.write_text(text + '{"status":"complete"}\n')
            (row,) = compare.tensor_memory_rows(temp)
            self.assertEqual(row["estimate"]["estimated_total_bytes"], 1000)
            self.assertEqual(row["peak_rss_bytes"], 4096)
            self.assertEqual(row["execution_peak_bytes"], 300)

    def test_memory_termination_is_not_a_completed_or_timed_result(self):
        import signal

        self.assertEqual(
            runner.memory_probe_status("composed", -signal.SIGXCPU), "cpu_limit"
        )
        self.assertEqual(
            runner.memory_probe_status("composed", -signal.SIGKILL),
            "killed_with_cpu_limit",
        )
        with self.assertRaises(RuntimeError):
            runner.memory_probe_status("composed", -signal.SIGSEGV)
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            row = dict(
                backend="composed",
                qubits=8,
                depth=100,
                status="cpu_limit",
                file="probe.log",
            )
            (p / "circuit-ad-memory-status.json").write_text(json.dumps([row]))
            (p / "probe.log").write_text(
                '{"phase":"circuit_ad_forward_tape","retained_additional_rust_heap_bytes":4096}\n2097152 maximum resident set size\n'
            )
            (result,) = compare.circuit_ad_memory_rows(p)
            self.assertEqual(result["peak_rss_bytes"], 2097152)
            self.assertIsNone(result["backward_peak_bytes"])
            row["status"] = "complete"
            (p / "circuit-ad-memory-status.json").write_text(json.dumps([row]))
            with self.assertRaisesRegex(ValueError, "without a completed probe"):
                compare.circuit_ad_memory_rows(p)

    def test_evolution_memory_rss_units_and_phase_boundaries(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "memory-evolution-test.log"
            records = [
                dict(model="ising", qubits=12, steps=16, gates=2208),
                dict(
                    phase="circuit_construction",
                    retained_additional_rust_heap_bytes=4096,
                ),
                dict(phase="native_execution", peak_additional_rust_heap_bytes=65536),
            ]
            prefix = "\n".join(json.dumps(row) for row in records) + "\n"
            for rss in [
                "2097152 maximum resident set size",
                "Maximum resident set size (kbytes): 2048",
            ]:
                path.write_text(prefix + rss)
                (row,) = compare.evolution_memory_rows(temp)
                self.assertEqual(row["peak_rss_bytes"], 2097152)
                self.assertEqual(row["circuit_retained_bytes"], 4096)
                self.assertEqual(row["execution_peak_bytes"], 65536)
            path.write_text(prefix)
            with self.assertRaisesRegex(ValueError, "Missing peak RSS"):
                compare.evolution_memory_rows(temp)

    def test_medians_and_threads_are_not_mixed(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / "metadata.json").write_text(
                json.dumps(dict(platform="test", runs=3))
            )
            for threads in (1, 4):
                for run, value in enumerate([100, 200, 900], 1):
                    prefix = f"{threads}t-run{run}"
                    (path / (prefix + "-rust.json")).write_text(
                        json.dumps(
                            [
                                dict(
                                    id="gate_8",
                                    backend="native",
                                    estimates=dict(
                                        median=dict(point_estimate=value * threads)
                                    ),
                                )
                            ]
                        )
                    )
                    (path / (prefix + "-julia.json")).write_text(
                        json.dumps(
                            dict(
                                threads=threads,
                                records=[
                                    dict(
                                        id="gate_8",
                                        median_ns=500,
                                        max_error=1e-14,
                                        approximation_error=0.01 * run * threads,
                                        yao_approximation_error=0.02 * run * threads,
                                    )
                                ],
                            )
                        )
                    )
            summary = compare.backend_report(path)
            native = {
                row["threads"]: row["median_ns"]
                for row in summary
                if row["backend"] == "native"
            }
            self.assertEqual(native, {1: 200, 4: 800})
            self.assertIn("2.50", (path / "report.md").read_text())
            self.assertIn("0.62", (path / "report.md").read_text())
            errors = {
                (row["threads"], row["backend"]): row["relative_state_error"]
                for row in summary
            }
            self.assertEqual(
                errors,
                {
                    (1, "native"): 0.03,
                    (1, "julia"): 0.06,
                    (4, "native"): 0.12,
                    (4, "julia"): 0.24,
                },
            )
            self.assertIn("1.00e-14", (path / "report.md").read_text())
            self.assertIn("evolution-error-time.svg", (path / "report.md").read_text())

    def test_missing_baseline_is_an_error(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / "metadata.json").write_text(
                json.dumps(dict(platform="test", runs=1))
            )
            (path / "1t-run1-rust.json").write_text(
                json.dumps(
                    [
                        dict(
                            id="gate_8",
                            backend="native",
                            estimates=dict(median=dict(point_estimate=1)),
                        )
                    ]
                )
            )
            with self.assertRaisesRegex(ValueError, "Missing Julia"):
                compare.backend_report(path)

    def test_missing_rust_and_incomplete_runs_are_errors(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / "metadata.json").write_text(
                json.dumps(dict(platform="test", runs=2, threads=[1]))
            )
            (path / "1t-run1-julia.json").write_text(
                json.dumps(
                    dict(
                        threads=1,
                        records=[dict(id="gate_8", median_ns=500, max_error=0)],
                    )
                )
            )
            with self.assertRaisesRegex(ValueError, "Missing native Rust"):
                compare.backend_report(path)
            (path / "1t-run1-rust.json").write_text(
                json.dumps(
                    [
                        dict(
                            id="gate_8",
                            backend="native",
                            estimates=dict(median=dict(point_estimate=1)),
                        )
                    ]
                )
            )
            with self.assertRaisesRegex(ValueError, "1/2"):
                compare.backend_report(path)

    def test_generated_workloads_have_unique_ids_and_valid_placements(self):
        cases = generate.cases()
        self.assertEqual(len(cases), len({c["id"] for c in cases}))
        self.assertTrue(any(c["circuit"]["num_qubits"] == 24 for c in cases))
        self.assertTrue(any(c["mode"] == "gradient" for c in cases))
        for case in cases:
            n = case["circuit"]["num_qubits"]
            for element in case["circuit"]["elements"]:
                sites = element.get("targets", element.get("locs", [])) + element.get(
                    "controls", []
                )
                self.assertTrue(all(0 <= s < n for s in sites))
                self.assertEqual(len(sites), len(set(sites)))
            if case["tensor"]:
                self.assertEqual(case["initial"], "zero")


if __name__ == "__main__":
    unittest.main()
