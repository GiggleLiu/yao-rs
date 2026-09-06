import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/(name+".py"))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
compare=load("compare")
generate=load("generate_cases")

class BackendReportTests(unittest.TestCase):
    def test_medians_and_threads_are_not_mixed(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)
            (path/"metadata.json").write_text(json.dumps(dict(platform="test",runs=3)))
            for threads in (1,4):
                for run,value in enumerate([100,200,900],1):
                    prefix=f"{threads}t-run{run}"
                    (path/(prefix+"-rust.json")).write_text(json.dumps([dict(id="gate_8",backend="native",estimates=dict(median=dict(point_estimate=value*threads)))]))
                    (path/(prefix+"-julia.json")).write_text(json.dumps(dict(threads=threads,records=[dict(id="gate_8",median_ns=500,max_error=1e-14)])))
            summary=compare.backend_report(path)
            native={row["threads"]:row["median_ns"] for row in summary if row["backend"]=="native"}
            self.assertEqual(native,{1:200,4:800})
            self.assertIn("2.50",(path/"report.md").read_text())
            self.assertIn("0.62",(path/"report.md").read_text())

    def test_missing_baseline_is_an_error(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)
            (path/"metadata.json").write_text(json.dumps(dict(platform="test",runs=1)))
            (path/"1t-run1-rust.json").write_text(json.dumps([dict(id="gate_8",backend="native",estimates=dict(median=dict(point_estimate=1)))]))
            with self.assertRaisesRegex(ValueError,"Missing Julia"):
                compare.backend_report(path)

    def test_generated_workloads_have_unique_ids_and_valid_placements(self):
        cases=generate.cases()
        self.assertEqual(len(cases),len({c["id"] for c in cases}))
        self.assertTrue(any(c["circuit"]["num_qubits"]==24 for c in cases))
        self.assertTrue(any(c["mode"]=="gradient" for c in cases))
        for case in cases:
            n=case["circuit"]["num_qubits"]
            for element in case["circuit"]["elements"]:
                sites=element.get("targets",element.get("locs",[]))+element.get("controls",[])
                self.assertTrue(all(0<=s<n for s in sites))
                self.assertEqual(len(sites),len(set(sites)))
            if case["tensor"]:
                self.assertEqual(case["initial"],"zero")

if __name__=="__main__":unittest.main()
