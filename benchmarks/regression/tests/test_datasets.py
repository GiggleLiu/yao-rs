import importlib.util
import json
from pathlib import Path
import unittest
import sys
import itertools

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(ROOT.parent))
from run_backend import measurement_order

prepare = load("prepare_datasets")
qulacs = load("qulacs_baseline")


class DatasetTests(unittest.TestCase):
    def test_asymmetric_custom_matrices_preserve_target_order_and_controls(self):
        n = 5
        for targets in ([0, 3], [3, 0], [4, 1, 3]):
            dimension = 1 << len(targets)
            matrix = np.roll(np.diag(np.exp(0.17j * np.arange(dimension))), 1, axis=0)
            for controlled in (False, True):
                controls = [q for q in range(n) if q not in targets][:2] if controlled else []
                configs = [False, True] if controlled else []
                with self.subTest(targets=targets, controls=controls):
                    element = dict(type='gate', gate='Custom', targets=targets,
                                   controls=controls, control_configs=configs,
                                   matrix=np.stack([matrix.real, matrix.imag], axis=-1).tolist())
                    spec = dict(num_qubits=n, elements=[element])
                    state = qulacs.initial(dict(circuit=spec, initial='deterministic'))
                    operation = UnitaryGate(matrix)
                    if controls:
                        operation = operation.control(len(controls), ctrl_state=2)
                    expected = Statevector(state.get_vector()).evolve(
                        operation, qargs=[n - 1 - q for q in controls + list(targets)]).data
                    qulacs.build(spec).update_quantum_state(state)
                    np.testing.assert_allclose(state.get_vector(), expected, rtol=1e-12, atol=1e-12)

    def test_six_runs_cover_every_implementation_order(self):
        self.assertEqual({measurement_order(i, True) for i in range(6)},
                         set(itertools.permutations(["rust", "julia", "qulacs"])))

    def test_complete_state_matches_before_and_after_normalization(self):
        circuit = QuantumCircuit(4)
        circuit.h(0)
        circuit.ry(0.4, 3)
        circuit.cp(-0.25, 3, 0)
        circuit.cx(0, 2)
        circuit.swap(1, 3)
        circuit.rz(0.7, 2)
        circuit.y(0)
        circuit.global_phase = 0.23
        spec, _ = prepare.normalize(circuit)
        case = dict(circuit=spec, initial="deterministic")
        state = qulacs.initial(case)
        reference = Statevector(state.get_vector()).evolve(circuit).data
        qulacs.build(spec).update_quantum_state(state)
        np.testing.assert_allclose(state.get_vector(), reference, rtol=1e-12, atol=1e-12)

    def test_terminal_readout_removed_and_recorded(self):
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.measure_all()
        spec, notes = prepare.normalize(circuit)
        self.assertEqual(notes["removed_terminal_measurements"], 2)
        self.assertEqual(len(spec["elements"]), 1)

    def test_dynamic_circuit_is_not_silently_changed(self):
        circuit = QuantumCircuit(1, 1)
        circuit.measure(0, 0)
        circuit.x(0)
        with self.assertRaisesRegex(ValueError, "unitary prefix"):
            prepare.normalize(circuit)

    def test_readout_before_gate_on_another_qubit_is_rejected(self):
        circuit = QuantumCircuit(2, 1)
        circuit.measure(0, 0)
        circuit.x(1)
        with self.assertRaisesRegex(ValueError, "unitary prefix"):
            prepare.normalize(circuit)

    def test_committed_cases_and_upstream_files_match_manifest(self):
        data = ROOT / "datasets"
        manifest = json.loads((data / "manifest.json").read_text())
        self.assertEqual(prepare.digest((data / "applications.json").read_bytes()), manifest["cases_sha256"])
        cases = json.loads((data / "applications.json").read_text())
        self.assertEqual(len(cases), 19)
        self.assertEqual(len({c["id"] for c in cases}), len(cases))
        for case, record in zip(cases, manifest["records"], strict=True):
            self.assertEqual(case["id"], record["id"])
            self.assertEqual(prepare.digest(json.dumps(case, sort_keys=True).encode()), record["case_sha256"])
            provenance = record["provenance"]
            expected = provenance.get("sha256", provenance.get("exported_qasm_sha256"))
            self.assertEqual(prepare.digest((data / "original" / (case["id"] + ".qasm")).read_bytes()), expected)


if __name__ == "__main__":
    unittest.main()
