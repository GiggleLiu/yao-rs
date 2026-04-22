import json
import math
import tempfile
import unittest
from pathlib import Path

import qflex_txt_to_yao_json as converter


class QflexTxtToYaoJsonTests(unittest.TestCase):
    def test_converts_small_circuit_to_native_yao_json(self):
        text = """4
0 h 0
1 t 0
2 cx 0 1
3 cz 2 3
4 x_1_2 0
5 y_1_2 1
"""

        circuit = converter.convert_text(text)

        self.assertEqual(circuit["num_qubits"], 4)
        self.assertEqual(
            circuit["elements"],
            [
                {"type": "gate", "gate": "H", "targets": [0]},
                {"type": "gate", "gate": "T", "targets": [0]},
                {"type": "gate", "gate": "X", "targets": [1], "controls": [0]},
                {"type": "gate", "gate": "Z", "targets": [3], "controls": [2]},
                {"type": "gate", "gate": "SqrtX", "targets": [0]},
                {"type": "gate", "gate": "SqrtY", "targets": [1]},
            ],
        )

    def test_remaps_sparse_raw_qubit_ids_by_first_use(self):
        text = """70
0 h 5
0 h 6
1 cz 18 19
2 h 5
"""

        circuit = converter.convert_text(text)

        self.assertEqual(circuit["num_qubits"], 4)
        self.assertEqual(circuit["elements"][0]["targets"], [0])
        self.assertEqual(circuit["elements"][1]["targets"], [1])
        self.assertEqual(circuit["elements"][2]["controls"], [2])
        self.assertEqual(circuit["elements"][2]["targets"], [3])
        self.assertEqual(circuit["elements"][3]["targets"], [0])

    def test_converts_parameterized_and_custom_gates(self):
        text = """2
0 rz(0.25) 0
1 fsim(1.5707963267948966,0.5235987755982988) 0 1
2 hz_1_2 0
"""

        circuit = converter.convert_text(text)

        self.assertEqual(
            circuit["elements"][0],
            {"type": "gate", "gate": "Rz", "targets": [0], "params": [0.25]},
        )
        self.assertEqual(
            circuit["elements"][1],
            {
                "type": "gate",
                "gate": "FSim",
                "targets": [0, 1],
                "params": [math.pi / 2, math.pi / 6],
            },
        )
        custom = circuit["elements"][2]
        self.assertEqual(custom["gate"], "Custom")
        self.assertEqual(custom["targets"], [0])
        self.assertEqual(custom["label"], "HZ12")
        self.assertFalse(custom["is_diagonal"])
        self.assertEqual(custom["matrix"][0][0], [0.5, 0.0])
        self.assertEqual(custom["matrix"][0][1], [0.5, 0.5])
        self.assertEqual(custom["matrix"][1][0], [0.5, -0.5])
        self.assertEqual(custom["matrix"][1][1], [0.5, 0.0])

    def test_writes_file_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "circuit.txt"
            output_path = Path(tmpdir) / "circuit.json"
            input_path.write_text("1\n0 h 0\n", encoding="utf-8")

            converter.convert_file(input_path, output_path)

            parsed = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(parsed["num_qubits"], 1)
            self.assertEqual(parsed["elements"][0]["gate"], "H")


if __name__ == "__main__":
    unittest.main()
