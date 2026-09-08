#!/usr/bin/env python3
"""Freeze public application circuits as shared, validated unitary workloads."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
import urllib.request

HERE = Path(__file__).resolve().parent
DATA = HERE / "datasets"
BASIS = ["rx", "ry", "rz", "p", "h", "x", "y", "z", "s", "t", "cx", "swap"]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize(circuit, seed=4137):
    from qiskit import transpile
    import numpy as np

    original = circuit.copy()
    terminal_readout = False
    for instruction in circuit.data:
        name = instruction.operation.name
        if terminal_readout and name not in {"measure", "barrier"}:
            raise ValueError("Only a unitary prefix with terminal measurements is supported")
        terminal_readout |= name == "measure"
    circuit = circuit.remove_final_measurements(inplace=False)
    forbidden = {"measure", "reset", "if_else", "while_loop", "for_loop", "switch_case"}
    if any(op.operation.name in forbidden or op.clbits for op in circuit.data):
        raise ValueError("Only a unitary prefix with terminal measurements is supported")
    circuit = transpile(circuit, basis_gates=BASIS, optimization_level=0, seed_transpiler=seed)
    n = circuit.num_qubits
    elements = []
    names = {name.lower(): name for name in ["Rx", "Ry", "Rz", "H", "X", "Y", "Z", "S", "T", "SWAP"]}
    names["p"] = "Phase"
    for instruction in circuit.data:
        op = instruction.operation
        if op.name == "barrier":
            continue
        targets = [n - 1 - circuit.find_bit(q).index for q in instruction.qubits]
        gate = {"type": "gate", "gate": "X" if op.name == "cx" else names[op.name], "targets": targets}
        if op.name == "cx":
            gate.update(targets=targets[1:], controls=targets[:1], control_configs=[True])
        if op.params:
            gate["params"] = [float(p) for p in op.params]
        elements.append(gate)
    phase = float(circuit.global_phase)
    if phase:
        value = complex(np.exp(1j * phase))
        z = [0.0, 0.0]
        v = [value.real, value.imag]
        elements.append(dict(type="gate", gate="Custom", targets=[0], label="Global phase",
                             matrix=[[v, z], [z, v]], is_diagonal=True))
    notes = dict(original_operations=len(original.data), normalized_operations=len(elements),
                 removed_terminal_measurements=original.count_ops().get("measure", 0),
                 basis=BASIS, optimization_level=0, seed_transpiler=seed,
                 qubit_mapping="yao_site = n - 1 - qiskit_bit", global_phase=phase)
    return dict(num_qubits=n, elements=elements), notes


def main():
    from mqt.bench import BenchmarkLevel, get_benchmark
    from qiskit import qasm2
    import importlib.metadata

    config = json.loads((DATA / "sources.json").read_text())
    records, cases = [], []
    raw = DATA / "original"
    raw.mkdir(exist_ok=True)
    upstream = config["qasmbench"]
    base = "https://raw.githubusercontent.com/pnnl/QASMBench/" + upstream["revision"] + "/"
    for name in ["LICENSE", "NOTICE"]:
        with urllib.request.urlopen(base + name, timeout=60) as response:
            (raw / ("QASMBench-" + name)).write_bytes(response.read())

    inputs = []
    for path in upstream["paths"]:
        with urllib.request.urlopen(base + path, timeout=60) as response:
            content = response.read()
        name = "qasmbench-" + Path(path).stem
        (raw / (name + ".qasm")).write_bytes(content)
        text = content.decode()
        # This track measures the unitary prefix. Some upstream UCCSD files have
        # invalid register names in their terminal readout; preserve the original
        # and remove only a contiguous terminal measurement suffix before parsing.
        suffix = re.search(r"(?:measure\s+[^;]+;\s*)+$", text)
        removed = 0 if suffix is None else suffix.group().count("measure")
        unitary = text if suffix is None else text[:suffix.start()]
        qc = qasm2.loads(unitary, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
        inputs.append((name, qc, dict(source=base + path, sha256=digest(content),
                                     terminal_measurements_removed_before_parsing=removed)))
    for algorithm in config["mqt"]["algorithms"]:
        for n in config["mqt"]["qubits"]:
            if algorithm == "grover" and n > 8:
                continue  # Exponential Grover repetitions are bounded in this versioned suite.
            qc = get_benchmark(algorithm, BenchmarkLevel.ALG, n)
            name = f"mqt-{algorithm}-{n}"
            # QASM 2 export first needs composite instructions expanded to standard gates.
            from qiskit import transpile
            exported = transpile(qc, basis_gates=BASIS, optimization_level=0, seed_transpiler=config["seed"])
            content = qasm2.dumps(exported).encode()
            (raw / (name + ".qasm")).write_bytes(content)
            inputs.append((name, qc, dict(source="mqt.bench", version=importlib.metadata.version("mqt.bench"),
                                         algorithm=algorithm, size=n, generator_seed=10,
                                         exported_qasm_sha256=digest(content))))
    for name, qc, provenance in inputs:
        circuit, notes = normalize(qc, config["seed"])
        case = dict(id=name, mode="state", tensor=False, initial="deterministic", circuit=circuit)
        cases.append(case)
        records.append(dict(id=name, provenance=provenance, normalization=notes,
                            case_sha256=digest(json.dumps(case, sort_keys=True).encode())))
    content = (json.dumps(cases, indent=2) + "\n").encode()
    (DATA / "applications.json").write_bytes(content)
    manifest = dict(schema_version=1, suite="applications-v1", cases_sha256=digest(content),
                    qiskit=importlib.metadata.version("qiskit"), records=records)
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(cases)} application cases")


if __name__ == "__main__":
    main()
