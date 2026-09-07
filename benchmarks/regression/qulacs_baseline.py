#!/usr/bin/env python3
"""Warmed Qulacs circuit execution, including the same input copy as native Rust."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np
import qulacs
from qulacs import QuantumCircuit, QuantumState
from qulacs.circuit import QuantumCircuitOptimizer
from qulacs import gate


def build(spec):
    n = spec["num_qubits"]
    circuit = QuantumCircuit(n)
    for element in spec["elements"]:
        if element["type"] != "gate":
            raise ValueError("Qulacs unitary track cannot time channels or annotations")
        name = element["gate"]
        sites = [n - 1 - q for q in element["targets"]]
        controls = [n - 1 - q for q in element.get("controls", [])]
        configs = element.get("control_configs", [True] * len(controls))
        params = element.get("params", [])
        if name == "X" and len(controls) == 1 and configs == [True]:
            circuit.add_gate(gate.CNOT(controls[0], sites[0]))
            continue
        if name in ("Rx", "Ry", "Rz"):
            # Qulacs named rotations use exp(+i theta sigma/2).
            operation = getattr(gate, name.upper())(sites[0], -params[0])
        elif name in ("X", "Y", "Z", "H", "S", "T"):
            operation = getattr(gate, name)(sites[0])
        elif name == "SWAP":
            operation = gate.SWAP(*sites)
        elif name == "Phase":
            operation = gate.DiagonalMatrix(sites, [1, np.exp(1j * params[0])])
        elif name == "FSim":
            theta, phi = params
            c, s = np.cos(theta), -1j * np.sin(theta)
            matrix = np.array([[1,0,0,0],[0,c,s,0],[0,s,c,0],[0,0,0,np.exp(-1j*phi)]])
            operation = gate.DenseMatrix(sites, matrix)
        elif name == "Custom":
            matrix = np.asarray(element["matrix"])
            matrix = matrix[...,0] + 1j * matrix[...,1]
            # Both libraries assign the first target to the matrix's low bit.
            operation = gate.DenseMatrix(sites, matrix)
        else:
            raise ValueError(f"Unsupported gate: {name}")
        if controls:
            operation = gate.to_matrix_gate(operation)
            for site, config in zip(controls, configs, strict=True):
                operation.add_control_qubit(site, int(config))
        circuit.add_gate(operation)
    return circuit


def initial(case):
    n = case["circuit"]["num_qubits"]
    state = QuantumState(n)
    if case["initial"] == "deterministic":
        k = np.arange(2**n, dtype=np.float64)
        values = np.cos(0.1*k) + 1j*np.sin(0.2*k)
        state.load(values / np.linalg.norm(values))
    elif case["initial"] != "zero":
        raise ValueError("Unsupported initial state")
    return state


def measure(call, samples):
    call()  # warm before calibration
    start = time.perf_counter_ns()
    call()
    duration = max(1, time.perf_counter_ns() - start)
    iterations = min(10000, max(1, int(10_000_000 / duration)))
    values = []
    for _ in range(samples):
        start = time.perf_counter_ns()
        for _ in range(iterations):
            call()
        values.append((time.perf_counter_ns() - start) / iterations)
    return values, iterations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("references", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--samples", type=int, default=30)
    args = parser.parse_args()
    if args.samples < 10:
        parser.error("at least 10 inner samples required")
    records = []
    for case in json.loads(args.cases.read_text()):
        if case["mode"] != "state":
            continue
        state = initial(case)
        expected = np.fromfile(args.references / (case["id"] + ".bin"), dtype="<c16")
        for backend, block_size in [("qulacs", None), ("qulacs_fused4", 4)]:
            circuit = build(case["circuit"])
            start = time.perf_counter_ns()
            if block_size:
                QuantumCircuitOptimizer().optimize(circuit, block_size)
            prepare_ns = time.perf_counter_ns() - start
            def execute():
                output = state.copy()
                circuit.update_quantum_state(output)
                return output
            got = execute().get_vector()
            np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-10, err_msg=case["id"])
            values, iterations = measure(execute, args.samples)
            records.append(dict(id=case["id"], backend=backend, median_ns=statistics.median(values),
                                samples_ns=values, iterations=iterations, max_error=float(np.max(np.abs(got-expected))),
                                preparation_ns=prepare_ns, gate_count=circuit.get_gate_count()))
    args.output.write_text(json.dumps(dict(qulacs=qulacs.__version__, threads=int(os.environ["QULACS_NUM_THREADS"]),
                                           records=records), indent=2) + "\n")


if __name__ == "__main__":
    main()
