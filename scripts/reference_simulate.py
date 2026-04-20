#!/usr/bin/env python3
"""Reference numpy simulator for yao-rs circuit JSON.

An independent state-vector simulator used by the tutorial pages to
cross-check the `yao` CLI. Supports only the gate set used in the docs
examples: H, X, Y, Z, Rx, Ry, Rz, Phase, SWAP, with optional `controls`
and `control_configs` (default active-high).

Usage:
    python3 scripts/reference_simulate.py CIRCUIT.json --probs
    python3 scripts/reference_simulate.py CIRCUIT.json --op "Z(0)Z(1)"
    cat CIRCUIT.json | python3 scripts/reference_simulate.py - --probs

Bit convention: qubit q_t occupies bit position (n-1-t) in the
probability-array index -- qubit 0 is the MSB. This matches the yao-rs
CLI convention documented at docs/src/conventions.md.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys

import numpy as np

S2 = 1.0 / math.sqrt(2.0)


def gate_matrix(name: str, params):
    if name == "H":
        return np.array([[S2, S2], [S2, -S2]], dtype=complex)
    if name == "X":
        return np.array([[0, 1], [1, 0]], dtype=complex)
    if name == "Y":
        return np.array([[0, -1j], [1j, 0]], dtype=complex)
    if name == "Z":
        return np.array([[1, 0], [0, -1]], dtype=complex)
    if name == "Rx":
        theta = float(params[0])
        c, s = math.cos(theta / 2), math.sin(theta / 2)
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)
    if name == "Ry":
        theta = float(params[0])
        c, s = math.cos(theta / 2), math.sin(theta / 2)
        return np.array([[c, -s], [s, c]], dtype=complex)
    if name == "Rz":
        theta = float(params[0])
        return np.array(
            [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]],
            dtype=complex,
        )
    if name == "Phase":
        phi = float(params[0])
        return np.array([[1, 0], [0, np.exp(1j * phi)]], dtype=complex)
    if name == "SWAP":
        return np.array(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
            dtype=complex,
        )
    raise ValueError(f"Gate not supported by reference simulator: {name}")


def apply_gate(state, matrix, targets, n, controls=None, control_configs=None):
    controls = list(controls or [])
    if control_configs is None:
        control_configs = [True] * len(controls)
    nt = len(targets)
    tpos = [n - 1 - t for t in targets]
    cpos = [n - 1 - c for c in controls]

    def extract_targets(idx):
        v = 0
        for k, p in enumerate(tpos):
            v |= ((idx >> p) & 1) << (nt - 1 - k)
        return v

    def set_targets(idx, v):
        for k, p in enumerate(tpos):
            b = (v >> (nt - 1 - k)) & 1
            idx = (idx & ~(1 << p)) | (b << p)
        return idx

    new_state = state.copy()
    for i in range(2 ** n):
        if any(((i >> cp) & 1) != int(cc) for cp, cc in zip(cpos, control_configs)):
            continue
        t_in = extract_targets(i)
        amp = 0j
        for t_out in range(2 ** nt):
            j = set_targets(i, t_out)
            amp += matrix[t_in, t_out] * state[j]
        new_state[i] = amp
    return new_state


def simulate(circuit):
    n = circuit["num_qubits"]
    state = np.zeros(2 ** n, dtype=complex)
    state[0] = 1.0
    for el in circuit["elements"]:
        if el.get("type") != "gate":
            continue
        M = gate_matrix(el["gate"], el.get("params", []))
        state = apply_gate(
            state,
            M,
            el["targets"],
            n,
            el.get("controls", []),
            el.get("control_configs"),
        )
    return state


def probs_from_state(state):
    return (state.conj() * state).real


_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def op_matrix(spec, n):
    tokens = re.findall(r"([XYZI])\((\d+)\)", spec)
    if not tokens:
        raise ValueError(f"Could not parse Pauli operator spec: {spec!r}")
    per_q = [np.eye(2, dtype=complex) for _ in range(n)]
    for letter, idx in tokens:
        i = int(idx)
        per_q[i] = _PAULI[letter] @ per_q[i]
    op = per_q[0]
    for m in per_q[1:]:
        op = np.kron(op, m)
    return op


def main():
    ap = argparse.ArgumentParser(
        description="Reference numpy simulator for yao-rs circuit JSON."
    )
    ap.add_argument("input", help="Circuit JSON path, or '-' for stdin")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--probs",
        action="store_true",
        help="Print the probability vector as JSON",
    )
    group.add_argument(
        "--op",
        type=str,
        help='Pauli operator spec, e.g. "Z(0)Z(1)"; prints the expectation value',
    )
    args = ap.parse_args()

    text = sys.stdin.read() if args.input == "-" else open(args.input).read()
    circuit = json.loads(text)
    state = simulate(circuit)

    if args.probs:
        p = probs_from_state(state).tolist()
        print(json.dumps({"probabilities": p, "num_qubits": circuit["num_qubits"]}))
    else:
        op = op_matrix(args.op, circuit["num_qubits"])
        ev = complex(np.vdot(state, op @ state))
        print(
            json.dumps(
                {
                    "expectation_value": {
                        "re": float(ev.real),
                        "im": float(ev.imag),
                    }
                }
            )
        )


if __name__ == "__main__":
    main()
