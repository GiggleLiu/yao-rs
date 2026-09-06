#!/usr/bin/env python3
"""Generate identical serialized circuit workloads for Rust and Julia runners."""
import argparse
import json
import math
from pathlib import Path


def gate(name, targets, params=None, controls=None, configs=None):
    result = dict(type="gate", gate=name, targets=targets)
    for key, value in [("params", params), ("controls", controls), ("control_configs", configs)]:
        if value is not None:
            result[key] = value
    return result


def cases(max_qubits=24):
    result = []
    def add(name, n, elements, mode="state", tensor=False):
        result.append(dict(id=f"{name}_{n}", mode=mode, tensor=tensor,
                           initial="deterministic" if mode=="state" and not tensor else "zero",
                           circuit=dict(num_qubits=n, elements=elements)))
    for n in [8, 12, 16, 20, 24]:
        if n > max_qubits:
            continue
        for name, elements in [
            ("rx", [gate("Rx", [n-1], [0.37])]),
            ("rz", [gate("Rz", [0], [-0.61])]),
            ("swap_far", [gate("SWAP", [0,n-1])]),
            ("cry_low_far", [gate("Ry", [n-1], [0.31], [0], [False])]),
            ("fsim_adjacent", [gate("FSim", [n-2,n-1], [0.21,-0.43])]),
        ]:
            add(name,n,elements)
        qft=[]
        for i in range(n):
            qft.append(gate("H",[i]))
            for j in range(1,n-i):
                qft.append(gate("Phase",[i],[2*math.pi/(2**(j+1))],[i+j]))
        add("qft",n,qft)
    for n in [4,8,12]:
        if n > max_qubits: continue
        for depth in [10,100]:
            ops=[]
            for layer in range(depth):
                for q in range(n):
                    ops.append(gate("Ry",[q],[0.1+0.01*q+0.001*layer]))
                for q in range(n-1): ops.append(gate("X",[q+1],controls=[q]))
            add(f"layers{depth}",n,ops)
            add(f"gradient{depth}",n,ops,mode="gradient")
    for n in [4,6,8,10]:
        if n > max_qubits: continue
        ops=[gate("H",[q]) for q in range(n)]
        ops += [gate("X",[q+1],controls=[q]) for q in range(n-1)]
        ops += [dict(type="channel",locs=[q],channel="Depolarizing",n=1,p=0.01) for q in range(n)]
        ops += [gate("Rz",[q],[0.3]) for q in range(n)]
        ops += [dict(type="channel",locs=[q],channel="AmplitudeDamping",gamma=0.05,excited_population=0.0) for q in range(n)]
        add("noisy",n,ops,mode="density",tensor=n<=6)
    for n in [4,8]:
        if n > max_qubits: continue
        ops=[gate("Ry",[q],[0.31+q*0.07]) for q in range(n)]
        ops += [gate("Rx",[q],[-0.17]) for q in range(n)]
        ops += [gate("X",[q+1],controls=[q]) for q in range(n-1)]
        add("tensor_state",n,ops,tensor=True)
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("output",type=Path)
    parser.add_argument("--max-qubits",type=int,default=24)
    args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(cases(args.max_qubits),indent=2)+"\n")
