#!/usr/bin/env python3
"""Compare Julia (Yao.jl) and Rust (yao-rs) benchmark timings."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "benchmarks" / "data"
CRITERION_DIR = ROOT / "target" / "criterion"


def load_julia_timings():
    path = DATA_DIR / "timings.json"
    if not path.exists():
        print(f"ERROR: {path} not found. Run Julia script first.")
        return None
    return json.loads(path.read_text())


def load_criterion_estimate(group: str, name: str, param: str):
    """Load Criterion's median point estimate in nanoseconds."""
    est_path = CRITERION_DIR / group / f"{name} {param}" / "new" / "estimates.json"
    if not est_path.exists():
        est_path = CRITERION_DIR / group / name / param / "new" / "estimates.json"
    if not est_path.exists():
        return None

    data = json.loads(est_path.read_text())
    return data.get("median", {}).get("point_estimate")


def main():
    julia = load_julia_timings()
    if julia is None:
        return

    rows = []

    for group_key, criterion_group in [
        ("single_gate_1q", "gates_1q"),
        ("single_gate_2q", "gates_2q"),
        ("single_gate_multi", "gates_multi"),
    ]:
        for gate_name, nq_data in julia.get(group_key, {}).items():
            for nq_str, julia_ns in nq_data.items():
                rust_ns = load_criterion_estimate(criterion_group, gate_name, nq_str)
                rows.append(("single_gate", gate_name, nq_str, julia_ns, rust_ns))

    for nq_str, julia_ns in julia.get("qft", {}).items():
        rust_ns = load_criterion_estimate("qft", "QFT", nq_str)
        rows.append(("qft", "QFT", nq_str, julia_ns, rust_ns))

    for nq_str, julia_ns in julia.get("noisy_dm", {}).items():
        rust_ns = load_criterion_estimate("noisy_dm", "noisy_dm", nq_str)
        rows.append(("noisy_dm", "full", nq_str, julia_ns, rust_ns))

    print(
        f"| {'Task':<14} | {'Gate/Circuit':<12} | {'Qubits':>6} | "
        f"{'Julia (ns)':>12} | {'Rust (ns)':>12} | {'Speedup':>8} |"
    )
    print(f"|{'-' * 16}|{'-' * 14}|{'-' * 8}|{'-' * 14}|{'-' * 14}|{'-' * 10}|")

    for task, name, nq, julia_ns, rust_ns in rows:
        julia_str = f"{julia_ns:>12.0f}" if julia_ns else "         N/A"
        if rust_ns:
            rust_str = f"{rust_ns:>12.0f}"
            speedup = f"{julia_ns / rust_ns:>7.1f}x" if julia_ns else "     N/A"
        else:
            rust_str = "         N/A"
            speedup = "     N/A"
        print(
            f"| {task:<14} | {name:<12} | {nq:>6} | {julia_str} | {rust_str} | {speedup} |"
        )


if __name__ == "__main__":
    main()
