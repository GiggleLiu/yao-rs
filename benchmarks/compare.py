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


def peak_rss_bytes(text):
    import re

    mac = re.search(r"(\d+)\s+maximum resident set size", text)
    linux = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
    if not mac and not linux:
        raise ValueError("Missing peak RSS")
    return int(mac[1]) if mac else int(linux[1]) * 1024


def circuit_ad_memory_rows(directory):
    path = Path(directory) / "circuit-ad-memory-status.json"
    if not path.exists():
        return []
    rows = json.loads(path.read_text())
    for row in rows:
        if "file" not in row:
            continue
        text = (Path(directory) / row["file"]).read_text()
        records = [
            json.loads(line) for line in text.splitlines() if line.startswith("{")
        ]
        if row["status"] == "complete" and not any(
            r.get("status") == "complete" for r in records
        ):
            raise ValueError(
                "AD memory result claims completion without a completed probe"
            )
        phases = {r["phase"]: r for r in records if "phase" in r}
        row["peak_rss_bytes"] = peak_rss_bytes(text)
        row["forward_retained_bytes"] = phases.get("circuit_ad_forward_tape", {}).get(
            "retained_additional_rust_heap_bytes"
        )
        row["backward_peak_bytes"] = phases.get(
            "circuit_ad_backward", phases.get("native_value_and_grad", {})
        ).get("peak_additional_rust_heap_bytes")
    return rows


def tensor_memory_rows(directory):
    rows = []
    for path in sorted(Path(directory).glob("memory-tensor-*.log")):
        text = path.read_text()
        records = [json.loads(line) for line in text.splitlines() if line.startswith("{")]
        if not any(r.get("status") == "complete" for r in records):
            raise ValueError(f"Incomplete tensor memory probe: {path.name}")
        metadata = next(r for r in records if "estimate" in r)
        phases = {r["phase"]: r for r in records if "phase" in r}
        rows.append(dict(metadata, peak_rss_bytes=peak_rss_bytes(text),
                         execution_peak_bytes=phases["slice_execution"]["peak_additional_rust_heap_bytes"],
                         input_retained_bytes=phases["matrix_inputs"]["retained_additional_rust_heap_bytes"]))
    return rows


def evolution_memory_rows(directory):
    """Read phase heap bytes and platform-specific time(1) peak RSS units."""

    rows = []
    for path in sorted(Path(directory).glob("memory-evolution-*.log")):
        text = path.read_text()
        records = [
            json.loads(line) for line in text.splitlines() if line.startswith("{")
        ]
        metadata = next(row for row in records if "model" in row)
        phases = {row["phase"]: row for row in records if "phase" in row}
        rows.append(
            dict(
                model=metadata["model"],
                qubits=metadata["qubits"],
                steps=metadata["steps"],
                gates=metadata["gates"],
                circuit_retained_bytes=phases["circuit_construction"][
                    "retained_additional_rust_heap_bytes"
                ],
                execution_peak_bytes=phases["native_execution"][
                    "peak_additional_rust_heap_bytes"
                ],
                peak_rss_bytes=peak_rss_bytes(text),
            )
        )
    return rows


def backend_report(directory):
    """Compare medians of independent runs; preserve individual confidence intervals."""
    import statistics
    from collections import defaultdict

    directory = Path(directory)
    values = defaultdict(list)
    errors = {}
    approximation_errors = {}
    for path in sorted(directory.glob("*t-run*-rust.json")):
        threads = int(path.name.split("t-", 1)[0])
        for row in json.loads(path.read_text()):
            values[(threads, row["id"], row["backend"])].append(
                row["estimates"]["median"]["point_estimate"]
            )
    for path in sorted(directory.glob("*t-run*-julia.json")):
        data = json.loads(path.read_text())
        for row in data["records"]:
            values[(data["threads"], row["id"], "julia")].append(row["median_ns"])
            errors[row["id"]] = max(errors.get(row["id"], 0), row["max_error"])
            if "approximation_error" in row:
                for backend, field in [
                    ("native", "approximation_error"),
                    ("julia", "yao_approximation_error"),
                ]:
                    key = (data["threads"], row["id"], backend)
                    approximation_errors[key] = max(
                        approximation_errors.get(key, 0), row[field]
                    )
    if not values:
        raise ValueError("No backend measurements found")
    metadata = json.loads((directory / "metadata.json").read_text())
    case_path = directory / "cases.json"
    expected_cases = (
        {case["id"] for case in json.loads(case_path.read_text())}
        if case_path.exists()
        else {case for (_, case, backend) in values if backend in ("native", "julia")}
    )
    expected_threads = metadata.get("threads", sorted({t for (t, _, _) in values}))
    for threads in expected_threads:
        for case in expected_cases:
            for backend in ("native", "julia"):
                count = len(values.get((threads, case, backend), []))
                if count != metadata["runs"]:
                    label = "Julia" if backend == "julia" else "native Rust"
                    raise ValueError(
                        f"Missing {label} runs for {case} at {threads} threads: {count}/{metadata['runs']}"
                    )
    summary = [
        dict(
            threads=t,
            id=case,
            backend=backend,
            median_ns=statistics.median(samples),
            min_run_median_ns=min(samples),
            max_run_median_ns=max(samples),
            runs=len(samples),
        )
        for (t, case, backend), samples in sorted(values.items())
    ]
    for row in summary:
        key = (row["threads"], row["id"], row["backend"])
        if key in approximation_errors:
            row["relative_state_error"] = approximation_errors[key]
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lookup = {(r["threads"], r["id"], r["backend"]): r for r in summary}
    lines = [
        "# CPU backend baseline",
        "",
        "Generated from raw Criterion and BenchmarkTools samples in this directory.",
        "",
        f"Platform: {metadata['platform']}. Precision: complex128. Independent runs: {metadata['runs']}.",
        "",
        *(
            ["Measurement notes: " + " ".join(metadata["measurement_notes"]), ""]
            if metadata.get("measurement_notes")
            else []
        ),
        "Times below are medians of per-process medians. Rust/Julia includes state copying; tensor phases are reported separately. A Julia/Rust ratio greater than one means native Rust was faster.",
        "",
        "| Threads | Case | Native Rust µs | Yao µs | Julia/Rust | Max output error |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        if row["backend"] != "native":
            continue
        other = lookup.get((row["threads"], row["id"], "julia"))
        if other is None:
            raise ValueError(f"Missing Julia measurement for {row['id']}")
        r, j = row["median_ns"], other["median_ns"]
        lines.append(
            f"| {row['threads']} | {row['id']} | {r / 1000:.3f} | {j / 1000:.3f} | {j / r:.2f} | {errors[row['id']]:.2e} |"
        )
    lines += [
        "",
        "## Tensor and extension phases",
        "",
        "Each row uses the named API boundary. `tenferro_from_arrays` includes conversion, automatic planning, execution and output conversion; `omeinsum` includes its conversion/planning/execution. `tenferro_warm` uses an already prepared plan. Those prototype rows use independent planning policies. Where present, `supported_planning` compiles an existing omeco greedy tree; `supported_warm` runs the supported CPU adapter including ndarray input/output adaptation; `supported_from_arrays` combines those two phases. `omeinsum_fixed_tree` executes the identical tree, including its internal preparation. These supported rows exclude tree search and CPU context creation.",
        "",
        "| Threads | Case | Phase | Median µs | Range of run medians µs |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for row in summary:
        if row["backend"] in ("native", "julia"):
            continue
        lines.append(
            f"| {row['threads']} | {row['id']} | {row['backend']} | {row['median_ns'] / 1000:.3f} | {row['min_run_median_ns'] / 1000:.3f}–{row['max_run_median_ns'] / 1000:.3f} |"
        )
    lines += [
        "",
        "Raw confidence intervals and samples are in `*-rust.json`; Julia trial samples are in `*-julia.json`. Memory logs report instrumented Rust allocations per phase and whole-process peak RSS separately. Timings from the allocation instrument are diagnostic only. See `metadata.json` and pinned manifests for reproducibility.",
        "",
    ]
    if approximation_errors:
        lines += [
            "## Product-formula accuracy",
            "",
            "Relative state error is measured against a dense exponential of the Hamiltonian built independently with Yao Pauli blocks. Timings compare the same lowered product formula on the same input; they exclude Hamiltonian/circuit construction and do not compare against adaptive Krylov execution.",
            "",
            "| Threads | Case | Native µs | Yao µs | Native relative error | Yao relative error |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for row in summary:
            if row["backend"] == "native" and "relative_state_error" in row:
                other = lookup[(row["threads"], row["id"], "julia")]
                lines.append(
                    f"| {row['threads']} | {row['id']} | {row['median_ns'] / 1000:.3f} | {other['median_ns'] / 1000:.3f} | {row['relative_state_error']:.3e} | {other['relative_state_error']:.3e} |"
                )
        lines += [
            "",
            "![Product-formula error versus execution time](evolution-error-time.svg)",
            "",
        ]
    elif metadata.get("suite") == "circuit-ad":
        lines += [
            "## Circuit AD costs and memory",
            "",
            "Every timing returns squared state-distance loss, real parameter gradients, and the complex input-state gradient. Native uses one forward pass plus its reversible sweep; the tenferro rows include input copies, eager graph construction, loss, backward and output collection, with context construction excluded. The ordinary-composition fixture uses 4×4 tensor matrices on the last two sites of the full asymmetric input; no gate fusion is applied.",
            "",
            "Repeated ordinary-composition timings cover 10 layers. The 100-layer cases are qualified separately with a 30-CPU-second cap, starting at 8 qubits; larger cases are skipped if that representative run fails to complete. The memory table records each outcome. Missing timings are not speedups. The circuit primitive and native/Yao baselines cover all six cases.",
            "",
            "![Circuit AD execution costs](circuit-ad-costs.svg)",
            "",
            "![Circuit AD memory](circuit-ad-memory.svg)",
            "",
        ]
    elif metadata.get("suite") == "tensor-memory":
        lines += [
            "## Polynomial expectations and sliced contraction",
            "",
            "Circuit rows compare the same complex polynomial expectation from a zero input, including simulation and expectation evaluation in native Rust/Yao. Yao uses sandwich for pure states and its dense operator trace formula without real projection for density matrices; native Rust applies each operator string without a dense operator matrix. Tensor rows share circuit tensors across terms. `observable_export` is separate; prepared tenferro execution excludes CPU context, tree search and compilation. omeinsum rows include executor preparation. Unsliced and term-sliced rows use the identical omeco greedy tree.",
            "",
            "Synthetic matrix-chain rows are a separate dense complex128 contraction workload, with no native/Yao simulator timing. `fixed_output` slices the first output index of the supplied ((A B) C) tree. `auto` explicitly allows omeco TreeSA replanning under the complete estimated budget; it is qualified at dimension 32 only. Heuristic slicing can produce many slice assignments. Larger fixed-path runs use dimensions 128/256. Separately labelled outer-product stress cases at dimensions 32/64 deliberately form an n^4 intermediate before contracting the third matrix; fixed output slicing is compared on that path, while a greedy unsliced path shows how planning can avoid the intermediate entirely. No automatic-planner speedup is inferred for those larger sizes.",
            "",
            "![Observable execution costs](observable-costs.svg)",
            "",
            "![Contraction time and memory](slicing-tradeoff.svg)",
            "",
        ]
    else:
        lines += [
            "## Plots",
            "",
            "![CPU scaling](cpu-scaling.svg)",
            "",
            "![Circuit and contraction costs](tensor-costs.svg)",
            "",
            "![AD memory](ad-memory.svg)",
            "",
        ]
        if any(row["backend"] == "supported_warm" for row in summary):
            lines += [
                "![Supported adapter, same contraction tree](supported-costs.svg)",
                "",
            ]
    memory = evolution_memory_rows(directory)
    if memory:
        (directory / "evolution-memory.json").write_text(
            json.dumps(memory, indent=2) + "\n"
        )
        lines += [
            "## Product-formula memory",
            "",
            "Second-order formulas, isolated native processes. Retained circuit heap is measured after construction; execution peak is additional Rust heap above the existing circuit/input state. Whole-process RSS includes startup and allocator retention. These small workloads do not establish a large-state memory limit. Allocation-instrumented times are diagnostic only.",
            "",
            "| Model | Qubits | Steps | Gates | Circuit retained KiB | Execution peak KiB | Process peak RSS MiB |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in memory:
            lines.append(
                f"| {row['model']} | {row['qubits']} | {row['steps']} | {row['gates']} | {row['circuit_retained_bytes'] / 1024:.2f} | {row['execution_peak_bytes'] / 1024:.2f} | {row['peak_rss_bytes'] / 1048576:.2f} |"
            )
        lines.append("")
    ad_memory = circuit_ad_memory_rows(directory)
    if ad_memory:
        (directory / "circuit-ad-memory.json").write_text(
            json.dumps(ad_memory, indent=2) + "\n"
        )
        lines += [
            "| Backend | Qubits | Layers | Status | Forward retained MiB | Backward peak extra MiB | Process peak RSS MiB |",
            "| --- | ---: | ---: | --- | ---: | ---: | ---: |",
        ]

        def mib(row, key):
            return "—" if row.get(key) is None else f"{row[key] / 1048576:.3f}"

        for row in ad_memory:
            lines.append(
                f"| {row['backend']} | {row['qubits']} | {row['depth']} | {row['status']} | {mib(row, 'forward_retained_bytes')} | {mib(row, 'backward_peak_bytes')} | {mib(row, 'peak_rss_bytes')} |"
            )
        lines += [
            "",
            "Native peak heap covers combined value/gradient execution. Other backward peaks are additional to retained forward storage. RSS includes startup/provider allocations and allocator retention; a terminated run's RSS is only its observed peak before termination. Rust heap counts exclude native-provider allocations. Allocation-instrumented times are diagnostic only.",
            "",
        ]
    tensor_memory = tensor_memory_rows(directory)
    if tensor_memory:
        (directory / "tensor-memory.json").write_text(json.dumps(tensor_memory, indent=2) + "\n")
        lines += [
            "## Isolated contraction memory",
            "",
            "One active slice; all storage columns are MiB. Input and full output storage remain allocated. Estimates conservatively count tensor buffers and a zero user workspace reserve; they exclude runtime metadata, compiled programs, allocator retention and unreported provider scratch. Process RSS is measured independently and is not bounded by that estimate. Heap-instrumented execution times are diagnostic only.",
            "",
            "| Backend | Fixture | Matrix dimension | Mode | Slices | Input | Output | omeco peak | Worker buffers | Total estimate | Execution extra heap | Process peak RSS |",
            "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in tensor_memory:
            e = row["estimate"]
            quantities = [e[k] for k in ("input_bytes", "output_bytes", "omeco_peak_bytes", "worker_buffer_bytes", "estimated_total_bytes")]
            quantities += [row["execution_peak_bytes"], row["peak_rss_bytes"]]
            lines.append(f"| {row['backend']} | {row['kind']} | {row['dimension']} | {row['mode']} | {e['slices']} | " + " | ".join(f"{x / 1048576:.3f}" for x in quantities) + " |")
        lines.append("")
    (directory / "report.md").write_text("\n".join(lines))
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--backend-results", type=Path)
    args = parser.parse_args()
    if args.backend_results:
        backend_report(args.backend_results)
    else:
        main()
