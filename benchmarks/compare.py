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


def trajectory_rows(directory):
    """Read completed diagnostics and deduplicate repeated timing-process seeds."""
    import math
    from collections import defaultdict

    unique = {}
    files = sorted(Path(directory).glob("*-trajectory-stats.jsonl"))
    for path in files:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get("status") != "complete":
                raise ValueError("Incomplete trajectory diagnostic")
            stats = row["statistics"]
            if stats["trajectories"] < 2 or stats["standard_error"] is None:
                raise ValueError("Trajectory diagnostic needs uncertainty")
            values = [*stats["mean"], *stats["standard_error"], *row["expected"], *row["error"]]
            if not all(math.isfinite(x) for x in values) or min(stats["standard_error"]) < 0:
                raise ValueError("Nonfinite or negative trajectory uncertainty")
            error = complex(*stats["mean"]) - complex(*row["expected"])
            if abs(error-complex(*row["error"])) > 1e-14:
                raise ValueError("Trajectory error disagrees with its mean/reference")
            key = (stats["threads"], row["id"], stats["trajectories"], stats["seed"])
            if key in unique and unique[key] != row:
                raise ValueError("Repeated trajectory seed is not reproducible")
            unique[key] = row
    groups = defaultdict(list)
    for (threads, case, count, seed), row in unique.items():
        groups[(threads, case, count)].append(row)
    rows = []
    for (threads, case, count), samples in sorted(groups.items()):
        mse = sum(abs(complex(*r["error"]))**2 for r in samples)/len(samples)
        predicted = sum(sum(x*x for x in r["statistics"]["standard_error"]) for r in samples)/len(samples)
        rows.append(dict(threads=threads, id=case, trajectories=count, seeds=len(samples),
                         rmse=math.sqrt(mse), predicted_rms_error=math.sqrt(predicted)))
    return rows


def trajectory_memory_rows(directory):
    rows = []
    for path in sorted(Path(directory).glob("memory-trajectory-*.log")):
        text = path.read_text()
        records = [json.loads(line) for line in text.splitlines() if line.startswith("{")]
        completed = [r for r in records if r.get("status") == "complete"]
        if len(completed) != 1:
            raise ValueError("Incomplete trajectory memory probe")
        row = dict(completed[0])
        phases = {r["phase"]: r for r in records if "phase" in r}
        phase = phases[row["backend"] + "_execute"]
        row.update(peak_rss_bytes=peak_rss_bytes(text),
                   execution_peak_bytes=phase["peak_additional_rust_heap_bytes"],
                   execution_retained_bytes=phase["retained_additional_rust_heap_bytes"])
        rows.append(row)
    return sorted(rows, key=lambda r: (r["backend"], r["qubits"], r["trajectories"], r["threads"]))


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


def krylov_diagnostics(directory, metadata, cases):
    """Require completed full-time diagnostics for every measured Krylov case."""
    import math

    directory = Path(directory)
    expected = {c["id"]: c["krylov"] for c in cases if c["mode"] == "krylov"}
    names = {f"{t}t-run{r}-krylov-stats.jsonl" for t in metadata["threads"]
             for r in range(1, metadata["runs"] + 1)}
    if {p.name for p in directory.glob("*-krylov-stats.jsonl")} != names:
        raise ValueError("Missing Krylov diagnostic processes")
    rows = []
    for name in sorted(names):
        records = [json.loads(line) for line in (directory / name).read_text().splitlines()]
        if len(records) != len(expected) or {r["id"] for r in records} != set(expected):
            raise ValueError("Missing or duplicate Krylov workload diagnostics")
        threads = int(name.split("t-", 1)[0])
        for row in records:
            info, spec = row["info"], expected[row["id"]]
            if row["status"] != "complete" or info["time_reached"] != spec["time"]:
                raise ValueError("Krylov diagnostic did not reach requested time")
            if (not all(math.isfinite(v) for v in info.values())
                    or not 0 <= info["estimated_error"] <= info["tolerance"]
                    or not 0 < info["tolerance"] <= spec["rtol"] * (1 + 1e-12)
                    or info["matvecs"] < 1 or info["steps"] < 1
                    or not 1 <= info["max_krylov_dim"] <= spec["krylov_dim"]):
                raise ValueError("Invalid Krylov convergence diagnostic")
            rows.append(dict(row, threads=threads, process=name))
        julia = json.loads((directory / name.replace("-krylov-stats.jsonl", "-julia.json")).read_text())
        selected = [r for r in julia["records"] if r["id"] in expected]
        if len(selected) != len(expected) or {r["id"] for r in selected} != set(expected):
            raise ValueError("Missing Krylov Julia qualification")
        for row in selected:
            vals = [row[k] for k in ("approximation_error", "yao_approximation_error", "tight_reference_error")]
            if (not all(math.isfinite(x) and x >= 0 for x in vals)
                    or row["tight_reference_error"] >= 1e-11
                    or row["yao_krylov"]["converged"] != 1):
                raise ValueError("Invalid Krylov Julia qualification")
    return rows


def krylov_memory_rows(directory):
    import math

    rows = []
    for path in sorted(Path(directory).glob("memory-krylov-*.log")):
        text = path.read_text()
        records = [json.loads(line) for line in text.splitlines() if line.startswith("{")]
        complete = [r for r in records if r.get("status") == "complete"]
        if len(complete) != 1:
            raise ValueError("Incomplete Krylov memory probe")
        row = complete[0]
        if row["info"]["time_reached"] != row["time"]:
            raise ValueError("Krylov memory probe returned a partial state")
        error, tolerance = row["info"]["estimated_error"], row["info"]["tolerance"]
        if not math.isfinite(error) or not math.isfinite(tolerance) or not 0 <= error <= tolerance:
            raise ValueError("Invalid Krylov memory error estimate")
        phases = {r["phase"]: r for r in records if "phase" in r}
        row.update(peak_rss_bytes=peak_rss_bytes(text),
                   execution_peak_bytes=phases["krylov_execute"]["peak_additional_rust_heap_bytes"],
                   execution_retained_bytes=phases["krylov_execute"]["retained_additional_rust_heap_bytes"])
        rows.append(row)
    return sorted(rows, key=lambda r: (r["model"], r["qubits"], r["krylov_dim"]))


def backend_report(directory):
    """Compare medians of independent runs; preserve individual confidence intervals."""
    import statistics
    from collections import defaultdict

    directory = Path(directory)
    values = defaultdict(list)
    errors = {}
    approximation_errors = {}
    for path in sorted([*directory.glob("*t-run*-rust.json"), *directory.glob("*t-run*-gpu.json")]):
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
    cuda_diagnostics = {case["id"] for case in json.loads(case_path.read_text()) if case.get("cuda_diagnostic_only", False)} if metadata.get("suite") == "cuda" else set()
    for threads in expected_threads:
        for case in expected_cases:
            required = ["native", "julia"]
            if metadata.get("suite") == "cuda" and case not in cuda_diagnostics:
                required += ["cuda_resident", "cuda_transfer_inclusive"]
            for backend in required:
                count = len(values.get((threads, case, backend), []))
                if count != metadata["runs"]:
                    label = {"julia": "Julia", "native": "native Rust"}.get(backend, backend)
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
        "# CUDA and CPU comparison" if metadata.get("suite") == "cuda" else "# CPU backend baseline",
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
        ("Times below are medians of per-process medians and include solver output allocation. Julia/Rust ratios compare different solver policies at the listed tolerances; consult achieved errors below before comparing efficiency. Product-formula rows compare the same circuit. Tensor phases are separate."
         if metadata.get("suite") == "krylov" else
         "Times below are medians of per-process medians. Rust/Julia includes state copying; tensor phases are reported separately. A Julia/Rust ratio greater than one means native Rust was faster."),
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
        "## Trajectory and tensor phases" if metadata.get("suite") == "trajectories" else "## Tensor and extension phases",
        "",
        ("Trajectory rows report complete ensembles; `trajectory_prepare` validates/prepares local gates and channels. Tenferro rows report exact expectation-network compilation and warm contraction. Their accuracy and timing boundaries are detailed below." if metadata.get("suite") == "trajectories" else
"Each row uses the named API boundary. `tenferro_from_arrays` includes conversion, automatic planning, execution and output conversion; `omeinsum` includes its conversion/planning/execution. `tenferro_warm` uses an already prepared plan. Those prototype rows use independent planning policies. Where present, `supported_planning` compiles an existing omeco greedy tree; `supported_warm` runs the supported CPU adapter including ndarray input/output adaptation; `supported_from_arrays` combines those two phases. `omeinsum_fixed_tree` executes the identical tree, including its internal preparation. These supported rows exclude tree search and CPU context creation."),
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
        ("Raw confidence intervals and samples are in `*-rust.json` and `*-gpu.json`; Julia trial samples are in `*-julia.json`. CUDA memory logs report device process snapshots and host peak RSS, with the measurement limits described below. See `metadata.json` and pinned manifests for reproducibility."
         if metadata.get("suite") == "cuda" else
         "Raw confidence intervals and samples are in `*-rust.json`; Julia trial samples are in `*-julia.json`. Memory logs report instrumented Rust allocations per phase and whole-process peak RSS separately. Timings from the allocation instrument are diagnostic only. See `metadata.json` and pinned manifests for reproducibility."),
        "",
    ]
    if approximation_errors:
        lines += [
            "## Evolution accuracy" if metadata.get("suite") == "krylov" else "## Product-formula accuracy",
            "",
            ("Krylov rows time the public Rust adaptive solver and Yao TimeEvolution on the same state/Hamiltonian. Model construction is excluded; solver buffers and Rust Pauli-mask preparation are included. Tolerances have different meanings: Rust targets a final norm error, whereas Yao/KrylovKit uses its own estimate and time scaling. Compare achieved errors, not the timing ratio alone. Four/eight-qubit oracles use a dense exponential; larger oracles use KrylovKit at tol=1e-13, independently checked against tighter Rust results. Product rows use the same four-qubit zero state and Suzuki circuit in both languages. Their tensor phases provide tenferro/omeinsum costs for that product approximation, not an adaptive tenferro Krylov implementation."
             if metadata.get("suite") == "krylov" else
             "Relative state error is measured against a dense exponential of the Hamiltonian built independently with Yao Pauli blocks. Timings compare the same lowered product formula on the same input; they exclude Hamiltonian/circuit construction and do not compare against adaptive Krylov execution."),
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
            "![Evolution error versus execution time](krylov-error-time.svg)" if metadata.get("suite") == "krylov" else "![Product-formula error versus execution time](evolution-error-time.svg)",
            "",
        ]
    if metadata.get("suite") == "krylov":
        cases = json.loads(case_path.read_text())
        diagnostics = krylov_diagnostics(directory, metadata, cases)
        memory_rows = krylov_memory_rows(directory)
        required = {(model, n, k) for model in ("ising", "heisenberg")
                    for n in (4, 8, 12, 16) for k in (8, 20, 40)}
        if len(memory_rows) != len(required) or {(r["model"], r["qubits"], r["krylov_dim"]) for r in memory_rows} != required:
            raise ValueError("Missing or duplicate Krylov memory workloads")
        (directory / "krylov-diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
        (directory / "krylov-memory.json").write_text(json.dumps(memory_rows, indent=2) + "\n")
        lines += ["## Krylov convergence and memory", "",
                  "Raw per-process Krylov diagnostics record completed time, operator applications, accepted steps, maximum basis dimension, truncation/defect estimate and tolerance. Julia records include its own convergence diagnostics and the discrepancy between tight references. Floating-point roundoff is not certified by either reported estimate.", "",
                  "| Model | Qubits | Basis cap | Basis used | Steps | Matvecs | Additional execution heap MiB | Process peak RSS MiB |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for r in memory_rows:
            i = r["info"]
            lines.append(f"| {r['model']} | {r['qubits']} | {r['krylov_dim']} | {i['max_krylov_dim']} | {i['steps']} | {i['matvecs']} | {r['execution_peak_bytes']/2**20:.3f} | {r['peak_rss_bytes']/2**20:.3f} |")
        lines += ["", "Memory probes use rtol=1e-8. The input state is already live before execution; additional heap includes basis/work/output storage. RSS includes input, startup and allocator retention. Instrumented times are excluded from timing tables.", "",
                  "![Krylov basis memory](krylov-memory.svg)", ""]
        lines += ["![Four-qubit product formulas and adaptive evolution](krylov-product-comparison.svg)", ""]
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
    elif not approximation_errors and metadata.get("suite") != "trajectories":
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
    if metadata.get("suite") == "trajectories":
        import math
        diagnostics = sorted(directory.glob("*-trajectory-stats.jsonl"))
        expected_names = {f"{t}t-run{r}-trajectory-stats.jsonl" for t in metadata["threads"] for r in range(1, metadata["runs"]+1)}
        if {p.name for p in diagnostics} != expected_names:
            raise ValueError("Missing trajectory diagnostic processes")
        for path in diagnostics:
            records = [json.loads(line) for line in path.read_text().splitlines()]
            if any(r["statistics"]["threads"] != int(path.name.split("t-")[0]) for r in records):
                raise ValueError("Trajectory diagnostic thread count disagrees with process")
            keys = {(r["id"], r["statistics"]["trajectories"], r["statistics"]["seed"]) for r in records}
            expected_keys = {(f"noisy_expectation_{n}", count, seed)
                             for n in [4, 6, 8] for count in [128, 512, 2048]
                             for seed in ([19, 7, 42, 73, 101, 137, 211, 307] if n == 6 else [19])}
            expected_keys |= {(f"product_noise_{n}", count, 19) for n in [12, 16] for count in [64, 256]}
            if len(records) != len(expected_keys) or keys != expected_keys:
                raise ValueError("Missing or duplicate trajectory diagnostic workloads")
        trajectory = trajectory_rows(directory)
        memory_rows = trajectory_memory_rows(directory)
        expected_memory = {("trajectory", n, count, t) for n in [4,8,10,12,16] for count in [64,256] for t in [1,4]}
        expected_memory |= {("density",n,0,1) for n in [4,8,10]}
        if len(memory_rows) != len(expected_memory) or {(r["backend"],r["qubits"],r["trajectories"],r["threads"]) for r in memory_rows} != expected_memory:
            raise ValueError("Missing trajectory memory workloads")
        (directory / "trajectory-statistics.json").write_text(json.dumps(trajectory, indent=2)+"\n")
        (directory / "trajectory-memory.json").write_text(json.dumps(memory_rows, indent=2)+"\n")
        lines += ["", "## Trajectory accuracy and time", "",
            "Exact native/Yao rows above evolve the same density matrix and complex polynomial. Tenferro exact warm rows contract a prepared expectation network; preparation excludes export and greedy order search. Trajectory timings include independent seeded evolution, observable evaluation, streaming moments, worker buffers and (at multiple threads) pool creation; local channel preparation is reported separately. These algorithms have different accuracy, so a timing ratio alone is not a speedup at equal error.", "",
            "The six-qubit case uses eight independent seeds per sample count; other rows use one. Repeated timing processes with the same seed are deduplicated for accuracy. RMSE combines real and imaginary errors. Predicted RMS error combines their standard errors. Large product-state cases use analytic expectations and are simpler workloads than the entangled exact-density cases.", "",
            "| Threads | Case | Trajectories | Independent seeds | Ensemble ms | Observed RMSE | Predicted RMS error |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
        for row in trajectory:
            timing = lookup.get((row["threads"], row["id"], f"trajectory_{row['trajectories']}"))
            if timing is None or timing["runs"] != metadata["runs"]:
                raise ValueError("Missing trajectory timing")
            assert math.isfinite(row["rmse"])
            lines.append(f"| {row['threads']} | {row['id']} | {row['trajectories']} | {row['seeds']} | {timing['median_ns']/1e6:.4f} | {row['rmse']:.4e} | {row['predicted_rms_error']:.4e} |")
        lines += ["", "![Trajectory sampling error versus time](trajectory-error-time.svg)", "",
            "## Trajectory memory", "",
            "Twenty trajectory probes and three exact-density probes run in isolation. Heap peaks describe the execution phase beyond live inputs/prepared channels; exact density storage is already live when its execution begins. RSS includes inputs, startup and allocator retention. Diagnostic instrumented times are excluded from timing tables. Qubits 4/8 use entangled fixtures; 10/12/16 use product fixtures. Native density execution is serial.", "",
            "| Backend | Qubits | Trajectories | Threads | Execution additional heap MiB | Peak RSS MiB |",
            "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for row in memory_rows:
            lines.append(f"| {row['backend']} | {row['qubits']} | {row['trajectories']} | {row['threads']} | {row['execution_peak_bytes']/2**20:.4f} | {row['peak_rss_bytes']/2**20:.3f} |")
        lines += ["", "![Trajectory state memory and sample count](trajectory-memory.svg)", ""]

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
    if metadata.get("suite") == "cuda":
        lines.extend(cuda_report_sections(directory, summary))
    (directory / "report.md").write_text("\n".join(lines))
    return summary


def cuda_report_sections(directory, summary):
    """Require successful full-output qualification before reporting GPU results."""
    import math
    import re

    directory = Path(directory)
    cases = json.loads((directory / "cases.json").read_text())
    lookup = {(row["id"], row["backend"]): row for row in summary}
    memory = []
    lines = ["", "## CUDA execution and transfers", "",
        "Synchronized resident execution includes host dispatch, allocation and fresh AD leaves, with inputs already on the device. Transfer-inclusive execution additionally uploads every input and downloads complete outputs. Both reuse prepared structure and constants. Context creation and preparation are excluded. CPU native/Yao rows include state copies. CPU tenferro AD uses the reversible custom primitive; GPU AD uses ordinary tensor composition.", "",
        "| Case | GPU resident ms | GPU transfer-inclusive ms | Native CPU / GPU resident | GPU max output error |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for case in cases:
        text = (directory / f"memory-{case['id']}.log").read_text()
        if not re.search(r"Exit status:\s*0\s*$", text):
            raise ValueError(f"GPU qualification process did not exit successfully: {case['id']}")
        records = [json.loads(line) for line in text.splitlines() if line.startswith("{")]
        complete = [row for row in records if row.get("status") == "complete"]
        if len(complete) != 1 or complete[0]["id"] != case["id"]:
            raise ValueError(f"Missing complete GPU qualification for {case['id']}")
        record = complete[0]
        output_errors = [record["max_error"], record["transfer_max_error"]]
        if any(not math.isfinite(value) or not 0 <= value <= 1e-9 for value in output_errors):
            raise ValueError(f"GPU qualification error for {case['id']}: {output_errors}")
        error = max(output_errors)
        samples = record.get("resident_samples_ns", [])
        if not samples or any(not math.isfinite(x) or x <= 0 for x in samples):
            raise ValueError(f"Invalid GPU diagnostic samples for {case['id']}")
        phases = {row["phase"]: row["device_process_bytes"] for row in records if "phase" in row}
        if set(phases) != {"context", "prepared", "first_result", "after_repeats"}:
            raise ValueError(f"Incomplete GPU memory snapshots for {case['id']}")
        if any(not math.isfinite(value) or value < 0 for value in phases.values()):
            raise ValueError("Invalid GPU memory snapshot")
        memory.append({**record, "snapshots": phases, "peak_rss_bytes": peak_rss_bytes(text)})
        if case.get("cuda_diagnostic_only", False):
            lines.append(f"| {case['id']} | Diagnostic only | Diagnostic only | — | {error:.2e} |")
            continue
        resident = lookup[(case["id"], "cuda_resident")]["median_ns"]
        transfer = lookup[(case["id"], "cuda_transfer_inclusive")]["median_ns"]
        native = lookup[(case["id"], "native")]["median_ns"]
        if any(not math.isfinite(x) or x <= 0 for x in [resident, transfer, native]):
            raise ValueError("Invalid CUDA comparison timing")
        lines.append(f"| {case['id']} | {resident / 1e6:.3f} | {transfer / 1e6:.3f} | {native / resident:.3g} | {error:.2e} |")
    lines += ["", "A CPU/GPU ratio greater than one means this GPU boundary was faster. These are medians of independent process medians; raw confidence intervals and samples are preserved. Transfer-inclusive figures are distinct from a complete setup-inclusive application run.", "",
        "## Process-cold setup and memory snapshots", "",
        "One process per case, with the persistent compiler/driver disk caches left in place. Context, preparation plus input uploads, and first synchronized execution are timed separately. NVIDIA process-memory snapshots include context, workspaces and the allocator pool. They are observed snapshots, not exact live tensor bytes or a continuous peak. Host peak RSS is a separate process metric and includes the native correctness reference.", "",
        "| Case | Context ms | Prepare + upload ms | First resident ms | Warm diagnostic ms | Context GPU MiB | Prepared GPU MiB | First result GPU MiB | After repeats GPU MiB | Host peak MiB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in memory:
        times = [row[k] / 1e6 for k in ["context_ns", "preparation_upload_ns", "first_resident_ns"]]
        import statistics
        times.append(statistics.median(row["resident_samples_ns"]) / 1e6)
        sizes = [row["snapshots"][k] / 1048576 for k in ["context", "prepared", "first_result", "after_repeats"]]
        sizes.append(row["peak_rss_bytes"] / 1048576)
        lines.append(f"| {row['id']} | " + " | ".join(f"{x:.3f}" for x in times + sizes) + " |")
    (directory / "cuda-qualification.json").write_text(json.dumps(memory, indent=2) + "\n")
    lines += ["", "Warm diagnostic values summarize samples within one probe process; they are not independent-run statistics. Deep cases marked diagnostic-only have one such sample and are excluded from the repeated GPU latency comparison.", "", "![CUDA and CPU latency](cuda-latency.svg)", "", "![Observed GPU memory versus depth](cuda-memory.svg)", ""]
    return lines


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--backend-results", type=Path)
    args = parser.parse_args()
    if args.backend_results:
        backend_report(args.backend_results)
    else:
        main()
