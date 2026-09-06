#!/usr/bin/env python3
"""Run CPU baseline processes serially and preserve raw samples and provenance."""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "benchmarks/tenferro-probe/Cargo.toml"
TARGET = ROOT / "target/probe"


def run(command, env, log=None):
    if log:
        with log.open("w") as output:
            subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT,
                check=True,
            )
    else:
        subprocess.run(command, cwd=ROOT, env=env, check=True)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_criterion():
    records = []
    for path in sorted((TARGET / "criterion").glob("**/new/estimates.json")):
        folder = path.parent
        if not (folder / "benchmark.json").exists():
            continue
        spec = json.loads((folder / "benchmark.json").read_text())
        records.append(
            dict(
                id=spec["group_id"],
                backend=spec["function_id"],
                estimates=json.loads(path.read_text()),
                sample=json.loads((folder / "sample.json").read_text()),
            )
        )
    return records


def measure_evolution_memory(output, env):
    """Separate processes: model/circuit construction and native execution heap/RSS."""
    for model in ["ising", "heisenberg"]:
        for n in [3, 12]:
            for steps in [1, 16]:
                run(
                    [
                        "/usr/bin/time",
                        "-l" if sys.platform == "darwin" else "-v",
                        str(TARGET / "release/memory"),
                        "evolution",
                        model,
                        str(n),
                        str(steps),
                    ],
                    env,
                    output / f"memory-evolution-{model}-{n}-{steps}.log",
                )


def measure_circuit_ad_memory(output, env):
    """Bound expensive ordinary composition; record incomplete runs explicitly."""
    records = []
    for n in [8, 12, 16]:
        for depth in [10, 100]:
            for backend in ["native", "custom", "composed"]:
                if (
                    backend == "composed"
                    and depth == 100
                    and n != 8
                    and any(
                        r["backend"] == "composed"
                        and r["qubits"] == 8
                        and r["depth"] == 100
                        and r["status"] != "complete"
                        for r in records
                    )
                ):
                    records.append(
                        dict(
                            backend=backend,
                            qubits=n,
                            depth=depth,
                            status="not_run_after_representative_limit",
                        )
                    )
                    continue
                name = f"memory-circuit-ad-{backend}-{n}-{depth}.log"
                command = [
                    str(TARGET / "release/memory"),
                    "circuit-ad",
                    backend,
                    str(n),
                    str(depth),
                ]
                if backend == "composed":
                    command = [
                        "sh",
                        "-c",
                        'ulimit -c 0; ulimit -t 30; exec "$@"',
                        "ad-memory-limit",
                    ] + command
                with (output / name).open("w") as log:
                    result = subprocess.run(
                        ["/usr/bin/time", "-l" if sys.platform == "darwin" else "-v"]
                        + command,
                        cwd=ROOT,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                status = memory_probe_status(backend, result.returncode)
                records.append(
                    dict(
                        backend=backend,
                        qubits=n,
                        depth=depth,
                        status=status,
                        returncode=result.returncode,
                        file=name,
                        cpu_seconds_limit=30 if backend == "composed" else None,
                    )
                )
                (output / "circuit-ad-memory-status.json").write_text(
                    json.dumps(records, indent=2) + "\n"
                )
    (output / "circuit-ad-memory-status.json").write_text(
        json.dumps(records, indent=2) + "\n"
    )


def memory_probe_status(backend, returncode):
    import signal

    if returncode == 0:
        return "complete"
    if backend == "composed":
        if returncode in [-signal.SIGXCPU, 128 + signal.SIGXCPU]:
            return "cpu_limit"
        if returncode in [-signal.SIGKILL, 128 + signal.SIGKILL]:
            return "killed_with_cpu_limit"  # SIGKILL alone does not prove its cause.
    raise RuntimeError(f"AD memory probe failed: {backend}, exit={returncode}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    p.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--max-qubits", type=int, default=24)
    p.add_argument("--julia", default="julia")
    p.add_argument(
        "--suite", choices=["circuits", "evolution", "circuit-ad"], default="circuits"
    )
    a = p.parse_args()
    if min(a.threads) < 1 or a.runs < 1 or not 4 <= a.max_qubits <= 24:
        p.error("positive threads/runs and 4..24 qubits required")
    output = a.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(TARGET)
    cases = output / "cases.json"
    if a.suite == "circuits":
        run(
            [
                sys.executable,
                "benchmarks/generate_cases.py",
                str(cases),
                "--max-qubits",
                str(a.max_qubits),
            ],
            env,
        )
    env["YAO_BENCH_CASES"] = str(cases)
    build = [
        "cargo",
        "build",
        "--release",
        "--locked",
        "--manifest-path",
        str(MANIFEST),
        "--bins",
        "--benches",
    ]
    run(build, env, output / "build.log")
    if a.suite == "evolution":
        run([str(TARGET / "release/evolution_cases"), str(cases)], env)
    elif a.suite == "circuit-ad":
        run([str(TARGET / "release/ad_cases"), str(cases)], env)
    run(
        ["cargo", "bench", "--locked", "--manifest-path", str(MANIFEST), "--no-run"],
        env,
        output / "bench-build.log",
    )
    run(
        [
            a.julia,
            "--project=benchmarks/julia",
            "-e",
            "using Pkg; Pkg.instantiate(); using Yao, BenchmarkTools",
        ],
        env,
        output / "julia-setup.log",
    )
    references = ROOT / "benchmarks/data" / ("reference-" + output.name)
    run(
        [str(TARGET / "release/reference"), str(references)],
        env,
        output / "reference.log",
    )
    source_paths = [
        *sorted((ROOT / "src").rglob("*.rs")),
        *sorted((ROOT / "benchmarks").glob("*.py")),
        *sorted((ROOT / "benchmarks/tenferro-probe").rglob("*.rs")),
        MANIFEST,
        MANIFEST.with_name("Cargo.lock"),
        ROOT / "Cargo.lock",
        ROOT / "benchmarks/julia/backend_baseline.jl",
    ]
    metadata = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        platform=platform.platform(),
        machine=platform.machine(),
        cpu=platform.processor(),
        threads=a.threads,
        runs=a.runs,
        suite=a.suite,
        cases_sha256=sha(cases),
        sources={str(x.relative_to(ROOT)): sha(x) for x in source_paths},
        git_head=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        rustc=subprocess.check_output(["rustc", "-Vv"], text=True),
        cargo=subprocess.check_output(["cargo", "-V"], text=True),
        precision="complex128",
        provider="tenferro cpu-faer; native yao-rs serial kernels; omeinsum defaults",
    )
    if sys.platform == "darwin":
        metadata["hardware"] = subprocess.check_output(
            ["sysctl", "machdep.cpu.brand_string", "hw.memsize", "hw.physicalcpu"],
            text=True,
        )
    else:
        metadata["hardware"] = subprocess.check_output(["lscpu"], text=True)
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    shutil.copyfile(
        ROOT / "benchmarks/julia/Manifest.toml", output / "julia-manifest.toml"
    )
    shutil.copyfile(MANIFEST.with_name("Cargo.lock"), output / "probe-cargo.lock")
    for threads in a.threads:
        env.update(
            {
                name: str(threads)
                for name in [
                    "YAO_BENCH_THREADS",
                    "RAYON_NUM_THREADS",
                    "JULIA_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                ]
            }
        )
        for index in range(a.runs):
            prefix = f"{threads}t-run{index + 1}"
            print(prefix, flush=True)
            # Criterion's output directory is shared; remove only previous benchmark results.
            shutil.rmtree(TARGET / "criterion", ignore_errors=True)
            run(
                [
                    "cargo",
                    "bench",
                    "--manifest-path",
                    str(MANIFEST),
                    "--locked",
                    "--bench",
                    "backend",
                ],
                env,
                output / (prefix + "-rust.log"),
            )
            (output / (prefix + "-rust.json")).write_text(
                json.dumps(collect_criterion(), indent=2) + "\n"
            )
            run(
                [
                    a.julia,
                    "--project=benchmarks/julia",
                    "benchmarks/julia/backend_baseline.jl",
                    str(cases),
                    str(references),
                    str(output / (prefix + "-julia.json")),
                ],
                env,
                output / (prefix + "-julia.log"),
            )
    for n in [8, 12, 16]:
        for depth in [10, 100]:
            prefix = f"memory-{n}-{depth}"
            time_args = ["/usr/bin/time", "-l" if sys.platform == "darwin" else "-v"]
            run(
                time_args + [str(TARGET / "release/memory"), str(n), str(depth)],
                env,
                output / (prefix + ".log"),
            )
    for depth in [10, 100]:
        time_args = ["/usr/bin/time", "-l" if sys.platform == "darwin" else "-v"]
        run(
            time_args + [str(TARGET / "release/memory"), "16", str(depth), "nonlinear"],
            env,
            output / f"memory-nonlinear-16-{depth}.log",
        )
    if a.suite == "evolution":
        measure_evolution_memory(output, env)
    elif a.suite == "circuit-ad":
        measure_circuit_ad_memory(output, env)
    run(
        [sys.executable, "benchmarks/compare.py", "--backend-results", str(output)], env
    )
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
