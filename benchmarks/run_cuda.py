#!/usr/bin/env python3
"""Serial CUDA/CPU/Yao comparison using shared fixtures and Criterion collectors."""

import argparse
import json
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from run_backend import MANIFEST, ROOT, collect_criterion, run, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--target", type=Path, default=ROOT / "target/probe")
    parser.add_argument("--julia", default="julia")
    parser.add_argument("--julia-project", type=Path, required=True)
    parser.add_argument("--yao-source", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--qualification-timeout", type=int, default=300)
    parser.add_argument("--repeat-deep-gradients", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.runs < 1 or args.qualification_timeout < 1:
        parser.error("runs and qualification timeout must be positive")
    output, target = args.output.resolve(), args.target.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(CARGO_TARGET_DIR=str(target), YAO_BENCH_SUITE="cuda")
    for name in [
        "YAO_BENCH_THREADS",
        "RAYON_NUM_THREADS",
        "JULIA_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
    ]:
        env[name] = "1"
    build = [
        "cargo",
        "build",
        "--release",
        "--locked",
        "--manifest-path",
        str(MANIFEST),
        "--features",
        "cuda",
        "--bins",
        "--benches",
    ]
    run(build, env, output / "build.log")
    cases_path = output / "cases.json"
    run(
        [str(target / "release/cuda_cases"), str(cases_path)]
        + (["--smoke"] if args.smoke else []),
        env,
    )
    env["YAO_BENCH_CASES"] = str(cases_path)
    cases = json.loads(cases_path.read_text())
    if args.repeat_deep_gradients:
        for case in cases:
            case["cuda_diagnostic_only"] = False
        cases_path.write_text(json.dumps(cases, indent=2) + "\n")
    sources = []
    for folder in [
        "src",
        "bitbasis/src",
        "benchmarks/tenferro-probe/src",
        "benchmarks/tenferro-probe/benches",
    ]:
        sources.extend((ROOT / folder).rglob("*.rs"))
    sources.extend(
        ROOT / name
        for name in [
            "Cargo.toml",
            "Cargo.lock",
            "bitbasis/Cargo.toml",
            "benchmarks/tenferro-probe/Cargo.toml",
            "benchmarks/tenferro-probe/Cargo.lock",
            "benchmarks/julia/backend_baseline.jl",
            "benchmarks/run_cuda.py",
            "benchmarks/run_backend.py",
            "benchmarks/compare.py",
        ]
    )

    def capture(command):
        return subprocess.check_output(command, cwd=ROOT, env=env, text=True).strip()

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "suite": "cuda",
        "qualification_timeout_seconds": args.qualification_timeout,
        "runs": args.runs,
        "threads": [1],
        "precision": "complex128",
        "git_head": capture(["git", "rev-parse", "HEAD"]),
        "git_status": capture(["git", "status", "--short"]),
        "sources": {str(p.relative_to(ROOT)): sha(p) for p in sorted(sources)},
        "cases_sha256": sha(cases_path),
        "platform": platform.platform(),
        "hardware": capture(["lscpu"]),
        "rustc": capture(["rustc", "-Vv"]),
        "gpus": capture(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,memory.total,memory.used,utilization.gpu",
                "--format=csv",
            ]
        ),
        "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
        "yao_commit": capture(["git", "-C", str(args.yao_source), "rev-parse", "HEAD"]),
        "yao_status": capture(["git", "-C", str(args.yao_source), "status", "--short"]),
        "julia": capture([args.julia, "--startup-file=no", "--version"]),
        "runtime_environment": {
            k: env.get(k)
            for k in [
                "CUDA_PATH",
                "TENFERRO_CUTENSOR_PATH",
                "LD_LIBRARY_PATH",
                "TENFERRO_EAGER_WHOLE_PROGRAM",
            ]
        },
        "provider": "tenferro 0.4.0 CUDA/cuTENSOR; CPU faer 1 thread; native serial kernels",
        "process_niceness": os.getpriority(os.PRIO_PROCESS, 0),
        "measurement_notes": [
            "CUDA and CPU tenferro custom-loss cases request parameter and input-state gradients with two targeted pullbacks; native Rust and Yao use their joint reversible pullback.",
            "GPU resident execution includes fresh device copies for tracked inputs, allocation and host dispatch. Transfers and prepared constants are separate boundaries.",
            "The host is shared. GPU inventory is captured before measurement, CPU affinity and GPU clocks are not pinned; use run-to-run dispersion when interpreting results.",
            "Cases marked cuda_diagnostic_only have one process-cold probe and one warm diagnostic sample, with full-output checks; their expensive deep gradients are excluded from repeated GPU timings.",
        ],
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    shutil.copyfile(
        args.julia_project / "Manifest.toml", output / "julia-manifest.toml"
    )
    shutil.copyfile(args.julia_project / "Project.toml", output / "julia-project.toml")
    shutil.copyfile(MANIFEST.with_name("Cargo.lock"), output / "probe-cargo.lock")
    references = output / "references"
    run(
        [str(target / "release/reference"), str(references)],
        env,
        output / "reference.log",
    )
    # Each cold/memory probe has an independent process and CUDA allocator.
    for case in cases:
        print("qualify", case["id"], flush=True)
        run(
            [
                "timeout",
                "--kill-after=10s",
                f"{args.qualification_timeout}s",
                "/usr/bin/time",
                "-v",
                str(target / "release/cuda_probe"),
                case["id"],
            ],
            env,
            output / f"memory-{case['id']}.log",
        )
    for index in range(1, args.runs + 1):
        for name, bench in [("rust", "backend"), ("gpu", "cuda")]:
            print("run", index, name, flush=True)
            criterion = output / f"criterion-{name}-run{index}"
            env["CRITERION_HOME"] = str(criterion)
            run(
                [
                    "cargo",
                    "bench",
                    "--locked",
                    "--manifest-path",
                    str(MANIFEST),
                    "--features",
                    "cuda",
                    "--bench",
                    bench,
                ],
                env,
                output / f"1t-run{index}-{name}.log",
            )
            records = collect_criterion(criterion)
            expected = {
                case["id"]
                for case in cases
                if name != "gpu" or not case.get("cuda_diagnostic_only", False)
            }
            if {row["id"] for row in records} != expected:
                raise ValueError(f"incomplete {name} Criterion results")
            (output / f"1t-run{index}-{name}.json").write_text(
                json.dumps(records, indent=2) + "\n"
            )
            shutil.rmtree(criterion)  # Full estimates and raw samples preserved above.
        print("run", index, "julia", flush=True)
        run(
            [
                args.julia,
                "--startup-file=no",
                f"--project={args.julia_project}",
                "benchmarks/julia/backend_baseline.jl",
                str(cases_path),
                str(references),
                str(output / f"1t-run{index}-julia.json"),
            ],
            env,
            output / f"1t-run{index}-julia.log",
        )
    # Full references are generated reproducibly; keep hashes rather than large binaries.
    (output / "reference-hashes.json").write_text(
        json.dumps({p.name: sha(p) for p in sorted(references.iterdir())}, indent=2)
        + "\n"
    )
    shutil.rmtree(references)
    print("complete", output, flush=True)


if __name__ == "__main__":
    main()
