#!/usr/bin/env python3
"""Run the curated suite through existing Rust/Julia adapters and Qulacs."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import fcntl
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from generate_cases import cases as native_cases
from run_backend import MANIFEST, TARGET, run, sha
import machine


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def prepare(track, profile, output, env):
    if track == "circuits":
        cases = native_cases(profile["max_qubits"])
        applications = json.loads((HERE / "datasets/applications.json").read_text())
        chosen = profile["application_ids"]
        cases += [c for c in applications if (chosen == "all" or c["id"] in chosen) and c["circuit"]["num_qubits"] <= profile["max_qubits"]]
    else:
        suite = json.loads((HERE / "suite.json").read_text())
        path = output / (track + "-generated.json")
        run([str(TARGET / "release" / suite["tracks"][track]["generator"]), str(path)], env)
        cases = json.loads(path.read_text())
        limit = profile.get("feature_max_qubits", {}).get(track, min(profile["max_qubits"], 16))
        cases = [c for c in cases if c["circuit"]["num_qubits"] <= limit]
        if track == "krylov":
            cases = [c for c in cases if c["mode"] == "krylov" and c["id"].endswith("tol7")]
        if profile["runs"] == 1:
            cases = cases[:1]
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError(f"Empty or duplicate suite: {track}")
    path = output / (track + "-cases.json")
    dump(path, cases)
    return cases, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--profile", choices=["smoke", "regression", "full"], default="regression")
    parser.add_argument("--julia", default="julia")
    parser.add_argument("--tracks", nargs="+", help="Explicit partial run; recorded in suite identity")
    args = parser.parse_args()
    spec = json.loads((HERE / "suite.json").read_text())
    profile = spec["profiles"][args.profile]
    tracks = args.tracks or list(spec["tracks"])
    if any(t not in spec["tracks"] for t in tracks) or len(set(tracks)) != len(tracks):
        parser.error("unknown or duplicate track")
    common = Path(subprocess.check_output(["git", "rev-parse", "--git-common-dir"], cwd=ROOT, text=True).strip())
    if not common.is_absolute():
        common = ROOT / common
    lock = (common / "yao-benchmark.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("another curated benchmark is running in this repository")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(CARGO_TARGET_DIR=str(TARGET), YAO_BENCH_CORE_ONLY="1", YAO_BENCH_FUSION="1")
    # All compilation and environment setup finish before any timed process.
    run(["cargo", "build", "--release", "--locked", "--manifest-path", str(MANIFEST), "--bins", "--benches"], env, output / "build.log")
    julia_project = HERE / "environment/julia"
    run([args.julia, "--startup-file=no", f"--project={julia_project}", "-e", "using Pkg; Pkg.instantiate(); using Yao, BenchmarkTools"], env, output / "julia-setup.log")
    selected = {track: prepare(track, profile, output, env) for track in tracks}
    hardware = machine.hardware()
    environment = dict(device=fingerprint([platform.node(), hardware]), hardware=hardware,
                       configuration=machine.configuration(ROOT),
                       platform=platform.platform(), precision=spec["precision"], python=platform.python_version(),
                       rustc=subprocess.check_output(["rustc", "-Vv"], text=True),
                       julia=subprocess.check_output([args.julia, "--version"], text=True),
                       python_lock=sha(HERE / "environment/uv.lock"), julia_lock=sha(julia_project / "Manifest.toml"),
                       probe_lock=sha(MANIFEST.with_name("Cargo.lock")),
                       timing=spec["timing"],
                       protocol_sha256=fingerprint({str(p.relative_to(ROOT)): sha(p) for p in [
                           Path(__file__), HERE / "machine.py", HERE / "qulacs_baseline.py", ROOT / "benchmarks/run_backend.py",
                           ROOT / "benchmarks/julia/backend_baseline.jl", ROOT / "benchmarks/tenferro-probe/benches/backend.rs"]}))
    suite_hash = fingerprint({track: cases for track, (cases, _) in selected.items()})
    metadata = dict(schema_version=1, suite=spec["name"], profile=args.profile, tracks=tracks,
                    suite_sha256=suite_hash, environment=environment, created_utc=datetime.now(timezone.utc).isoformat(),
                    source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    sources={str(p.relative_to(ROOT)): sha(p) for directory in [ROOT / "src", ROOT / "bitbasis/src", HERE, ROOT / "benchmarks/tenferro-probe/src", ROOT / "benchmarks/tenferro-probe/benches"]
                             for p in directory.rglob("*") if p.suffix in (".rs", ".py") and ".venv" not in p.parts})
    dump(output / "metadata.json", metadata)
    samples = defaultdict(list)
    required = set()
    providers = set()
    accuracy = defaultdict(list)
    for track, (cases, case_path) in selected.items():
        print(f"Running {track}: {len(cases)} cases", flush=True)
        folder = output / track
        command = [sys.executable, "benchmarks/run_backend.py", str(folder), "--suite", track,
                   "--cases", str(case_path), "--runs", str(profile["runs"]), "--threads", *map(str, profile["threads"]),
                   "--julia", args.julia, "--julia-project", str(julia_project), "--skip-memory", "--no-report", "--qulacs"]
        run(command, env, output / (track + ".log"))
        for threads in profile["threads"]:
            for case in cases:
                for backend in ["native", "julia"] + (["qulacs", "qulacs_fused4"] if case["mode"] == "state" else []):
                    required.add((track + "/" + case["id"], backend, threads, "execute"))
                if case["mode"] == "state":
                    for phase in ["fused2_prepare", "fused2_execute", "fused4_prepare", "fused4_execute"]:
                        required.add((track + "/" + case["id"], "native", threads, phase))
            for index in range(1, profile["runs"] + 1):
                prefix = f"{threads}t-run{index}"
                for row in json.loads((folder / (prefix + "-rust.json")).read_text()):
                    phase = "execute" if row["backend"] == "native" else row["backend"]
                    samples[(track + "/" + row["id"], "native", threads, phase)].append(row["estimates"]["median"]["point_estimate"])
                julia_result = json.loads((folder / (prefix + "-julia.json")).read_text())
                providers.add(julia_result["blas"])
                for row in julia_result["records"]:
                    samples[(track + "/" + row["id"], "julia", threads, "execute")].append(row["median_ns"])
                    if "approximation_error" in row:
                        accuracy[track + "/" + row["id"]].append({
                            key: row[key] for key in ["approximation_error", "yao_approximation_error", "oracle", "tight_reference_error"]})
                qpath = folder / (prefix + "-qulacs.json")
                for row in json.loads(qpath.read_text())["records"]:
                    samples[(track + "/" + row["id"], row["backend"], threads, "execute")].append(row["median_ns"])
    if not required.issubset(samples):
        raise ValueError(f"Missing required results: {required - samples.keys()}")
    if any(len(v) != profile["runs"] for v in samples.values()):
        raise ValueError("Incomplete independent runs")
    if len(providers) != 1:
        raise ValueError(f"Julia BLAS provider changed between processes: {providers}")
    environment["julia_blas"] = next(iter(providers))
    metadata["accuracy"] = dict(accuracy)
    dump(output / "metadata.json", metadata)
    records = [dict(case=k[0], backend=k[1], threads=k[2], phase=k[3], correctness="passed", run_medians_ns=v)
               for k, v in sorted(samples.items())]
    dump(output / "results.json", dict(**metadata, expected_records=[list(k) for k in sorted(samples)], records=records))
    lines = ["# Curated CPU comparison", "", f"Profile: {args.profile}. Independent runs: {profile['runs']}.", "",
             "Times are medians of independent process medians. Each library uses its fastest measured execution mode. Ratios above 1 favor yao-rs. Fusion preparation is outside warmed execution.", "",
             "| Case | Threads | yao-rs mode | yao-rs ms | Fastest measured competitor | Competitor ms | Competitor / yao-rs |", "|---|---:|---|---:|---|---:|---:|"]
    for case, backend, threads, phase in sorted(required):
        if backend != "native" or phase != "execute":
            continue
        rivals = [(statistics.median(samples[k]), k[1]) for k in required if k[0] == case and k[2] == threads and k[1] != "native"]
        rival, name = min(rivals)
        choices = [(statistics.median(v), k[3]) for k, v in samples.items()
                   if k[:3] == (case, backend, threads) and k[3] in ["execute", "fused2_execute", "fused4_execute"]]
        native, mode = min(choices)
        lines.append(f"| {case} | {threads} | {mode} | {native/1e6:.4f} | {name} | {rival/1e6:.4f} | {rival/native:.2f}× |")
    lines += ["", "This table is descriptive. Run the regression gate against a compatible saved run; it rejects incomplete or inconclusive comparisons.",
              "Feature-specific phases and all raw samples are retained in results.json and the track directories."]
    if accuracy:
        lines += ["", "## Achieved evolution accuracy", "",
                  "The solver parameter is rtol=1e-7; the validated global relative-error budget is 1e-6. Achieved errors differ, so timing at this budget is not an equal-error comparison.", "",
                  "| Case | yao-rs relative error (maximum) | Yao.jl relative error (maximum) |",
                  "|---|---:|---:|"]
        for case, values in sorted(accuracy.items()):
            lines.append(f"| {case} | {max(x['approximation_error'] for x in values):.3e} | {max(x['yao_approximation_error'] for x in values):.3e} |")
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print(f"Results: {output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
