# Reproducible performance benchmarks

This folder defines the curated CPU suite and the environment used to run it.
It reuses the Rust/Julia feature adapters in `benchmarks/tenferro-probe` and adds
pinned QASMBench/MQT application circuits plus Qulacs comparisons.
[Survey and selection rationale](SURVEY.md) describe the coverage.

From the repository root, with Rust (1.96+), Julia, and uv installed:

```bash
make benchmark-setup
make benchmark-test
make benchmark BENCH_PROFILE=smoke BENCH_OUT=benchmarks/results/smoke-local
make benchmark BENCH_OUT=benchmarks/results/baseline-local
# After changing yao-rs, on the same host and environment:
make benchmark BENCH_OUT=benchmarks/results/candidate-local
make benchmark-check \
  BENCH_BASELINE=benchmarks/results/baseline-local/results.json \
  BENCH_CANDIDATE=benchmarks/results/candidate-local/results.json
make benchmark-qualify BENCH_RESULT=benchmarks/results/candidate-local/results.json
```

Set `JULIA=/path/to/julia` if it is not on PATH. The committed Julia environment
was resolved with Julia 1.12.4; use that version to reproduce this environment.
Python 3.13.12 and its dependencies are managed by uv. Rust dependencies use the
probe's `Cargo.lock` with `--locked`. Compiler versions, dependency locks, host
identity, suite contents, thread budgets, and adapter code are recorded; the
comparison command rejects incompatible runs instead of pooling them.
Compiler flags and Cargo configuration hashes are retained. Julia startup files
are disabled, and its BLAS provider is checked across processes. Linux CPU
identity excludes changing clock readings while retaining topology and affinity.

## Profiles

| Profile | Independent runs | Thread budgets | Use |
|---|---:|---|---|
| `smoke` | 1 | 1 | End-to-end correctness and harness setup; cannot pass a performance gate |
| `regression` (default) | 6 | 1 | Native fixtures through 16 qubits and the bounded application/feature tracks |
| `full` | 7 | 1, 4 | Native fixtures through 24 qubits, all 19 application cases, and the feature tracks |

`suite.json` is the versioned selection policy. Each run stores the actual
selected cases, source hashes, per-process samples, `results.json`, and a
readable `report.md`. Runs are serialized across worktrees of this repository;
keep unrelated builds and workloads off the measurement host as well.
Output directories must be new. No result directory is automatically replaced.

The 19 public application cases are frozen in `datasets/`; normal benchmarking
requires no dataset downloads. Their manifest records source revision, SHA-256,
qubit mapping, gate lowering, and removed final measurements. Original files
and licenses are retained. QASMBench UCCSD's invalid terminal readout register
names are preserved in the original file; only the unitary prefix is timed.
Mid-circuit measurement, reset, or classical control is rejected. MQT generation
uses its pinned deterministic defaults (seed 10), followed by level-0 common
basis lowering (transpiler seed 4137). Global phase is preserved explicitly.
Regenerate only when intentionally revising the dataset:

```bash
uv run --frozen --project benchmarks/regression/environment \
  python benchmarks/regression/prepare_datasets.py
make benchmark-test
```

## Timing and comparisons

Implementation order is counterbalanced across the six default runs.
Warmed state execution includes a fresh state copy in each implementation.
Complete outputs are checked outside timing. yao-rs has direct, two-qubit-fused,
and four-qubit-fused execution rows; Qulacs has ordinary and four-qubit-fused
rows. Fusion preparation is recorded separately. The report identifies each
library's fastest measured execution mode and retains all timings. Julia warms each operation for 0.2 seconds, then takes 30 calibrated batches
targeting 10 ms each. Batch sizes and GC timings are saved. Gradients, density matrices, tensor phases, trajectories, and Krylov
operations retain their feature-specific execution boundaries from the
[parent benchmark guide](../README.md).

`benchmark-check` compares the same implementation and phase across revisions.
It bootstraps independent process medians, using a 95% interval for the
candidate/baseline time ratio. The default tolerance is 5%; configure it with
`BENCH_TOLERANCE`. Fewer than five runs, missing cases, correctness failures,
nonfinite samples, incompatible environments, and inconclusive intervals cannot
pass. A mean across workloads never masks an individual regression.

A comparison against a prior yao-rs revision is a regression check, not a claim
of state-of-the-art performance. Qualifying that claim additionally requires
strong applicable competitors on every main feature, matched accuracy and
outputs, and a same-device run. CPU results do not qualify CUDA performance.
The existing CUDA suite remains documented in the parent guide.

`benchmark-qualify` checks the fastest measured yao-rs execution mode against
every measured competitor, using the same confidence interval and tolerance.
It identifies the selected mode and excludes preparation timings. Each comparison
must pass; a fast result on one workload cannot offset a slow result elsewhere.
This command covers the named execution comparisons in the saved run. Tensor
phases, stochastic accuracy, and GPU performance need their own qualification.

Krylov results record achieved errors against an independently checked tight
reference. The current solver parameter is `rtol=1e-7`, with a validated global
relative-error budget of `1e-6`. Timing at this budget is not an equal-error
comparison; consult the reported errors before making a solver-performance claim.
