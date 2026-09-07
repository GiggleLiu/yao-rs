# CPU backend comparison

The initial tenferro experiment is an isolated, unpublished Cargo fixture at
`tenferro-probe`. It pins the published tenferro 0.4.0 crates and requires Rust
1.96 or newer. It does not change the library's dependencies or default backend.

Run from the repository root:

```sh
CARGO_TARGET_DIR=target/probe cargo test --manifest-path benchmarks/tenferro-probe/Cargo.toml --locked
python3 -m unittest discover -s benchmarks/tests
python3 benchmarks/run_backend.py benchmarks/results/mac-cpu-2026-09-07 --julia ~/.juliaup/bin/julia
python3 benchmarks/compare.py --backend-results benchmarks/results/mac-cpu-2026-09-07
uv run --with matplotlib benchmarks/plot_backend.py benchmarks/results/mac-cpu-2026-09-07
```

The runner builds before timing, runs three independent processes at each of 1
and 4 threads, and executes Rust and Julia serially. Use `--max-qubits 8 --runs 1
--threads 1` with a new result directory for a shorter run. Do not run other
benchmarks/builds concurrently. A complete 24-qubit sweep creates several GiB
of full-state oracle files under ignored `benchmarks/data/`; those can be
removed after the recorded Julia comparisons pass. The script refuses to
overwrite an existing result directory.

The shared JSON fixtures include gate placements, parameters, initial-state
choice, and workload identity. Non-tensor unitary workloads use a deterministic
asymmetric complex input. Rust writes complete binary reference states,
density matrices and value/gradient vectors; Julia validates every entry before
timing. Julia maps site `q` to `n-q`, so flat vectors agree without an additional
permutation. Density matrices are compared in Rust row-major order.

Gradient timings return both value and parameter gradient. The Julia wrapper
uses Yao's reversible `apply_back` routine and computes the value from the same
forward state/cotangent; it avoids a second forward circuit evaluation. The
wrapper follows Yao's Apache-2.0-licensed `expect_g` algorithm, with an added inner
product for the value. The tests also exercise the public expectation adjoint
when checking numerical agreement.

Tensor workloads compare native simulation, omeinsum contraction, and tenferro
contraction with independently chosen plans. Separate rows measure conversion,
planning, prepared execution, and conversion+planning+execution+output-layout
conversion from existing arrays. Export and cold execution are additionally
measured by the diagnostic memory probe. These rows do not imply identical
contraction trees; the explicit-path correctness test is separate.

The custom-operation experiment embeds the existing X kernel in a tenferro
traced runtime and compares it with tensor composition. The eager AD microcase
measures input copying, graph construction, forward loss, and backward together.
It is a tensor loss, not yet full circuit AD. The separate memory probe reports
AD input, retained forward tape and backward allocation phases across depths.
Both a linear conjugation chain and a nonlinear `0.3*sin(x)` chain are measured: their derivative rules have different primal-state retention needs.

`metadata.json` records compiler, source hashes, machine and thread controls;
lockfiles pin Rust and Julia dependencies. Raw Criterion JSON retains samples
and confidence intervals. Generated tables use medians of per-process medians,
not Julia minima versus Rust medians. Ratios greater than one mean native Rust
was faster. Native state-vector kernels are serial in this fixture; setting a
thread budget does not make them parallel. Provider-specific comparisons are
future additions.

Memory logs separate instrumented Rust heap allocation/retention from
whole-process peak RSS reported by `/usr/bin/time`. Rust allocator accounting
excludes native-provider allocations; RSS includes startup and allocator
retention. Allocation-instrumented timings are diagnostic, not used for speed
ratios. Shared CI checks correctness and compilation; it does not enforce
wall-clock performance thresholds.

The legacy `make bench-*` and comparison commands remain available. The new
shared-fixture runner is the reference for tenferro backend decisions.


### Supported CPU adapter (PR #48)

The fixture now also benchmarks `yao_rs::tenferro::CpuContractor`.
`supported_planning` validates and compiles a supplied omeco greedy tree;
`supported_warm` executes it with ndarray input/output adaptation;
`supported_from_arrays` combines both phases. `omeinsum_fixed_tree` executes
that identical tree, including omeinsum's internal preparation and adaptation.
These phases exclude tree search and context creation. Each result is checked
against the existing contractor before measurement. They are distinct from the
`tenferro_*` prototype phases, which use tenferro's own automatic planning and
owned inputs.

For a smaller CPU follow-up sweep, use `--max-qubits 8` with the same runner.
Run compilation/tests first and keep other build jobs off the measurement host
during the final timing passes. Preserve each report's metadata and source
hashes; do not overwrite the earlier 24-qubit baseline. `plot_backend.py` adds a
separate `supported-costs.svg`/`.png` comparison when these phases are present.

### Hamiltonian product formulas

```sh
python3 benchmarks/run_backend.py benchmarks/results/mac-evolution-cpu-2026-09-07 --suite evolution --julia ~/.juliaup/bin/julia
uv run --with matplotlib benchmarks/plot_evolution.py benchmarks/results/mac-evolution-cpu-2026-09-07
```

The evolution suite uses the Rust model builders to emit shared circuits for
three-qubit Ising and XYZ Hamiltonians, first/second order formulas, and
1, 2, 4, 8, 16 steps. Julia independently builds the Hamiltonian from Pauli
blocks and computes its dense exponential outside timing. The report separates
Rust–Yao circuit agreement from each formula's relative state error against
that exponential. Timings cover execution of the same lowered circuits, with
construction measured separately; they do not benchmark adaptive Krylov action.
Two additional zero-state cases exercise both tensor contractors, including
the supported tenferro adapter with the same supplied tree as omeinsum.
The runner also measures model/circuit construction and native execution heap
in separate processes at 3/12 qubits and 1/16 second-order steps. Each raw
`memory-evolution-*.log` includes phase heap accounting and process peak RSS.
`compare.py` generates the memory table alongside the accuracy/cost comparison.

### Circuit custom-loss AD

```sh
python3 benchmarks/run_backend.py benchmarks/results/mac-circuit-ad-cpu-2026-09-07 --suite circuit-ad --julia ~/.juliaup/bin/julia
uv run --with matplotlib benchmarks/plot_circuit_ad.py benchmarks/results/mac-circuit-ad-cpu-2026-09-07
```

This suite returns squared state-distance loss, real parameter gradients and
complex input-state gradients for 8/12/16 qubits and 10/100 layers. Each layer
has Ry, Rz, CX and Rx acting on the final two sites of the same full asymmetric
complex input. The target is independently generated from the recorded formula.
This workload compares derivative implementations, not every circuit topology.
Yao uses `apply_back` with seed `2(output-target)` and the same site mapping.

`native` shares the existing reversible sweep. `tenferro_circuit_ad` embeds it
as a circuit primitive; `tenferro_composed_ad` constructs ordinary 4×4 tensor
matrices and multiplies the state in tenferro. Tensor rows include input copies,
eager graph construction, forward loss, backward and result collection; context
creation is excluded. Outputs are checked before timing. No fusion is applied.

Repeated tensor-composition timings cover 10 layers. Separate processes qualify
100-layer composition with a 30-CPU-second limit, beginning at 8 qubits. Larger
100-layer composition probes are skipped if that representative run does not
complete. Limits and skipped cases are recorded explicitly, without timing
ratios. `memory-circuit-ad-*.log` separates retained forward heap from additional
backward peak heap and process RSS. A terminated run's RSS is only its observed
peak before termination. CPU-limit diagnostics and allocation-instrumented
times are not treated as successful benchmark measurements.


The committed [circuit AD M4 report](results/mac-circuit-ad-cpu-2026-09-07/report.md)
contains the six shared gradient cases, three runs at one/four threads, and
bounded isolated memory probes. See the report for complete versus CPU-limited
composition runs and the measured overhead of the tenferro circuit operation.

## Polynomial expectations and slicing

```bash
python3 benchmarks/run_backend.py benchmarks/results/mac-tensor-memory-cpu-2026-09-07 --suite tensor-memory --julia ~/.juliaup/bin/julia
uv run --with matplotlib benchmarks/plot_slicing.py benchmarks/results/mac-tensor-memory-cpu-2026-09-07
```

The shared circuit fixtures evaluate five-term complex polynomials at 4/6
qubits, with and without a bit-flip channel. Native Rust/Yao include state
copying, circuit execution and the complete expectation. Yao uses `sandwich`
for pure states and its density-matrix trace formula without real projection;
the latter constructs the dense operator inside the timed call, matching its
existing density expectation algorithm. Rust applies each operator string.
The difference in algorithms is part of the reported workflow cost.

Tensor exports share the circuit tensors across all terms. Export is timed
separately. Unsliced and term-sliced contractions use the same omeco greedy
tree; tenferro compilation and warm execution are separate, while omeinsum
prepares its executor for each call/slice. Context creation and tree search
are excluded from those execution rows.

Synthetic dense complex128 matrix chains supply a separate time/memory sweep:
32/128/256 square matrices with a supplied `((A B) C)` path, compared unsliced
and with one fixed output index. Automatic omeco TreeSA slicing/replanning is
qualified only at dimension 32; it can require many more assignments. Each
process records its actual selected execution tree/slices in `*-plans.jsonl`.
These matrix rows have no native/Yao quantum-simulation counterpart. Never
extrapolate their ratios to circuit evolution or infer unmeasured automatic
planner results at larger sizes.

Isolated process logs separate input heap, planning, compilation, execution
heap and whole-process peak RSS. Each record also includes the static tensor
estimate, a zero user workspace reserve, and one active slice. Provider scratch,
compiled graphs and allocator retention can put RSS above the estimate. The
report rejects incomplete logs instead of presenting them as completed results.

The initial ordinary-chain qualification reduced estimates with little change
in RSS at those sizes. The suite therefore retains those cases and also includes
32/64-dimensional **outer-product stress paths**: first form `A[i,j] B[k,l]`,
then contract `C[j,k]`, producing an `n^4` intermediate. Fixed output slicing
keeps that order; a separately named greedy unsliced plan demonstrates that
better planning can avoid the intermediate entirely. This deliberately poor
supplied path tests memory controls; it is not presented as an optimized baseline.


### Noisy trajectories

Run `benchmarks/run_backend.py RESULTS --suite trajectories --julia PATH_TO_JULIA`
for repeated exact native/Yao density expectations, prepared tenferro expectation
contractions and seeded trajectory ensembles. Three entangled fixtures use
4/6/8 qubits and amplitude damping plus depolarization. The 6-qubit sample-count
sweep uses eight seeds; other cases use one. Larger 12/16-qubit product fixtures
have an analytic expectation and are explicitly simpler workloads.

`*-trajectory-stats.jsonl` records means, references, errors, variances and
standard errors. Repeated timing-process seeds are deduplicated for accuracy.
Twenty trajectory and three exact-density memory probes report heap and isolated
process RSS. The runner omits unrelated generic AD memory probes for this suite.
Generate standalone figures using `uv run --with matplotlib benchmarks/plot_trajectories.py RESULTS`.
Timing includes buffers, moment reduction and thread-pool creation; local channel
preparation is separate. Compare time at achieved statistical error, not as if
trajectories and exact density evolution had identical accuracy.

### Matrix-free Krylov evolution

Run `benchmarks/run_backend.py RESULTS --suite krylov --julia PATH_TO_JULIA`.
The suite compares adaptive Rust evolution with pinned Yao `TimeEvolution` for
Ising/XYZ models at 4/8/12/16 qubits, using relative tolerances 1e-4/1e-7/1e-10.
Rust caps each basis at 20 vectors; Yao retains its public eager/default basis
policy (up to 1000 vectors). Model construction is excluded from execution;
state/work buffers and Rust mask preparation are included. Both use complex128.
Rust vector kernels are serial; one/four-thread runs also record surrounding
provider settings. These are distinct solver policies with distinct achieved
errors, not equal-accuracy speedups inferred from the nominal tolerance.

Small oracles use an independent Yao dense exponential. Larger oracles use
KrylovKit at tol=1e-13 and require agreement with a separate Rust solve at
rtol=1e-13/basis cap 40. Diagnostics qualify convergence outside timing and
record actual state errors. Asymmetric complex inputs are checked at four
qubits; the size/tolerance sweep uses a zero state. Four-qubit Suzuki circuits
with 2/8/32 steps add tenferro and omeinsum execution at explicitly reported
product-formula errors on that same zero input. No adaptive tenferro Krylov
backend is implied by those tensor timings.

Twenty-four isolated memory probes vary model, qubits and basis cap (8/20/40)
at rtol=1e-8. They separate additional execution heap from whole-process RSS.
Generate standalone plots with
`uv run --with matplotlib benchmarks/plot_krylov.py RESULTS`.

The [recorded M4 report](results/mac-krylov-cpu-2026-09-07/report.md) includes
all 32 cases in six timing processes and all 24 memory probes, with raw
convergence diagnostics and standalone error/time and memory figures.

## CUDA and same-host CPU comparison

Enable the optional `cuda` feature and configure the runtime described in
[the CUDA guide](../docs/src/cuda.md). Measurements use complex128 throughout.
`cuda_cases` produces shared unitary, custom-loss gradient and exact noisy
fixtures; the existing native/tenferro/omeinsum CPU and Yao runners consume
those same circuits. CPU tenferro AD uses the reversible custom operation;
GPU AD uses ordinary tensor composition. The CPU composition feasibility
microbenchmark is excluded from this suite.

Create an isolated Yao environment from the pinned source, without changing a
personal Julia environment:

```bash
git clone https://github.com/QuantumBFS/Yao.jl.git /tmp/yao-cuda-source
git -C /tmp/yao-cuda-source checkout --detach 31c7c1333b14b1e89123c511eff5742e7ac24edd
julia --startup-file=no benchmarks/julia/setup_source.jl /tmp/yao-cuda-source /tmp/yao-cuda-env
CUDA_VISIBLE_DEVICES=0 python3 benchmarks/run_cuda.py /tmp/yao-cuda-results \
  --julia-project /tmp/yao-cuda-env --yao-source /tmp/yao-cuda-source
python3 benchmarks/compare.py --backend-results /tmp/yao-cuda-results
uv run --with matplotlib --with numpy python benchmarks/plot_cuda.py /tmp/yao-cuda-results
```

Use `--smoke --runs 1` for three small end-to-end qualification cases. The full
suite runs three independent processes per backend, with one configured CPU
thread. Select an idle GPU and keep other work off the measured CPU cores.
`--target` can reuse an existing build directory; Criterion outputs live in the
new result directory and raw samples are preserved in JSON. Each invocation
requires a new result directory. `--julia` accepts an explicit Julia executable.

CUDA resident timings synchronize before and after execution, include host
scheduling, device allocation and fresh AD leaves, and exclude user transfers.
Transfer-inclusive timings add uploads of every input and full output downloads;
both reuse prepared constants. Process-cold probes separate context creation,
preparation plus upload, and first execution, while retaining compiler/driver
disk caches. NVIDIA process-memory snapshots include allocator retention and
workspaces and are not continuous peaks. Host RSS is reported separately.
Metadata records source/lockfile hashes, hardware, runtime paths and the pinned
Julia environment. Preserve the runtime package versions alongside the report.

The CUDA gradient runner requests parameter and input-state gradients explicitly
with `runtime.grad` (two targeted pullbacks). Tenferro 0.4.0's stateful
`backward()` also requests pullbacks for retained intermediates, which is extra
work beyond this benchmark's output contract. Each qualification process has a
300-second wall-time limit; adjust `--qualification-timeout` explicitly for a
larger study. A failed/timed-out qualification aborts timing and is not reported
as a performance result.

By default, 40-layer gradient cases carry `cuda_diagnostic_only: true`: their
cold probe, full-output agreement, memory snapshots, and one warm diagnostic
sample are recorded, while repeated GPU Criterion timing is reserved for the
10-layer cases. CPU/Yao timings still cover every case. Diagnostic values are
labeled separately and do not enter CPU/GPU timing ratios or latency bars.
Use `--repeat-deep-gradients --qualification-timeout 600` to request the longer
repeated study. The memory-versus-depth plot includes the completed deep probes.

The [recorded A800 comparison](results/a800-cuda-2026-09-07/README.md) contains
the repeated same-host CPU/GPU/Yao results, all 14 successful qualification
processes, raw samples, source hashes, runtime packages and standalone figures.
