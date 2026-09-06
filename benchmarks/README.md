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
