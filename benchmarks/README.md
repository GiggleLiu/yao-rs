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
wrapper follows Yao's MIT-licensed `expect_g` algorithm, with an added inner
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
