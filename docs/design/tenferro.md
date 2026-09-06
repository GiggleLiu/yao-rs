# Tenferro integration decision and execution record

The approved roadmap is implemented as sequential, squash-merged PRs. Initial
integration preserves existing public APIs and specialized CPU kernels. GPU
validation targets the NVIDIA A800 host reachable as `ssh gpu`.

## Milestones

| Milestone | Required evidence | Status |
| --- | --- | --- |
| 1. Published backend feasibility / CPU baseline | Complex/layout/AD/custom-operation tests; reproducible CPU report and memory measurements | Merged [PR #47](https://github.com/GiggleLiu/yao-rs/pull/47) (`6bad156`) |
| 2. Supported tenferro CPU backend | Library/CLI adapter, explicit and reusable plans, numerical/feature/package checks | Implemented in [PR #48](https://github.com/GiggleLiu/yao-rs/pull/48); validated, reports attached; awaiting merge |
| 3. Hamiltonian evolution | Pauli rotations, model builders, Trotter/Suzuki, physical parameter bindings and convergence tests | Pending |
| 4. Differentiable circuits | Custom losses/input-state VJP, tenferro integration, shared parameters, numerical and memory tests | Pending |
| 5. Tensor memory / observables | Polynomial expectations, omeco slicing, estimates, versioned plans and budget tests | Pending |
| 6. Noisy trajectories | Seeded Kraus sampling, uncertainty, exact-density comparisons and memory scaling | Pending |
| 7. Matrix-free exponential action | Community implementation qualification, error/convergence diagnostics, Yao comparison | Pending |
| 8. GPU execution | Resident tensor/circuit/AD execution, explicit transfers, correctness and timings via `ssh gpu` | Pending |

Reusable subcircuits, batching, measurement/feedforward, symbolic algebra,
qudit state-vector simulation and distributed execution remain the roadmap's
later backlog rather than hidden requirements for this sequence.

## Published dependency and API findings

The isolated `benchmarks/tenferro-probe` crate resolves published tenferro 0.4.0
on Rust 1.98.0. The crate minimum is Rust 1.96. A committed fixture lockfile pins
transitive dependencies, including omeco 0.2.6. No development Git dependency is
needed. The initial CPU provider is faer. The fixture is unpublished and
excluded from library packages by the existing `benchmarks/` exclusion.

Complex128 tensors, negative-label remapping, arbitrary qudit tensor export,
density matrices, conjugation and real-Hermitian VJPs are exercised against
independent expected values and existing circuit simulation. Zero-copy access
is not assumed: the prototype performs an explicit ndarray logical-axis to
column-major conversion, and accounts for that conversion separately.

The published `ConcreteEinsumPlan` prepares reusable inputs of fixed shapes and
dtypes but chooses an automatic contraction tree. Caller-supplied paths are
available through `TraceContextEinsumExt::einsum_subscripts_with` and
`EinsumOptimize::Path`/`Tree`. The supported adapter must therefore use an
explicit traced path or independently prepared tree nodes for an existing
omeco plan. Silently dropping an explicit order is unacceptable.

The custom-operation prototype hosts an existing X kernel in a tenferro traced
runtime. It has a distinct family identity, validated graph input shapes and
explicit CPU runtime registration. Unregistered AD rules produce an error.
This proves the integration seam; the full differentiable circuit operation
and its first-order rules belong to milestone 4.

Tenferro's complex cotangent convention is
`dL = Re(sum(conj(x_bar) * dx))`. The probe verifies `grad(sum(abs(x)^2)) = 2x`
and a non-real seed for `x*x`. The circuit adjoint adapter must reconcile the
factor of two with the existing `expect_grad` implementation.

## Backend policy

Keep native state-vector kernels and the current omeinsum contractor available
while adding tenferro behind an explicit feature/backend boundary. Prefer
borrowed layout-compatible storage in the supported adapter; retain explicit
conversion where ownership/strides require it. Own backend sessions and
bounded plans at the caller/context level, avoiding nested CPU backend entry.

Reuse omeco for slicing/planning and argmin for optimizer examples. Investigate
Krylov community packages independently before committing to a maintenance
burden. Do not add a second simulator as a runtime dependency. The full
performance report, not a single GEMM number, will guide any default change.

## GPU host reconnaissance

The host has six A800 80 GB GPUs, a dual-socket Xeon Platinum 8378A CPU and
approximately 1 TiB RAM. Other GPUs are in use; select an idle GPU explicitly
at test time. Rust 1.96.0 is already installed as a non-default rustup toolchain.
Use `cargo +1.96.0` (or a task-local newer toolchain) without changing other
users' environments.

The installed driver is 535.230.02, reporting CUDA 12.2, with a CUDA 12.1
toolkit. This needs validation against tenferro's CUDA requirements. NVIDIA's
[forward-compatibility documentation](https://docs.nvidia.com/deploy/cuda-compatibility/latest/forward-compatibility.html)
describes supported user-space compatibility libraries on data-center GPUs;
evaluate that task-local option before considering any system driver change.
Host access alone is not evidence that GPU execution works.

## Validation and performance record

The [Apple M4 CPU report](../../benchmarks/results/mac-cpu-2026-09-07/report.md)
contains three independent runs each at one and four threads, all 48 shared
workloads, raw samples/confidence intervals, pinned environments and plots.
Every Julia output comparison passed; the largest absolute discrepancy was
1.94e-14. `make check-all`, eight tenferro probe tests, fixture Clippy, four
report-generator tests and all GitHub CI jobs passed before the results update.
The [x86-64 CPU report](../../benchmarks/results/linux-cpu-2026-09-07/report.md)
now contains all 48 workloads and six independent processes per language.
All 288 output comparisons passed (maximum error 2.16e-14). This shared
Xeon Platinum 8378A host was not pinned to exclusive cores; use the raw run
ranges. For example, one-thread QFT at 24 qubits takes 5.968 s in native Rust
and 5.725 s in Yao; the 10-qubit noisy density case takes 1.318 s and 0.585 s.
This report measures the PR #47 prototype at `6e804e8`, not the supported adapter.

Measured one-thread examples (medians of three run medians):

| Workload | Native yao-rs | Yao.jl | Interpretation |
| --- | ---: | ---: | --- |
| Rx, 24 qubits | 21.95 ms | 15.60 ms | Yao is faster on this large gate workload |
| QFT, 24 qubits | 2.427 s | 2.415 s | Similar full-circuit cost on this host |
| 100-layer gradient, 12 qubits | 23.31 ms | 23.66 ms | Similar value/gradient cost |
| Noisy density matrix, 10 qubits | 198.42 ms | 155.45 ms | Yao is faster for this exact-noise case |

For the 8-qubit tensor-state fixture, native simulation takes 4.48 µs,
omeinsum contraction from arrays 181.67 µs, tenferro from arrays 306.62 µs,
and prepared tenferro execution 82.36 µs. These are different algorithms and
planning boundaries, not interchangeable speedup claims. The initial policy
therefore keeps native execution and adds tenferro explicitly. The custom X
extension takes 66.40 µs versus 91.68 µs for prepared tensor composition at
16 qubits, supporting further specialized-operation experiments.

Memory retention depends on AD rules: the 16-qubit-sized nonlinear chain
retains 10.06 MiB after 10 layers and 100.55 MiB after 100 layers, whereas
conjugation mainly retains graph metadata. Whole-process RSS is reported
separately. This is evidence for testing the actual circuit differentiation
rules and their memory, not a universal assertion that every AD operation
retains every intermediate state. See `benchmarks/README.md` for shared fixtures, numerical equivalence,
thread controls, phase boundaries and the distinction between Rust heap and
process RSS. GPU timings must later distinguish synchronized resident execution
from transfer-inclusive execution, and report unsupported dtype/operations
without silent CPU fallback.


## Supported CPU adapter (PR #48)

`yao_rs::tenferro::CpuContractor` owns an explicit faer thread configuration and
CPU runtime. `PreparedContraction` holds fixed-shape compiled code without
retaining input values. The scoped read-only runtime borrows compatible ndarray
storage; the adapter copies noncontiguous/reversed inputs explicitly. Each
omeco tree node becomes a distinct traced einsum, retaining its intermediate
axes and grouping. Unary expressions and empty scalar networks are preserved.
Library and CLI tree validation share one implementation. The CLI rejects
nonbinary trees unsupported by the previous omeinsum provider rather than
letting that provider panic.

The [supported-adapter M4 report](../../benchmarks/results/mac-supported-cpu-2026-09-07/report.md)
compares an identical omeco tree within each process, separates compilation
from repeated execution, and retains three runs at one/four threads. All 114
Yao comparisons for the 19 shared circuit cases passed (maximum error 1.80e-15);
the supported adapter is also checked against the previous contractor before
each benchmark. Source hashes match the recorded `b1b262c` commit.

One-thread medians of run medians, including ndarray adaptation:

| Workload | omeinsum, supplied tree | tenferro prepared | tenferro compile + execute |
| --- | ---: | ---: | ---: |
| 8-qubit tensor state | 74.24 µs | 502.26 µs | 824.54 µs |
| 4-qubit density matrix | 111.27 µs | 425.48 µs | 600.52 µs |

The traced runtime adapter has substantial overhead on these small networks;
the prototype's prepared timing has a different execution/ownership boundary.
The tensor-state case also varies substantially across processes, shown in the
plots and raw samples. These measurements support keeping the default provider
unchanged. They are not stable performance regression thresholds or evidence
of a universal tenferro speedup. Runtime/graph overhead is an optimization
opportunity for later backend work.

Validation: `make check-all` (607 tests), tenferro-only and no-default-feature
workspace tests, documentation builds, default/all-feature package verification,
fixture/reporting checks, and Linux/macOS CI. CPU-only builds need no GPU runtime.
