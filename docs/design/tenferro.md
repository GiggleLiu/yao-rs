# Tenferro integration decision and execution record

The approved roadmap is implemented as sequential, squash-merged PRs. Initial
integration preserves existing public APIs and specialized CPU kernels. GPU
validation targets the NVIDIA A800 host reachable as `ssh gpu`.

## Milestones

| Milestone | Required evidence | Status |
| --- | --- | --- |
| 1. Published backend feasibility / CPU baseline | Complex/layout/AD/custom-operation tests; reproducible CPU report and memory measurements | Merged [PR #47](https://github.com/GiggleLiu/yao-rs/pull/47) (`6bad156`) |
| 2. Supported tenferro CPU backend | Library/CLI adapter, explicit and reusable plans, numerical/feature/package checks | Implemented in [PR #48](https://github.com/GiggleLiu/yao-rs/pull/48); merged (`3ec7e69`) |
| 3. Hamiltonian evolution | Pauli rotations, model builders, Trotter/Suzuki, physical parameter bindings and convergence tests | Merged [PR #49](https://github.com/GiggleLiu/yao-rs/pull/49) (`a1f24f3`) |
| 4. Differentiable circuits | Custom losses/input-state VJP, tenferro integration, shared parameters, numerical and memory tests | Merged [PR #50](https://github.com/GiggleLiu/yao-rs/pull/50) (`bd00084`) |
| 5. Tensor memory / observables | Polynomial expectations, omeco slicing, estimates, versioned plans and budget tests | Merged [PR #51](https://github.com/GiggleLiu/yao-rs/pull/51) (`53897a4`) |
| 6. Noisy trajectories | Seeded Kraus sampling, uncertainty, exact-density comparisons and memory scaling | Merged [PR #52](https://github.com/GiggleLiu/yao-rs/pull/52) (`f09b722`) |
| 7. Matrix-free exponential action | Community implementation qualification, error/convergence diagnostics, Yao comparison | Merged [PR #53](https://github.com/GiggleLiu/yao-rs/pull/53) (`ceb372f`) |
| 8. GPU execution | Resident tensor/circuit/AD execution, explicit transfers, correctness and timings via `ssh gpu` | Implemented in [PR #54](https://github.com/GiggleLiu/yao-rs/pull/54); hardware qualification and repeated A800 report complete |

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

## GPU host qualification

The host has six A800 80 GB GPUs, a dual-socket Xeon Platinum 8378A CPU and
approximately 1 TiB RAM. Other GPUs are in use; select an idle GPU explicitly
at test time. Rust 1.96.0 is already installed as a non-default rustup toolchain.
Use `cargo +1.96.0` (or a task-local newer toolchain) without changing other
users' environments.

The installed driver is 535.230.02, reporting CUDA 12.2, with a CUDA 12.1
toolkit. The qualified task-local runtime uses CUDA 12.8.90, cuBLAS 12.8.5.5,
NVRTC 12.8.93, cuTENSOR 2.6.0.4 and NVIDIA's CUDA 12.8 forward-compatibility
package 570.211.01. Published tenferro 0.4.0 executes complex128 circuits,
first-order derivatives and contractions on GPU zero with these libraries.
System drivers and other users' environments were not changed. NVIDIA's
[forward-compatibility documentation](https://docs.nvidia.com/deploy/cuda-compatibility/latest/forward-compatibility.html)
describes the compatibility requirements; this measurement qualifies one
configuration, rather than every CUDA installation. See the
[CUDA guide](../src/cuda.md) for explicit runtime selection, transfers and
unsupported upstream operations.

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

## Hamiltonian evolution (PR #49)

Pauli rotations, Ising/XYZ builders and first/second-order product formulas
lower to the existing small gates. `BoundCircuit` exposes fixed/scaled/product
bindings and their first-order Jacobian actions. Time and couplings stay shared
across expanded gates; zero time retains the graph for differentiation. Identity
terms retain global phase, including under control. No full Hamiltonian matrix
is constructed by the library builders or native simulation.

The [evolution M4 report](../../benchmarks/results/mac-evolution-cpu-2026-09-07/report.md)
contains 22 shared cases, three independent runs at one/four threads, and 132
passing Yao state comparisons (maximum discrepancy 5.70e-14). A separately built
dense Yao Hamiltonian supplies the exact exponential oracle for three-qubit
accuracy checks. Doubling steps from 8 to 16 reduces first-order error by about
two and second-order error by about four for both noncommuting models.

At 16 second-order steps, one-thread native/Yao execution medians are
12.01/10.56 µs for Ising and 53.09/76.02 µs for XYZ, with relative errors
3.94e-4 and 8.41e-4 respectively. These are identical lowered formulas, not
adaptive Krylov comparisons. The zero-state XYZ tensor workload takes 0.70 ms
with omeinsum's supplied tree and 3.67 ms with prepared tenferro using the same
tree; tenferro compilation plus execution takes 8.54 ms. Native kernels remain
the appropriate low-overhead path for these small states.

Isolated memory probes separate circuit storage and execution heap from RSS.
At 12 qubits, XYZ circuit storage grows from 101.15 KiB at one step to
1616.38 KiB at 16 steps, while additional native execution heap stays at
64.06 KiB. The execution path does not retain a state per gate. These small
cases do not establish a large-state memory limit; raw logs include process RSS.

Validation: `make check-all` (618 tests), no-default-feature tests, warnings-denied
rustdoc and mdBook, the native/tenferro example, fixture Clippy, and five report
tests. Timing sources are pinned at `2c97de3`; later memory/report source hashes
are recorded separately. GPU execution remains a later milestone.


## Differentiable circuits (PR #50)

`DifferentiableCircuit` generalizes the existing reversible adjoint sweep to
arbitrary complex output cotangents and returns both physical-parameter and
input-state gradients. Its JVP applies the same binding Jacobian. The optional
`tenferro-ad` feature exposes this as an eager/traced operation whose trainable
parameters are tensor inputs. Tenferro differentiates surrounding real losses;
the circuit operation retains its final state and parameter values. First-order
rules reject higher-order transforms explicitly. Channels and nonunitary custom
matrices are outside the reversible domain. An optional argmin L-BFGS example
optimizes a complex state-distance loss to below `1e-20`.

The [circuit AD M4 report](../../benchmarks/results/mac-circuit-ad-cpu-2026-09-07/report.md)
contains six shared cases at 8/12/16 qubits and 10/100 layers, three independent
runs each at one/four configured threads, and 36 passing Yao comparisons of
loss, every real parameter gradient, and every complex input gradient. The
maximum discrepancy is `2.20e-14`. Timing and memory sources are pinned at
`6a949d7`; a later plot-label adjustment is recorded separately in metadata.

One-thread medians of run medians, returning the same loss and gradients:

| Workload | Native | Yao | Tenferro circuit operation | Ordinary tensor composition |
| --- | ---: | ---: | ---: | ---: |
| 12 qubits, 10 layers | 0.535 ms | 0.615 ms | 1.429 ms | 42.637 ms |
| 16 qubits, 10 layers | 8.542 ms | 9.339 ms | 20.120 ms | 103.422 ms |
| 16 qubits, 100 layers | 83.395 ms | 89.391 ms | 185.119 ms | Not repeatedly timed |

Tenferro rows include input copies, eager graph construction, loss, backward,
and gradient collection; reusable context creation is excluded. Its custom
operation substantially reduces the cost of this unfused tensor-composition
fixture but remains slower than native/Yao execution. Four threads do not
consistently help these workloads; native circuit kernels remain serial.

In isolated memory probes, the 16-qubit custom operation retains 3.022 MiB
after the forward phase at 10 layers and 3.026 MiB at 100 layers; process RSS
is 28.70 and 28.50 MiB respectively. Ordinary composition at 16 qubits and
10 layers peaks at 235.64 MiB RSS. Its 8-qubit, 100-layer backward reaches the
30 CPU-second limit with 3138.70 MiB observed peak RSS; larger 100-layer
composition probes are explicitly skipped. This incomplete run supplies no
completed timing or speedup ratio. Rust heap and whole-process RSS measure
different storage, and allocator/provider retention affects the latter.

Validation: `make check-all` (631 tests), nine fixture tests, six report tests,
warnings-denied rustdoc and mdBook, successful optimizer example, and a Linux
Rust 1.96 check of all workspace targets/features. Numerical tests include
finite-difference step sweeps, real/imaginary input perturbations, JVP/VJP
duality, non-real cotangents, active-low controls, FSim, shared and zero-time
parameters, empty parameter tensors, and invalid derivative domains.


## Tensor memory and polynomial expectations (PR #51)

Pure/noisy polynomial expectations share their circuit tensors and a summed
term-selection index. Identity/zero coefficients are preserved. `SlicedPlan`
validates fixed slices, checked byte estimates and assignment limits; omeco
TreeSA supplies optional automatic slicing/replanning. A single active slice
makes reduction order deterministic and keeps concurrent tensor storage bounded.
Tenferro prepares one reusable slice shape. CLI version-two plans retain their
labels and limits; an actual preceding-version CLI rejects them before execution,
with [recorded compatibility evidence](../../benchmarks/results/mac-tensor-memory-cpu-2026-09-07/reader-compatibility.json).

The [tensor-memory M4 report](../../benchmarks/results/mac-tensor-memory-cpu-2026-09-07/report.md)
contains three independent processes each at one/four threads. All 24 Yao
complex-expectation comparisons pass (maximum discrepancy `1.12e-16`). The
reference preserves complex values through Yao's sandwich/trace formulas;
Yao's density formula builds a dense operator, while Rust applies each word.
Measured sources are pinned at `8b91c3c`; later plot presentation changes have
separate hashes. Selected execution trees are recorded per timing process.

At six qubits, one-thread medians for the five-term pure expectation are
3.52 µs native, 4.58 µs Yao, 359.77 µs omeinsum and 1358.48 µs prepared
tenferro. The noisy density expectation takes 280.06/234.06 µs in native/Yao;
its tensor contractions take 360.42/1357.31 µs. Slicing the five-term index on
the same tree multiplies contraction work; tenferro takes about 6.75–6.80 ms.
These small circuits favor specialized simulation and retain existing defaults.

All 26 isolated matrix memory probes completed. For an ordinary 256-square
matrix chain, tenferro's fixed output slicing changes its tensor estimate from
14.00 to 9.02 MiB but RSS remains about 12 MiB; execution grows from 5.71 to
15.37 ms. Slicing is not automatically a process-memory or time improvement.

The separately labelled 64-square outer-product stress path intentionally
creates a large intermediate. Tenferro RSS falls from 518.86 to 19.38 MiB with
fixed output slicing, while execution changes from 25.82 to 27.06 ms. A greedy
unsliced order avoids that intermediate: 6.81 MiB RSS and 0.122 ms. omeinsum
shows the same qualitative result (515.25/11.61/3.70 MiB and
26.09/23.41/0.226 ms for supplied/sliced/greedy paths). This is evidence for
choosing a good order before slicing, not a comparison against an optimized
unsliced baseline. Automatic TreeSA slicing is separately qualified at dimension
32; it chooses 1024 assignments in these runs, with its actual trees recorded.

The estimate is a tensor-storage model plus a user workspace reserve, not a
hard RSS bound. The default zero reserve does not imply zero provider scratch;
measured heap/RSS can exceed it. Raw records separate inputs, output, omeco's
live tensor estimate, worker buffers, reserve, execution heap and process RSS.

Validation: `make check-all` (643 tests), 11 fixture tests and Clippy, seven
report tests, warnings-denied rustdoc/mdBook, runnable sliced-expectation example,
Linux Rust 1.96 all-target/all-feature compilation, and CLI version/pipeline
checks. GPU, stochastic trajectories and matrix-free evolution remain later
milestones.


## Seeded noisy trajectories (PR #52)

`TrajectoryCircuit` validates qubit/unitary structure and local CPTP Kraus maps,
then samples normalized branches with native state-vector kernels. Rand's existing
ChaCha8 streams use trajectory IDs; optional Rayon workers reduce scalar moments
in trajectory order. Real/imaginary variance, standard error and covariance are
streamed with bounded state buffers. The CLI selects this mode explicitly with
`run --trajectories N --op ... --seed ...`; measurement shots stay separate.
Built-in register-wide depolarization samples Pauli words without materializing
its dense Kraus list.

Thermal relaxation now matches analytic population/coherence decay, fixing the
omitted survival factor in the former Julia-derived conversion. Tests cover
positive infinite time constants and long durations without cancellation in the
coherence amplitude. The historical Julia Kraus-entry fixture is superseded by
physical channel-map tests; other channel comparisons remain intact.

Implementation validation: `make check-all` (656 tests), 12 benchmark-fixture tests
and Clippy, nine report tests, warnings-denied API docs/mdBook, a runnable noise
example, CLI pure-input/density-input checks, and serial/parallel reproducibility.
An independent embedded complex matrix qualifies three-target Kraus execution.
The full workspace suite also passes on the GPU host using Linux Rust 1.96.

The [trajectory M4 CPU report](../../benchmarks/results/mac-trajectories-cpu-2026-09-07/report.md)
records three independent processes at one/four threads, with raw samples,
source hashes and 23 isolated memory probes. All 18 Yao exact complex-expectation
comparisons pass (maximum discrepancy `3.13e-16`); tenferro exact contractions
also agree with direct density evolution. Measured implementation is `5cbef8a`;
subsequent report presentation/validation changes have separate hashes.

For the six-qubit entangled case, eight-seed RMSE falls from 0.02450 at 128
trajectories to 0.01045 at 512 and 0.00370 at 2048. Predicted RMS sampling errors
are 0.01932/0.00952/0.00481. One-worker ensemble times are 0.94/3.80/15.06 ms;
exact native/Yao density expectations take 1.06/1.14 ms. Trajectories trade
sampling error and work for memory, and are not automatically faster.

At eight qubits the exact prepared tenferro contraction takes 1.98 ms versus
20.43 ms native density and 19.33 ms Yao (one thread). This tensor boundary
excludes export, order search and its separately measured 1.42 ms compilation;
it does not measure a tenferro trajectory executor. Existing defaults remain.

For the simpler 16-qubit product fixture, 256 trajectories take 1385.66 ms with
one worker and 455.99 ms with four. Additional execution heap stays 2.0001 MiB
at both 64 and 256 samples for one worker, and about 8.03 MiB for four. Process
RSS is 5.50/11.89 MiB for the 256-sample runs. The 10-qubit exact density probe
uses 48.00 MiB additional execution heap and 66.50 MiB RSS; one-worker
trajectories on that same product fixture use 0.0313 MiB and 2.48 MiB. These
memory measurements include neither an accuracy equivalence claim nor a
claim that heap equals process RSS. Parallel overhead slows the four-qubit
fixture, while benefiting the larger workloads.


## Matrix-free Hermitian evolution

`PauliHamiltonian::evolve_krylov` applies Pauli sums through compiled bit masks.
`evolution::exponential_action` exposes the same CPU solver to fixed, linear,
Hermitian complex callbacks. Adaptive Lanczos uses twice-modified Gram–Schmidt,
pairwise reductions, and the existing faer eigensolver on the small real
tridiagonal projection. No full Hilbert-space square matrix is allocated.
The basis cap and total operator-application limit bound storage and work.

The local truncation criterion follows Jawecki, Auzinger and Koch,
[Theorem 1](https://doi.org/10.1007/s10543-019-00771-6), with an added estimate
for discarded reorthogonalization corrections. The public diagnostics report
accepted time, work, steps and accumulated estimate; they do not certify all
floating-point roundoff. Failed convergence returns the last accepted state
inside an error, never a successful state at the wrong time. Zero/stationary
inputs, negative time, unnormalized vectors and identity phases are supported.
There is no derivative through this adaptive solver; fixed product formulas
remain available for tenferro circuit differentiation.

Community qualification found that the inspected ORMATEX Rust API uses real
operators and lacks the required returned convergence failure, while the
inspected scirs2 interfaces lack the required complex callback/error-control
combination. The implementation is independently written, with KrylovKit
[v0.10.2 / 775546b](https://github.com/Jutho/KrylovKit.jl/tree/775546bccc5053193ce72d66725aaabe93b8d6ca)
as a design reference. It is not a translation of KrylovKit's phi-function
integrator and adds no runtime dependency. See the Hamiltonian guide for the
algorithm and callback contract.

The [repeated CPU report](../../benchmarks/results/mac-krylov-cpu-2026-09-07/report.md)
records 32 workloads at one/four threads, three independent processes each,
and 24 isolated memory probes. The measured source is `b3bb513`; all 192
Julia qualification records pass and tight references differ by at most
3.46e-13. The report includes achieved-error/time plots, the complete tolerance
sweep, raw convergence diagnostics, and separate tensor planning/execution.

On the M4 at one thread and nominal tolerance 1e-10, 16-qubit Ising takes
273.93 ms in Rust with relative state error 5.67e-11, versus 282.38 ms and
4.27e-10 in Yao. The corresponding XYZ case takes 506.87 ms / 6.02e-11 versus
631.62 ms / 1.63e-10. Results vary by workload: 12-qubit Ising takes 11.93 ms
in Rust and 8.96 ms in Yao, with errors 6.36e-11 and 1.95e-10 respectively.
The solvers use different stopping criteria and basis policies, so nominal
tolerance ratios are not equal-accuracy speedups. Native vector kernels remain
serial; four-thread provider settings do not make them parallel.

For the 16-qubit XYZ memory probe at rtol=1e-8, basis caps 8/20/40 use
11.04/23.05/43.09 MiB additional execution heap and 279/74/51 operator
applications. Whole-process RSS is 14.95/27.16/47.42 MiB. The cap therefore
exposes a measurable storage/work tradeoff, and heap remains distinct from RSS.

The bounded four-qubit tensor comparison executes fixed Suzuki circuits.
At 32 XYZ steps, native/Yao execution takes 0.176/0.223 ms, the supported
tenferro adapter takes 96.24 ms with a prepared tree, and omeinsum takes
16.77 ms with that same tree. The product-formula state error is 4.57e-5.
These small-state circuits expose general tensor execution overhead; they do
not measure an adaptive tenferro Krylov backend or establish a default backend
change. The zero-state four-qubit XYZ adaptive case terminates in a
four-dimensional invariant subspace; asymmetric inputs are tested separately.


## CUDA circuits, differentiation and tensor contraction (PR #54)

`CudaSimulator` owns one explicit tenferro CUDA runtime. `CudaCircuit` reuses
validated unitary structure, physical bindings and existing gate generators;
parameters and state values remain device inputs. Controls form batch axes of
local gate tensors so each gate applies one state contraction. A gate with
`c` controls and `k` targets stores `2^c * 4^k` local complex entries, separate
from state and derivative storage. Preparation uploads constants, while
execution evaluates parameter-dependent trigonometric factors on the device.

`PreparedCudaContraction` validates shapes without device initialization and
preserves each supplied omeco tree node. N-ary nodes lower left to right.
Contraction shares CPU shape/tree validation, supports changed tensor values,
and executes fixed-noise density networks and arbitrary qudit exports.
Upload/download boundaries are explicit. Context validation reads borrowed
placement metadata; it does not copy GPU values merely to inspect them.

The seven opt-in hardware tests cover named/custom gates, reordered targets,
mixed active-low/high controls, scalar phases, physical parameter bindings,
finite-difference step sweeps, complex input VJPs, JVP/VJP pairing, custom
losses and device parameter updates. Tensor tests cover empty/unary/hyperedge
networks, explicit contraction order, qudits and exact Kraus noise. All pass
on the A800 in release mode and debug mode with an 8 MiB test-thread stack.
Regular CPU-only CI compiles all features and runs device-independent plan
validation without initializing CUDA.

The pinned upstream backend has explicit limits. Trace/diagonal forward
contraction works, but tracked networks with repeated input labels are
rejected because their pullback reaches a host-only padding path. Complex
`abs` differentiation reaches an unsupported sign operation; the example uses
the smooth real squared norm. The upstream whole-program eager prototype is
rejected for contraction. CUDA AD uses ordinary tensor composition and retains
intermediates; the CPU reversible operation's bounded state memory does not
apply. GPU slicing, trajectory execution, adaptive Krylov and differentiation
of noise parameters remain unsupported.

The [A800 report](../../benchmarks/results/a800-cuda-2026-09-07/README.md)
records three independent processes per runner: 168 CPU phase records, 66 GPU
timing records, 42 passing Yao comparisons and all 14 isolated GPU qualification
processes. All 84 measured source hashes match `5e825920`; later report validation
and plotting changes have separate hashes. Full-output CUDA/native differences
are at most `1.15e-14`, and Yao/native differences at most `6.66e-15`.

At 24 qubits and ten layers, native/Yao state execution takes 1585.27/1488.15 ms,
versus 76.27 ms resident on CUDA and 812.82 ms including transfers. At 20 qubits,
the ten-layer loss and all gradients take 515.98/349.75 ms native/Yao,
1477.83 ms through the CPU tenferro custom operation, and 899.17/910.28 ms
resident/transfer-inclusive on CUDA. Smaller cases strongly favor native CPU
kernels. The gradient fixture repeats four gates on two sites, rather than a
full-width variational ansatz.

The ten-qubit exact-noise fixture takes 1513.95/539.87 ms with direct native/Yao
density evolution. The existing omeinsum CPU contractor takes 22.39 ms and
prepared tenferro CPU 45.72 ms, compared with 20.70 ms resident CUDA and
153.16 ms including transfers. At eight qubits omeinsum is substantially faster
than CUDA. These comparisons include algorithm and boundary differences; they
do not establish a universal GPU or tenferro advantage. Defaults remain intact.

Forty-layer gradients have bounded, separately labeled GPU diagnostics and full
output checks: first execution takes about 132–135 seconds, with one warm sample
of 7.58–15.60 seconds per case. They are excluded from repeated GPU timing
ratios. Observed device process memory after repeats is 856/920/1112 MiB at
8/16/20 qubits for both depths. Allocator-inclusive snapshots do not establish
depth-independent peak memory. Host peak RSS, process-cold context/preparation,
runtime package hashes, hardware test logs and standalone figures are retained.
The shared host uses one configured CPU thread, unpinned clocks/affinity and
niceness 5; raw independent-run ranges remain part of the report.

Validation: `make check-all` (668 passing workspace tests), 12 fixture tests,
12 report tests including rejection of incomplete/non-finite qualifications,
warnings-denied rustdoc/mdBook, and Linux/macOS feature-matrix CI. The seven
hardware tests are opt-in so CPU-only installation remains usable.
