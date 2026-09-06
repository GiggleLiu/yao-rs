# Tenferro integration decision and execution record

The approved roadmap is implemented as sequential, squash-merged PRs. Initial
integration preserves existing public APIs and specialized CPU kernels. GPU
validation targets the NVIDIA A800 host reachable as `ssh gpu`.

## Milestones

| Milestone | Required evidence | Status |
| --- | --- | --- |
| 1. Published backend feasibility / CPU baseline | Complex/layout/AD/custom-operation tests; reproducible CPU report and memory measurements | In progress |
| 2. Supported tenferro CPU backend | Library/CLI adapter, explicit and reusable plans, numerical/feature/package checks | Pending |
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

Milestone 1 checks and raw CPU results will be linked here before its PR is
merged. See `benchmarks/README.md` for shared fixtures, numerical equivalence,
thread controls, phase boundaries and the distinction between Rust heap and
process RSS. GPU timings must later distinguish synchronized resident execution
from transfer-inclusive execution, and report unsupported dtype/operations
without silent CPU fallback.
