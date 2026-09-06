# Hamiltonian evolution

The `hamiltonian` module builds Hermitian Pauli sums and approximates
`exp(-i H t)` with existing one- and two-qubit gates. No full Hamiltonian matrix
is allocated. The builders require at least one qubit.

```rust
use yao_rs::{ArrayReg, Op, OperatorPolynomial, expect_grad};
use yao_rs::hamiltonian::{Boundary, ProductFormula, ising};

let h = ising(3, -0.7, 0.4, Boundary::Open).unwrap();
let mut evolution = h.evolve(0.8, 8, ProductFormula::Suzuki2).unwrap();
let initial = ArrayReg::zero_state(3);
let state = evolution.apply(&initial).unwrap();
let z = OperatorPolynomial::single(0, Op::Z, 1.0.into());
let (_, angle_gradient) = expect_grad(&z, evolution.circuit(), &initial);
let physical_gradient = evolution.pullback(&angle_gradient).unwrap();
assert_eq!(physical_gradient.len(), 3); // [time, J, h]
evolution.dispatch(&[0.9, -0.7, 0.45]).unwrap();
```

Run the complete example with:

```sh
cargo run --release --example hamiltonian --features tenferro
```

The example checks the same lowered circuit through tenferro tensor
contraction. Export with `circuit_to_einsum_with_boundary(evolution.circuit(),
&[])` for zero-state inputs, then prepare/execute with `CpuContractor`.
Circuit JSON captures the lowered gate angles; physical bindings remain in the
Rust `BoundCircuit` object.

## Pauli rotations and identity terms

`pauli_rotation(n, &word, theta)` builds
`R_P(theta) = exp(-i theta P / 2)` and exposes `[theta]` as its physical vector.
X/Y basis changes and controlled-X parity accumulation lower a long Pauli word
to small gates. Arbitrary nonadjacent sites are supported. Sites must be in
range, unique and sorted; `OperatorString::new` sorts sites and removes explicit
identities. Deserialized words are validated and identities normalized too.
Projectors and ladder operators are outside the Hermitian Pauli-sum API.

An identity word emits `Rz(theta)` followed by `Phase(-theta)`, retaining the
global factor `exp(-i theta/2)`. This also gives the correct relative phase when
the lowered gates are controlled. Identity terms are never discarded.

## Models and conventions

| Builder | Hamiltonian | Coupling parameter order |
| --- | --- | --- |
| `ising(n, J, h, boundary)` | `J Σ Z_i Z_(i+1) + h Σ X_i` | `[J, h]` |
| `heisenberg(n, [Jx,Jy,Jz], h, boundary)` | `Σ (Jx XX + Jy YY + Jz ZZ) + h Σ Z_i` | `[Jx,Jy,Jz,h]` |

These are Pauli matrices, not spin matrices divided by two. Choose signed
couplings explicitly; for example negative Ising J favors aligned Z spins.
An open chain has `n-1` bonds; a periodic chain adds the end bond and requires
at least three sites, avoiding ambiguous self-bonds or doubled two-site bonds.

`PauliHamiltonian::new(n, &polynomial)` accepts the existing numeric
`OperatorPolynomial` representation and treats each term coefficient as an
independent physical parameter. `with_shared_coefficients(n, terms, values)`
lets multiple `(OperatorString, parameter_index)` entries share one coupling.
This also supports a single isotropic Heisenberg coupling shared across X/Y/Z.
Numeric equality alone does not tie independently declared parameters.

## Product formulas and errors

`evolve(time, steps, formula)` requires finite real time/couplings and positive
steps. A Lie–Trotter step applies each term in input order with angle
`2 * coupling * time / steps`. A symmetric second-order Suzuki step applies
forward half-steps, then reversed half-steps. Duplicate terms are kept in input
order; they are not rearranged or coalesced across noncommuting terms.

For noncommuting sums, global approximation errors generally scale as
`O(1/steps)` and `O(1/steps²)` respectively at fixed time. These are not error
bounds or a requested-tolerance solver. Refine steps to check convergence.
Commuting terms are exact apart from roundoff. Negative times are supported.

Zero Hamiltonians and zero time are handled without losing the derivative at
zero: the bound object retains its parameterized graph, while its `apply`
method returns the input exactly when every emitted rotation angle is zero.
Executing the lowered circuit directly or through tensor contraction may still
incur basis-change roundoff. An empty Hamiltonian emits no gates, regardless of
the requested number of steps.

## Physical parameter Jacobians

`BoundCircuit` keeps a validated circuit and one `ParameterBinding` per gate
angle. Bindings are fixed, scaled parameters, or products of two parameters
(such as time × coupling). Evolution vectors have order `[time, couplings...]`.
A fixed number of steps fixes the structure; dispatch updates all occurrences
without rebuilding that structure.

`pullback(angle_gradient)` accumulates the binding Jacobian transpose, including
shared couplings and repeated time steps. `pushforward(physical_tangent)` applies
the same Jacobian in the forward direction. Fixed basis-change angles contribute
no physical gradient. Invalid counts, indices, nonfinite parameters, overflowing
angles and unaddressable circuit lengths produce errors. Failed dispatch leaves
the current circuit and physical vector unchanged.

These Jacobians compose with the existing unitary `expect_grad`; they do not
implement a second general differentiation engine. They also compose with
the custom-loss and input-state VJPs described in the differentiation guide.

## Matrix-free Krylov evolution

Use `evolve_krylov` to request an accuracy tolerance without choosing a product
formula or constructing a circuit:

```rust
use yao_rs::{ArrayReg, evolution::EvolutionOptions};
use yao_rs::hamiltonian::{ising, Boundary};
let h = ising(12, -0.7, 0.4, Boundary::Open).unwrap();
let result = h.evolve_krylov(
    &ArrayReg::zero_state(12), 0.8,
    EvolutionOptions { rtol: 1e-9, ..Default::default() },
).unwrap();
let state = ArrayReg::from_vec(12, result.state);
assert_eq!(result.info.time_reached, 0.8);
println!("{} operator applications; error estimate {}",
    result.info.matvecs, result.info.estimated_error);
```

The CPU solver applies Pauli sums directly through bit masks. Each adaptive
step builds at most `krylov_dim` complex basis vectors and diagonalizes a small
real tridiagonal matrix with the existing faer dependency. Storage grows as
`O(krylov_dim * 2^n)`, plus state/work buffers and `O(krylov_dim²)` projected
workspace. There is no full `2^n`-by-`2^n` Hamiltonian. The basis buffers are
reused across time steps. Identity phases and negative time are preserved;
zero time, zero vectors, and stationary vectors retain their input exactly.

The tolerance is `atol + rtol * norm(initial)`. Defaults are `atol=1e-12`,
`rtol=1e-10`, `krylov_dim=30`, and `max_matvecs=10_000`. Diagnostics give the
accepted time, completed steps, operator applications, largest basis used,
accumulated error estimate, and requested tolerance. The estimate adds a
Lanczos truncation bound to the discarded reorthogonalization corrections.
It does not fully bound floating-point errors in the callback, inner products,
or eigensolver; very tight tolerances can fail or reach a roundoff floor.

Work exhaustion returns `EvolutionError` with kind `WorkLimit`, and an
unattainable step returns `PrecisionLimit`. Its optional `partial` field holds
the last accepted vector and actual `time_reached`. An unfinished basis counts
against the work limit but never advances that vector. A successful return
always reaches the requested time. Resuming a partial result requires the
remaining time and a separately chosen remaining error budget.

`evolution::exponential_action(&input, time, options, callback)` supports other
complex, matrix-free Hermitian operators, including non-qubit dimensions.
The callback receives input/output slices and must overwrite every output with
`H * input`. It must be a fixed linear Hermitian map; subspace checks detect
some violations but cannot prove this contract for an arbitrary callback.
Callback errors and nonfinite outputs are explicit failures. Inputs need not
have unit norm. This solver is CPU-only and provides no automatic derivatives;
use fixed-step product formulas for the existing tenferro circuit AD path.

### Community references and algorithm choice

Yao's `TimeEvolution` uses KrylovKit. The Rust solver follows the same broad
Lanczos projection/restart approach, but uses the computable defect bound in
[Jawecki, Auzinger and Koch, Theorem 1](https://doi.org/10.1007/s10543-019-00771-6)
for step selection. With residual factors `beta_j`, its truncation estimate is
`norm(x) * product(beta_1..beta_m) * abs(dt)^m / m!`. Full reorthogonalization
controls basis drift; logarithmic evaluation avoids factorial/product overflow.

The source reference was
[KrylovKit v0.10.2, commit 775546b](https://github.com/Jutho/KrylovKit.jl/tree/775546bccc5053193ce72d66725aaabe93b8d6ca),
licensed MIT. This is an independent implementation; it does not copy its
phi-function integrator or claim identical stopping criteria. ORMATEX's inspected
Rust implementation uses real-valued operators and lacks the required returned
convergence diagnostics. The inspected scirs2 exponential-action interfaces
also lack the required complex callback/error-control combination. No new
runtime dependency was adopted for this solver.

## CPU comparison

The [recorded CPU report](https://github.com/GiggleLiu/yao-rs/blob/main/benchmarks/results/mac-evolution-cpu-2026-09-07/report.md)
compares the same Ising/XYZ circuits with Yao.jl, separates approximation error
from cross-language agreement, and includes execution costs and isolated memory
measurements. The sweep uses 1–16 product steps and three runs each at one/four
threads. Reproduce it with the evolution suite in `benchmarks/README.md`.

The [Krylov CPU report](https://github.com/GiggleLiu/yao-rs/blob/main/benchmarks/results/mac-krylov-cpu-2026-09-07/report.md)
compares adaptive Rust and Yao evolution at 4–16 qubits, three tolerances and
three independent runs per thread setting. Read achieved state error alongside
execution time: the two solvers have different stopping criteria. Its 24 memory
probes show the basis-cap storage/work tradeoff, separating additional execution
heap from process RSS. Four-qubit Suzuki tensor timings are reported separately.
