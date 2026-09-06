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
implement a second general differentiation engine. A later milestone adds
custom losses and input-state VJPs through tenferro.

## CPU comparison

The [recorded CPU report](https://github.com/GiggleLiu/yao-rs/blob/main/benchmarks/results/mac-evolution-cpu-2026-09-07/report.md)
compares the same Ising/XYZ circuits with Yao.jl, separates approximation error
from cross-language agreement, and includes execution costs and isolated memory
measurements. The sweep uses 1–16 product steps and three runs each at one/four
threads. Reproduce it with the evolution suite in `benchmarks/README.md`.
