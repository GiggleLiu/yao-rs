# Variational Quantum Eigensolver (VQE)

> Minimize a small Heisenberg Hamiltonian with a parameterized circuit and
> adjoint-mode gradients. This example builds the Hamiltonian as an
> `OperatorPolynomial`, constructs an `Ry` ansatz with a CNOT chain, and trains
> the circuit parameters with plain stochastic-gradient-style updates.

## Background

The Variational Quantum Eigensolver prepares a parameterized trial state
\\( |\psi(\boldsymbol\theta)\rangle \\), evaluates the energy
\\( \langle\psi(\boldsymbol\theta)|H|\psi(\boldsymbol\theta)\rangle \\),
and uses a classical optimizer to reduce that energy. This page uses the
4-qubit open-chain Heisenberg-like Hamiltonian
\\( \sum_i Z_iZ_{i+1} + X_iX_{i+1} \\), a shallow hardware-style ansatz, and
`expect_grad` to compute the energy and gradient in one adjoint-mode pass.

The full runnable example lives at `examples/vqe.rs`.

## Building the Hamiltonian

The Hamiltonian is a sum of Pauli products, represented as an
`OperatorPolynomial`. Each nearest-neighbor edge contributes one `ZZ` term and
one `XX` term.

```rust
use num_complex::Complex64;
use yao_rs::operator::{Op, OperatorPolynomial, OperatorString};

fn heisenberg(n: usize) -> OperatorPolynomial {
    let one = Complex64::new(1.0, 0.0);
    let mut h = OperatorPolynomial::zero();
    for i in 0..n - 1 {
        let zz = OperatorPolynomial::new(
            vec![one],
            vec![OperatorString::new(vec![(i, Op::Z), (i + 1, Op::Z)])],
        );
        let xx = OperatorPolynomial::new(
            vec![one],
            vec![OperatorString::new(vec![(i, Op::X), (i + 1, Op::X)])],
        );
        h = &(&h + &zz) + &xx;
    }
    h
}
```

The `OperatorString` site indices follow the same qubit numbering used by
circuits: qubit 0 is the most significant qubit in state-vector index order.

## Building the ansatz

The ansatz starts from a layer of Hadamards, then repeats a trainable `Ry`
layer followed by a fixed CNOT chain. For `n` qubits and `layers` layers, the
parameter count is `n * layers`, because only the `Ry` gates are trainable.

```rust
use yao_rs::circuit::{Circuit, control, put};
use yao_rs::gate::Gate;

fn ansatz(n: usize, layers: usize) -> Circuit {
    let mut elems = Vec::new();
    for q in 0..n {
        elems.push(put(vec![q], Gate::H));
    }
    for _ in 0..layers {
        for q in 0..n {
            elems.push(put(vec![q], Gate::Ry(0.0)));
        }
        for q in 0..n - 1 {
            elems.push(control(vec![q], vec![q + 1], Gate::X));
        }
    }
    Circuit::qubits(n, elems).unwrap()
}
```

`Circuit::parameters()` reads the trainable angles in element order.
`Circuit::dispatch(&params)` writes a new parameter vector back into the same
canonical order.

## Training loop

`expect_grad(&h, &circuit, &psi0)` returns both the current energy and the
gradient with respect to `circuit.parameters()`. The example keeps parameters
in a plain `Vec<f64>`, dispatches them into the circuit before each evaluation,
and then applies an SGD update.

```rust
use yao_rs::ad::expect_grad;
use yao_rs::register::ArrayReg;

let n = 4usize;
let h = heisenberg(n);
let psi0 = ArrayReg::zero_state(n);
let mut circuit = ansatz(n, 3);

let mut params: Vec<f64> = (0..circuit.num_params())
    .map(|i| 0.01 + 0.007 * (i as f64))
    .collect();

let lr = 0.05_f64;
for _step in 1..=300 {
    circuit.dispatch(&params);
    let (_energy, grad) = expect_grad(&h, &circuit, &psi0);
    for (p, g) in params.iter_mut().zip(&grad) {
        *p -= lr * g;
    }
}
```

For a complete copy-pasteable program, use the repository example:

```bash
cargo run --release --example vqe
```

The run prints an energy trace. On this repository's deterministic
initialization, it starts near `2.715930` and drops below `-4.3` after 300
steps.

## Reproducing the run

From the repo root:

```bash
cargo run --release --example vqe
```

Expected output has this shape:

```text
step 0: energy = 2.715930
step 50: energy = -4.199266
step 100: energy = -4.227495
step 150: energy = -4.235043
step 200: energy = -4.242315
step 250: energy = -4.260200
step 300: energy = -4.320383
```

The exact last digits can vary with compiler and math-library choices, but the
energy should decrease substantially over the loop.

## Comparison

Yao.jl's VQE tutorial #7 follows the same algorithmic pattern: build a Pauli
Hamiltonian, prepare a parameterized ansatz, evaluate the energy, compute
gradients, and update parameters with a classical optimizer.
