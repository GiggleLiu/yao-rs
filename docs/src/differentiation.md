# Differentiable simulation

yao-rs computes expectation gradients with adjoint-mode differentiation:
one forward circuit evaluation and one reverse sweep. Like Yao.jl's
`expect'(H, state => circuit)`, this uses reversibility to avoid retaining a
state vector for every gate. The Rust `expect_grad` returns the expectation
value and the circuit parameter gradients; it does not return an input-state
gradient or integrate with a general-purpose AD framework.

```rust
use yao_rs::{ArrayReg, Circuit, Gate, Op, OperatorPolynomial, expect_grad, put};

let mut circuit = Circuit::qubits(1, vec![put(vec![0], Gate::Rx(0.3))]).unwrap();
let observable = OperatorPolynomial::single(0, Op::Z, 1.0.into());
let initial = ArrayReg::zero_state(1);
let (value, gradient) = expect_grad(&observable, &circuit, &initial);
assert!((value - 0.3_f64.cos()).abs() < 1e-12);
assert!((gradient[0] + 0.3_f64.sin()).abs() < 1e-12);

let updated: Vec<f64> = circuit.parameters().iter().zip(&gradient)
    .map(|(parameter, derivative)| parameter - 0.05 * derivative).collect();
circuit.dispatch(&updated);
```

Supported trainable gates are `Rx`, `Ry`, `Rz`, `Phase`, and `FSim` (theta,
then phi), including controlled and active-low controlled gates. Parameter
order follows circuit element order and each gate's parameter order. Fixed
gates and annotations contribute no parameters.

The observable must be Hermitian, gates must be unitary, and the input must
be an `ArrayReg` with matching qubit count. Noise-channel differentiation is
unsupported and is rejected; noisy forward simulation is available through
`DensityMatrix` and the CLI. Custom matrices have no trainable entries.

See the [VQE example](./examples/vqe.md) for a complete optimization loop.
Tests compare analytic derivatives and independent finite differences,
including active-low controls and FSim. A committed Yao.jl reference fixture
checks values, parameter order, and gradients against the local upstream
implementation; regenerate it with `scripts/generate_ad_reference.jl`. The API follows the reverse-mode
workflow in [Yao.jl's differentiation guide](https://docs.yaoquantum.org/stable/man/automatic_differentiation.html).
