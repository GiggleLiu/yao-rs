# Differentiable simulation

yao-rs computes expectation gradients with adjoint-mode differentiation:
one forward circuit evaluation and one reverse sweep. Like Yao.jl's
`expect'(H, state => circuit)`, this uses reversibility to avoid retaining a
state vector for every gate. The Rust `expect_grad` returns the expectation
value and the circuit parameter gradients. `DifferentiableCircuit` additionally
provides input-state gradients and arbitrary output cotangents; the optional
`tenferro-ad` feature composes these derivatives with tenferro custom losses.

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

## Parameters and input-state derivatives

`DifferentiableCircuit::from_circuit` treats each gate angle independently.
`DifferentiableCircuit::new(bound)` preserves a `BoundCircuit`'s physical
bindings, including shared, scaled and product parameters from Hamiltonian
evolution. Construct once, then pass new real parameter values on each call.

```rust
use yao_rs::{ArrayReg, Circuit, Gate, put};
use yao_rs::differentiable::DifferentiableCircuit;
let circuit = Circuit::qubits(1, vec![put(vec![0], Gate::Ry(0.3))]).unwrap();
let prepared = DifferentiableCircuit::from_circuit(circuit).unwrap();
let input = ArrayReg::zero_state(1);
let seed = ArrayReg::zero_state(1); // loss = Re(output[0])
let gradient = prepared.vjp(&[0.3], &input, &seed).unwrap();
assert!((gradient.parameters[0] + 0.5 * (0.15_f64).sin()).abs() < 1e-12);
assert_eq!(gradient.input.state.len(), 2);
```

The convention is the **real Hermitian pairing**:

```text
dL = Re(sum(conj(input_bar) * dinput)) + dot(parameter_bar, dparameters)
```

Real and imaginary input components are independent real directions; inputs
need not be normalized. For a Hermitian expectation, seed with `2 H output`.
For squared state distance, seed with `2 (output - target)`. `jvp` returns both
the output and its directional derivative for physical and input-state
tangents. `value_and_grad` accepts an analytic loss callback returning its
value and output cotangent, avoiding an extra circuit forward pass.

The reversible path validates qubit dimensions, finite parameters/states,
custom-matrix unitarity and diagonal hints. Channels and nonunitary custom
matrices return errors. Custom matrices are fixed, not trainable matrix inputs.
Physical bindings accumulate all shared occurrences. Fixed-only circuits and
empty parameter vectors still produce input-state derivatives.

## Custom losses with tenferro

Enable `tenferro-ad` and create a context with
`yao_rs::tenferro_ad::eager_cpu_runtime(threads)`. Create real F64 parameter and
complex C64 state tensors using `EagerTensor::requires_grad_in`, then pass them
to `circuit_apply_eager(Arc<DifferentiableCircuit>, &parameters, &input)`.
The returned tensor participates in ordinary tenferro operations:

```rust,ignore
let output = circuit_apply_eager(prepared, &parameters, &input)?;
let difference = output.sub(&target)?;
let magnitude = difference.abs()?;
let loss = magnitude.mul(&magnitude)?.reduce_sum(Some(&[0]))?;
let gradients = loss.backward()?;
let parameter_gradient = parameters.grad()?.unwrap(); // F64
let input_gradient = input.grad()?.unwrap();           // C64
```

`backward()` follows tenferro's accumulation semantics. Use fresh tracked
leaves per optimization evaluation, or clear gradients when reusing leaves.
Drop evaluation tensors and gradient owners to release retained tape/storage.
Reuse the context and circuit descriptor across evaluations. Trainable values
are tensor inputs, so updates do not require changing opaque graph payloads.

The traced equivalent is `circuit_apply`, `ad_context`, and `cpu_runtime`.
The registered first-order rules support both JVPs and VJPs; higher-order
transforms of the derivative primitives return unsupported errors. Runtime
shape constraints and execution validation reject wrong dimensions/dtypes.
Inputs use compact rank-one CPU tensors; GPU circuit kernels are a later
milestone and there is no implicit device-to-host fallback.

Reverse execution retains physical parameters and the final circuit state.
The kernel reconstructs earlier states with adjoint gates, using a fixed number
of state-sized buffers. Circuit and parameter metadata still grow with circuit
size. Arbitrary tensor losses have their own retention needs, so the entire
loss graph does not have a universal constant-memory guarantee.

## Optimizer example

```sh
cargo run --release --example custom_loss --features optimizer-example
```

This fits a complex entangled target state using a tenferro squared-distance
loss and [argmin](https://docs.rs/argmin/0.11.0/argmin/)'s L-BFGS and line search.
The adapter caches the value/gradient pair when argmin requests them separately
at identical parameters. It checks that the final squared error is below
`1e-10`. The optional optimizer dependencies are confined to this example
feature; the core library does not implement an optimizer.
