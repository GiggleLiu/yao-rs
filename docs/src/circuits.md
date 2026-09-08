# Circuits & gates

Build a circuit by placing gates on numbered sites. A gate describes an
operation; its placement specifies the targets and any controls. The circuit
checks that those placements match the register dimensions.

## Build a circuit

```rust
use yao_rs::{Circuit, Gate, control, put};

let circuit = Circuit::qubits(2, vec![
    put(vec![0], Gate::H),
    control(vec![0], vec![1], Gate::X),
]).unwrap();
```

`put` applies a gate to its target sites. `control` applies it only when all
control sites are in state 1. Operations run in the order you add them.
See the [gate reference](gates.md) for supported gates and parameters.

The CLI reads the same circuit from [JSON](conventions.md#circuit-json).
Use `yao inspect circuit.json` to check its structure before running it.

## Choose controls

A controlled X is a CNOT. Two controls make a Toffoli:

```rust
use yao_rs::{Gate, control};

let toffoli = control(vec![0, 1], vec![2], Gate::X);
```

To trigger on state 0, set the corresponding `control_configs` entry to `false`:

```rust
use yao_rs::{Circuit, CircuitElement, Gate, control};

let mut gate = control(vec![0], vec![1], Gate::X);
if let CircuitElement::Gate(ref mut positioned) = gate {
    positioned.control_configs[0] = false;
}
let circuit = Circuit::qubits(2, vec![gate]).unwrap();
```

## Validate placement

Construction returns `Result<Circuit, CircuitError>`. It checks that sites
exist, targets and controls do not overlap, controls are qubits, and gate
matrices match the target dimensions.

```rust
use yao_rs::{Circuit, Gate, put};

let result = Circuit::qubits(2, vec![put(vec![5], Gate::H)]);
assert!(result.is_err()); // There is no qubit 5.
```

Validation checks circuit structure; it does not prove that a custom matrix
is unitary. See the [error reference](api/yao_rs/circuit/enum.CircuitError.html)
for individual error cases.

## Change parameters

Parameterized gates store angles in radians. A circuit exposes them in element
order so an optimizer can update an experiment without rebuilding it:

```rust
use yao_rs::{Circuit, Gate, put};

let mut circuit = Circuit::qubits(1, vec![put(vec![0], Gate::Ry(0.0))]).unwrap();
circuit.dispatch(&[std::f64::consts::FRAC_PI_2]);
assert_eq!(circuit.parameters().len(), 1);
```

The [VQE example](examples/vqe.md) uses this interface with `expect_grad`
to minimize an energy.

## Use higher-dimensional sites

For a mixed register, `Circuit::new` takes a dimension for each site:
`vec![2, 3, 2]` describes a qubit, a qutrit, and a qubit.
Custom gates can act on these sites when their matrix dimensions match.
Named gates and controls require qubits.

Use [tensor network export](tensor-networks.md) to evaluate qudit circuits.
Direct simulation with `ArrayReg` requires every site to have dimension 2.
