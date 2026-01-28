# Circuits

A `Circuit` represents a sequence of positioned gates applied to a register of qudits. Each gate in the circuit is wrapped in a `PositionedGate` that specifies which sites the gate acts on and which sites control its activation.

## PositionedGate

```rust
pub struct PositionedGate {
    pub gate: Gate,
    pub target_locs: Vec<usize>,
    pub control_locs: Vec<usize>,
    pub control_configs: Vec<bool>,
}
```

- **`gate`**: The gate to apply.
- **`target_locs`**: Sites the gate matrix acts on (0-indexed).
- **`control_locs`**: Sites that control the gate activation.
- **`control_configs`**: Which state triggers each control (`true` = |1>).

## Builder API

### `put`

Places a gate on target locations with no controls.

```rust
use yao_rs::{put, Gate};

// H gate on qubit 0
let h = put(vec![0], Gate::H);

// SWAP on qubits 1 and 2
let swap = put(vec![1, 2], Gate::SWAP);
```

### `control`

Places a controlled gate. All controls are active-high, triggering on |1>.

```rust
use yao_rs::{control, Gate};

// CNOT: control on qubit 0, X on qubit 1
let cnot = control(vec![0], vec![1], Gate::X);

// Toffoli: controls on qubits 0,1, X on qubit 2
let toffoli = control(vec![0, 1], vec![2], Gate::X);
```

Note: All controls are active-high (trigger on |1>). The `control_configs` are automatically set to `vec![true; ctrl_locs.len()]`.

## Building a Circuit

```rust
use yao_rs::{Circuit, Gate, put, control};

let gates = vec![
    put(vec![0], Gate::H),
    control(vec![0], vec![1], Gate::X),
];
let circuit = Circuit::new(vec![2, 2], gates).unwrap();
```

`Circuit::new` validates all gates and returns `Result<Circuit, CircuitError>`.

## Validation Rules

The 6 validation rules checked by `Circuit::new`:

1. **control_configs length must match control_locs length** — Each control site needs a configuration.
2. **All locations must be in range** — Every loc in `target_locs` and `control_locs` must be < `dims.len()`.
3. **No overlap between target and control** — A site cannot be both a target and a control.
4. **Control sites must be qubits (d=2)** — Controlled gates only support qubit control sites.
5. **Named gate targets must be qubits** — Non-Custom gates require target sites with d=2.
6. **Gate matrix size must match target dimensions** — The gate's matrix dimension must equal the product of target site dimensions.

Example of a validation error:

```rust
use yao_rs::{Circuit, Gate, put};

// This fails: location 5 is out of range for a 2-qubit circuit
let result = Circuit::new(vec![2, 2], vec![put(vec![5], Gate::H)]);
assert!(result.is_err());
```

## Qudit Support

The `dims` vector specifies per-site dimensions. For qubits use 2, for qutrits use 3, etc.

```rust
// Mixed qubit-qutrit circuit
let dims = vec![2, 3, 2]; // qubit, qutrit, qubit
```

Custom gates can target non-qubit sites, but named gates (X, Y, Z, H, etc.) require d=2.
