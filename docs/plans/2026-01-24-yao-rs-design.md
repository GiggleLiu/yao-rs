# yao-rs Design

A Rust port of [Yao.jl](https://github.com/QuantumBFS/Yao.jl) focused on circuit description and tensor network export via [omeco](https://crates.io/crates/omeco).

## Goals (from issue #1)

1. Circuit description + einsum export (primary)
2. Generic `apply!` for testing correctness (secondary)
3. Port relevant tests from Yao.jl
4. mdBook documentation (later)
5. PyTorch tensor contraction (later)
6. Manifold optimization on unitary manifolds (later)

## Design Decisions

- **Circuit model**: Simplified — a circuit is a list of `PositionedGate`s. No separate PutBlock/KronBlock types; put is a PositionedGate with empty controls.
- **Gate representation**: Enum with qubit-only named variants + `Custom` fallback for arbitrary qudits.
- **Parameters**: Rotation angles are `f64`, matrices are `Complex<f64>`.
- **Qudit support**: Per-site dimensions. Each site can have a different dimension.
- **Control**: Qubit-only (d=2). Control is forbidden for sites with d!=2.
- **Diagonal gates**: `Custom` variant has `is_diagonal` flag, which changes tensor network structure (no index doubling).
- **omeco**: Used as a crates.io dependency for contraction order optimization.

## Data Structures

```rust
use num_complex::Complex64;
use ndarray::{Array1, Array2, ArrayD};

/// Quantum gate representation
pub enum Gate {
    // Qubit-only (d=2) named gates
    X, Y, Z, H, S, T,
    SWAP,
    Rx(f64), Ry(f64), Rz(f64),
    // General qudit gate
    Custom {
        matrix: Array2<Complex64>,
        is_diagonal: bool,
    },
}

/// A gate placed at specific sites with optional controls
pub struct PositionedGate {
    pub gate: Gate,
    pub target_locs: Vec<usize>,
    pub control_locs: Vec<usize>,      // must have d=2
    pub control_configs: Vec<bool>,    // true=|1>, false=|0>
}

/// A quantum circuit: sequence of positioned gates on a qudit register
pub struct Circuit {
    pub dims: Vec<usize>,              // dimension of each site
    pub gates: Vec<PositionedGate>,
}

/// Quantum state as a dense vector
pub struct State {
    pub dims: Vec<usize>,
    pub data: Array1<Complex64>,
}

/// Result of converting a circuit to a tensor network
pub struct TensorNetwork {
    pub code: EinCode<usize>,
    pub tensors: Vec<ArrayD<Complex64>>,
}
```

## Validation Rules

- Named gate variants error if any target site has d!=2
- `control_locs` must reference sites with d=2
- `target_locs` and `control_locs` must not overlap
- Gate matrix size must match product of target site dimensions

## Einsum Export Algorithm

1. Assign an initial label to each site (input state indices)
2. For each `PositionedGate` in sequence:
   - **Non-diagonal gate/control sites**: allocate new output labels; tensor has legs `[out_0, out_1, ..., in_0, in_1, ...]`
   - **Diagonal gate sites**: no new labels; tensor has legs `[label_0, label_1, ...]` (shared input/output)
3. Final labels become the output indices of the EinCode
4. Collect all index lists -> `EinCode`, all gate matrices (reshaped to tensors) -> `tensors`

### Example: CNOT on 2-qubit circuit

```
Sites: [d=2, d=2]
Initial labels: [0, 1]

Gate: PositionedGate { X, target=[1], control=[0], config=[true] }
  -> control site 0: new label 2 (legs: [2, 0])
  -> target site 1: new label 3 (legs: [3, 1])
  -> tensor rank 4: [2, 3, 0, 1]

Output labels: [2, 3]
EinCode: ixs=[[2,3,0,1]], iy=[2,3]
```

## Generic Apply (for testing)

Unoptimized implementation: for each gate, build the full-space matrix (expand with identity on uninvolved sites) and multiply the state vector.

## Testing Strategy

Core correctness test: contracting the tensor network must equal applying the circuit directly.

Coverage targets:
- Each named gate variant produces correct matrix
- Single-gate circuits (no controls)
- Controlled gates (CNOT, Toffoli)
- Multi-gate circuits
- Qudit circuits (d>2) with Custom gates
- Diagonal vs non-diagonal Custom gates produce different einsum structures
- Port relevant tests from Yao.jl

## Project Structure

```
yao-rs/
├── Cargo.toml
├── src/
│   ├── lib.rs              # Public API exports
│   ├── gate.rs             # Gate enum + matrix generation
│   ├── circuit.rs          # PositionedGate, Circuit
│   ├── einsum.rs           # circuit_to_einsum -> TensorNetwork (EinCode)
│   ├── tensors.rs          # Gate-to-tensor conversion
│   ├── state.rs            # State struct
│   └── apply.rs            # Generic apply for testing
├── tests/
│   ├── test_gates.rs       # Gate matrix correctness
│   ├── test_einsum.rs      # Einsum export correctness
│   └── test_apply.rs       # apply vs einsum contraction consistency
└── docs/                   # mdBook (later)
```

## Dependencies

- `omeco` — einsum contraction order optimization
- `num-complex` — Complex64
- `ndarray` — matrices and tensors
