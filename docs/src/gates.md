# Gate reference

Use these gates with the [circuit builders](circuits.md) or their names in
[JSON](conventions.md#gate-elements). Angles are in radians.

## Gate names

| Rust gate / JSON name | Sites | Parameters | Operation |
|---|---|---|---|
| `X`, `Y`, `Z` | 1 | — | Pauli gates |
| `H` | 1 | — | Hadamard |
| `S` | 1 | — | Phase of π/2 on state 1 |
| `T` | 1 | — | Phase of π/4 on state 1 |
| `SqrtX`, `SqrtY` | 1 | — | Square roots of X and Y |
| `SqrtW` | 1 | — | π/2 rotation about the (X + Y)/√2 axis |
| `Rx`, `Ry`, `Rz` | 1 | `theta` | Rotation about X, Y, or Z |
| `Phase` | 1 | `theta` | Diagonal matrix `diag(1, exp(i*theta))` |
| `SWAP` | 2 | — | Exchange two sites |
| `ISWAP` | 2 | — | Exchange states 01 and 10 with phase i |
| `FSim` | 2 | `theta, phi` | Fermionic simulation gate |
| `Custom` | Matrix-dependent | — | User-supplied matrix |

In Rust, write `Gate::Ry(theta)` or `Gate::FSim(theta, phi)`.
In JSON, use `"params": [theta]` or `"params": [theta, phi]` alongside the gate name.
Named gates require qubit targets.

## Inspect a matrix

```rust
use yao_rs::Gate;

let matrix = Gate::H.matrix();
assert_eq!(matrix.shape(), &[2, 2]);
```

The [generated gate API](api/yao_rs/gate/enum.Gate.html) documents matrix and
parameter access.

## Custom gates

Supply a square complex matrix, a display label, and an accurate diagonal flag:

```rust
use yao_rs::Gate;

let gate = Gate::Custom {
    matrix: Gate::H.matrix(),
    is_diagonal: false,
    label: "My H".into(),
};
```

A gate's matrix size must equal the product of its target dimensions. For
example, a gate on two qutrits needs a 9 × 9 matrix. Supply a unitary matrix
for unitary evolution; the circuit constructor does not check unitarity.

Set `is_diagonal: true` only if all off-diagonal entries are zero. This flag
enables specialized simulation and [tensor export](tensor-networks.md#control-computation-cost)
paths. The named diagonal gates are `Z`, `S`, `T`, `Phase`, and `Rz`.
