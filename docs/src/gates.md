# Gates

Gates in yao-rs are represented by the `Gate` enum, covering standard qubit gates, parameterized rotations, and custom matrices.

```rust
pub enum Gate {
    X, Y, Z, H, S, T, SWAP,
    Phase(f64),
    Rx(f64), Ry(f64), Rz(f64),
    Custom { matrix: Array2<Complex64>, is_diagonal: bool },
}
```

## Named Qubit Gates

| Gate | Matrix | Diagonal |
|------|--------|----------|
| `X` | `[[0, 1], [1, 0]]` | No |
| `Y` | `[[0, -i], [i, 0]]` | No |
| `Z` | `[[1, 0], [0, -1]]` | Yes |
| `H` | `1/sqrt(2) [[1, 1], [1, -1]]` | No |
| `S` | `[[1, 0], [0, i]]` | Yes |
| `T` | `[[1, 0], [0, e^(i*pi/4)]]` | Yes |

## SWAP Gate

The 2-qubit `Gate::SWAP` acts on 2 sites. Its 4x4 matrix swaps the |01> and |10> basis states.

## Rotation Gates

- **Rx(theta):** `[[cos(theta/2), -i*sin(theta/2)], [-i*sin(theta/2), cos(theta/2)]]`
- **Ry(theta):** `[[cos(theta/2), -sin(theta/2)], [sin(theta/2), cos(theta/2)]]`
- **Rz(theta):** `[[e^(-i*theta/2), 0], [0, e^(i*theta/2)]]` -- diagonal

## Phase Gate

`Gate::Phase(theta)` represents `diag(1, e^(i*theta))`. This is equivalent to Yao.jl's `shift(theta)`.

Special cases:

- `Phase(pi)` = Z
- `Phase(pi/2)` = S
- `Phase(pi/4)` = T

The Phase gate is diagonal.

## Custom Gates

```rust
Gate::Custom {
    matrix: Array2<Complex64>,
    is_diagonal: bool,
}
```

Provide an arbitrary unitary matrix. The `is_diagonal` flag tells the tensor network exporter to use the diagonal optimization (shared legs instead of separate input/output legs).

The matrix dimension determines how many sites the gate acts on: a d^n x d^n matrix acts on n sites of dimension d.

## Diagonal Gate Optimization

In tensor networks, diagonal gates have one shared leg per site instead of separate input and output legs. This reduces the tensor rank and can improve contraction efficiency.

Gates that are diagonal: `Z`, `S`, `T`, `Phase(theta)`, `Rz(theta)`, and any `Custom` gate with `is_diagonal: true`.

## Getting the Matrix

```rust
let mat = Gate::H.matrix(2);  // d=2 for qubits
```

Named gates require `d=2` (will panic otherwise). Custom gates work with any dimension.
