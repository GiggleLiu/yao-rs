# States

A `State` represents a quantum state vector over a register of qudits. Each qudit has a local dimension (2 for qubits, 3 for qutrits, etc.), and the full state vector lives in the tensor product of the individual Hilbert spaces.

## State Vector Representation

The `State` struct has two fields:

- **`dims`**: a vector of local dimensions, one per site. For example, `[2, 2, 2]` describes three qubits.
- **`data`**: a complex amplitude vector of length `dims[0] * dims[1] * ... * dims[n-1]`.

States are stored as flat vectors in row-major order. The amplitude of a computational basis state `|i_0, i_1, ..., i_{n-1}>` is found at a single flat index computed from the local indices and dimensions.

## Creating States

### Zero State

`zero_state` creates the all-zeros computational basis state `|0,0,...,0>`, which has amplitude 1 at index 0 and amplitude 0 everywhere else.

```rust
use yao_rs::State;

// 3-qubit zero state |000>
let state = State::zero_state(&[2, 2, 2]);
assert_eq!(state.total_dim(), 8);
assert_eq!(state.data[0].re, 1.0); // |000> amplitude = 1
```

### Product State

`product_state` creates a computational basis state `|i_0, i_1, ..., i_{n-1}>` where each qudit is in a definite level.

```rust
use yao_rs::State;

// |01> on 2 qubits
let state = State::product_state(&[2, 2], &[0, 1]);
// Index = 0*2 + 1 = 1, so data[1] = 1
assert_eq!(state.data[1].re, 1.0);
```

## Row-Major Index Ordering

The flat index for state `|i_0, i_1, ..., i_{n-1}>` is computed as:

```
index = i_0 * (d_1 * d_2 * ... * d_{n-1})
      + i_1 * (d_2 * ... * d_{n-1})
      + ...
      + i_{n-1}
```

Example for 2 qubits (d=2 each):

| State   | Index |
|---------|-------|
| \|00\>  | 0     |
| \|01\>  | 1     |
| \|10\>  | 2     |
| \|11\>  | 3     |

## Multi-Qudit Examples

The state representation generalizes beyond qubits. Any combination of local dimensions is supported.

```rust
use yao_rs::State;

// Qutrit (d=3): |2>
let state = State::product_state(&[3], &[2]);
assert_eq!(state.data[2].re, 1.0);

// Qubit + qutrit: |1,2>
// Index = 1*3 + 2 = 5
let state = State::product_state(&[2, 3], &[1, 2]);
assert_eq!(state.total_dim(), 6);
assert_eq!(state.data[5].re, 1.0);
```

## Applying Circuits

Once you have a state, you can evolve it by applying a circuit with the `apply` function.

```rust
use yao_rs::{State, Circuit, Gate, put, apply};

let circuit = Circuit::new(vec![2, 2], vec![put(vec![0], Gate::X)]).unwrap();
let state = State::zero_state(&[2, 2]);
let result = apply(&circuit, &state);
// X flips qubit 0: |00> -> |10>
assert_eq!(result.data[2].re, 1.0);
```

The `apply` function builds full matrices and performs state-vector simulation. The result preserves the norm.
