# Quantum Registers

An `ArrayReg` is a qubit-only register backed by a dense state vector (`Vec<Complex64>`). It is the primary type for state-vector simulation in yao-rs.

## Structure

`ArrayReg` has two fields:

- **`state`** (public): a `Vec<Complex64>` of length `2^n` holding the amplitudes.
- **`nbits`** (private): the number of qubits, accessible via `nqubits()`.

## Creating Registers

### Zero State

`zero_state(n)` creates the computational basis state |00...0> on `n` qubits.

```rust
use yao_rs::ArrayReg;

let reg = ArrayReg::zero_state(3);
assert_eq!(reg.nqubits(), 3);
assert_eq!(reg.state.len(), 8); // 2^3
assert_eq!(reg.state[0].re, 1.0); // |000> has amplitude 1
```

### Product State

`product_state` creates a computational basis state from a `BitStr<N>`. The const generic `N` determines the number of qubits.

```rust
use yao_rs::ArrayReg;
use bitbasis::BitStr;

// |01> on 2 qubits: bit string value 1 means qubit 1 is |1>
let reg = ArrayReg::product_state(BitStr::<2>::new(0b01));
assert_eq!(reg.state[1].re, 1.0); // index 1 = |01>

// |10> on 2 qubits: bit string value 2 means qubit 0 is |1>
let reg = ArrayReg::product_state(BitStr::<2>::new(0b10));
assert_eq!(reg.state[2].re, 1.0); // index 2 = |10>
```

### Uniform State

`uniform_state(n)` creates the equal superposition state where every basis state has amplitude `1/sqrt(2^n)`.

```rust
use yao_rs::ArrayReg;

let reg = ArrayReg::uniform_state(2);
// All 4 amplitudes equal 1/sqrt(4) = 0.5
for amp in reg.state_vec() {
    assert!((amp.re - 0.5).abs() < 1e-12);
}
```

### GHZ State

`ghz_state(n)` creates the Greenberger-Horne-Zeilinger state `(|00...0> + |11...1>) / sqrt(2)`.

```rust
use yao_rs::ArrayReg;

let reg = ArrayReg::ghz_state(3);
let amp = 1.0 / 2.0_f64.sqrt();
assert!((reg.state[0].re - amp).abs() < 1e-12);  // |000>
assert!((reg.state[7].re - amp).abs() < 1e-12);  // |111>
```

### From a Raw Vector

`from_vec(nbits, data)` wraps an existing amplitude vector. The vector length must be exactly `2^nbits`.

```rust
use yao_rs::ArrayReg;
use num_complex::Complex64;

let amps = vec![
    Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
    Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
];
let reg = ArrayReg::from_vec(1, amps);
assert_eq!(reg.nqubits(), 1);
```

### Random State

`rand_state(nbits, rng)` generates a normalized state with random complex amplitudes.

```rust
use yao_rs::ArrayReg;
use rand::rngs::StdRng;
use rand::SeedableRng;

let mut rng = StdRng::seed_from_u64(42);
let reg = ArrayReg::rand_state(3, &mut rng);
assert!((reg.norm() - 1.0).abs() < 1e-12); // always normalized
```

## Bit Ordering

yao-rs uses **big-endian** ordering: qubit 0 is the most significant bit. For a 2-qubit register the basis states map to vector indices as follows:

| State   | Index |
|---------|-------|
| \|00\>  | 0     |
| \|01\>  | 1     |
| \|10\>  | 2     |
| \|11\>  | 3     |

So applying an X gate to qubit 0 on state |00> produces |10> (index 2), not |01>.

## Utility Methods

| Method | Description |
|--------|-------------|
| `nqubits()` | Number of qubits |
| `state_vec()` | Borrow the amplitude slice |
| `state_vec_mut()` | Mutably borrow the amplitude slice |
| `norm()` | L2 norm of the state vector |
| `normalize()` | Normalize the state in place |
| `fidelity(&other)` | Squared overlap `|<self|other>|^2` between two registers |

## Applying Circuits

Use `apply` to evolve a register through a circuit, returning a new `ArrayReg`. Use `apply_inplace` to modify a register in place.

```rust
use yao_rs::{ArrayReg, Circuit, Gate, put, apply, apply_inplace};

// Build a 2-qubit circuit that flips qubit 0
let circuit = Circuit::new(vec![2, 2], vec![put(vec![0], Gate::X)]).unwrap();

// apply returns a new register
let reg = ArrayReg::zero_state(2);
let result = apply(&circuit, &reg);
// X on qubit 0: |00> -> |10> (index 2)
assert_eq!(result.state[2].re, 1.0);

// apply_inplace modifies the register directly
let mut reg = ArrayReg::zero_state(2);
apply_inplace(&circuit, &mut reg);
assert_eq!(reg.state[2].re, 1.0);
```

The simulation backend operates directly on the state vector using per-gate instruction kernels -- it never constructs the full `2^n x 2^n` unitary matrix.

## Qudit Support

`ArrayReg` is qubit-only (every site has dimension 2). For circuits with non-qubit dimensions (qutrits, etc.), yao-rs supports construction and tensor-network export via `circuit_to_einsum`, but direct state-vector simulation with `apply` requires all dimensions to be 2.
