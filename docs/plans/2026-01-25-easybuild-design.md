# EasyBuild Design (Issue #4)

Port all prebuilt circuits from Yao.jl's `EasyBuild` module (circuit builders only, no Hamiltonians).

## New Gate Variants

Add to `Gate` enum in `src/gate.rs`:

```rust
SqrtX,              // √X: single-qubit
SqrtY,              // √Y: single-qubit
SqrtW,              // √W = rot((X+Y)/√2, π/2): single-qubit, non-Clifford
ISWAP,              // iSWAP: two-qubit
FSim(f64, f64),     // FSim(θ, φ): two-qubit, Google's fermionic simulation gate
```

Also update `Custom` to require a label:

```rust
Custom {
    matrix: Array2<Complex64>,
    is_diagonal: bool,
    label: String,  // Required display name
}
```

### Gate Matrices

- `SqrtX`: `(1+i)/2 * [[1, -i], [-i, 1]]`
- `SqrtY`: `(1+i)/2 * [[1, -1], [1, 1]]`
- `SqrtW`: `rot((X+Y)/√2, π/2)` matrix
- `ISWAP`: permutes |01⟩↔|10⟩ with phase `i`, 4×4 matrix
- `FSim(θ,φ)`: `[[1,0,0,0],[0,cos θ,-i sin θ,0],[0,-i sin θ,cos θ,0],[0,0,0,e^{-iφ}]]`

## Module Structure

Single file: `src/easybuild.rs`, exported from `lib.rs`.

### Entanglement Layouts

```rust
pub fn pair_ring(n: usize) -> Vec<(usize, usize)>
pub fn pair_square(m: usize, n: usize, periodic: bool) -> Vec<(usize, usize)>
```

### Deterministic Circuit Builders

```rust
pub fn qft_circuit(n: usize) -> Circuit
pub fn phase_estimation_circuit(unitary: Gate, n_reg: usize, n_b: usize) -> Circuit
pub fn variational_circuit(n: usize, nlayer: usize, pairs: &[(usize, usize)]) -> Circuit
pub fn hadamard_test_circuit(unitary: Gate, phi: f64) -> Circuit
pub fn swap_test_circuit(nbit: usize, nstate: usize, phi: f64) -> Circuit
pub fn general_u2(theta1: f64, theta2: f64, theta3: f64) -> Vec<PositionedGate>
pub fn general_u4(params: &[f64; 15]) -> Vec<PositionedGate>
```

### Random Circuit Builders

```rust
pub fn rand_supremacy2d(nx: usize, ny: usize, depth: usize, rng: &mut impl Rng) -> Circuit
pub fn rand_google53(depth: usize, nbits: usize, rng: &mut impl Rng) -> Circuit
```

Dependency: `rand` crate.

## Circuit Builder Details

### `qft_circuit(n)`
Move existing example logic into easybuild. For each qubit i: apply H, then controlled-Phase(2π/2^(j-i+1)) from qubit j for j > i.

### `phase_estimation_circuit(unitary, n_reg, n_b)`
Total qubits = n_reg + n_b.
1. H on all register qubits (0..n_reg)
2. For i in 0..n_reg: controlled-U^(2^i) on target qubits, controlled by qubit i. U^(2^i) computed by repeated matrix squaring, stored as Custom gates.
3. Inverse QFT on register qubits (conjugate-transpose phases, reversed order)

### `variational_circuit(n, nlayer, pairs)`
Hardware-efficient ansatz, all angles initialized to zero:
1. First rotor layer: Rx(0), Rz(0) on each qubit (no leading Rz)
2. For each of nlayer middle layers: CNOT entangler on pairs, then Rz(0), Rx(0), Rz(0) on each qubit
3. Final rotor layer: Rz(0), Rx(0) on each qubit (no trailing Rz)

### `hadamard_test_circuit(unitary, φ)`
N+1 qubits (qubit 0 is ancilla):
1. H on qubit 0
2. Rz(φ) on qubit 0
3. Controlled-U on qubits 1..N, controlled by qubit 0
4. H on qubit 0

### `swap_test_circuit(nbit, nstate, φ)`
nstate×nbit + 1 qubits:
1. H on ancilla (qubit 0)
2. Rz(φ) on ancilla
3. Controlled-SWAP between consecutive state registers, controlled by ancilla
4. H on ancilla

### `general_u2(θ1, θ2, θ3)`
Returns [Rz(θ3), Ry(θ2), Rz(θ1)] on a single qubit (indices relative, caller embeds).

### `general_u4(params: &[f64; 15])`
15-parameter SU(4) decomposition on 2 qubits:
- general_u2(params[0..3]) on qubit 0
- general_u2(params[3..6]) on qubit 1
- CNOT(1→0)
- Rz(params[6]) on qubit 0, Ry(params[7]) on qubit 1
- CNOT(0→1)
- Ry(params[8]) on qubit 1
- CNOT(1→0)
- general_u2(params[9..12]) on qubit 0
- general_u2(params[12..15]) on qubit 1

### `rand_supremacy2d(nx, ny, depth, rng)`
nx×ny qubit grid:
1. H on all qubits
2. Each layer: CZ entangler on cycling pattern (pair_supremacy layout), random gate from {T, SqrtX, SqrtY} on non-entangled qubits (no repeat, T first)
3. Final layer: H on all qubits

Helper: `pair_supremacy(nx, ny)` — 8 entanglement patterns (checkerboard CZ layout).

### `rand_google53(depth, nbits, rng)`
Sycamore circuit:
1. Lattice53 struct: 5×12 grid with holes matching Google's chip topology
2. Cycles through patterns ['A','B','C','D','C','D','A','B']
3. Each layer: random gate from {SqrtX, SqrtY, SqrtW} on each qubit, then FSim(π/2, π/6) entanglers

Helper: `Lattice53` struct with pattern-to-pairs mapping.

## Testing

1. **Gate matrices**: unitarity, known eigenvalues, FSim(π/2, 0) ≈ iSWAP
2. **QFT**: uniform superposition from |0⟩, phase progression from |1⟩
3. **QPE**: recover eigenphases of diagonal unitary
4. **Hadamard test**: recover ⟨ψ|U|ψ⟩ from ancilla
5. **Swap test**: overlap=1 for identical states, 0 for orthogonal
6. **general_u2**: unitarity, Euler decomposition correctness
7. **general_u4**: SU(4) coverage, 3 CNOTs
8. **Variational**: correct parameter count, normalized output
9. **Random circuits**: correct qubit count, depth, normalized output (seeded RNG)
10. **Entanglement layouts**: pair_ring(4) = [(0,1),(1,2),(2,3),(3,0)], pair_square checks
