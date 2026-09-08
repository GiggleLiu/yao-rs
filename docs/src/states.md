# Simulation & measurement

Simulation applies a circuit to an initial state. You can then inspect exact
probabilities, sample measurements, or compute an observable's expectation value.

## Run from the CLI

`yao run` starts from the all-zero state and simulates a circuit before sampling:

```bash
yao run circuit.json --shots 1024
```

To reuse the resulting state across several queries, save it once:

```bash
yao simulate circuit.json --output state.bin
yao probs state.bin
yao measure state.bin --shots 1024
yao expect state.bin --op "Z(0)Z(1)"
```

`probs` computes the distribution without sampling noise. `measure` draws random
outcomes. `expect` evaluates an [operator expression](conventions.md#operator-syntax).
Add `--locs 0,1` to `probs` or `measure` to select a subset of qubits.

To start from a saved state, pass `--input state.bin` to `yao simulate`.
See the [command reference](cli.md) for all options.

## Simulate from Rust

`ArrayReg` stores a qubit state as complex amplitudes. `apply` returns a new
register; `apply_inplace` updates an existing one.

```rust
use yao_rs::{ArrayReg, Circuit, Gate, apply, probs, put};

let circuit = Circuit::qubits(1, vec![put(vec![0], Gate::H)]).unwrap();
let initial = ArrayReg::zero_state(1);
let result = apply(&circuit, &initial);
let probabilities = probs(&result, None);
assert!((probabilities[0] - 0.5).abs() < 1e-12);
```

Choose an initial state to match your experiment:

| Constructor | Initial state |
|---|---|
| `ArrayReg::zero_state(n)` | All qubits in 0 |
| `ArrayReg::uniform_state(n)` | Equal superposition of all basis states |
| `ArrayReg::ghz_state(n)` | Equal superposition of all-zero and all-one states |
| `ArrayReg::product_state(bitstr)` | A basis state described by a `BitStr<N>` |
| `ArrayReg::from_vec(n, amplitudes)` | Your own vector of exactly `2^n` amplitudes |
| `ArrayReg::rand_state(n, rng)` | A normalized random state |

`from_vec` checks the vector length; supply normalized amplitudes or call
`normalize()` before interpreting the state as probabilities.
Use `state_vec()` to inspect amplitudes and `nqubits()` for the register size.
The shared [bit ordering convention](conventions.md#bit-ordering) defines which
basis state each array index represents.

## Reuse a fixed circuit

For repeated runs with the same parameters, prepare fused gates once:

```rust
let prepared = circuit.fused(2).unwrap();
let result = apply(&prepared, &initial);
```

Fusion combines consecutive gates into matrices on at most two qubits here.
Compare it with ordinary execution for your workload; larger blocks can be
slower. Parameter values are fixed in the prepared circuit, so prepare again
after changing them and use the original circuit for differentiation.

## Measure and post-process

The Rust [`measure_with_postprocess`](api/yao_rs/measure/fn.measure_with_postprocess.html)
function supports sampling without changing the register, resetting measured
qubits, or removing them. Select the behavior with `PostProcess`.
Use [`expect_arrayreg`](api/yao_rs/expect/fn.expect_arrayreg.html) with an
`OperatorPolynomial` to compute an expectation value directly.

## Mixed states and noise

Use [`DensityMatrix`](api/yao_rs/density_matrix/struct.DensityMatrix.html) for
mixed states and noise channels. The CLI automatically uses a density matrix when a circuit contains noise
channels or its input is already a density matrix. `probs`, `measure`, and
`expect` accept both saved representations.

For streaming observable estimates with uncertainty, use [noisy trajectories](trajectories.md).
Other guides cover [Hamiltonian evolution](hamiltonian.md),
[differentiable simulation](differentiation.md), and [CUDA execution](cuda.md).

State-vector simulation is qubit-only and stores `2^n` complex amplitudes.
For larger structured circuits or higher-dimensional sites, consider
[tensor networks](tensor-networks.md).
