# Noisy trajectories

A density matrix gives exact channel evolution, but stores `4^n` complex values.
A trajectory samples a pure-state realization using `2^n` values. Averaging
observable values across independent trajectories estimates the mixed-state
expectation and reports Monte Carlo uncertainty.

```rust
use yao_rs::{ArrayReg, Circuit, NoiseChannel, Op, OperatorPolynomial, channel};
use yao_rs::trajectories::{TrajectoryCircuit, TrajectoryOptions};

let circuit = Circuit::qubits(1, vec![
    channel(vec![0], NoiseChannel::BitFlip { p: 0.25 }),
])?;
let simulation = TrajectoryCircuit::new(circuit)?;
let observable = OperatorPolynomial::single(0, Op::Z, 1.0.into());
let result = simulation.expectation(
    &ArrayReg::zero_state(1), &observable,
    TrajectoryOptions { trajectories: 4096, seed: 19, threads: 1 },
)?;
// Estimates 0.5. A standard error describes sampling uncertainty.
println!("{} +/- {:?}", result.mean, result.standard_error);
# Ok::<(), Box<dyn std::error::Error>>(())
```

`TrajectoryCircuit` validates the circuit and prepares local Kraus matrices once.
Input states must be normalized, finite qubit registers. Channels must satisfy
`sum K†K = I` to `1e-12` per local entry; custom gates must be unitary. Duplicate
or out-of-range locations, invalid probabilities and malformed matrices return
errors. `NoiseChannel::try_kraus_operators()` exposes the same fallible channel
check for other callers.

For a Kraus channel, the simulator computes each `p_k = ||K_k psi||²`, samples a
branch with Rand's `WeightedIndex`, then normalizes `K_k psi / sqrt(p_k)`. One
scratch state is reused for probabilities and observable evaluation. Built-in
multi-qubit depolarization samples Pauli words directly, avoiding an exponentially
large list of dense Kraus matrices. Arbitrary custom channels still store their
local Kraus operators; a dense channel spanning the entire register can be costly.

`sample(input, seed, trajectory_id)` returns one explicit realization. It does
not represent the exact mixed state. `expectation` streams polynomial values;
it retains neither all final states nor a sample history. Non-Hermitian words
and complex coefficients are supported. The result contains:

- `mean`: complex sample mean.
- `sample_variance`: unbiased variances of the real and imaginary components.
- `standard_error`: componentwise square roots of variance divided by trajectory count.
- `covariance`: sample covariance between real and imaginary components.
- `trajectories`, `seed`, and `threads`: execution settings.

Uncertainty fields are `None` (`null` in JSON) with one sample. A standard error
is an estimate, not a guaranteed bound; rare events may be absent from a small
sample and give a misleadingly small estimated variance. Increase samples and
compare independent seeds when estimating rare events. Sampling error generally
scales as `1/sqrt(N)`, so halving it requires about four times the work. These
fields exclude model error and floating-point error.

## CLI

```bash
yao run noisy.json --trajectories 4096 --op 'Z(0)' --seed 19
cargo install --path yao-cli --features parallel
yao run noisy.json --trajectories 4096 --op 'Z(0)' --seed 19 --threads 4
```

`--trajectories` requires `--op` and conflicts with measurement `--shots`.
The output has `"mode": "trajectories"` and a `statistics` object; it is not a
serialized state. Complex values are JSON `[real, imaginary]` pairs. Omitting
`--seed` chooses a random seed and records it in the result. `--input` accepts
a normalized pure-state file; exact density input belongs to the exact simulator.
Existing `run --op` without `--trajectories` uses exact simulation.

## Reproducibility, concurrency and backend

ChaCha8 streams use the master seed and trajectory ID, following the
[Rand parallel-stream guidance](https://rust-random.github.io/book/guide-parallel.html).
Workers consume independent streams and scalar results are reduced in trajectory
order. Changing thread count preserves the values and statistics on the same
version/platform. Reproducibility across library versions or different floating
point implementations is not promised.

The optional `parallel` feature enables a dedicated Rayon pool. The requested
thread count bounds the active workers; memory is `O(threads * 2^n)` plus circuit,
local Kraus storage and a bounded scalar batch. There is no slice/trajectory
history proportional to the number of samples. CPU trajectories use the native
state-vector kernels, consistent with the existing specialized simulation path.
Tenferro provides the separate tensor-network backend. GPU trajectories and
stochastic differentiation are not implemented.

## Thermal-relaxation correction

Thermal relaxation now obeys population decay `exp(-time/T1)` and coherence decay
`exp(-time/T2)`, including positive infinite time constants, zero time, and the
boundary `T2 = 2*T1`. Time must be finite and nonnegative. The previous
Julia-derived conversion omitted a survival factor in the phase probability;
it gave the wrong coherence and could produce NaNs for valid long durations.
The pinned Yao fixture contains the same error, so analytic channel-map tests
replace equality with those Kraus entries. This correction changes results for
`ThermalRelaxation`; other damping parameters retain their existing definitions.
The analytic map is also used by
[Qiskit Aer's thermal channel](https://github.com/Qiskit/qiskit-aer/blob/0.17.1/qiskit_aer/noise/errors/standard_errors.py).


## CPU measurements

The [M4 report](https://github.com/GiggleLiu/yao-rs/blob/main/benchmarks/results/mac-trajectories-cpu-2026-09-07/report.md)
compares achieved sampling error, exact native/Yao/tenferro expectations and
isolated memory. It shows constant state-buffer memory as sample count grows,
with useful parallelism on larger cases and overhead on small ones. High-sample
trajectory estimates can take longer than exact simulation when a density
matrix still fits comfortably.
