# Benchmark selection

Surveyed 8 September 2026. The target is classical simulator performance and
repeatable regression detection on the same physical device.

| Candidate | Strength | Decision |
|---|---|---|
| [QASMBench](https://github.com/pnnl/QASMBench) | Fixed OpenQASM 2 application circuits: arithmetic, chemistry, optimization, and phase estimation | Include a pinned subset spanning gate mixes and widths. Preserve original inputs and record every normalization. |
| [MQT Bench](https://github.com/munich-quantum-toolkit/bench) | Scalable algorithms at algorithmic, target-independent, native-gate, and mapped levels | Include seeded, target-independent circuits. Freeze generated files and generator versions so compiler changes cannot silently change workloads. |
| [NWQBench](https://github.com/pnnl/nwqbench) | Scalable generators in several framework representations | Reserve for expansion; overlaps the initial QASMBench/MQT algorithm coverage. |
| [SupermarQ](https://arxiv.org/abs/2202.11045) | Application-level hardware benchmarks and fidelity metrics | Useful workload inspiration; hardware scores are not simulator throughput scores. Exclude dynamic/error-correction cases from the unitary track. |
| [Benchpress](https://github.com/Qiskit/benchpress) | Quantum SDK circuit creation, manipulation, and compilation | Separate future front-end track. Do not mix transpiler speed with state evolution. |
| [Einsum Benchmark](https://benchmark.einsum.org/) | 168 tensor expressions, including 32 quantum-computing instances, with reference contraction paths | Candidate for broader contraction-engine qualification. Keep the initial tensor track tied to yao-rs circuit exports; a public-network extension needs bounded intermediate memory, matched data types, and full-output validation. |
| [qsim](https://github.com/quantumlib/qsim) and [Qulacs](https://github.com/qulacs/qulacs) benchmark workloads | Random circuits and optimized state-vector simulation baselines | Include seeded random layers alongside structured applications; report fusion/preparation policies explicitly. |
| Existing yao-rs shared fixtures | Gradients, density matrices, trajectories, tensor planning/contraction, and Hamiltonian evolution | Retain as feature-specific tracks that OpenQASM circuit collections do not cover. |

## Curated coverage

Choose cases before timing. Keep both favorable and unfavorable results.

| Dataset | Frozen application selection | Cases |
|---|---|---:|
| QASMBench | Adder (10 qubits), QAOA (6), QPE (9), Ising (10), UCCSD (8), QFT (18) | 6 |
| MQT Bench | GHZ, QFT, QAOA, and random circuits, each at 8, 12, and 16 qubits | 12 |
| MQT Bench | Grover at 8 qubits; bound its exponentially growing repetition count | 1 |

The default regression profile includes 18 application cases; the full profile
also includes the 18-qubit QASMBench QFT. Feature fixtures add the following:

- Kernel diagnostics: low/high/nonadjacent targets, active-low controls,
  diagonal rotations, two-qubit matrices, and increasing state sizes.
- Differentiation: value and parameter gradient; custom scalar loss with
  parameter and input-state gradients. Fix parameter bindings and output work.
- Noise: exact density matrices separately from trajectories. Compare sampling
  at fixed statistical accuracy as well as fixed trajectory count.
- Tensors: export, planning, and prepared contraction as distinct measurements;
  fixed-tree execution separately from an optimized end-to-end workflow.
- Hamiltonians: product formulas and adaptive Krylov action. Require convergence
  and measured error before comparing time.
- CUDA: a separate same-GPU track, with device-resident execution and host/device
  transfers reported separately. A CPU run cannot qualify GPU performance.

## Comparison rules

Use complex128, identical serialized circuits and initial states, aligned qubit
ordering, equal requested outputs, fixed seeds, and explicit thread budgets.
Validate complete outputs before timing. Exclude compilation from warmed
execution but preserve preparation costs as a separate metric. Record raw
samples, independent-process medians, source hashes, toolchains, dependency
locks, device identity, and execution boundaries. Run implementations serially.

The initial comparison pool is Yao.jl and Qulacs for native CPU simulation,
Yao.jl for reversible circuit differentiation and noisy density evolution,
and Yao.jl for Hamiltonian evolution. Tensor cases compare tenferro and omeinsum
phases on the exported networks; these are backend regression measurements.
Broader tensor qualification should add a validated
[cotengra contraction adapter](https://cotengra.readthedocs.io/en/stable/autoapi/cotengra/index.html)
with preparation and path-search costs reported separately.
Passing this pool means competitive performance against the named, tested
baselines; it does not establish an unrestricted state-of-the-art claim.

qsim is a useful strong simulator reference, but its documented
[single-precision execution](https://github.com/quantumlib/qsim) and
[complex64 Python interface](https://github.com/quantumlib/qsim/blob/main/qsimcirq/qsim_simulator.py)
do not match this suite's complex128 contract. A precision-matched adapter or a
separately versioned complex64 track is needed before including its timings.

A release gate must reject missing cases, failed correctness, incompatible
precision/device/thread settings, and statistically unresolved regressions.
Do not hide individual failures behind a geometric mean. Repeated measurements
on a dedicated host are required for a performance decision; shared CI checks
harness correctness and never substitutes for that measurement.

Use calibrated inner batches for short operations, with sample and evaluation
counts retained. The [BenchmarkTools manual](https://juliaci.github.io/BenchmarkTools.jl/stable/manual/)
explains why timing individual evaluations can be unreliable and distinguishes
the sample limit from a time budget. Confidence intervals here resample whole
process medians, rather than treating inner evaluations as independent runs.
