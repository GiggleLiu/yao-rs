# Yao Examples And Algorithm Zoo Coverage

**Date:** 2026-04-19
**Goal:** Cover the examples from Yao.jl documentation and `/tmp/QuAlgorithmZoo.jl` in yao-rs, prioritizing examples that can be run through the `yao` CLI today and recording unsupported algorithm families as actionable GitHub issues.

## Context

Current `yao example` support is limited to `bell`, `ghz`, and `qft`. The library already has `easybuild` support for QFT, phase estimation, hadamard test, swap test, hardware-efficient variational circuits, and random supremacy circuits. The CLI can inspect, simulate, measure, compute probabilities, compute Pauli expectations, render SVG, export OpenQASM, export tensor networks, optimize contraction order, and contract tensor networks when built with `omeinsum`.

The source material to cover is:

- Yao documentation examples: GHZ, QFT, phase estimation, and QCBM.
- `/tmp/QuAlgorithmZoo.jl`: QFT, phase estimation, hadamard test, swap test, VQE, QSVD, ground-state solvers, HHL, QAOA, QCBM, QuGAN, Shor, Grover, gate learning, QMPS, Mermin Magic Square, Bernstein-Vazirani, and related port examples.

## Coverage Strategy

Use a tiered plan rather than a flat port. Each source example must be accounted for, but only examples that map cleanly to circuit JSON and existing CLI workflows are implemented in the first waves.

### Tier 1: Direct CLI Examples

Tier 1 examples can be generated as circuit JSON with existing code and demonstrated through current CLI commands.

| Source | Example | Planned coverage |
|--------|---------|------------------|
| Yao docs | GHZ | Keep `yao example ghz`; expand docs with `inspect`, `run`, `probs`, `visualize`, and tensor-network pipeline examples. |
| Yao docs / QuAlgorithmZoo README | QFT | Keep `yao example qft`; expand docs with simulation, SVG, OpenQASM, and tensor-network workflows. |
| QuAlgorithmZoo README | Existing easybuild catalog | Add docs that connect `qft`, phase estimation, hadamard test, and swap test to the CLI example catalog. |

Tier 1 should not add new algorithm semantics. It is primarily CLI exposure, docs, tests, and naming cleanup.

### Tier 2: Small Builders Then CLI Examples

Tier 2 examples need a small reusable builder or a thin CLI parameter layer, but do not require optimizers, autodiff, external chemistry data, or large arithmetic subroutines.

| Source | Example | Planned coverage |
|--------|---------|------------------|
| Yao docs | Phase estimation | Expose `easybuild::phase_estimation_circuit` through `yao example phase-estimation` with small unitary presets such as `z`, `phase`, or `t`. |
| QuAlgorithmZoo README | Hadamard test | Expose `easybuild::hadamard_test_circuit` through `yao example hadamard-test` with a small unitary preset and phase option. |
| QuAlgorithmZoo README | Swap test | Expose `easybuild::swap_test_circuit` through `yao example swap-test` with `--nqubits-per-state`, `--nstates`, and optional phase. |
| QuAlgorithmZoo | Bernstein-Vazirani | Add a builder for a phase-oracle Bernstein-Vazirani circuit from a secret bit string, then expose `yao example bernstein-vazirani --secret 10101`. |
| QuAlgorithmZoo | Grover | Add a small marked-basis-state Grover circuit, such as `yao example grover --nqubits 4 --marked 15 --iterations auto`. Defer inference-oracle and variational-generator versions to Tier 3. |
| QuAlgorithmZoo | QAOA | Add a static MaxCut ansatz for a small built-in graph or explicit edge list. Expose circuit generation only; defer parameter optimization. |
| Yao docs / QuAlgorithmZoo | QCBM | Add static QCBM-style ansatz generation using existing variational-circuit structure; defer MMD training and parameter updates. |
| QuAlgorithmZoo | Mermin Magic Square | Read the Julia source carefully and add a fixed circuit/demo if it can be represented by current gates and measurement/probability workflows. |

Tier 2 builders should live in `src/easybuild.rs` only when reusable by Rust callers. Pure CLI fixture circuits can stay in `yao-cli/src/commands/example.rs` if they do not deserve public library API.

### Tier 3: Deferred Issue-Tracked Examples

Tier 3 examples require missing capabilities. They must not disappear from the plan. The implementation plan must create GitHub issues for these deferred groups, with a checklist of affected source examples and acceptance criteria.

Default issue shape:

```text
Title: Track deferred QuAlgorithmZoo example coverage: <capability group>
Labels: examples, algorithm-zoo, deferred
Body:
- Source examples covered
- Missing yao-rs capability
- Minimal implementation target
- CLI surface expected after capability exists
- Validation strategy against Yao.jl or known analytical behavior
```

Deferred issue groups:

| Capability group | Source examples | Missing capability |
|------------------|-----------------|--------------------|
| Optimization and training loops | VQE, VQE OpenFermion, QCBM training, QuGAN, GateLearning, QMPS | Parameter storage/dispatch, optimizers, gradient or parameter-shift workflows, benchmarkable training loops. |
| Hamiltonian and time evolution algorithms | GroundStateSolvers, DiffEq, HHL, QSVD | Hamiltonian abstractions, controlled time evolution, block encodings, linear-system or singular-value workflows. |
| Arithmetic and number-theory oracles | Shor | Modular multiplication/exponentiation circuits, classical preprocessing/postprocessing, scalable oracle construction. |
| Rich Grover oracle families | Grover inference and variational-generator variants | Oracle constructors beyond marked-basis states, subspace predicates, and validation helpers. |
| External data and chemistry workflows | VQE OpenFermion | Data import, operator conversion, and dependency strategy for chemistry instances. |

The first implementation pass should create these issues before closing the example-coverage epic. If the repository does not have a suitable epic issue yet, create one umbrella issue for “Yao docs and QuAlgorithmZoo example coverage” and link the Tier 3 issues from it.

## CLI Design

Keep `yao example <name>` as the entry point for circuit-producing examples. Do not add separate top-level algorithm commands until an example needs runtime behavior beyond emitting circuit JSON.

Proposed names:

```text
bell
ghz
qft
phase-estimation
hadamard-test
swap-test
bernstein-vazirani
grover
qaoa-maxcut
qcbm
mermin-magic
```

Example-specific options should be conservative:

| Option | Used by | Purpose |
|--------|---------|---------|
| `--nqubits` | qft, grover, qcbm | Circuit size. |
| `--secret <bits>` | bernstein-vazirani | Secret bit string. |
| `--marked <index>` | grover | Marked computational-basis state. |
| `--iterations <N|auto>` | grover | Grover iteration count. |
| `--depth <N>` | qaoa-maxcut, qcbm | Ansatz depth. |
| `--preset <name>` | phase-estimation, hadamard-test, qaoa-maxcut | Small built-in unitary or graph. |
| `--phase <theta>` | phase-estimation, hadamard-test, swap-test | Phase parameter where relevant. |

The CLI should continue to emit human-readable circuit display in terminals and JSON when piped or when `--json` is set.

## Documentation Design

Add a docs page for the example catalog and link it from `docs/src/SUMMARY.md`.

Each supported example should include:

- Source inspiration: Yao docs or QuAlgorithmZoo path.
- One `yao example` command.
- A simulation or probability pipeline.
- A visualization command.
- A tensor-network pipeline where meaningful.
- A short expected-result note, such as “Bernstein-Vazirani concentrates probability on the secret bit string.”

The docs should also include a coverage matrix with Tier 1, Tier 2, and Tier 3 status so readers understand what has been implemented and what is tracked by issues.

## Testing Design

Tests should cover both library builders and CLI behavior.

Library tests:

- Add focused `easybuild` tests for any new public builder.
- Validate qubit count, gate structure, norm preservation, and at least one algorithm-specific behavior.
- Prefer analytical checks where possible, such as Bernstein-Vazirani output probability on the secret state.

CLI tests:

- Add `yao example <name>` integration tests for each supported example.
- Parse emitted JSON and inspect `num_qubits`, gate count, and required gate features.
- For small examples, pipe through `run`, `probs`, or `expect` and check expected outcomes.
- Add negative tests for invalid parameters such as empty secrets, out-of-range marked states, and zero qubits where unsupported.

Docs validation:

- Keep command examples executable where practical.
- Run at least the CLI test suite and targeted `cargo test -p yao-rs easybuild` after implementation.

## Implementation Boundaries

This project should not implement full optimizers, autodiff, chemistry import, HHL, or Shor in the same PR as the example catalog. Those belong in Tier 3 issues.

This project may add small algorithm-specific builders when the output is still a finite circuit over existing `Gate` variants. The first PR should prefer correctness, clear CLI behavior, and documentation over broad algorithm depth.

## Success Criteria

- Every Yao docs and `/tmp/QuAlgorithmZoo.jl` example is listed in the coverage matrix.
- Tier 1 and selected Tier 2 examples are available through `yao example`.
- Supported examples have CLI integration tests and docs workflows.
- Tier 3 examples are recorded as GitHub issues with missing capability, affected source examples, and acceptance criteria.
- The final implementation keeps existing CLI output conventions and does not break existing `bell`, `ghz`, or `qft` examples.
