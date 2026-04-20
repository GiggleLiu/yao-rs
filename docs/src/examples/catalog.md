# Example Catalog

The example catalogue walks through eight worked quantum algorithms at
textbook depth. Each page motivates the algorithm, derives the key
identity, shows the circuit (SVG + annotated JSON excerpt + download link),
runs it from the CLI, and interprets the result.

Start with [Getting Started](../getting-started.md) and the
[Circuit JSON Conventions](../conventions.md) if you have not read them.

## Foundational

- [Entangled States](./entangled-states.md) — Bell pair and GHZ state;
  the first taste of entanglement.
- [Quantum Fourier Transform](./qft.md) — the phase-encoding primitive
  underlying phase estimation, Shor, and HHL.

## Standard subroutines

- [Phase Estimation](./phase-estimation.md) — read eigenphases into an
  ancilla register using an inverse QFT.
- [Ancilla Protocols](./ancilla-protocols.md) — Hadamard test and swap
  test; ancilla + controlled-U + basis change estimates overlaps.

## Algorithms

- [Bernstein–Vazirani](./bernstein-vazirani.md) — single-query learning of
  a linear Boolean function.
- [Grover Search](./grover-search.md) — amplitude amplification for
  unstructured search.

## Variational

- [QAOA for MaxCut](./qaoa-maxcut.md) — variational algorithm for
  combinatorial optimization (static parameters; training deferred).
- [Quantum Circuit Born Machine](./qcbm.md) — parametric circuit as a
  probability generator (static parameters; training deferred).

## Deferred coverage

The following categories need features tracked in open issues:

| Group | Examples | Issue |
|---|---|---|
| Optimization and training loops | VQE, QCBM training, QuGAN, GateLearning | [#31](https://github.com/GiggleLiu/yao-rs/issues/31) |
| Hamiltonian and time evolution | Ground-state solvers, HHL, QSVD, ODE-style evolutions | [#32](https://github.com/GiggleLiu/yao-rs/issues/32) |
| Rich Grover oracles | Grover inference, variational-generator variants | [#33](https://github.com/GiggleLiu/yao-rs/issues/33) |
| Measurement games and external workflows | Mermin magic square, chemistry import | [#34](https://github.com/GiggleLiu/yao-rs/issues/34) |
| Arithmetic oracles | Shor order finding | [#35](https://github.com/GiggleLiu/yao-rs/issues/35) |

## Regenerating all artifacts

All embedded SVGs, probability JSONs, and plots live under
`docs/src/examples/generated/`. Every example page lists a copy-paste bash
block to regenerate its own artifacts. To rebuild everything at once:

```bash
cargo build -p yao-cli --no-default-features
YAO_ARTIFACT_DIR=docs/src/examples/generated YAO_BIN=target/debug/yao bash examples/cli/generate_artifacts.sh
python3 scripts/plot_cli_results.py docs/src/examples/generated/results docs/src/examples/generated/plots
```
