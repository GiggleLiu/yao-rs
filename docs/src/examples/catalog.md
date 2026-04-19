# Example Catalog

This page maps Yao.jl documentation and QuAlgorithmZoo.jl examples to yao-rs CLI workflows.

## Supported CLI Workflows

| Example | Source | Command | Notes |
|---------|--------|---------|-------|
| [Bell State](./cli/bell.md) | yao-rs starter example | `yao example bell` | Produces a 2-qubit Bell circuit. |
| [GHZ 4](./cli/ghz4.md) | Yao docs GHZ | `yao example ghz --nqubits 4` | Produces GHZ entanglement. |
| [QFT 4](./cli/qft4.md) | Yao docs QFT | `yao example qft --nqubits 4` | Builds the CLI QFT ladder and generated visualization. |
| [Phase Estimation Z](./cli/phase-estimation-z.md) | Yao docs / QuAlgorithmZoo | `bash examples/cli/phase_estimation_z.sh` | Builds a small Z-eigenphase circuit in bash and runs `yao simulate | yao probs`. |
| [Hadamard Test Z](./cli/hadamard-test-z.md) | QuAlgorithmZoo README | `bash examples/cli/hadamard_test_z.sh` | Builds a Z Hadamard-test circuit in bash and runs `yao simulate | yao probs`. |
| [Swap Test](./cli/swap-test.md) | QuAlgorithmZoo README | `bash examples/cli/swap_test.sh` | Builds a one-qubit swap test in bash and runs `yao simulate | yao probs`. |
| [Bernstein-Vazirani 1011](./cli/bernstein-vazirani-1011.md) | QuAlgorithmZoo | `bash examples/cli/bernstein_vazirani.sh 1011` | Builds the phase-oracle circuit in bash; probability concentrates on the secret. |
| [Grover Marked State 5](./cli/grover-marked-5.md) | QuAlgorithmZoo | `bash examples/cli/grover_marked_state.sh 5` | Builds a 3-qubit marked-state Grover circuit in bash using primitive gates. |
| [QAOA MaxCut Line-4 Depth 2](./cli/qaoa-maxcut-line4-depth2.md) | QuAlgorithmZoo | `bash examples/cli/qaoa_maxcut_line4.sh 2` | Builds a static line-graph ansatz in bash and evaluates `Z(0)Z(1)`. |
| [QCBM Static Depth 2](./cli/qcbm-static-depth2.md) | Yao docs / QuAlgorithmZoo | `bash examples/cli/qcbm_static.sh 2` | Builds a static variational ansatz in bash and emits probabilities; no training. |

## Typical Workflows

```bash
cargo build -p yao-cli --no-default-features
YAO_BIN=target/debug/yao bash examples/cli/bernstein_vazirani.sh 1011
YAO_BIN=target/debug/yao bash examples/cli/grover_marked_state.sh 5
target/debug/yao example qft --nqubits 4 --json --output qft4.json
target/debug/yao visualize qft4.json --output qft4.svg
YAO_BIN=target/debug/yao bash examples/cli/qaoa_maxcut_line4.sh 1
```

## Generated Visualization

For copy-paste commands, generated circuit SVGs, generated plots, and generated
result summaries, see [CLI Example Visualization](./cli-visualization.md).

## Deferred Coverage

The following examples are tracked by GitHub issues because they require capabilities beyond circuit JSON generation:

| Group | Source examples | Issue |
|-------|-----------------|-------|
| Optimization and training | VQE, VQE OpenFermion, QCBM training, QuGAN, GateLearning, QMPS, PortZygote | [#31](https://github.com/GiggleLiu/yao-rs/issues/31) |
| Hamiltonian and time evolution | GroundStateSolvers, DiffEq, HHL, QSVD | [#32](https://github.com/GiggleLiu/yao-rs/issues/32) |
| Arithmetic oracles | Shor | [#35](https://github.com/GiggleLiu/yao-rs/issues/35) |
| Rich Grover oracles | Grover inference and variational-generator variants | [#33](https://github.com/GiggleLiu/yao-rs/issues/33) |
| Measurement games and external workflows | Mermin Magic Square, PortQuantumInformation, external chemistry import | [#34](https://github.com/GiggleLiu/yao-rs/issues/34) |
