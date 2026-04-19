# Example Catalog

This page maps Yao.jl documentation and QuAlgorithmZoo.jl examples to yao-rs CLI workflows.

## Supported CLI Workflows

| Example | Source | Command | Notes |
|---------|--------|---------|-------|
| Bell | yao-rs starter example | `yao example bell` | Produces a 2-qubit Bell circuit. |
| GHZ | Yao docs GHZ | `yao example ghz --nqubits 5` | Produces GHZ entanglement. |
| QFT | Yao docs QFT | `yao example qft --nqubits 6` | Matches Yao EasyBuild QFT without final swaps. |
| Phase estimation | Yao docs / QuAlgorithmZoo | `bash examples/cli/phase_estimation_z.sh` | Builds a small Z-eigenphase circuit in bash and runs `yao simulate | yao probs`. |
| Hadamard test | QuAlgorithmZoo README | `bash examples/cli/hadamard_test_z.sh` | Builds a Z Hadamard-test circuit in bash and runs `yao simulate | yao probs`. |
| Swap test | QuAlgorithmZoo README | `bash examples/cli/swap_test.sh` | Builds a one-qubit swap test in bash and runs `yao simulate | yao probs`. |
| Bernstein-Vazirani | QuAlgorithmZoo | `bash examples/cli/bernstein_vazirani.sh 1011` | Builds the phase-oracle circuit in bash; probability concentrates on the secret. |
| Grover | QuAlgorithmZoo | `bash examples/cli/grover_marked_state.sh 5` | Builds a 3-qubit marked-state Grover circuit in bash using primitive gates. |
| QAOA MaxCut | QuAlgorithmZoo | `bash examples/cli/qaoa_maxcut_line4.sh 2` | Builds a static line-graph ansatz in bash and evaluates `Z(0)Z(1)`. |
| QCBM | Yao docs / QuAlgorithmZoo | `bash examples/cli/qcbm_static.sh 2` | Builds a static variational ansatz in bash and emits probabilities; no training. |

## Typical Workflows

```bash
bash examples/cli/bernstein_vazirani.sh 1011
bash examples/cli/grover_marked_state.sh 5
yao example qft --nqubits 4 | yao visualize - --output qft.svg
bash examples/cli/qaoa_maxcut_line4.sh 1
```

## Generated Visualization

For copy-paste commands, generated circuit SVGs, and generated result
summaries, see [CLI Example Visualization](./cli-visualization.md).

## Deferred Coverage

The following examples are tracked by GitHub issues because they require capabilities beyond circuit JSON generation:

| Group | Source examples | Issue |
|-------|-----------------|-------|
| Optimization and training | VQE, VQE OpenFermion, QCBM training, QuGAN, GateLearning, QMPS, PortZygote | [#31](https://github.com/GiggleLiu/yao-rs/issues/31) |
| Hamiltonian and time evolution | GroundStateSolvers, DiffEq, HHL, QSVD | [#32](https://github.com/GiggleLiu/yao-rs/issues/32) |
| Arithmetic oracles | Shor | [#35](https://github.com/GiggleLiu/yao-rs/issues/35) |
| Rich Grover oracles | Grover inference and variational-generator variants | [#33](https://github.com/GiggleLiu/yao-rs/issues/33) |
| Measurement games and external workflows | Mermin Magic Square, PortQuantumInformation, external chemistry import | [#34](https://github.com/GiggleLiu/yao-rs/issues/34) |
