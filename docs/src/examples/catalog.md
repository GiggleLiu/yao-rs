# Example Catalog

This page maps Yao.jl documentation and QuAlgorithmZoo.jl examples to yao-rs CLI workflows.

## Supported CLI Examples

| Example | Source | Command | Notes |
|---------|--------|---------|-------|
| Bell | yao-rs starter example | `yao example bell` | Produces a 2-qubit Bell circuit. |
| GHZ | Yao docs GHZ | `yao example ghz --nqubits 5` | Produces GHZ entanglement. |
| QFT | Yao docs QFT | `yao example qft --nqubits 6` | Matches Yao EasyBuild QFT without final swaps. |
| Phase estimation | Yao docs / QuAlgorithmZoo | `yao example phase-estimation --nqubits 3 --preset z` | Emits circuit JSON for a small preset unitary. |
| Hadamard test | QuAlgorithmZoo README | `yao example hadamard-test --preset z` | Uses the existing easybuild circuit. |
| Swap test | QuAlgorithmZoo README | `yao example swap-test --nqubits-per-state 2 --nstates 2` | Uses the existing easybuild circuit. |
| Bernstein-Vazirani | QuAlgorithmZoo | `yao example bernstein-vazirani --secret 1011` | Measurement concentrates on the secret. |
| Grover | QuAlgorithmZoo | `yao example grover --nqubits 3 --marked 5 --iterations auto` | Dense custom-gate marked-state demo, limited to small sizes. |
| QAOA MaxCut | QuAlgorithmZoo | `yao example qaoa-maxcut --preset line4 --depth 2` | Static ansatz only, no optimizer. |
| QCBM | Yao docs / QuAlgorithmZoo | `yao example qcbm --nqubits 6 --depth 2` | Static variational ansatz only, no training. |

## Typical Workflows

```bash
yao example bernstein-vazirani --secret 1011 | yao run - --shots 128
yao example grover --nqubits 3 --marked 5 --iterations auto | yao probs -
yao example qft --nqubits 4 | yao visualize - --output qft.svg
yao example qaoa-maxcut --preset line4 --depth 1 | yao toeinsum - --mode overlap | yao optimize - | yao contract -
```

## Deferred Coverage

The following examples are tracked by GitHub issues because they require capabilities beyond circuit JSON generation:

| Group | Source examples | Issue |
|-------|-----------------|-------|
| Optimization and training | VQE, VQE OpenFermion, QCBM training, QuGAN, GateLearning, QMPS, PortZygote | [#31](https://github.com/GiggleLiu/yao-rs/issues/31) |
| Hamiltonian and time evolution | GroundStateSolvers, DiffEq, HHL, QSVD | [#32](https://github.com/GiggleLiu/yao-rs/issues/32) |
| Arithmetic oracles | Shor | [#35](https://github.com/GiggleLiu/yao-rs/issues/35) |
| Rich Grover oracles | Grover inference and variational-generator variants | [#33](https://github.com/GiggleLiu/yao-rs/issues/33) |
| Measurement games and external workflows | Mermin Magic Square, PortQuantumInformation, external chemistry import | [#34](https://github.com/GiggleLiu/yao-rs/issues/34) |
