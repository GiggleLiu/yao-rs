# CLI Example Visualization

This page collects the CLI commands that generate the checked-in example
circuits, SVG circuit diagrams, result JSON files, and result plots.

Regenerate every artifact from the repository root:

```bash
cargo build -p yao-cli --no-default-features
YAO_BIN=target/debug/yao bash examples/cli/generate_artifacts.sh docs/src/examples/generated
python3 scripts/plot_cli_results.py docs/src/examples/generated/results docs/src/examples/generated/plots
```

The generator writes the full artifact index to
[generated/manifest.md](./generated/manifest.md). For quick inspection, open
[generated/svg/qft4.svg](./generated/svg/qft4.svg),
[generated/results/grover-marked-5-probs.json](./generated/results/grover-marked-5-probs.json),
or [generated/plots/grover-marked-5-probs.svg](./generated/plots/grover-marked-5-probs.svg).

## Script Commands

```bash
YAO_BIN=target/debug/yao bash examples/cli/phase_estimation_z.sh
YAO_BIN=target/debug/yao bash examples/cli/hadamard_test_z.sh
YAO_BIN=target/debug/yao bash examples/cli/swap_test.sh
YAO_BIN=target/debug/yao bash examples/cli/bernstein_vazirani.sh 1011
YAO_BIN=target/debug/yao bash examples/cli/grover_marked_state.sh 5
YAO_BIN=target/debug/yao bash examples/cli/qaoa_maxcut_line4.sh 2
YAO_BIN=target/debug/yao bash examples/cli/qcbm_static.sh 2
```

Artifact-mode commands:

```bash
YAO_ARTIFACT_DIR=docs/src/examples/generated YAO_BIN=target/debug/yao bash examples/cli/phase_estimation_z.sh
YAO_ARTIFACT_DIR=docs/src/examples/generated YAO_BIN=target/debug/yao bash examples/cli/hadamard_test_z.sh
YAO_ARTIFACT_DIR=docs/src/examples/generated YAO_BIN=target/debug/yao bash examples/cli/swap_test.sh
YAO_ARTIFACT_DIR=docs/src/examples/generated YAO_BIN=target/debug/yao bash examples/cli/bernstein_vazirani.sh 1011
YAO_ARTIFACT_DIR=docs/src/examples/generated YAO_BIN=target/debug/yao bash examples/cli/grover_marked_state.sh 5
YAO_ARTIFACT_DIR=docs/src/examples/generated YAO_BIN=target/debug/yao bash examples/cli/qaoa_maxcut_line4.sh 2
YAO_ARTIFACT_DIR=docs/src/examples/generated YAO_BIN=target/debug/yao bash examples/cli/qcbm_static.sh 2
```

The Grover marked-state example reaches about `0.9453` probability on the
marked state. The QAOA line-4 example reports about `0.3074` for the checked
expectation value. The QCBM page uses a static zero-parameter-free schedule.

## Example Pages

- [Bell State](./cli/bell.md)
- [GHZ 4](./cli/ghz4.md)
- [QFT 4](./cli/qft4.md)
- [Phase Estimation Z](./cli/phase-estimation-z.md)
- [Hadamard Test Z](./cli/hadamard-test-z.md)
- [Swap Test](./cli/swap-test.md)
- [Bernstein-Vazirani 1011](./cli/bernstein-vazirani-1011.md)
- [Grover Marked State 5](./cli/grover-marked-5.md)
- [QAOA MaxCut Line-4 Depth 2](./cli/qaoa-maxcut-line4-depth2.md)
- [QCBM Static Depth 2](./cli/qcbm-static-depth2.md)
