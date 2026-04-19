# CLI Example Visualization

This page indexes the CLI commands used to generate the example circuits, SVG
diagrams, result summaries, and result plots committed under
[`generated/`](./generated/manifest.md).

## Regenerate Artifacts

Build the CLI and regenerate the checked-in example artifacts from the
repository root:

```bash
cargo build -p yao-cli --no-default-features
YAO_BIN=target/debug/yao bash examples/cli/generate_artifacts.sh docs/src/examples/generated
```

To refresh only the result plots after editing or replacing result JSON, run:

```bash
python3 scripts/plot_cli_results.py docs/src/examples/generated/results docs/src/examples/generated/plots
```

The generated artifact index is available in
[generated/manifest.md](./generated/manifest.md).

## Example Pages

Each page below is a copy-paste walkthrough that rebuilds the CLI, regenerates
the example data, refreshes the Python-generated plot, and links the committed
artifacts used by mdBook.

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

## CLI Commands

Built-in examples are available directly from `yao example`:

```bash
target/debug/yao example bell
target/debug/yao example ghz --nqubits 4
target/debug/yao example qft --nqubits 4
```

Scripted examples can emit result JSON to stdout:

```bash
YAO_BIN=target/debug/yao bash examples/cli/phase_estimation_z.sh
YAO_BIN=target/debug/yao bash examples/cli/hadamard_test_z.sh
YAO_BIN=target/debug/yao bash examples/cli/swap_test.sh
YAO_BIN=target/debug/yao bash examples/cli/bernstein_vazirani.sh 1011
YAO_BIN=target/debug/yao bash examples/cli/grover_marked_state.sh 5
YAO_BIN=target/debug/yao bash examples/cli/qaoa_maxcut_line4.sh 2
YAO_BIN=target/debug/yao bash examples/cli/qcbm_static.sh 2
```

Use `YAO_ARTIFACT_DIR` when a script should also write checked-in circuit,
SVG, and result artifacts:

```bash
YAO_ARTIFACT_DIR=docs/src/examples/generated YAO_BIN=target/debug/yao bash examples/cli/phase_estimation_z.sh
YAO_ARTIFACT_DIR=docs/src/examples/generated YAO_BIN=target/debug/yao bash examples/cli/hadamard_test_z.sh
YAO_ARTIFACT_DIR=docs/src/examples/generated YAO_BIN=target/debug/yao bash examples/cli/swap_test.sh
YAO_ARTIFACT_DIR=docs/src/examples/generated YAO_BIN=target/debug/yao bash examples/cli/bernstein_vazirani.sh 1011
YAO_ARTIFACT_DIR=docs/src/examples/generated YAO_BIN=target/debug/yao bash examples/cli/grover_marked_state.sh 5
YAO_ARTIFACT_DIR=docs/src/examples/generated YAO_BIN=target/debug/yao bash examples/cli/qaoa_maxcut_line4.sh 2
YAO_ARTIFACT_DIR=docs/src/examples/generated YAO_BIN=target/debug/yao bash examples/cli/qcbm_static.sh 2
```

## Circuit Gallery

| Example | Circuit | Plot | Result |
|---------|---------|------|--------|
| Bell State | [generated/svg/bell.svg](./generated/svg/bell.svg) | [generated/plots/bell-probs.svg](./generated/plots/bell-probs.svg) | [generated/results/bell-probs.json](./generated/results/bell-probs.json) |
| GHZ 4 | [generated/svg/ghz4.svg](./generated/svg/ghz4.svg) | [generated/plots/ghz4-probs.svg](./generated/plots/ghz4-probs.svg) | [generated/results/ghz4-probs.json](./generated/results/ghz4-probs.json) |
| QFT 4 | [generated/svg/qft4.svg](./generated/svg/qft4.svg) | [generated/plots/qft4-probs.svg](./generated/plots/qft4-probs.svg) | [generated/results/qft4-probs.json](./generated/results/qft4-probs.json) |
| Phase Estimation Z | [generated/svg/phase-estimation-z.svg](./generated/svg/phase-estimation-z.svg) | [generated/plots/phase-estimation-z-probs.svg](./generated/plots/phase-estimation-z-probs.svg) | [generated/results/phase-estimation-z-probs.json](./generated/results/phase-estimation-z-probs.json) |
| Hadamard Test Z | [generated/svg/hadamard-test-z.svg](./generated/svg/hadamard-test-z.svg) | [generated/plots/hadamard-test-z-probs.svg](./generated/plots/hadamard-test-z-probs.svg) | [generated/results/hadamard-test-z-probs.json](./generated/results/hadamard-test-z-probs.json) |
| Swap Test | [generated/svg/swap-test.svg](./generated/svg/swap-test.svg) | [generated/plots/swap-test-probs.svg](./generated/plots/swap-test-probs.svg) | [generated/results/swap-test-probs.json](./generated/results/swap-test-probs.json) |
| Bernstein-Vazirani 1011 | [generated/svg/bernstein-vazirani-1011.svg](./generated/svg/bernstein-vazirani-1011.svg) | [generated/plots/bernstein-vazirani-1011-probs.svg](./generated/plots/bernstein-vazirani-1011-probs.svg) | [generated/results/bernstein-vazirani-1011-probs.json](./generated/results/bernstein-vazirani-1011-probs.json) |
| Grover Marked State 5 | [generated/svg/grover-marked-5.svg](./generated/svg/grover-marked-5.svg) | [generated/plots/grover-marked-5-probs.svg](./generated/plots/grover-marked-5-probs.svg) | [generated/results/grover-marked-5-probs.json](./generated/results/grover-marked-5-probs.json) |
| QAOA MaxCut Line-4 Depth 2 | [generated/svg/qaoa-maxcut-line4-depth2.svg](./generated/svg/qaoa-maxcut-line4-depth2.svg) | [generated/plots/qaoa-maxcut-line4-depth2-expect.svg](./generated/plots/qaoa-maxcut-line4-depth2-expect.svg) | [generated/results/qaoa-maxcut-line4-depth2-expect.json](./generated/results/qaoa-maxcut-line4-depth2-expect.json) |
| QCBM Static Depth 2 | [generated/svg/qcbm-static-depth2.svg](./generated/svg/qcbm-static-depth2.svg) | [generated/plots/qcbm-static-depth2-probs.svg](./generated/plots/qcbm-static-depth2-probs.svg) | [generated/results/qcbm-static-depth2-probs.json](./generated/results/qcbm-static-depth2-probs.json) |

## Result Highlights

Bell and GHZ concentrate on the all-zero and all-one branches. QFT 4 produces a
uniform 16-state distribution. Grover Marked State 5 amplifies `101` to about
`0.9453`, and QAOA MaxCut Line-4 Depth 2 reports a `Z(0)Z(1)` real expectation
around `0.3074`. QCBM Static Depth 2 is a static zero-parameter demo, not a
training workflow.
