# QCBM Static Depth 2

Run from the repository root. This walkthrough rebuilds the local CLI, runs the
bash workflow for depth `2`, and refreshes the plot from the generated result
JSON.

This QCBM example is a static zero-parameter ansatz demonstration. It borrows
the layered circuit shape used in QCBM examples, but it is not a training
workflow and does not fit a data distribution.

## 1. Build the CLI

```bash
cargo build -p yao-cli --no-default-features
```

## 2. Generate the artifacts

```bash
YAO_ARTIFACT_DIR=docs/src/examples/generated YAO_BIN=target/debug/yao bash examples/cli/qcbm_static.sh 2
```

## 3. Refresh the plot

```bash
python3 scripts/plot_cli_results.py docs/src/examples/generated/results docs/src/examples/generated/plots
```

## 4. Inspect the generated result

```bash
python3 -m json.tool docs/src/examples/generated/results/qcbm-static-depth2-probs.json
```

## Generated Artifacts

![QCBM Static Depth 2 circuit](../generated/svg/qcbm-static-depth2.svg)

[QCBM Static Depth 2 result JSON](../generated/results/qcbm-static-depth2-probs.json)

![QCBM Static Depth 2 probability plot](../generated/plots/qcbm-static-depth2-probs.svg)

The generated probability JSON has 64 entries for six qubits and keeps this
demo's mass on the all-zero basis state.
