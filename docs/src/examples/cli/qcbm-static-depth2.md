# QCBM Static Depth 2

Run from the repository root.

## 1. Build the CLI

```bash
cargo build -p yao-cli --no-default-features
```

## 2. Generate the artifacts

```bash
YAO_ARTIFACT_DIR=docs/src/examples/generated YAO_BIN=target/debug/yao bash examples/cli/qcbm_static.sh 2
```

Circuit: [../generated/svg/qcbm-static-depth2.svg](../generated/svg/qcbm-static-depth2.svg)

Result: [../generated/results/qcbm-static-depth2-probs.json](../generated/results/qcbm-static-depth2-probs.json)

Plot: [../generated/plots/qcbm-static-depth2-probs.svg](../generated/plots/qcbm-static-depth2-probs.svg)

## 3. Refresh the plot

```bash
python3 scripts/plot_cli_results.py docs/src/examples/generated/results docs/src/examples/generated/plots
```

## 4. Inspect the generated result

```bash
python3 -m json.tool docs/src/examples/generated/results/qcbm-static-depth2-probs.json
```
