# Grover Marked State 5

Run from the repository root.

## 1. Build the CLI

```bash
cargo build -p yao-cli --no-default-features
```

## 2. Generate the artifacts

```bash
YAO_ARTIFACT_DIR=docs/src/examples/generated YAO_BIN=target/debug/yao bash examples/cli/grover_marked_state.sh 5
```

Circuit: [../generated/svg/grover-marked-5.svg](../generated/svg/grover-marked-5.svg)

Result: [../generated/results/grover-marked-5-probs.json](../generated/results/grover-marked-5-probs.json)

Plot: [../generated/plots/grover-marked-5-probs.svg](../generated/plots/grover-marked-5-probs.svg)

## 3. Refresh the plot

```bash
python3 scripts/plot_cli_results.py docs/src/examples/generated/results docs/src/examples/generated/plots
```

## 4. Inspect the generated result

```bash
python3 -m json.tool docs/src/examples/generated/results/grover-marked-5-probs.json
```
