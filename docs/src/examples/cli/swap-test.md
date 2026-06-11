# Swap Test

Run from the repository root.

## 1. Build the CLI

```bash
cargo build -p yao-cli --no-default-features
```

## 2. Generate the artifacts

```bash
YAO_ARTIFACT_DIR=docs/src/examples/generated YAO_BIN=target/debug/yao bash examples/cli/swap_test.sh
```

Circuit: [../generated/svg/swap-test.svg](../generated/svg/swap-test.svg)

Result: [../generated/results/swap-test-probs.json](../generated/results/swap-test-probs.json)

Plot: [../generated/plots/swap-test-probs.svg](../generated/plots/swap-test-probs.svg)

## 3. Refresh the plot

```bash
python3 scripts/plot_cli_results.py docs/src/examples/generated/results docs/src/examples/generated/plots
```

## 4. Inspect the generated result

```bash
python3 -m json.tool docs/src/examples/generated/results/swap-test-probs.json
```
