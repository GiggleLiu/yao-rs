# Hadamard Test Z

Run from the repository root.

## 1. Build the CLI

```bash
cargo build -p yao-cli --no-default-features
```

## 2. Generate the artifacts

```bash
YAO_ARTIFACT_DIR=docs/src/examples/generated YAO_BIN=target/debug/yao bash examples/cli/hadamard_test_z.sh
```

Circuit: [../generated/svg/hadamard-test-z.svg](../generated/svg/hadamard-test-z.svg)

Result: [../generated/results/hadamard-test-z-probs.json](../generated/results/hadamard-test-z-probs.json)

Plot: [../generated/plots/hadamard-test-z-probs.svg](../generated/plots/hadamard-test-z-probs.svg)

## 3. Refresh the plot

```bash
python3 scripts/plot_cli_results.py docs/src/examples/generated/results docs/src/examples/generated/plots
```

## 4. Inspect the generated result

```bash
python3 -m json.tool docs/src/examples/generated/results/hadamard-test-z-probs.json
```
