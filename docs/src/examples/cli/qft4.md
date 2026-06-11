# QFT 4

Run from the repository root.

## 1. Build the CLI

```bash
cargo build -p yao-cli --no-default-features
```

## 2. Generate the artifacts

```bash
target/debug/yao example qft --nqubits 4 --json --output docs/src/examples/generated/circuits/qft4.json
target/debug/yao visualize docs/src/examples/generated/circuits/qft4.json --output docs/src/examples/generated/svg/qft4.svg
target/debug/yao simulate docs/src/examples/generated/circuits/qft4.json | target/debug/yao probs - > docs/src/examples/generated/results/qft4-probs.json
```

Circuit: [../generated/svg/qft4.svg](../generated/svg/qft4.svg)

Result: [../generated/results/qft4-probs.json](../generated/results/qft4-probs.json)

Plot: [../generated/plots/qft4-probs.svg](../generated/plots/qft4-probs.svg)

## 3. Refresh the plot

```bash
python3 scripts/plot_cli_results.py docs/src/examples/generated/results docs/src/examples/generated/plots
```

## 4. Inspect the generated result

```bash
python3 -m json.tool docs/src/examples/generated/results/qft4-probs.json
```
