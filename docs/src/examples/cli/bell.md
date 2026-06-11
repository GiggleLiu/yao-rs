# Bell State

Run from the repository root.

## 1. Build the CLI

```bash
cargo build -p yao-cli --no-default-features
```

## 2. Generate the artifacts

```bash
target/debug/yao example bell --json --output docs/src/examples/generated/circuits/bell.json
target/debug/yao visualize docs/src/examples/generated/circuits/bell.json --output docs/src/examples/generated/svg/bell.svg
target/debug/yao simulate docs/src/examples/generated/circuits/bell.json | target/debug/yao probs - > docs/src/examples/generated/results/bell-probs.json
```

Circuit: [../generated/svg/bell.svg](../generated/svg/bell.svg)

Result: [../generated/results/bell-probs.json](../generated/results/bell-probs.json)

Plot: [../generated/plots/bell-probs.svg](../generated/plots/bell-probs.svg)

## 3. Refresh the plot

```bash
python3 scripts/plot_cli_results.py docs/src/examples/generated/results docs/src/examples/generated/plots
```

## 4. Inspect the generated result

```bash
python3 -m json.tool docs/src/examples/generated/results/bell-probs.json
```
