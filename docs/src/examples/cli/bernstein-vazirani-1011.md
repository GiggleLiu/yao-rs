# Bernstein-Vazirani 1011

Run from the repository root.

## 1. Build the CLI

```bash
cargo build -p yao-cli --no-default-features
```

## 2. Generate the artifacts

```bash
YAO_ARTIFACT_DIR=docs/src/examples/generated YAO_BIN=target/debug/yao bash examples/cli/bernstein_vazirani.sh 1011
```

Circuit: [../generated/svg/bernstein-vazirani-1011.svg](../generated/svg/bernstein-vazirani-1011.svg)

Result: [../generated/results/bernstein-vazirani-1011-probs.json](../generated/results/bernstein-vazirani-1011-probs.json)

Plot: [../generated/plots/bernstein-vazirani-1011-probs.svg](../generated/plots/bernstein-vazirani-1011-probs.svg)

## 3. Refresh the plot

```bash
python3 scripts/plot_cli_results.py docs/src/examples/generated/results docs/src/examples/generated/plots
```

## 4. Inspect the generated result

```bash
python3 -m json.tool docs/src/examples/generated/results/bernstein-vazirani-1011-probs.json
```
