# Phase Estimation Z

Run from the repository root.

## 1. Build the CLI

```bash
cargo build -p yao-cli --no-default-features
```

## 2. Generate the artifacts

```bash
YAO_ARTIFACT_DIR=docs/src/examples/generated YAO_BIN=target/debug/yao bash examples/cli/phase_estimation_z.sh
```

Circuit: [../generated/svg/phase-estimation-z.svg](../generated/svg/phase-estimation-z.svg)

Result: [../generated/results/phase-estimation-z-probs.json](../generated/results/phase-estimation-z-probs.json)

Plot: [../generated/plots/phase-estimation-z-probs.svg](../generated/plots/phase-estimation-z-probs.svg)

## 3. Refresh the plot

```bash
python3 scripts/plot_cli_results.py docs/src/examples/generated/results docs/src/examples/generated/plots
```

## 4. Inspect the generated result

```bash
python3 -m json.tool docs/src/examples/generated/results/phase-estimation-z-probs.json
```
