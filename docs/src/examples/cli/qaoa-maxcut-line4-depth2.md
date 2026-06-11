# QAOA MaxCut Line-4 Depth 2

Run from the repository root.

## 1. Build the CLI

```bash
cargo build -p yao-cli --no-default-features
```

## 2. Generate the artifacts

```bash
YAO_ARTIFACT_DIR=docs/src/examples/generated YAO_BIN=target/debug/yao bash examples/cli/qaoa_maxcut_line4.sh 2
```

Circuit: [../generated/svg/qaoa-maxcut-line4-depth2.svg](../generated/svg/qaoa-maxcut-line4-depth2.svg)

Result: [../generated/results/qaoa-maxcut-line4-depth2-expect.json](../generated/results/qaoa-maxcut-line4-depth2-expect.json)

Plot: [../generated/plots/qaoa-maxcut-line4-depth2-expect.svg](../generated/plots/qaoa-maxcut-line4-depth2-expect.svg)

## 3. Refresh the plot

```bash
python3 scripts/plot_cli_results.py docs/src/examples/generated/results docs/src/examples/generated/plots
```

## 4. Inspect the generated result

```bash
python3 -m json.tool docs/src/examples/generated/results/qaoa-maxcut-line4-depth2-expect.json
```
