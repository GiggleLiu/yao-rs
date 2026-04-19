# QAOA MaxCut Line-4 Depth 2

Run from the repository root. This walkthrough rebuilds the local CLI, runs the
bash workflow for depth `2`, and refreshes the plot from the generated
expectation JSON.

The circuit builds a fixed QAOA-style ansatz for a four-node line graph. It
alternates edge phase separators with single-qubit mixers, then evaluates the
`Z(0)Z(1)` expectation. The script uses fixed parameters and does not run a
classical optimizer.

## 1. Build the CLI

```bash
cargo build -p yao-cli --no-default-features
```

## 2. Generate the artifacts

```bash
YAO_ARTIFACT_DIR=docs/src/examples/generated YAO_BIN=target/debug/yao bash examples/cli/qaoa_maxcut_line4.sh 2
```

## 3. Refresh the plot

```bash
python3 scripts/plot_cli_results.py docs/src/examples/generated/results docs/src/examples/generated/plots
```

## 4. Inspect the generated result

```bash
python3 -m json.tool docs/src/examples/generated/results/qaoa-maxcut-line4-depth2-expect.json
```

## Generated Artifacts

![QAOA MaxCut Line-4 Depth 2 circuit](../generated/svg/qaoa-maxcut-line4-depth2.svg)

[QAOA MaxCut Line-4 Depth 2 result JSON](../generated/results/qaoa-maxcut-line4-depth2-expect.json)

![QAOA MaxCut Line-4 Depth 2 expectation plot](../generated/plots/qaoa-maxcut-line4-depth2-expect.svg)

The expectation JSON reports `Z(0)Z(1)` with real value about `0.3074`.
