# Swap Test

Run from the repository root. This walkthrough rebuilds the local CLI, runs the
bash workflow, and refreshes the plot from the generated result JSON.

The swap test compares two prepared states through an ancilla-controlled swap
pattern. This static CLI version is useful for checking both the circuit shape
and the distribution produced by the controlled swaps.

## 1. Build the CLI

```bash
cargo build -p yao-cli --no-default-features
```

## 2. Generate the artifacts

```bash
YAO_ARTIFACT_DIR=docs/src/examples/generated YAO_BIN=target/debug/yao bash examples/cli/swap_test.sh
```

## 3. Refresh the plot

```bash
python3 scripts/plot_cli_results.py docs/src/examples/generated/results docs/src/examples/generated/plots
```

## 4. Inspect the generated result

```bash
python3 -m json.tool docs/src/examples/generated/results/swap-test-probs.json
```

## Generated Artifacts

![Swap Test circuit](../generated/svg/swap-test.svg)

[Swap Test result JSON](../generated/results/swap-test-probs.json)

![Swap Test probability plot](../generated/plots/swap-test-probs.svg)

The nonzero states are `001`, `010`, `101`, and `110`, each with probability
`0.25`.
