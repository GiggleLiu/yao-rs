# Hadamard Test Z

Run from the repository root. This walkthrough rebuilds the local CLI, runs the
bash workflow, and refreshes the plot from the generated result JSON.

The circuit keeps the Hadamard-test shape deliberately small: prepare an
ancilla, apply controlled Z phase behavior, then read out the resulting
two-qubit probabilities.

## 1. Build the CLI

```bash
cargo build -p yao-cli --no-default-features
```

## 2. Generate the artifacts

```bash
YAO_ARTIFACT_DIR=docs/src/examples/generated YAO_BIN=target/debug/yao bash examples/cli/hadamard_test_z.sh
```

## 3. Refresh the plot

```bash
python3 scripts/plot_cli_results.py docs/src/examples/generated/results docs/src/examples/generated/plots
```

## 4. Inspect the generated result

```bash
python3 -m json.tool docs/src/examples/generated/results/hadamard-test-z-probs.json
```

## Generated Artifacts

![Hadamard Test Z circuit](../generated/svg/hadamard-test-z.svg)

[Hadamard Test Z result JSON](../generated/results/hadamard-test-z-probs.json)

![Hadamard Test Z probability plot](../generated/plots/hadamard-test-z-probs.svg)

For this minimal Z case, the generated probabilities intentionally match the
phase-estimation Z demo and place probability `1.0` on state `11`.
