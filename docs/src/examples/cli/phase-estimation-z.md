# Phase Estimation Z

Run from the repository root. This walkthrough rebuilds the local CLI, runs the
bash workflow, and refreshes the plot from the generated result JSON.

The script uses a compact Z phase-estimation circuit: prepare the eigenstate,
kick phase back through a controlled Z, apply the readout Hadamard, and inspect
the two-qubit probability distribution.

## 1. Build the CLI

```bash
cargo build -p yao-cli --no-default-features
```

## 2. Generate the artifacts

```bash
YAO_ARTIFACT_DIR=docs/src/examples/generated YAO_BIN=target/debug/yao bash examples/cli/phase_estimation_z.sh
```

## 3. Refresh the plot

```bash
python3 scripts/plot_cli_results.py docs/src/examples/generated/results docs/src/examples/generated/plots
```

## 4. Inspect the generated result

```bash
python3 -m json.tool docs/src/examples/generated/results/phase-estimation-z-probs.json
```

## Generated Artifacts

![Phase Estimation Z circuit](../generated/svg/phase-estimation-z.svg)

[Phase Estimation Z result JSON](../generated/results/phase-estimation-z-probs.json)

![Phase Estimation Z probability plot](../generated/plots/phase-estimation-z-probs.svg)

The result places all probability on state `11`, index `3`.
