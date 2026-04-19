# Phase Estimation Z

This compact phase-estimation example uses a Z phase pattern with two qubits.
It prepares phase kickback, applies the inverse readout shape, and checks the
measured phase register.

## Commands

```bash
cargo build -p yao-cli --no-default-features
YAO_ARTIFACT_DIR=docs/src/examples/generated YAO_BIN=target/debug/yao bash examples/cli/phase_estimation_z.sh
python3 scripts/plot_cli_results.py docs/src/examples/generated/results docs/src/examples/generated/plots
```

## Generated Artifacts

![Phase Estimation Z circuit](../generated/svg/phase-estimation-z.svg)

[Phase Estimation Z result JSON](../generated/results/phase-estimation-z-probs.json)

![Phase Estimation Z probability plot](../generated/plots/phase-estimation-z-probs.svg)

The result places all probability on state `11`, index `3`.
