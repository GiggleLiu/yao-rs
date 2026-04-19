# Grover Marked State 5

The Grover walkthrough follows the usual oracle-and-diffusion rhythm for a
three-qubit search space. The script marks basis index `5`, binary state
`101`, and applies two Grover iterations.

## Commands

```bash
cargo build -p yao-cli --no-default-features
YAO_ARTIFACT_DIR=docs/src/examples/generated YAO_BIN=target/debug/yao bash examples/cli/grover_marked_state.sh 5
python3 scripts/plot_cli_results.py docs/src/examples/generated/results docs/src/examples/generated/plots
```

## Generated Artifacts

![Grover Marked State 5 circuit](../generated/svg/grover-marked-5.svg)

[Grover Marked State 5 result JSON](../generated/results/grover-marked-5-probs.json)

![Grover Marked State 5 probability plot](../generated/plots/grover-marked-5-probs.svg)

The marked `101` state is amplified to about `0.9453`.
