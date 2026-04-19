# Swap Test

The swap test compares two prepared states through an ancilla-controlled swap
pattern. This static CLI version is useful for checking both the circuit shape
and the distribution produced by the controlled swaps.

## Commands

```bash
cargo build -p yao-cli --no-default-features
YAO_ARTIFACT_DIR=docs/src/examples/generated YAO_BIN=target/debug/yao bash examples/cli/swap_test.sh
python3 scripts/plot_cli_results.py docs/src/examples/generated/results docs/src/examples/generated/plots
```

## Generated Artifacts

![Swap Test circuit](../generated/svg/swap-test.svg)

[Swap Test result JSON](../generated/results/swap-test-probs.json)

![Swap Test probability plot](../generated/plots/swap-test-probs.svg)

The nonzero states are `001`, `010`, `101`, and `110`, each with probability
`0.25`.
