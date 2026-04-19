# GHZ 4

The GHZ 4 example extends the Bell pattern across four qubits. One Hadamard
creates the branch, and the following controlled operations tie every qubit to
the same classical parity branch.

## Commands

```bash
cargo build -p yao-cli --no-default-features
target/debug/yao example ghz --nqubits 4 --json --output docs/src/examples/generated/circuits/ghz4.json
target/debug/yao visualize docs/src/examples/generated/circuits/ghz4.json --output docs/src/examples/generated/svg/ghz4.svg
target/debug/yao simulate docs/src/examples/generated/circuits/ghz4.json | target/debug/yao probs - > docs/src/examples/generated/results/ghz4-probs.json
python3 scripts/plot_cli_results.py docs/src/examples/generated/results docs/src/examples/generated/plots
```

## Generated Artifacts

![GHZ 4 circuit](../generated/svg/ghz4.svg)

[GHZ 4 result JSON](../generated/results/ghz4-probs.json)

![GHZ 4 probability plot](../generated/plots/ghz4-probs.svg)

The result has only `0000` and `1111` populated, each with probability `0.5`.
