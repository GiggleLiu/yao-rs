# Bell State

The Bell example prepares a two-qubit entangled state. The first qubit is put
into superposition, and a controlled operation ties the second qubit to the
same parity branch.

## Commands

```bash
cargo build -p yao-cli --no-default-features
target/debug/yao example bell --json --output docs/src/examples/generated/circuits/bell.json
target/debug/yao visualize docs/src/examples/generated/circuits/bell.json --output docs/src/examples/generated/svg/bell.svg
target/debug/yao simulate docs/src/examples/generated/circuits/bell.json | target/debug/yao probs - > docs/src/examples/generated/results/bell-probs.json
python3 scripts/plot_cli_results.py docs/src/examples/generated/results docs/src/examples/generated/plots
```

## Generated Artifacts

![Bell circuit](../generated/svg/bell.svg)

[Bell result JSON](../generated/results/bell-probs.json)

![Bell probability plot](../generated/plots/bell-probs.svg)

The result puts probability `0.5` on `00` and `0.5` on `11`, with no mass on
the middle basis states.
