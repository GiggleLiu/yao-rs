# QFT 4

The QFT 4 example builds the four-qubit Fourier-transform ladder from Hadamard
and controlled phase rotations, then finishes with swaps for the usual output
wire order. Starting from the zero state, this circuit produces a uniform
distribution.

## Commands

```bash
cargo build -p yao-cli --no-default-features
target/debug/yao example qft --nqubits 4 --json --output docs/src/examples/generated/circuits/qft4.json
target/debug/yao visualize docs/src/examples/generated/circuits/qft4.json --output docs/src/examples/generated/svg/qft4.svg
target/debug/yao simulate docs/src/examples/generated/circuits/qft4.json | target/debug/yao probs - > docs/src/examples/generated/results/qft4-probs.json
python3 scripts/plot_cli_results.py docs/src/examples/generated/results docs/src/examples/generated/plots
```

## Generated Artifacts

![QFT 4 circuit](../generated/svg/qft4.svg)

[QFT 4 result JSON](../generated/results/qft4-probs.json)

![QFT 4 probability plot](../generated/plots/qft4-probs.svg)

The probability output is uniform across all 16 basis states, with probability
`0.0625` per state.
