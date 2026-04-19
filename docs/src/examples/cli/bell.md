# Bell State

Run from the repository root. This walkthrough rebuilds the local CLI,
regenerates the Bell circuit artifacts, and refreshes the plot from the
generated result JSON.

The circuit follows the standard Bell tutorial pattern: a Hadamard puts qubit 0
into superposition, then a controlled X ties qubit 1 to the same branch.

## 1. Build the CLI

```bash
cargo build -p yao-cli --no-default-features
```

## 2. Generate the artifacts

```bash
target/debug/yao example bell --json --output docs/src/examples/generated/circuits/bell.json
target/debug/yao visualize docs/src/examples/generated/circuits/bell.json --output docs/src/examples/generated/svg/bell.svg
target/debug/yao simulate docs/src/examples/generated/circuits/bell.json | target/debug/yao probs - > docs/src/examples/generated/results/bell-probs.json
```

## 3. Refresh the plot

```bash
python3 scripts/plot_cli_results.py docs/src/examples/generated/results docs/src/examples/generated/plots
```

## 4. Inspect the generated result

```bash
python3 -m json.tool docs/src/examples/generated/results/bell-probs.json
```

## Generated Artifacts

![Bell circuit](../generated/svg/bell.svg)

[Bell result JSON](../generated/results/bell-probs.json)

![Bell probability plot](../generated/plots/bell-probs.svg)

The result puts probability `0.5` on `00` and `0.5` on `11`, with no mass on
the middle basis states.
