# Bernstein-Vazirani 1011

Run from the repository root. This walkthrough rebuilds the local CLI, runs the
bash workflow for secret string `1011`, and refreshes the plot from the
generated result JSON.

The circuit puts the register into superposition, applies the phase oracle for
the secret bits, and uses final Hadamards to recover the secret directly in the
probability distribution.

## 1. Build the CLI

```bash
cargo build -p yao-cli --no-default-features
```

## 2. Generate the artifacts

```bash
YAO_ARTIFACT_DIR=docs/src/examples/generated YAO_BIN=target/debug/yao bash examples/cli/bernstein_vazirani.sh 1011
```

## 3. Refresh the plot

```bash
python3 scripts/plot_cli_results.py docs/src/examples/generated/results docs/src/examples/generated/plots
```

## 4. Inspect the generated result

```bash
python3 -m json.tool docs/src/examples/generated/results/bernstein-vazirani-1011-probs.json
```

## Generated Artifacts

![Bernstein-Vazirani 1011 circuit](../generated/svg/bernstein-vazirani-1011.svg)

[Bernstein-Vazirani 1011 result JSON](../generated/results/bernstein-vazirani-1011-probs.json)

![Bernstein-Vazirani 1011 probability plot](../generated/plots/bernstein-vazirani-1011-probs.svg)

The result places probability `1.0` on `1011`, index `11`.
