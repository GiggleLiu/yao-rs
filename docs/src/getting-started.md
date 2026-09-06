# Getting Started

This guide installs the `yao` CLI tool, runs your first circuit, and points
at the example catalogue. You do not need to write any Rust — circuits are
plain JSON. See [Circuit JSON Conventions](./conventions.md) for the schema,
and if you want to embed yao-rs as a Rust library, see the Rust API pages
linked from the sidebar.

## Install the CLI

Install the published CLI with a current stable Rust toolchain:

```bash
cargo install yao-cli --locked
yao --help
```

To install the development version from source:

```bash
git clone https://github.com/GiggleLiu/yao-rs.git
cd yao-rs
cargo install --path yao-cli --locked
```

Both commands install `yao` into Cargo's binary directory (normally
`~/.cargo/bin`), which must be on your `PATH`.

## Your first circuit: Bell state

Ask the CLI for a built-in Bell circuit, render it to SVG, and simulate:

```bash
yao example bell > bell.json
yao visualize bell.json --output bell.svg
yao simulate bell.json | yao probs - --json
```

The `probs` output is:

```json
{"num_qubits": 2, "locs": null, "probabilities": [0.5, 0.0, 0.0, 0.5]}
```

Probability 0.5 on indices 0 and 3, nothing in the middle. Under the
qubit-0-MSB convention (see
[Bit ordering](./conventions.md#bit-ordering)) index 3 is
\\( |q_0 q_1\rangle = |11\rangle \\) — both qubits in
\\( |1\rangle \\). The [Entangled States](./examples/entangled-states.md)
example builds on this.

## Inspect a circuit without running it

```bash
yao inspect bell.json
```

Prints the number of qubits and gate counts in a human-readable form. Add
`--json` if you want the inspection itself as JSON for piping.

## Measurement samples

```bash
yao run bell.json --shots 1024
```

JSON output contains `counts` keyed by bit strings and `outcomes` as arrays
of measured bits in qubit order. Add `--json` to force JSON in a terminal.
Use `--seed 42` to repeat a sample sequence with the same binary version.

## Expectation values

For any Hermitian Pauli product:

```bash
yao run bell.json --op "Z(0)Z(1)" --json
```

Returns:

```json
{"operator": "Z(0)Z(1)", "expectation_value": {"re": 1.0, "im": 0.0}}
```

See the [Operator syntax](./conventions.md#operator-syntax) section of the
conventions page for the full grammar.

## Next steps

- [CLI Tool](./cli.md) — full command reference.
- [Circuit JSON Conventions](./conventions.md) — schema, gate names, bit
  ordering, result formats.
- [Example Catalog](./examples/catalog.md) — eight worked algorithms from
  Bell pairs to QCBM.
