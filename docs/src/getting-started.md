# Getting Started

This guide installs the `yao` CLI tool, runs your first circuit, and points
at the example catalogue. You do not need to write any Rust — circuits are
plain JSON. See [Circuit JSON Conventions](./conventions.md) for the schema,
and if you want to embed yao-rs as a Rust library, see the Rust API pages
linked from the sidebar.

## Install the CLI

Clone the repository and build the CLI:

```bash
git clone https://github.com/QuantumBFS/yao-rs.git
cd yao-rs
cargo build -p yao-cli --release
```

The built binary is `target/release/yao`. Either add it to your `PATH` or
run it by full path:

```bash
./target/release/yao --help
```

All commands below use the bare `yao` name — replace with the full path if
your `PATH` does not include it.

## Your first circuit: Bell state

Ask the CLI for a built-in Bell circuit, render it to SVG, and simulate:

```bash
yao example bell > bell.json
yao visualize bell.json --output bell.svg
yao simulate bell.json | yao probs -
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

Returns an array of measurement outcomes. Each sample is an integer whose
bit pattern encodes a computational-basis state under the qubit-0-MSB
convention.

## Expectation values

For any Hermitian Pauli product:

```bash
yao run bell.json --op "Z(0)Z(1)"
```

Returns:

```json
{"operator": "Z(0)Z(1)", "value": 1.0}
```

See the [Operator syntax](./conventions.md#operator-syntax) section of the
conventions page for the full grammar.

## Next steps

- [CLI Tool](./cli.md) — full command reference.
- [Circuit JSON Conventions](./conventions.md) — schema, gate names, bit
  ordering, result formats.
- [Example Catalog](./examples/catalog.md) — eight worked algorithms from
  Bell pairs to QCBM.
