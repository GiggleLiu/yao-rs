# OpenQASM

Exchange circuits with tools that read or write OpenQASM 2.0.
The CLI includes this support by default; Rust projects need the
[`qasm` feature](installation.md#optional-features).

## Import and run

```bash
yao fromqasm circuit.qasm --output circuit.json
yao run circuit.json --shots 1024
```

Import expands gate definitions into primitive operations. The resulting
circuit can be simulated, rendered, or exported as a tensor network.

## Export a circuit

```bash
yao example bell | yao toqasm - --output bell.qasm
```

`--output` writes raw QASM. When piped without `--output`, `toqasm` returns
a JSON object with a `qasm` string.

Export uses standard `qelib1.inc` gate names where possible. Unsupported
operations return an error.

## Use from Rust

```rust
use yao_rs::qasm::{from_qasm, to_qasm};

let source = r#"
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
cx q[0], q[1];
"#;
let imported = from_qasm(source).unwrap();
let exported = to_qasm(&imported.circuit).unwrap();
```

Measurements are returned separately in `imported.measurements` as
`(qubit, classical_bit)` pairs. They are not embedded in the executable circuit.
The CLI warns when it ignores imported measurements; `yao run --shots` samples
the final state.

## Try a benchmark circuit

```bash
yao fetch qasmbench list
yao fetch qasmbench grover | yao fromqasm - | yao run - --shots 100
```

Fetching requires a network connection. See the [command reference](cli.md#yao-fetch)
for benchmark scale selection.
