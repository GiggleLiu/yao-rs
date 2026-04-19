# Circuit JSON Conventions

Every CLI example in yao-rs consumes and produces JSON documents. This page
documents the schema, the gate names, the bit ordering, and the result
formats. If you want to write your own circuit from Python, a notebook, or a
text editor, this is the only page you need.

## Schema at a glance

A circuit is a JSON object with two fields:

```json
{
  "num_qubits": 2,
  "elements": [
    {"type": "gate", "gate": "H", "targets": [0]},
    {"type": "gate", "gate": "X", "targets": [1], "controls": [0]}
  ]
}
```

- `num_qubits`: the integer number of qubits. (The underlying `Circuit` type
  also supports qudit dimensions, but the CLI simulator is qubit-only.)
- `elements`: an ordered array of circuit elements applied left-to-right.

Each element has a `type` field. For CLI examples, `type` is always
`"gate"`; the `"annotation"` and `"channel"` variants exist for diagram
labels and noise channels but are not used in the example catalog.

## Gate elements

A gate element has these fields:

| Field | Required? | Type | Meaning |
|---|---|---|---|
| `type` | yes | `"gate"` | Marks this as an applied gate. |
| `gate` | yes | string | Gate name (see table below). |
| `targets` | yes | `int[]` | Qubits the gate matrix acts on. |
| `controls` | no | `int[]` | Control qubits. Omit for uncontrolled gates. |
| `control_configs` | no | `bool[]` | One flag per control; `true` = trigger on \|1⟩, `false` = trigger on \|0⟩. Defaults to all `true`. |
| `params` | for `Rx`, `Ry`, `Rz`, `Phase` | `float[]` | Gate parameters in radians. |

Worked example — Bell state (H on qubit 0, then CNOT with 0 as control and
1 as target):

```json
{
  "num_qubits": 2,
  "elements": [
    {"type": "gate", "gate": "H", "targets": [0]},
    {"type": "gate", "gate": "X", "targets": [1], "controls": [0]}
  ]
}
```

The `CNOT` gate is not a primitive; it is an `X` with a control.

## Gate names

| Name | Arity | Description |
|---|---|---|
| `X`, `Y`, `Z` | 1 | Pauli gates. |
| `H` | 1 | Hadamard. |
| `S`, `T` | 1 | Diagonal phase gates (S = diag(1, i); T = diag(1, e^{iπ/4})). |
| `SqrtX`, `SqrtY`, `SqrtW` | 1 | √X, √Y, √W gates. |
| `Rx`, `Ry`, `Rz` | 1 | Rotations about X/Y/Z; `params: [theta]` in radians. |
| `Phase` | 1 | Diagonal phase gate; `params: [phi]`; matrix diag(1, e^{iφ}). |
| `SWAP` | 2 | Swap two qubits. |
| `ISWAP` | 2 | iSWAP entangling gate. |
| `FSim` | 2 | fSim(θ, φ); `params: [theta, phi]`. |

For the full Rust-side enum and matrix definitions, see
[Gates](./gates.md).

## Bit ordering (little-endian)

The basis state labeled by integer `k` is
\\( \|q_{n-1}\,q_{n-2}\,\dots\,q_1\,q_0\rangle \\) with `q_0` the least
significant bit. Probability arrays and measurement samples follow the same
order: `probabilities[5]` on a three-qubit register is the probability of
\\( \|101\rangle \\), which means \\( q_2=1,\,q_1=0,\,q_0=1 \\).

This bites everyone once. When a circuit acts on qubit `0` it is acting on
the *least-significant* qubit in the basis-state label.

## Result JSON

`yao simulate` produces a state vector; `yao probs` post-processes into:

```json
{"num_qubits": 2, "locs": null, "probabilities": [0.5, 0.0, 0.0, 0.5]}
```

`yao run --shots N` produces measurement samples:

```json
{"num_qubits": 2, "samples": [0, 3, 0, 3, 3, 0, ...]}
```

`yao run --op "..."` produces an expectation value:

```json
{"operator": "Z(0)Z(1)", "value": -1.0}
```

## Operator syntax

For `yao run --op` and related commands, the operator string is a product
of single-qubit Paulis. `Z(0)Z(1)` is \\( Z_0 \otimes Z_1 \\) extended by
identity on every other qubit. Supported single-qubit symbols are `I`,
`X`, `Y`, `Z`. Real coefficients can be prepended, e.g. `0.5*Z(0)`.

## Where to get JSON

- `yao example <name>` emits a ready-made circuit for common cases
  (`bell`, `ghz`, `qft`). See [CLI Tool](./cli.md) for the catalog.
- `yao fromqasm circuit.qasm` imports an OpenQASM 2.0 file.
- Hand-write the JSON in any editor.
- Generate the JSON programmatically from any language — the schema is
  plain data, so `json.dump(...)` from Python, `JSON.stringify` from
  JavaScript, or `jq` pipelines all work.
