# Formats & conventions

This page defines the data exchanged by the CLI. Text results are human-readable
in a terminal and JSON when piped; `--json` forces JSON. Saved states from
`simulate` are always binary, including density matrices for noisy circuits.

## Circuit JSON

A circuit specifies the qubit count and an ordered list of operations:

```json
{
  "num_qubits": 2,
  "elements": [
    {"type": "gate", "gate": "H", "targets": [0]},
    {"type": "gate", "gate": "X", "targets": [1], "controls": [0]}
  ]
}
```

Operations execute in list order. Site indices start at zero. JSON circuits use
qubits; use the Rust `Circuit` API for higher-dimensional sites.

## Gate elements

| Field | Required | Meaning |
|---|---|---|
| `type` | Yes | `"gate"` |
| `gate` | Yes | A [gate name](gates.md#gate-names) |
| `targets` | Yes | Target site indices |
| `controls` | No | Control site indices |
| `control_configs` | No | One boolean per control; defaults to all `true` (trigger on 1) |
| `params` | For parameterized gates | Angles in radians, in the order listed in the gate reference |
| `matrix` | For `Custom` | Rows of complex entries, each encoded as `[real, imaginary]` |
| `is_diagonal` | No | Custom gate diagonal optimization; defaults to `false` |
| `label` | No | Custom gate display label |

`CNOT` and `CX` are aliases for `X`; specify `controls` explicitly.
Visual annotations use `type: "label"`, as shown in the
[visualization guide](visualization.md#label-a-step).

Noise elements use `type: "channel"`, `locs`, a `channel` name, and its
parameters. For example:

```json
{"type": "channel", "locs": [0], "channel": "BitFlip", "p": 0.01}
```

The CLI automatically selects a density matrix for circuits containing channels.
Use `yao toeinsum --mode dm` for density-matrix tensor export, or
[trajectories](trajectories.md) for stochastic observable estimates. The [noise API](api/yao_rs/noise/index.html)
describes the available channel parameters.

## Bit ordering

Qubit 0 is the **most significant bit** in a state-vector index. Read a basis
label from left to right as \\( |q_0 q_1 \dots q_{n-1}\rangle \\):

\\[ k = \sum_{i=0}^{n-1} q_i 2^{n-1-i}. \\]

For three qubits, flipping qubits 0 and 1 prepares `110`, at array index 6.
This convention also determines the order of the probability array.
For a subset of sites, `locs` determines the order of the returned bits.

## Result JSON

`yao probs` returns probabilities indexed by basis state (values below are rounded):

```json
{"num_qubits": 2, "locs": null, "probabilities": [0.5, 0.0, 0.0, 0.5]}
```

`yao run --shots` and `yao measure` return individual bit arrays and their counts.
One possible two-shot result is:

```json
{
  "num_qubits": 2,
  "shots": 2,
  "locs": null,
  "counts": {"00": 1, "11": 1},
  "outcomes": [[0, 0], [1, 1]]
}
```

`yao run --op` and `yao expect` return a complex expectation value:

```json
{"operator": "Z(0)Z(1)", "expectation_value": {"re": 1.0, "im": 0.0}}
```

## Operator syntax

The `--op` argument accepts sums of weighted operator products, such as
`"0.5*Z(0)Z(1) + X(0)"`, including tensor-network expectations. Each term must
use distinct, in-range sites.

### Supported Operators

| Name | Matrix | Description |
|------|--------|-------------|
| `I` | identity | Identity |
| `X` | \|0><1\| + \|1><0\| | Pauli X |
| `Y` | -i\|0><1\| + i\|1><0\| | Pauli Y |
| `Z` | \|0><0\| - \|1><1\| | Pauli Z |
| `P0` | \|0><0\| | Projector onto \|0> |
| `P1` | \|1><1\| | Projector onto \|1> |
| `Pu` | \|0><1\| | Raising operator (sigma+) |
| `Pd` | \|1><0\| | Lowering operator (sigma-) |

### Syntax

```
term [+/- term ...]
term = [coeff *] Op(site)[Op(site)...]
```


## State file format

State files use a compact binary format with a JSON header:

```
[JSON header line]\n
[binary payload: Complex64 array in little-endian]
```

Header example:

```json
{"format":"yao-state-v1","num_qubits":4,"dims":[2,2,2,2],"num_elements":16,"dtype":"complex128"}
```

Each complex amplitude is stored as two 64-bit little-endian floats (real, imaginary), 16 bytes per element. The total binary payload size is `num_elements * 16` bytes.
Pure states use `yao-state-v1` with `2^num_qubits` entries. Density matrices use
`yao-density-v1` with `4^num_qubits` entries in row-major order; `dims` still lists
one dimension per physical qubit. Headers must agree on dimensions and element
count. Truncated payloads and non-finite entries are rejected.


## Tensor network JSON

Unsliced tensor networks use `format: "yao-tn-v1"`. Sliced plans use
`yao-tn-v2`; see [observables and memory controls](tensor-memory.md).

| Field | Meaning |
|---|---|
| `mode` | `"pure"` or `"dm"`; overlap and state exports also serialize as `"pure"` |
| `eincode.input_indices` | A list of index labels for each tensor |
| `eincode.output_indices` | Open indices of the result |
| `tensors` | Tensor objects with `shape`, `data_re`, and `data_im` |
| `size_dict` | The dimension of each index label |
| `contraction_order` | Nested contraction tree added by `yao optimize`; initially `null` |

Labels are strings. Density-matrix exports use negative labels for bra indices.
Tensor data is flattened in row-major order, with real and imaginary values
stored separately. See [tensor networks](tensor-networks.md) to choose boundaries
and contract a network.
