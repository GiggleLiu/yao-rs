# CLI Tool

The `yao` CLI provides a command-line interface for quantum circuit simulation, measurement, and tensor network export. It wraps the yao-rs library so you can work with circuits without writing Rust code.

## Installation

Build from source (requires Rust toolchain):

```bash
cargo install --path yao-cli
```

## Quick Start: Bell State

Create a circuit file `bell.json`:

```json
{
  "num_qubits": 2,
  "elements": [
    { "type": "gate", "gate": "H", "targets": [0] },
    { "type": "gate", "gate": "CNOT", "targets": [1], "controls": [0] }
  ]
}
```

Inspect, simulate, and measure:

```bash
# View circuit structure
yao inspect bell.json

# Simulate and measure in one step
yao run bell.json --shots 1024

# Compute expectation value
yao run bell.json --op "Z(0)Z(1)"

# Pipeline: simulate, then compute probabilities
yao simulate bell.json | yao probs -
```

## Output Modes

Output is human-readable in a terminal, JSON when piped. Use `--json` to force JSON in interactive mode.

```bash
yao inspect bell.json              # human-readable
yao inspect bell.json --json       # force JSON
yao inspect bell.json | jq .       # auto-JSON when piped
```

Global flags available on all commands:

| Flag | Description |
|------|-------------|
| `--json` | Force JSON output |
| `-q`, `--quiet` | Suppress informational messages on stderr |
| `-o`, `--output <file>` | Write output to file |

## Commands

### `yao inspect`

Display circuit information: qubit count, gate count, gate list.

```bash
yao inspect circuit.json
yao inspect circuit.json --json
cat circuit.json | yao inspect -
```

### `yao simulate`

Simulate a circuit and output the resulting quantum state.

```bash
yao simulate circuit.json --output state.bin
yao simulate circuit.json --input initial.bin --output final.bin
yao simulate circuit.json | yao measure - --shots 100
```

Without `--output`, writes binary state data to stdout (suitable for piping to other commands).

| Option | Description |
|--------|-------------|
| `--input <file>` | Input state file (defaults to \|0...0>) |
| `--output <file>` | Save state to file |

### `yao measure`

Sample measurement outcomes from a state.

```bash
yao measure state.bin --shots 1024
yao measure state.bin --shots 100 --locs 0,1
yao simulate circuit.json | yao measure - --shots 1024
```

| Option | Description |
|--------|-------------|
| `--shots <N>` | Number of measurement shots (default: 1024) |
| `--locs <i,j,...>` | Qubit indices for partial measurement (comma-separated) |

### `yao probs`

Compute the probability distribution from a state.

```bash
yao probs state.bin
yao probs state.bin --locs 0,1
yao simulate circuit.json | yao probs -
```

| Option | Description |
|--------|-------------|
| `--locs <i,j,...>` | Qubit indices for marginal probabilities (comma-separated) |

### `yao expect`

Compute the expectation value of an operator on a state.

```bash
yao expect state.bin --op "Z(0)"
yao expect state.bin --op "0.5*Z(0)Z(1) + X(0)"
yao simulate circuit.json | yao expect - --op "Z(0)"
```

| Option | Description |
|--------|-------------|
| `--op <expr>` | Operator expression (see [Operator DSL](#operator-dsl) below) |

### `yao run`

All-in-one command: simulate a circuit and optionally post-process, without intermediate files.

```bash
yao run circuit.json --shots 1024
yao run circuit.json --op "Z(0)Z(1)"
yao run circuit.json --shots 100 --locs 0,1
yao run circuit.json --output state.bin
```

| Option | Description |
|--------|-------------|
| `--input <file>` | Input state file (defaults to \|0...0>) |
| `--shots <N>` | Simulate then measure (mutually exclusive with `--op`) |
| `--op <expr>` | Simulate then compute expectation (mutually exclusive with `--shots`) |
| `--locs <i,j,...>` | Qubit indices for partial measurement (used with `--shots`) |
| `--output <file>` | Save final state to file |

Without `--shots`, `--op`, or `--output`, the state is written to stdout in binary format.

### `yao toeinsum`

Export a circuit as a tensor network in einsum format.

```bash
yao toeinsum circuit.json
yao toeinsum circuit.json --output tn.json
yao toeinsum circuit.json --mode dm
yao toeinsum circuit.json --mode overlap
yao toeinsum circuit.json --mode state
yao toeinsum circuit.json --op "Z(0)Z(1)"
```

| Option | Description |
|--------|-------------|
| `--mode <pure\|dm\|overlap\|state>` | Export mode: `pure` (default), `dm` (density matrix), `overlap` (scalar ⟨0\|U\|0⟩), or `state` (state vector with \|0⟩ boundary tensors) |
| `--op <expr>` | Operator expression for expectation value TN (overrides `--mode`) |
| `--output <file>` | Save tensor network JSON to file |

See [Tensor Network JSON Format](#tensor-network-json-format) below for the output schema.

### `yao optimize`

Optimize contraction order for a tensor network. Requires the `omeinsum` feature.

```bash
yao optimize tn.json
yao optimize tn.json --method treesa --ntrials 20
yao toeinsum circuit.json --mode overlap | yao optimize -
```

| Option | Description |
|--------|-------------|
| `--method <greedy\|treesa>` | Optimization method (default: `greedy`) |
| `--alpha <f64>` | [greedy] Output-vs-input size balance weight (default: 0.0) |
| `--temperature <f64>` | [greedy] Temperature for stochastic selection; 0 = deterministic (default: 0.0) |
| `--ntrials <N>` | [treesa] Number of independent SA trials (default: 10) |
| `--niters <N>` | [treesa] Iterations per temperature level (default: 50) |
| `--betas <start:step:stop>` | [treesa] Inverse temperature schedule (default: "0.01:0.05:15.0") |
| `--sc-target <f64>` | [treesa] Space complexity target threshold (default: 20.0) |
| `--tc-weight <f64>` | [treesa] Time complexity weight (default: 1.0) |
| `--sc-weight <f64>` | [treesa] Space complexity weight (default: 1.0) |
| `--rw-weight <f64>` | [treesa] Read-write complexity weight (default: 0.0) |

Adds a `contraction_order` field to the TN JSON, ready for `yao contract`.

### `yao contract`

Contract a pre-optimized tensor network. Requires the `omeinsum` feature. Input must have a `contraction_order` field (produced by `yao optimize`).

```bash
yao toeinsum circuit.json | yao optimize - | yao contract -
yao toeinsum circuit.json --mode overlap | yao optimize - | yao contract -
yao toeinsum circuit.json --op "Z(0)Z(1)" | yao optimize - | yao contract -
```

### `yao fromqasm`

Convert an OpenQASM 2.0 file to circuit JSON. Requires the `qasm` feature.

```bash
yao fromqasm circuit.qasm
yao fromqasm circuit.qasm --output circuit.json
yao fromqasm circuit.qasm | yao run - --shots 1024
```

### `yao toqasm`

Export a circuit as OpenQASM 2.0. Requires the `qasm` feature.

```bash
yao toqasm circuit.json
yao example bell | yao toqasm -
```

### `yao fetch`

Download benchmark circuits from online repositories.

```bash
yao fetch qasmbench list                  # List all circuits
yao fetch qasmbench list --scale small    # List only small circuits
yao fetch qasmbench grover               # Download by name (auto-detect scale)
yao fetch qasmbench qft_n4 -o qft.qasm   # Save to file
yao fetch qasmbench medium/shor_n5        # Explicit scale/name path
```

| Option | Description |
|--------|-------------|
| `--scale <small\|medium\|large>` | Filter by scale (used with `list`) |

Pipeline example:

```bash
yao fetch qasmbench grover | yao fromqasm - | yao run - --shots 100
```

### `yao example`

Print example circuit JSON to stdout.

```bash
yao example bell
yao example bell > bell.json
yao example qft --nqubits 6
```

Available examples: `bell`, `ghz`, `qft`.

| Option | Description |
|--------|-------------|
| `--nqubits <N>` | Number of qubits (default: 2 for bell, 3 for ghz, 4 for qft) |

See the [Example Catalog](./examples/catalog.md) for bash scripts that reproduce algorithm examples through CLI workflows.

### `yao visualize`

Render a circuit diagram as SVG.

```bash
yao visualize circuit.json --output circuit.svg
```

The `--output` flag is required. Only SVG output is supported.

### `yao completions`

Generate shell completion scripts.

```bash
eval "$(yao completions)"        # auto-detect shell
yao completions bash >> ~/.bashrc
yao completions zsh > _yao
```

## Circuit JSON Format

Circuits are specified as JSON with the following structure:

```json
{
  "num_qubits": 2,
  "elements": [
    { "type": "gate", "gate": "H", "targets": [0] },
    { "type": "gate", "gate": "X", "targets": [1], "controls": [0] }
  ]
}
```

Each element has `"type": "gate"` and the following fields:

| Field | Required | Description |
|-------|----------|-------------|
| `gate` | yes | Gate name (see table below) |
| `targets` | yes | Target qubit indices |
| `controls` | no | Control qubit indices |
| `control_configs` | no | Control activation states (default: all `true` = active-high) |
| `params` | no | Gate parameters (for parameterized gates) |
| `matrix` | no | Custom gate matrix (for `Custom` gates) |
| `is_diagonal` | no | Whether custom gate is diagonal (default: `false`) |
| `label` | no | Display label for custom gates |

### Gate Names

| Name | Description | Parameters |
|------|-------------|------------|
| `H` | Hadamard | -- |
| `X`, `CNOT`, `CX` | Pauli X (use with controls for CNOT) | -- |
| `Y` | Pauli Y | -- |
| `Z` | Pauli Z | -- |
| `S` | Phase gate (sqrt Z) | -- |
| `T` | T gate (fourth-root Z) | -- |
| `SWAP` | Swap (2-qubit) | -- |
| `SqrtX` | Square root of X | -- |
| `SqrtY` | Square root of Y | -- |
| `SqrtW` | Square root of W | -- |
| `ISWAP` | iSWAP (2-qubit) | -- |
| `Phase` | Phase shift diag(1, e^{i*theta}) | `params: [theta]` |
| `Rx` | X rotation | `params: [theta]` |
| `Ry` | Y rotation | `params: [theta]` |
| `Rz` | Z rotation | `params: [theta]` |
| `FSim` | Fermionic simulation (2-qubit) | `params: [theta, phi]` |
| `Custom` | Arbitrary unitary | `matrix`, optional `is_diagonal`, `label` |

Controlled gates are specified by adding `controls` to any gate. For example, CNOT is X with a control:

```json
{ "type": "gate", "gate": "X", "targets": [1], "controls": [0] }
```

The aliases `CNOT` and `CX` are accepted as shorthand for X (with controls expected).

Annotations can be added for visualization:

```json
{ "type": "label", "text": "QFT block", "loc": 0 }
```

## Operator DSL

The `--op` flag accepts operator expressions built from Pauli operators and projectors.

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

### Examples

```bash
# Single Pauli
yao run circuit.json --op "Z(0)"

# Multi-site product
yao run circuit.json --op "Z(0)Z(1)"

# Weighted sum
yao run circuit.json --op "0.5*Z(0)Z(1) + 0.3*X(0)"

# Difference
yao run circuit.json --op "X(0)Y(1) - Y(0)X(1)"

# Negative leading term
yao run circuit.json --op "-Z(0)"

# Projectors
yao run circuit.json --op "P0(0) + P1(1)"
```

## State File Format

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

## Tensor Network JSON Format

The `toeinsum` command outputs a JSON tensor network DTO:

```json
{
  "format": "yao-tn-v1",
  "mode": "pure",
  "eincode": {
    "input_indices": [["2", "0"], ["3", "4", "2", "1"]],
    "output_indices": ["3", "4"]
  },
  "tensors": [
    {
      "shape": [2, 2],
      "data_re": [0.707, 0.707, 0.707, -0.707],
      "data_im": [0.0, 0.0, 0.0, 0.0]
    }
  ],
  "size_dict": { "0": 2, "1": 2, "2": 2, "3": 2, "4": 2 },
  "contraction_order": null
}
```

| Field | Description |
|-------|-------------|
| `format` | Always `"yao-tn-v1"` |
| `mode` | `"pure"`, `"dm"`, `"overlap"`, or `"state"` |
| `eincode.input_indices` | Index labels for each tensor (list of lists) |
| `eincode.output_indices` | Open indices of the final state |
| `tensors` | Gate tensors with shape and split real/imaginary data |
| `size_dict` | Maps each index label to its dimension |
| `contraction_order` | Nested binary tree for contraction order (added by `yao optimize`, null otherwise) |

Index labels are strings. In density matrix mode (`dm`), bra indices use negative labels (e.g., `"-1"`, `"-2"`).

## Typical Workflows

### Quick simulation and measurement

```bash
yao run circuit.json --shots 1024
```

### Pipeline with saved state

```bash
yao simulate circuit.json --output state.bin
yao measure state.bin --shots 1000
yao expect state.bin --op "X(0) + Z(0)"
yao probs state.bin --locs 0,1
```

### Pipeline without intermediate files

```bash
yao simulate circuit.json | yao measure - --shots 1024
yao simulate circuit.json | yao probs -
yao simulate circuit.json | yao expect - --op "Z(0)Z(1)"
```

### Tensor network export and contraction

```bash
# Export only
yao toeinsum circuit.json --output tn.json
yao toeinsum circuit.json --mode dm --output tn_dm.json

# Full pipeline: export → optimize → contract
yao toeinsum circuit.json --mode state | yao optimize - | yao contract -
yao toeinsum circuit.json --mode overlap | yao optimize - | yao contract -
yao toeinsum circuit.json --op "Z(0)Z(1)" | yao optimize - | yao contract -
```

### OpenQASM import/export

```bash
yao fromqasm circuit.qasm | yao run - --shots 1024
yao example bell | yao toqasm -
yao fetch qasmbench grover | yao fromqasm - | yao run - --shots 100
```

### Circuit visualization

```bash
yao visualize circuit.json --output circuit.svg
```
