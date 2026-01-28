# Visualization Design (Issue #5)

Dump/load circuit to/from JSON, and render with quill in typst.

## JSON Schema

Pure circuit data, no styling:

```json
{
  "num_qubits": 4,
  "gates": [
    { "gate": "H", "targets": [0] },
    { "gate": "Phase", "params": [1.5708], "targets": [1] },
    { "gate": "X", "targets": [1], "controls": [0], "control_configs": [true] },
    {
      "gate": "Custom",
      "label": "U_oracle",
      "targets": [0, 1],
      "matrix": [[[1.0, 0.0], [0.0, 0.0], ...], ...],
      "is_diagonal": false
    },
    { "gate": "FSim", "params": [1.5708, 0.5236], "targets": [2, 3] }
  ]
}
```

### Gate Entry Fields

- `gate`: String — gate type name ("H", "X", "Phase", "Rx", "Custom", "FSim", etc.)
- `params`: Optional `Vec<f64>` — for parametric gates (Phase, Rx, Ry, Rz, FSim)
- `targets`: `Vec<usize>` — target qubit indices
- `controls`: Optional `Vec<usize>` — control qubit indices
- `control_configs`: Optional `Vec<bool>` — active-high (true=|1⟩) per control
- `label`: Optional String — required for Custom gates, display name
- `matrix`: Optional `Vec<Vec<[f64; 2]>>` — row-major complex matrix, each element [re, im]. Required for Custom gates.
- `is_diagonal`: Optional bool — for Custom gates

## Rust Side (`src/json.rs`)

Dependencies: `serde`, `serde_json`

### Serde Types

```rust
#[derive(Serialize, Deserialize)]
struct CircuitJson {
    num_qubits: usize,
    gates: Vec<GateJson>,
}

#[derive(Serialize, Deserialize)]
struct GateJson {
    gate: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    params: Option<Vec<f64>>,
    targets: Vec<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    controls: Option<Vec<usize>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    control_configs: Option<Vec<bool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    matrix: Option<Vec<Vec<[f64; 2]>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    is_diagonal: Option<bool>,
}
```

### Public API

```rust
pub fn circuit_to_json(circuit: &Circuit) -> String
pub fn circuit_from_json(json: &str) -> Result<Circuit, ...>
```

Full round-trip: Custom gates store their complete complex matrix in JSON.

## Typst Script (`visualization/circuit.typ`)

Uses quill's tequila submodule for automatic layout.

```typst
#import "@preview/quill:0.5.0": *

// Implementation — takes parsed JSON data
#let render-circuit-impl(data, gate-style: gate-name => (label: gate-name)) = {
  // maps JSON gates to tequila ops, applying gate-style() for display
  // ...
}

// Convenience — takes filename, reads JSON
#let render-circuit(filename, gate-style: gate-name => (label: gate-name)) = {
  let data = json(filename)
  render-circuit-impl(data, gate-style: gate-style)
}
```

### Gate Mapping (JSON → tequila)

- Named single-qubit: `H` → `tq.h(target)`, `X` → `tq.x(target)`, etc.
- Parametric: `Rx(θ)` → `tq.rx(θ, target)`, `Phase(θ)` → `tq.p(θ, target)`
- Controlled-X → `tq.cx(ctrl, target)`, Controlled-Z → `tq.cz(ctrl, target)`
- SWAP → `tq.swap(a, b)`
- Multi-qubit / Custom → `tq.mqgate(label, targets)` with gate-style applied
- FSim, SqrtX, etc. → `tq.gate(label, targets)` with gate-style applied

### Usage Example

```typst
#let my-style(gate-name) = {
  if gate-name == "H" { (label: $H$, fill: blue.lighten(80%)) }
  else if gate-name == "Phase" { (label: $P$, fill: orange.lighten(80%)) }
  else if gate-name == "U_oracle" { (label: "Oracle", fill: purple.lighten(80%), stroke: (dash: "dashed")) }
  else { (label: gate-name) }
}

#render-circuit("circuit.json", gate-style: my-style)
```

## Testing

1. **Round-trip**: `circuit_to_json` → `circuit_from_json` produces equivalent Circuit for all gate types
2. **Named gates**: correct JSON field mapping (gate name, params)
3. **Custom gates**: matrix serialization preserves complex values (f64 precision)
4. **Controls**: control_locs and control_configs serialize/deserialize correctly
5. **Typst script**: manual verification — export QFT circuit JSON, render with typst, visually confirm
