# Annotation Gate Design

Issue: #16 - Add helper block to allow users annotate a line

## Summary

Add a separate `Annotation` type (not a Gate variant) for visual markers in circuit diagrams. Annotations are no-ops in execution but render as floating labels on qubit wires in PDF output.

## Core Types

```rust
// src/circuit.rs (or new src/annotation.rs)

/// Annotation variants for circuit visualization
#[derive(Debug, Clone, PartialEq)]
pub enum Annotation {
    /// A text label displayed on the circuit diagram
    Label(String),
}

/// An annotation placed at a specific qubit location
#[derive(Debug, Clone)]
pub struct PositionedAnnotation {
    pub annotation: Annotation,
    pub loc: usize,  // single qubit only
}

/// Elements that can appear in a circuit sequence
#[derive(Debug, Clone)]
pub enum CircuitElement {
    Gate(PositionedGate),
    Annotation(PositionedAnnotation),
}
```

## Circuit Structure Change

```rust
pub struct Circuit {
    pub dims: Vec<usize>,
    pub elements: Vec<CircuitElement>,  // was: gates: Vec<PositionedGate>
}
```

## Builder API

```rust
// All builders return CircuitElement directly (breaking change, acceptable pre-1.0)

pub fn label(loc: usize, text: impl Into<String>) -> CircuitElement {
    CircuitElement::Annotation(PositionedAnnotation {
        annotation: Annotation::Label(text.into()),
        loc,
    })
}

pub fn put(target_locs: Vec<usize>, gate: Gate) -> CircuitElement {
    CircuitElement::Gate(PositionedGate::new(gate, target_locs, vec![], vec![]))
}

pub fn control(ctrl_locs: Vec<usize>, target_locs: Vec<usize>, gate: Gate) -> CircuitElement {
    let configs = vec![true; ctrl_locs.len()];
    CircuitElement::Gate(PositionedGate::new(gate, target_locs, ctrl_locs, configs))
}
```

**Usage:**
```rust
let circuit = Circuit::new(vec![2, 2], vec![
    put(vec![0], Gate::H),
    label(0, "Bell prep"),
    control(vec![0], vec![1], Gate::X),
]).unwrap();
```

## Validation

Annotations only require `loc < dims.len()`. Reuse existing `CircuitError::LocOutOfRange`.

```rust
impl Circuit {
    pub fn new(dims: Vec<usize>, elements: Vec<CircuitElement>) -> Result<Self, CircuitError> {
        let num_sites = dims.len();
        for element in &elements {
            match element {
                CircuitElement::Gate(pg) => { /* existing validation */ }
                CircuitElement::Annotation(pa) => {
                    if pa.loc >= num_sites {
                        return Err(CircuitError::LocOutOfRange {
                            loc: pa.loc,
                            num_sites,
                        });
                    }
                }
            }
        }
        Ok(Circuit { dims, elements })
    }
}
```

## Execution Behavior

Annotations are no-ops in all execution paths:

**apply.rs:**
```rust
for element in &circuit.elements {
    match element {
        CircuitElement::Gate(pg) => { /* apply gate */ }
        CircuitElement::Annotation(_) => { /* skip */ }
    }
}
```

**einsum.rs:**
```rust
for element in &circuit.elements {
    match element {
        CircuitElement::Gate(pg) => { /* create tensor */ }
        CircuitElement::Annotation(_) => { /* skip, no tensor */ }
    }
}
```

## JSON Serialization

```rust
#[derive(Serialize, Deserialize)]
#[serde(tag = "type")]
enum ElementJson {
    #[serde(rename = "gate")]
    Gate(GateJson),
    #[serde(rename = "label")]
    Label { text: String, loc: usize },
}

#[derive(Serialize, Deserialize)]
struct CircuitJson {
    num_qubits: usize,
    elements: Vec<ElementJson>,  // was: gates
}
```

## Typst Rendering

Update `visualization/circuit.typ` to handle label elements:

```typst
for entry in data.elements {
  if entry.type == "label" {
    // Render floating text on qubit entry.loc
    // Use quill's annotation/label functionality
  } else {
    // existing gate rendering logic
  }
}
```

Label renders as floating text above/on the qubit wire at that position in the circuit.

## Files to Modify

1. `src/circuit.rs` - Add `Annotation`, `PositionedAnnotation`, `CircuitElement`; change `Circuit.gates` to `Circuit.elements`; update `put()`, `control()` return types; add `label()` builder
2. `src/apply.rs` - Match on `CircuitElement`, skip annotations
3. `src/einsum.rs` - Match on `CircuitElement`, skip annotations
4. `src/json.rs` - Update serialization for `CircuitElement` with tagged enum
5. `src/typst.rs` - No changes needed (just passes JSON)
6. `visualization/circuit.typ` - Handle label elements in rendering loop
7. `src/lib.rs` - Export new types and `label()` function
8. Tests - Update all tests using `Circuit::new` to use new element-based API
