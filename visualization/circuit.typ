#import "@preview/quill:0.7.2": *

/// Render a quantum circuit from parsed JSON data.
///
/// Parameters:
/// - data: Parsed JSON object with fields `num_qubits` and `elements`
///   where elements is an array of gates and labels
/// - gate-style: Function mapping gate name (string) to a dictionary of
///   styling options (label, fill, stroke, etc.)
#let round2(x) = calc.round(x, digits: 2)

#let param-label(name, params) = {
  let vals = params.map(x => str(round2(x)))
  name + "(" + vals.join(", ") + ")"
}

#let render-circuit-impl(data, gate-style: gate-name => (label: gate-name)) = {
  let n = data.num_qubits
  let ops = ()

  for entry in data.elements {
    // Handle label annotations
    if entry.type == "label" {
      // Render floating text on the qubit wire at entry.loc
      // Using gate with transparent fill/stroke so labels are not rendered as gates
      ops.push(tequila.gate(entry.loc, entry.text, fill: none, stroke: none))
      continue
    }

    // For gate type, extract gate-specific fields
    let gate-name = entry.gate
    let targets = entry.targets
    let controls = entry.at("controls", default: none)
    let params = entry.at("params", default: none)
    let label = entry.at("label", default: gate-name)
    let style = gate-style(gate-name)
    let fill = style.at("fill", default: none)
    // For parametric gates, default label includes the angle
    let display-label = if gate-name == "Phase" and params != none and style.at("label", default: none) == none {
      $phi.alt (#str(round2(params.at(0))))$
    } else if params != none and style.at("label", default: none) == none {
      param-label(label, params)
    } else {
      style.at("label", default: label)
    }

    if controls == none or controls.len() == 0 {
      // Uncontrolled gates
      if targets.len() == 1 {
        let t = targets.at(0)
        if gate-name == "H" {
          ops.push(tequila.h(t))
        } else if gate-name == "X" {
          ops.push(tequila.x(t))
        } else if gate-name == "Y" {
          ops.push(tequila.y(t))
        } else if gate-name == "Z" {
          ops.push(tequila.z(t))
        } else if gate-name == "S" {
          ops.push(tequila.s(t))
        } else if gate-name == "T" {
          ops.push(tequila.t(t))
        } else if gate-name == "SqrtX" {
          ops.push(tequila.sx(t))
        } else if gate-name == "Phase" and params != none {
          ops.push(tequila.p(round2(params.at(0)), t))
        } else if gate-name == "Rx" and params != none {
          ops.push(tequila.rx(round2(params.at(0)), t))
        } else if gate-name == "Ry" and params != none {
          ops.push(tequila.ry(round2(params.at(0)), t))
        } else if gate-name == "Rz" and params != none {
          ops.push(tequila.rz(round2(params.at(0)), t))
        } else {
          // Generic single-qubit gate
          if fill != none {
            ops.push(tequila.gate(t, display-label, fill: fill))
          } else {
            ops.push(tequila.gate(t, display-label))
          }
        }
      } else if targets.len() == 2 {
        let t0 = targets.at(0)
        let t1 = targets.at(1)
        if gate-name == "SWAP" {
          ops.push(tequila.swap(t0, t1))
        } else {
          // Multi-qubit gate (FSim, ISWAP, Custom, etc.)
          if fill != none {
            ops.push(tequila.mqgate(t0, n: t1 - t0 + 1, display-label, fill: fill))
          } else {
            ops.push(tequila.mqgate(t0, n: t1 - t0 + 1, display-label))
          }
        }
      } else {
        // 3+ qubit gate
        let t-min = calc.min(..targets)
        let t-max = calc.max(..targets)
        if fill != none {
          ops.push(tequila.mqgate(t-min, n: t-max - t-min + 1, display-label, fill: fill))
        } else {
          ops.push(tequila.mqgate(t-min, n: t-max - t-min + 1, display-label))
        }
      }
    } else {
      // Controlled gates
      if targets.len() == 1 and controls.len() == 1 {
        let ctrl = controls.at(0)
        let t = targets.at(0)
        if gate-name == "X" {
          ops.push(tequila.cx(ctrl, t))
        } else if gate-name == "Z" {
          ops.push(tequila.cz(ctrl, t))
        } else {
          // Generic single-controlled gate (ctrl dot + gate box)
          if fill != none {
            ops.push(tequila.ca(ctrl, t, display-label, fill: fill))
          } else {
            ops.push(tequila.ca(ctrl, t, display-label))
          }
        }
      } else if controls.len() == 1 {
        // Single control, multi-target
        let ctrl = controls.at(0)
        let all-locs = controls + targets
        let q-min = calc.min(..all-locs)
        let q-max = calc.max(..all-locs)
        if fill != none {
          ops.push(tequila.mqgate(q-min, n: q-max - q-min + 1, display-label, fill: fill))
        } else {
          ops.push(tequila.mqgate(q-min, n: q-max - q-min + 1, display-label))
        }
      } else if controls.len() == 2 and targets.len() == 1 {
        // Double control (Toffoli-like)
        let t = targets.at(0)
        if gate-name == "X" {
          ops.push(tequila.ccx(controls.at(0), controls.at(1), t))
        } else if gate-name == "Z" {
          ops.push(tequila.ccz(controls.at(0), controls.at(1), t))
        } else {
          ops.push(tequila.cca(controls.at(0), controls.at(1), t, display-label))
        }
      } else {
        // General multi-controlled gate
        let t = targets.at(0)
        if fill != none {
          ops.push(tequila.multi-controlled-gate(controls, t, gate.with(display-label, fill: fill)))
        } else {
          ops.push(tequila.multi-controlled-gate(controls, t, gate.with(display-label)))
        }
      }
    }
  }

  quantum-circuit(..tequila.build(n: n, ..ops))
}

/// Render a quantum circuit from a JSON file.
///
/// Parameters:
/// - filename: Path to the JSON file (relative to the typst document)
/// - gate-style: Function mapping gate name to styling dictionary
#let render-circuit(filename, gate-style: gate-name => (label: gate-name)) = {
  let data = json(filename)
  render-circuit-impl(data, gate-style: gate-style)
}
