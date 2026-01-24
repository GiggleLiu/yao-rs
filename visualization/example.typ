#import "circuit.typ": render-circuit

// Custom gate styling
#let my-style(gate-name) = {
  if gate-name == "H" { (label: $H$, fill: blue.lighten(80%)) }
  else if gate-name == "Phase" { (fill: orange.lighten(80%)) }
  else if gate-name == "Custom" { (label: "U", fill: purple.lighten(80%)) }
  else { (label: gate-name) }
}

= QFT Circuit (3 qubits)
#render-circuit("example-qft.json", gate-style: my-style)

= Basic Gates
#render-circuit("example-basic.json")
