use crate::circuit::{Circuit, CircuitElement, PositionedGate, control, label, put};
use crate::gate::Gate;

#[test]
fn renders_basic_h_gate_to_svg() {
    let circuit = Circuit::new(vec![2], vec![put(vec![0], Gate::H)]).unwrap();

    let svg = circuit.to_svg();

    assert!(svg.starts_with("<svg"));
    assert!(svg.contains("<line "));
    assert!(svg.contains(">H</text>"));
    assert!(svg.contains("viewBox="));
    assert!(svg.ends_with("</svg>"));
    assert_eq!(svg, crate::svg::to_svg(&circuit));
}

#[test]
fn renders_controlled_x_with_connector_and_target_marker() {
    let circuit = Circuit::new(vec![2, 2], vec![control(vec![0], vec![1], Gate::X)]).unwrap();
    let svg = crate::svg::to_svg(&circuit);

    assert!(svg.contains("class=\"control\""));
    assert!(svg.contains("class=\"target-x\""));
    assert!(svg.contains("class=\"control-link\""));
}

#[test]
fn renders_active_low_controls_as_open_circles() {
    let gate = PositionedGate::new(Gate::X, vec![1], vec![0], vec![false]);
    let circuit = Circuit::new(vec![2, 2], vec![CircuitElement::Gate(gate)]).unwrap();
    let svg = crate::svg::to_svg(&circuit);

    assert!(svg.contains("class=\"control-open\""));
}

#[test]
fn escapes_label_text_for_xml() {
    let circuit = Circuit::new(vec![2], vec![label(0, "<Bell & test>")]).unwrap();
    let svg = crate::svg::to_svg(&circuit);

    assert!(svg.contains("&lt;Bell &amp; test&gt;"));
}
