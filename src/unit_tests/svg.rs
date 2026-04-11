use crate::circuit::{Circuit, put};
use crate::gate::Gate;

#[test]
fn renders_basic_h_gate_to_svg() {
    let circuit = Circuit::new(vec![2], vec![put(vec![0], Gate::H)]).unwrap();

    let svg = crate::svg::to_svg(&circuit);

    assert!(svg.starts_with("<svg"));
    assert!(svg.contains(">H</text>"));
    assert!(svg.contains("viewBox="));
}
