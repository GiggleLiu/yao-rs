use crate::circuit::{Circuit, CircuitElement};

/// Render a circuit as a minimal SVG.
///
/// This Task 1 implementation only emits a valid SVG shell with one wire and
/// the first gate label it encounters. Later tasks will expand layout and
/// placement-aware rendering.
pub fn to_svg(circuit: &Circuit) -> String {
    let width = 120;
    let height = 40;
    let wire_y = 20;

    let label = circuit
        .elements
        .iter()
        .find_map(|element| match element {
            CircuitElement::Gate(pg) => Some(pg.gate.to_string()),
            _ => None,
        })
        .unwrap_or_default();

    format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}"><line x1="0" y1="{wire_y}" x2="{width}" y2="{wire_y}" stroke="black"/><text x="60" y="24" text-anchor="middle">{label}</text></svg>"#
    )
}

#[cfg(test)]
#[path = "unit_tests/svg.rs"]
mod tests;
