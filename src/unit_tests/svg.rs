use crate::circuit::{Circuit, CircuitElement, PositionedGate, channel, control, label, put};
use crate::gate::Gate;
use crate::noise::NoiseChannel;
use ndarray::Array2;
use num_complex::Complex64;

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

#[test]
fn does_not_panic_on_valid_targetless_gate() {
    let gate = Gate::Custom {
        matrix: Array2::from_shape_vec((1, 1), vec![Complex64::new(1.0, 0.0)]).unwrap(),
        is_diagonal: true,
        label: "ScalarPhase".to_string(),
    };
    let circuit = Circuit::new(vec![2], vec![put(vec![], gate)]).unwrap();

    let svg = crate::svg::to_svg(&circuit);

    assert!(svg.starts_with("<svg"));
    assert!(svg.contains("ScalarPhase"));
}

#[test]
fn renders_swap_with_two_markers_and_connector() {
    let circuit = Circuit::new(vec![2, 2], vec![put(vec![0, 1], Gate::SWAP)]).unwrap();
    let svg = crate::svg::to_svg(&circuit);

    assert_eq!(count_occurrences(&svg, "class=\"swap-marker\""), 4);
    assert_eq!(count_occurrences(&svg, "class=\"control-link\""), 1);
}

#[test]
fn renders_multi_target_gate_as_tall_box() {
    let circuit = Circuit::new(vec![2, 2, 2], vec![put(vec![0, 2], Gate::ISWAP)]).unwrap();
    let svg = crate::svg::to_svg(&circuit);
    let height = extract_attr_from_tag(&svg, "data-label=\"ISWAP\"", "height");

    assert!(height > 28.0);
}

#[test]
fn renders_annotation_above_the_target_wire() {
    let circuit = Circuit::new(vec![2], vec![label(0, "Bell prep")]).unwrap();
    let svg = crate::svg::to_svg(&circuit);
    let annotation_y = extract_attr_from_tag(&svg, "class=\"annotation-label\"", "y");
    let wire_y = extract_attr_from_tag(&svg, "class=\"wire\"", "y1");

    assert!(svg.contains(">Bell prep</text>"));
    assert!(annotation_y < wire_y);
}

#[test]
fn renders_channel_as_dashed_box_with_label() {
    let circuit = Circuit::new(
        vec![2],
        vec![channel(
            vec![0],
            NoiseChannel::PhaseAmplitudeDamping {
                amplitude: 0.2,
                phase: 0.1,
                excited_population: 0.0,
            },
        )],
    )
    .unwrap();
    let svg = crate::svg::to_svg(&circuit);

    assert!(svg.contains("class=\"channel-box\""));
    assert!(svg.contains(">PhaseAmplitudeDamping</text>"));
}

#[test]
fn emits_one_wire_per_site_and_one_gate_box_per_column() {
    let circuit = Circuit::new(
        vec![2, 2, 2],
        vec![put(vec![0], Gate::H), put(vec![2], Gate::Z)],
    )
    .unwrap();
    let svg = crate::svg::to_svg(&circuit);

    assert_eq!(count_occurrences(&svg, "class=\"wire\""), 3);
    assert_eq!(count_occurrences(&svg, "class=\"gate-box\""), 2);
}

#[test]
fn widens_gate_box_and_viewbox_for_long_labels() {
    let circuit = Circuit::new(vec![2], vec![put(vec![0], Gate::Phase(1.2345))]).unwrap();
    let svg = crate::svg::to_svg(&circuit);
    let gate_width = extract_attr_from_tag(&svg, "data-label=\"Phase(1.2345)\"", "width");
    let viewbox_width = extract_viewbox_width(&svg);

    assert!(gate_width > 42.0);
    assert!(viewbox_width > 136.0);
}

#[test]
fn widens_channel_column_for_long_channel_labels() {
    let circuit = Circuit::new(
        vec![2],
        vec![channel(
            vec![0],
            NoiseChannel::PhaseAmplitudeDamping {
                amplitude: 0.2,
                phase: 0.1,
                excited_population: 0.0,
            },
        )],
    )
    .unwrap();
    let svg = crate::svg::to_svg(&circuit);
    let viewbox_width = extract_viewbox_width(&svg);

    assert!(viewbox_width > 136.0);
}

fn count_occurrences(haystack: &str, needle: &str) -> usize {
    haystack.match_indices(needle).count()
}

fn extract_attr_from_tag(svg: &str, marker: &str, attr: &str) -> f32 {
    let marker_start = svg.find(marker).unwrap();
    let tag_start = svg[..marker_start].rfind('<').unwrap();
    let tag_end = svg[marker_start..].find('>').unwrap() + marker_start;
    let tag = &svg[tag_start..=tag_end];
    let attr_start = tag.find(&format!("{attr}=\"")).unwrap() + attr.len() + 2;
    let attr_end = tag[attr_start..].find('"').unwrap() + attr_start;

    tag[attr_start..attr_end].parse().unwrap()
}

fn extract_viewbox_width(svg: &str) -> f32 {
    let viewbox_start = svg.find("viewBox=\"").unwrap() + "viewBox=\"".len();
    let viewbox_end = svg[viewbox_start..].find('"').unwrap() + viewbox_start;
    let parts: Vec<&str> = svg[viewbox_start..viewbox_end].split_whitespace().collect();

    parts[2].parse().unwrap()
}
