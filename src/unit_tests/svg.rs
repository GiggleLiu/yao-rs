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
    assert_eq!(extract_attr_from_tag(&svg, "xmlns=", "width"), 136.0);
    assert_eq!(extract_attr_from_tag(&svg, "xmlns=", "height"), 56.0);
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
fn packs_disjoint_gates_into_the_same_column() {
    let circuit = Circuit::new(
        vec![2, 2, 2],
        vec![put(vec![0], Gate::H), put(vec![2], Gate::Z)],
    )
    .unwrap();
    let svg = crate::svg::to_svg(&circuit);

    let h_x = extract_attr_from_tag(&svg, "data-label=\"H\"", "x");
    let z_x = extract_attr_from_tag(&svg, "data-label=\"Z\"", "x");
    let viewbox_width = extract_viewbox_width(&svg);

    assert_eq!(h_x, z_x);
    assert_eq!(viewbox_width, 136.0);
}

#[test]
fn keeps_overlapping_gates_in_separate_columns() {
    let circuit =
        Circuit::new(vec![2], vec![put(vec![0], Gate::H), put(vec![0], Gate::X)]).unwrap();
    let svg = crate::svg::to_svg(&circuit);

    let h_x = extract_attr_from_tag(&svg, "data-label=\"H\"", "x");
    let x_x = extract_attr_from_tag(&svg, "data-label=\"X\"", "x");

    assert!(h_x < x_x);
}

#[test]
fn extends_controlled_x_connector_to_marker_edges() {
    let circuit = Circuit::new(vec![2, 2], vec![control(vec![0], vec![1], Gate::X)]).unwrap();
    let svg = crate::svg::to_svg(&circuit);

    let link_y1 = extract_attr_from_tag(&svg, "class=\"control-link\"", "y1");
    let link_y2 = extract_attr_from_tag(&svg, "class=\"control-link\"", "y2");
    let control_y = extract_attr_from_tag(&svg, "class=\"control\"", "cy");
    let target_y = extract_attr_from_tag(&svg, "class=\"target-x\"", "cy");

    assert!(link_y1 <= control_y - super::CONTROL_RADIUS);
    assert!(link_y2 >= target_y + super::TARGET_X_RADIUS);
}

#[test]
fn widens_gate_box_and_viewbox_for_long_labels() {
    let gate = Gate::Custom {
        matrix: Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
        )
        .unwrap(),
        is_diagonal: false,
        label: "LongCustomLabel".to_string(),
    };
    let circuit = Circuit::new(vec![2], vec![put(vec![0], gate)]).unwrap();
    let svg = crate::svg::to_svg(&circuit);
    let gate_width = extract_attr_from_tag(&svg, "data-label=\"LongCustomLabel\"", "width");
    let viewbox_width = extract_viewbox_width(&svg);

    assert!(gate_width > 42.0);
    assert!(viewbox_width > 136.0);
}

#[test]
fn renders_rotation_gate_labels_compactly_with_two_decimals() {
    let circuit = Circuit::new(vec![2], vec![put(vec![0], Gate::Rx(1.2345))]).unwrap();
    let svg = crate::svg::to_svg(&circuit);
    let gate_width = extract_attr_from_tag(&svg, "data-label=\"Rx(1.2345)\"", "width");
    let viewbox_width = extract_viewbox_width(&svg);

    assert!(svg.contains(">Rx</text>"));
    assert!(svg.contains(">1.23</text>"));
    assert!(!svg.contains(">Rx(1.2345)</text>"));
    assert_eq!(gate_width, 42.0);
    assert_eq!(viewbox_width, 136.0);
}

#[test]
fn renders_negative_rotation_gate_parameters_with_two_decimals() {
    let circuit = Circuit::new(vec![2], vec![put(vec![0], Gate::Ry(-0.236))]).unwrap();
    let svg = crate::svg::to_svg(&circuit);

    assert!(svg.contains(">Ry</text>"));
    assert!(svg.contains(">-0.24</text>"));
    assert!(!svg.contains(">Ry(-0.2360)</text>"));
}

#[test]
fn renders_standalone_phase_with_standard_p_label() {
    let circuit = Circuit::new(vec![2], vec![put(vec![0], Gate::Phase(1.2345))]).unwrap();
    let svg = crate::svg::to_svg(&circuit);
    let gate_width = extract_attr_from_tag(&svg, "data-label=\"Phase(1.2345)\"", "width");

    assert!(svg.contains(">P</text>"));
    assert!(svg.contains(">1.23</text>"));
    assert!(!svg.contains(">Phase</text>"));
    assert!(!svg.contains(">Phase(1.2345)</text>"));
    assert!(gate_width < 94.0);
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

#[test]
fn renders_controlled_phase_as_connected_dots_in_either_direction() {
    for (control_site, target_site) in [(0, 1), (1, 0), (0, 2), (2, 0)] {
        let circuit = Circuit::qubits(
            3,
            vec![control(
                vec![control_site],
                vec![target_site],
                Gate::Phase(std::f64::consts::FRAC_PI_2),
            )],
        )
        .unwrap();
        let svg = circuit.to_svg();
        assert_eq!(count_occurrences(&svg, "class=\"control\""), 1);
        assert_eq!(count_occurrences(&svg, "class=\"phase-target\""), 1);
        assert_eq!(count_occurrences(&svg, "class=\"control-link\""), 1);
        assert!(!svg.contains("class=\"gate-box\""));
        assert!(svg.contains(">π/2</text>"));
        let control_y = extract_attr_from_tag(&svg, "class=\"control\"", "cy");
        let target_y = extract_attr_from_tag(&svg, "class=\"phase-target\"", "cy");
        let label_y = extract_attr_from_tag(&svg, "class=\"phase-label\"", "y");
        assert!(label_y > control_y.min(target_y) && label_y < control_y.max(target_y));
        let y1 = extract_attr_from_tag(&svg, "class=\"control-link\"", "y1");
        let y2 = extract_attr_from_tag(&svg, "class=\"control-link\"", "y2");
        assert_eq!(y1, control_y.min(target_y) - super::CONTROL_RADIUS);
        assert_eq!(y2, control_y.max(target_y) + super::CONTROL_RADIUS);
    }
}

#[test]
fn preserves_open_control_for_controlled_phase() {
    let gate = PositionedGate::new(
        Gate::Phase(-std::f64::consts::FRAC_PI_4),
        vec![1],
        vec![0],
        vec![false],
    );
    let circuit = Circuit::qubits(2, vec![CircuitElement::Gate(gate)]).unwrap();
    let svg = circuit.to_svg();
    assert_eq!(count_occurrences(&svg, "class=\"control-open\""), 1);
    assert_eq!(count_occurrences(&svg, "class=\"phase-target\""), 1);
    assert!(!svg.contains("class=\"control\""));
    assert!(svg.contains(">−π/4</text>"));
}

#[test]
fn uses_boxed_p_for_multiple_controls_and_keeps_rz_distinct() {
    let circuit = Circuit::qubits(
        3,
        vec![
            control(vec![0, 1], vec![2], Gate::Phase(std::f64::consts::PI)),
            control(vec![0], vec![1], Gate::Rz(0.5)),
        ],
    )
    .unwrap();
    let svg = circuit.to_svg();
    assert_eq!(count_occurrences(&svg, "class=\"gate-box\""), 2);
    assert!(svg.contains(">P</text>"));
    assert!(svg.contains(">π</text>"));
    assert!(svg.contains(">Rz</text>"));
    assert!(!svg.contains("class=\"phase-target\""));
}

#[test]
fn formats_phase_angles_without_misidentifying_decimal_parameters() {
    use std::f64::consts::PI;
    for (angle, label) in [
        (0.0, "0"),
        (PI, "π"),
        (-PI, "−π"),
        (2.0 * PI, "2π"),
        (PI / 2.0, "π/2"),
        (PI / 4.0, "π/4"),
        (PI / 8.0, "π/8"),
        (PI / 3.0, "π/3"),
        (3.0 * PI / 4.0, "3π/4"),
        (-PI / 1024.0, "−π/1024"),
        (1.2345, "1.23"),
        (PI / 2.0 + 0.0001, "1.57"),
        (0.000001, "1.00e-6"),
    ] {
        assert_eq!(super::phase_angle_label(angle), label, "angle {angle}");
    }
}

#[test]
fn reserves_space_for_long_phase_angles_between_columns() {
    let theta = -std::f64::consts::PI / 1024.0;
    let circuit = Circuit::qubits(
        2,
        vec![
            control(vec![0], vec![1], Gate::Phase(theta)),
            put(vec![0], Gate::H),
        ],
    )
    .unwrap();
    let svg = circuit.to_svg();
    let label_x = extract_attr_from_tag(&svg, "class=\"phase-label\"", "x");
    let h_x = extract_attr_from_tag(&svg, "data-label=\"H\"", "x");
    let label_right = label_x + super::text_width("−π/1024");
    assert!(label_right < h_x);
    assert!(label_right < extract_viewbox_width(&svg));
}
