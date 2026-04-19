use crate::circuit::{Annotation, Circuit, CircuitElement, PositionedChannel, PositionedGate};
use crate::gate::Gate;
use crate::noise::NoiseChannel;

struct LayoutConfig {
    left_pad: f32,
    top_pad: f32,
    col_width: f32,
    row_height: f32,
    gate_width: f32,
}

struct ColumnLayout {
    center_x: f32,
}

struct CircuitLayout {
    elements: Vec<ColumnLayout>,
    total_width: f32,
}

struct Frontier {
    rows: Vec<usize>,
    header: usize,
}

struct Occupancy {
    row_span: Option<(usize, usize)>,
    header: bool,
}

enum RenderNode {
    Wire {
        y: f32,
        x1: f32,
        x2: f32,
    },
    GateBox {
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        label: String,
        class: &'static str,
    },
    Text {
        x: f32,
        y: f32,
        label: String,
        class: &'static str,
    },
    Circle {
        x: f32,
        y: f32,
        r: f32,
        class: &'static str,
    },
    Line {
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
        class: &'static str,
    },
}

const RIGHT_PAD: f32 = 32.0;
const GATE_HEIGHT: f32 = 28.0;
const CONTROL_RADIUS: f32 = 5.0;
const TARGET_X_RADIUS: f32 = 12.0;
const TARGET_X_ARM: f32 = 8.0;
const SWAP_ARM: f32 = 8.0;
const LABEL_CHAR_WIDTH: f32 = 7.5;
const BOX_LABEL_PADDING: f32 = 18.0;
const COLUMN_GUTTER: f32 = 18.0;

impl Default for LayoutConfig {
    fn default() -> Self {
        Self {
            left_pad: 32.0,
            top_pad: 28.0,
            col_width: 72.0,
            row_height: 48.0,
            gate_width: 42.0,
        }
    }
}

pub fn to_svg(circuit: &Circuit) -> String {
    let config = LayoutConfig::default();
    let layout = layout_circuit(circuit, &config);
    let width = config.left_pad + layout.total_width + RIGHT_PAD;
    let height = if circuit.nbits == 0 {
        config.top_pad * 2.0
    } else {
        config.top_pad * 2.0 + (circuit.nbits.saturating_sub(1) as f32) * config.row_height
    };
    let wire_x1 = config.left_pad * 0.5;
    let wire_x2 = width - RIGHT_PAD * 0.5;

    let mut wires = Vec::with_capacity(circuit.nbits);
    for site in 0..circuit.nbits {
        wires.push(RenderNode::Wire {
            y: wire_y(site, &config),
            x1: wire_x1,
            x2: wire_x2,
        });
    }

    let mut nodes = Vec::new();
    for (column, element) in layout.elements.iter().zip(&circuit.elements) {
        let x = column.center_x;
        match element {
            CircuitElement::Gate(pg) => layout_gate(pg, x, &config, &mut nodes),
            CircuitElement::Annotation(pa) => {
                let Annotation::Label(text) = &pa.annotation;
                nodes.push(RenderNode::Text {
                    x,
                    y: wire_y(pa.loc, &config) - 18.0,
                    label: text.clone(),
                    class: "annotation-label",
                });
            }
            CircuitElement::Channel(pc) => layout_channel(pc, x, &config, &mut nodes),
        }
    }

    let mut svg = String::new();
    svg.push_str(&format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}">"#,
        width, height
    ));
    svg.push_str(
        r#"<style>
.wire { stroke: #444; stroke-width: 2; }
.gate-box { fill: #fff; stroke: #111; stroke-width: 2; }
.channel-box { fill: #fff; stroke: #111; stroke-width: 2; stroke-dasharray: 6 4; }
.gate-label, .channel-label, .annotation-label { fill: #111; font-family: monospace; text-anchor: middle; dominant-baseline: middle; }
.annotation-label { dominant-baseline: auto; }
.control { fill: #111; stroke: #111; stroke-width: 2; }
.control-open { fill: #fff; stroke: #111; stroke-width: 2; }
.control-link { stroke: #111; stroke-width: 2; }
.target-x { fill: none; stroke: #111; stroke-width: 2; }
.swap-marker { stroke: #111; stroke-width: 2; }
</style>"#,
    );

    for node in wires.iter().chain(nodes.iter()) {
        push_node(&mut svg, node);
    }

    svg.push_str("</svg>");
    svg
}

fn layout_gate(pg: &PositionedGate, x: f32, config: &LayoutConfig, nodes: &mut Vec<RenderNode>) {
    if let Some((y1, y2)) = connector_span(pg, config) {
        nodes.push(RenderNode::Line {
            x1: x,
            y1,
            x2: x,
            y2,
            class: "control-link",
        });
    }

    for (&loc, &is_closed) in pg.control_locs.iter().zip(&pg.control_configs) {
        nodes.push(RenderNode::Circle {
            x,
            y: wire_y(loc, config),
            r: CONTROL_RADIUS,
            class: if is_closed { "control" } else { "control-open" },
        });
    }

    if matches!(pg.gate, Gate::SWAP) {
        for &loc in &pg.target_locs {
            push_swap_marker(x, wire_y(loc, config), nodes);
        }
        return;
    }

    if matches!(pg.gate, Gate::X) && !pg.control_locs.is_empty() && pg.target_locs.len() == 1 {
        push_target_x(x, wire_y(pg.target_locs[0], config), nodes);
        return;
    }

    let label = pg.gate.to_string();
    let box_width = box_width_for_label(&label, config);
    let (top, height) = gate_box_frame(pg, config);
    nodes.push(RenderNode::GateBox {
        x: x - box_width * 0.5,
        y: top,
        width: box_width,
        height,
        label: label.clone(),
        class: "gate-box",
    });
    nodes.push(RenderNode::Text {
        x,
        y: top + height * 0.5,
        label,
        class: "gate-label",
    });
}

fn layout_channel(
    pc: &PositionedChannel,
    x: f32,
    config: &LayoutConfig,
    nodes: &mut Vec<RenderNode>,
) {
    let label = channel_label(&pc.channel).to_string();
    let box_width = box_width_for_label(&label, config);
    let (top, height) = box_frame_for_locs(&pc.locs, config);
    nodes.push(RenderNode::GateBox {
        x: x - box_width * 0.5,
        y: top,
        width: box_width,
        height,
        label: label.clone(),
        class: "channel-box",
    });
    nodes.push(RenderNode::Text {
        x,
        y: top + height * 0.5,
        label: label.clone(),
        class: "channel-label",
    });
}

fn push_target_x(x: f32, y: f32, nodes: &mut Vec<RenderNode>) {
    nodes.push(RenderNode::Circle {
        x,
        y,
        r: TARGET_X_RADIUS,
        class: "target-x",
    });
    nodes.push(RenderNode::Line {
        x1: x - TARGET_X_ARM,
        y1: y,
        x2: x + TARGET_X_ARM,
        y2: y,
        class: "target-x",
    });
    nodes.push(RenderNode::Line {
        x1: x,
        y1: y - TARGET_X_ARM,
        x2: x,
        y2: y + TARGET_X_ARM,
        class: "target-x",
    });
}

fn push_swap_marker(x: f32, y: f32, nodes: &mut Vec<RenderNode>) {
    nodes.push(RenderNode::Line {
        x1: x - SWAP_ARM,
        y1: y - SWAP_ARM,
        x2: x + SWAP_ARM,
        y2: y + SWAP_ARM,
        class: "swap-marker",
    });
    nodes.push(RenderNode::Line {
        x1: x - SWAP_ARM,
        y1: y + SWAP_ARM,
        x2: x + SWAP_ARM,
        y2: y - SWAP_ARM,
        class: "swap-marker",
    });
}

fn layout_circuit(circuit: &Circuit, config: &LayoutConfig) -> CircuitLayout {
    let mut frontier = Frontier::new(circuit.nbits);
    let mut assignments = Vec::with_capacity(circuit.elements.len());
    let mut column_widths = Vec::new();

    for element in &circuit.elements {
        let column = frontier.reserve(occupancy_for_element(circuit.nbits, element));
        let width = column_width_for_element(element, config);
        if column_widths.len() <= column {
            column_widths.resize(column + 1, 0.0);
        }
        column_widths[column] = f32::max(column_widths[column], width);
        assignments.push(column);
    }

    let mut cursor = config.left_pad;
    let column_centers: Vec<f32> = column_widths
        .iter()
        .map(|&width| {
            let center = cursor + width * 0.5;
            cursor += width;
            center
        })
        .collect();

    let elements = assignments
        .into_iter()
        .map(|column| ColumnLayout {
            center_x: column_centers[column],
        })
        .collect();

    CircuitLayout {
        elements,
        total_width: column_widths.iter().sum(),
    }
}

impl Frontier {
    fn new(nrows: usize) -> Self {
        Self {
            rows: vec![0; nrows],
            header: 0,
        }
    }

    fn reserve(&mut self, occupancy: Occupancy) -> usize {
        let mut column = if occupancy.header { self.header } else { 0 };

        if let Some((start, end)) = occupancy.row_span {
            let row_column = self.rows[start..=end]
                .iter()
                .copied()
                .max()
                .unwrap_or(0);
            column = column.max(row_column);
        }

        if occupancy.header {
            self.header = column + 1;
        }

        if let Some((start, end)) = occupancy.row_span {
            for row in &mut self.rows[start..=end] {
                *row = column + 1;
            }
        }

        column
    }
}

fn occupancy_for_element(nbits: usize, element: &CircuitElement) -> Occupancy {
    match element {
        CircuitElement::Gate(pg) => occupancy_for_locs(nbits, &pg.all_locs(), pg.target_locs.is_empty()),
        CircuitElement::Annotation(pa) => Occupancy {
            row_span: Some((pa.loc, pa.loc)),
            header: false,
        },
        CircuitElement::Channel(pc) => occupancy_for_locs(nbits, &pc.locs, pc.locs.is_empty()),
    }
}

fn occupancy_for_locs(nbits: usize, locs: &[usize], header: bool) -> Occupancy {
    if let Some((min_loc, max_loc)) = min_max(locs.iter().copied()) {
        Occupancy {
            row_span: Some((min_loc, max_loc)),
            header,
        }
    } else {
        Occupancy {
            row_span: (nbits > 0).then_some((0, nbits - 1)),
            header: true,
        }
    }
}

fn column_width_for_element(element: &CircuitElement, config: &LayoutConfig) -> f32 {
    match element {
        CircuitElement::Gate(pg) if is_symbol_only_gate(pg) => config.col_width,
        CircuitElement::Gate(pg) => {
            let label = pg.gate.to_string();
            config
                .col_width
                .max(box_width_for_label(&label, config) + COLUMN_GUTTER)
        }
        CircuitElement::Annotation(pa) => match &pa.annotation {
            Annotation::Label(text) => config.col_width.max(text_width(text) + COLUMN_GUTTER),
        },
        CircuitElement::Channel(pc) => {
            let label = channel_label(&pc.channel);
            config
                .col_width
                .max(box_width_for_label(label, config) + COLUMN_GUTTER)
        }
    }
}

fn is_symbol_only_gate(pg: &PositionedGate) -> bool {
    matches!(pg.gate, Gate::SWAP)
        || (matches!(pg.gate, Gate::X) && !pg.control_locs.is_empty() && pg.target_locs.len() == 1)
}

fn connector_span(pg: &PositionedGate, config: &LayoutConfig) -> Option<(f32, f32)> {
    if pg.control_locs.is_empty() && !matches!(pg.gate, Gate::SWAP) {
        return None;
    }

    if pg.target_locs.is_empty() {
        let mut ys: Vec<f32> = pg
            .control_locs
            .iter()
            .map(|&loc| wire_y(loc, config))
            .collect();
        ys.push(header_center_y(config));

        let min_y = ys.iter().copied().reduce(f32::min)?;
        let max_y = ys.iter().copied().reduce(f32::max)?;
        return Some((min_y, max_y));
    }

    let (min_loc, max_loc) = min_max(pg.all_locs().into_iter())?;
    let min_y = wire_y(min_loc, config) - connector_endpoint_padding(pg, min_loc);
    let max_y = wire_y(max_loc, config) + connector_endpoint_padding(pg, max_loc);
    Some((min_y, max_y))
}

fn connector_endpoint_padding(pg: &PositionedGate, loc: usize) -> f32 {
    let control_padding = pg
        .control_locs
        .contains(&loc)
        .then_some(CONTROL_RADIUS)
        .unwrap_or(0.0);

    let target_padding = if pg.target_locs.contains(&loc) {
        if matches!(pg.gate, Gate::X) && !pg.control_locs.is_empty() && pg.target_locs.len() == 1 {
            TARGET_X_RADIUS
        } else if matches!(pg.gate, Gate::SWAP) {
            SWAP_ARM
        } else {
            GATE_HEIGHT * 0.5
        }
    } else {
        0.0
    };

    f32::max(control_padding, target_padding)
}

fn gate_box_frame(pg: &PositionedGate, config: &LayoutConfig) -> (f32, f32) {
    if let Some((min_target, max_target)) = min_max(pg.target_locs.iter().copied()) {
        let top = wire_y(min_target, config) - GATE_HEIGHT * 0.5;
        let height = GATE_HEIGHT + (max_target - min_target) as f32 * config.row_height;
        (top, height)
    } else {
        (header_center_y(config) - GATE_HEIGHT * 0.5, GATE_HEIGHT)
    }
}

fn box_frame_for_locs(locs: &[usize], config: &LayoutConfig) -> (f32, f32) {
    if let Some((min_loc, max_loc)) = min_max(locs.iter().copied()) {
        let top = wire_y(min_loc, config) - GATE_HEIGHT * 0.5;
        let height = GATE_HEIGHT + (max_loc - min_loc) as f32 * config.row_height;
        (top, height)
    } else {
        (header_center_y(config) - GATE_HEIGHT * 0.5, GATE_HEIGHT)
    }
}

fn box_width_for_label(label: &str, config: &LayoutConfig) -> f32 {
    config.gate_width.max(text_width(label) + BOX_LABEL_PADDING)
}

fn text_width(label: &str) -> f32 {
    label.chars().count() as f32 * LABEL_CHAR_WIDTH
}

fn header_center_y(config: &LayoutConfig) -> f32 {
    config.top_pad * 0.5
}

fn wire_y(site: usize, config: &LayoutConfig) -> f32 {
    config.top_pad + site as f32 * config.row_height
}

fn min_max<I>(mut locs: I) -> Option<(usize, usize)>
where
    I: Iterator<Item = usize>,
{
    let first = locs.next()?;
    Some(locs.fold((first, first), |(min_loc, max_loc), loc| {
        (min_loc.min(loc), max_loc.max(loc))
    }))
}

fn push_node(svg: &mut String, node: &RenderNode) {
    match node {
        RenderNode::Wire { y, x1, x2 } => svg.push_str(&format!(
            r#"<line class="wire" x1="{}" y1="{}" x2="{}" y2="{}"/>"#,
            x1, y, x2, y
        )),
        RenderNode::GateBox {
            x,
            y,
            width,
            height,
            label,
            class,
        } => svg.push_str(&format!(
            r#"<rect class="{}" x="{}" y="{}" width="{}" height="{}" rx="6" ry="6" data-label="{}"/>"#,
            class,
            x,
            y,
            width,
            height,
            escape_xml(label)
        )),
        RenderNode::Text { x, y, label, class } => svg.push_str(&format!(
            r#"<text class="{}" x="{}" y="{}">{}</text>"#,
            class,
            x,
            y,
            escape_xml(label)
        )),
        RenderNode::Circle { x, y, r, class } => svg.push_str(&format!(
            r#"<circle class="{}" cx="{}" cy="{}" r="{}"/>"#,
            class, x, y, r
        )),
        RenderNode::Line {
            x1,
            y1,
            x2,
            y2,
            class,
        } => svg.push_str(&format!(
            r#"<line class="{}" x1="{}" y1="{}" x2="{}" y2="{}"/>"#,
            class, x1, y1, x2, y2
        )),
    }
}

fn escape_xml(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '"' => escaped.push_str("&quot;"),
            '\'' => escaped.push_str("&apos;"),
            _ => escaped.push(ch),
        }
    }
    escaped
}

fn channel_label(channel: &NoiseChannel) -> &'static str {
    match channel {
        NoiseChannel::BitFlip { .. } => "BitFlip",
        NoiseChannel::PhaseFlip { .. } => "PhaseFlip",
        NoiseChannel::Depolarizing { .. } => "Depolarizing",
        NoiseChannel::PauliChannel { .. } => "PauliChannel",
        NoiseChannel::Reset { .. } => "Reset",
        NoiseChannel::AmplitudeDamping { .. } => "AmplitudeDamping",
        NoiseChannel::PhaseDamping { .. } => "PhaseDamping",
        NoiseChannel::PhaseAmplitudeDamping { .. } => "PhaseAmplitudeDamping",
        NoiseChannel::ThermalRelaxation { .. } => "ThermalRelaxation",
        NoiseChannel::Coherent { .. } => "Coherent",
        NoiseChannel::Custom { .. } => "Custom",
    }
}

#[cfg(test)]
#[path = "unit_tests/svg.rs"]
mod tests;
