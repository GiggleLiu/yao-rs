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
    let width = config.left_pad + circuit.elements.len() as f32 * config.col_width + RIGHT_PAD;
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
    for (col, element) in circuit.elements.iter().enumerate() {
        let x = config.left_pad + col as f32 * config.col_width + config.col_width * 0.5;
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
    if needs_vertical_connector(pg) {
        let (min_loc, max_loc) = min_max(pg.all_locs().iter().copied());
        nodes.push(RenderNode::Line {
            x1: x,
            y1: wire_y(min_loc, config),
            x2: x,
            y2: wire_y(max_loc, config),
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

    let (min_target, max_target) = min_max(pg.target_locs.iter().copied());
    let top = wire_y(min_target, config) - GATE_HEIGHT * 0.5;
    let height = GATE_HEIGHT + (max_target - min_target) as f32 * config.row_height;
    nodes.push(RenderNode::GateBox {
        x: x - config.gate_width * 0.5,
        y: top,
        width: config.gate_width,
        height,
        label: pg.gate.to_string(),
    });
    nodes.push(RenderNode::Text {
        x,
        y: top + height * 0.5,
        label: pg.gate.to_string(),
        class: "gate-label",
    });
}

fn layout_channel(
    pc: &PositionedChannel,
    x: f32,
    config: &LayoutConfig,
    nodes: &mut Vec<RenderNode>,
) {
    let (min_loc, max_loc) = min_max(pc.locs.iter().copied());
    let top = wire_y(min_loc, config) - GATE_HEIGHT * 0.5;
    let height = GATE_HEIGHT + (max_loc - min_loc) as f32 * config.row_height;
    let left = x - config.gate_width * 0.5;
    let right = x + config.gate_width * 0.5;
    let bottom = top + height;

    nodes.push(RenderNode::Line {
        x1: left,
        y1: top,
        x2: right,
        y2: top,
        class: "channel-box",
    });
    nodes.push(RenderNode::Line {
        x1: left,
        y1: bottom,
        x2: right,
        y2: bottom,
        class: "channel-box",
    });
    nodes.push(RenderNode::Line {
        x1: left,
        y1: top,
        x2: left,
        y2: bottom,
        class: "channel-box",
    });
    nodes.push(RenderNode::Line {
        x1: right,
        y1: top,
        x2: right,
        y2: bottom,
        class: "channel-box",
    });
    nodes.push(RenderNode::Text {
        x,
        y: top + height * 0.5,
        label: channel_label(&pc.channel).to_string(),
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

fn needs_vertical_connector(pg: &PositionedGate) -> bool {
    !pg.control_locs.is_empty() || matches!(pg.gate, Gate::SWAP)
}

fn wire_y(site: usize, config: &LayoutConfig) -> f32 {
    config.top_pad + site as f32 * config.row_height
}

fn min_max<I>(mut locs: I) -> (usize, usize)
where
    I: Iterator<Item = usize>,
{
    let first = locs.next().expect("layout requires at least one location");
    locs.fold((first, first), |(min_loc, max_loc), loc| {
        (min_loc.min(loc), max_loc.max(loc))
    })
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
        } => svg.push_str(&format!(
            r#"<rect class="gate-box" x="{}" y="{}" width="{}" height="{}" rx="6" ry="6" data-label="{}"/>"#,
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
