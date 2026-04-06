use crate::output::OutputConfig;
use crate::output::{fmt_bold, fmt_cyan, fmt_dim};
use anyhow::Result;
use yao_rs::CircuitElement;

pub fn inspect(input: &str, out: &OutputConfig) -> Result<()> {
    let circuit = super::load_circuit(input)?;

    let num_qubits = circuit.num_sites();
    let mut gate_count = 0usize;
    let mut annotation_count = 0usize;
    let mut channel_count = 0usize;

    for element in &circuit.elements {
        match element {
            CircuitElement::Gate(_) => gate_count += 1,
            CircuitElement::Annotation(_) => annotation_count += 1,
            CircuitElement::Channel(_) => channel_count += 1,
        }
    }

    let json_value = serde_json::json!({
        "num_qubits": num_qubits,
        "dims": &circuit.dims,
        "total_elements": circuit.elements.len(),
        "gate_count": gate_count,
        "annotation_count": annotation_count,
        "channel_count": channel_count,
        "gates": circuit
            .elements
            .iter()
            .filter_map(|element| {
                if let CircuitElement::Gate(pg) = element {
                    Some(serde_json::json!({
                        "gate": pg.gate.to_string(),
                        "target_locs": &pg.target_locs,
                        "control_locs": &pg.control_locs,
                    }))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>(),
    });

    let mut human = String::new();
    human.push_str(&format!("{}\n", fmt_bold("Circuit Info")));
    human.push_str(&format!("  {} {}\n", fmt_cyan("Qubits:"), num_qubits));
    human.push_str(&format!("  {} {:?}\n", fmt_cyan("Dims:"), circuit.dims));
    human.push_str(&format!("  {} {}\n", fmt_cyan("Gates:"), gate_count));
    if annotation_count > 0 {
        human.push_str(&format!(
            "  {} {}\n",
            fmt_cyan("Annotations:"),
            annotation_count
        ));
    }
    if channel_count > 0 {
        human.push_str(&format!("  {} {}\n", fmt_cyan("Channels:"), channel_count));
    }

    human.push_str(&format!("\n{}\n", fmt_bold("Gate List")));
    for (index, element) in circuit.elements.iter().enumerate() {
        if let CircuitElement::Gate(pg) = element {
            let ctrl = if pg.control_locs.is_empty() {
                String::new()
            } else {
                format!(" {}", fmt_dim(&format!("ctrl={:?}", pg.control_locs)))
            };
            human.push_str(&format!(
                "  {:>3}. {} on {:?}{}\n",
                index + 1,
                pg.gate,
                pg.target_locs,
                ctrl
            ));
        }
    }

    out.emit(&human, &json_value)
}
