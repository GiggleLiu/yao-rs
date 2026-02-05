use serde::{Serialize, Deserialize};
use ndarray::Array2;
use num_complex::Complex64;
use crate::gate::Gate;
use crate::circuit::{Circuit, PositionedGate, CircuitElement, PositionedAnnotation, Annotation};

#[derive(Serialize, Deserialize)]
struct CircuitJson {
    num_qubits: usize,
    elements: Vec<ElementJson>,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "type")]
enum ElementJson {
    #[serde(rename = "gate")]
    Gate(GateJson),
    #[serde(rename = "label")]
    Label { text: String, loc: usize },
}

#[derive(Serialize, Deserialize)]
struct GateJson {
    gate: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    params: Option<Vec<f64>>,
    targets: Vec<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    controls: Option<Vec<usize>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    control_configs: Option<Vec<bool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    matrix: Option<Vec<Vec<[f64; 2]>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    is_diagonal: Option<bool>,
}

/// Helper function to convert a PositionedGate to GateJson
fn positioned_gate_to_json(pg: &PositionedGate) -> GateJson {
    let (gate_name, params, label, matrix, is_diagonal) = match &pg.gate {
        Gate::X => ("X".to_string(), None, None, None, None),
        Gate::Y => ("Y".to_string(), None, None, None, None),
        Gate::Z => ("Z".to_string(), None, None, None, None),
        Gate::H => ("H".to_string(), None, None, None, None),
        Gate::S => ("S".to_string(), None, None, None, None),
        Gate::T => ("T".to_string(), None, None, None, None),
        Gate::SWAP => ("SWAP".to_string(), None, None, None, None),
        Gate::SqrtX => ("SqrtX".to_string(), None, None, None, None),
        Gate::SqrtY => ("SqrtY".to_string(), None, None, None, None),
        Gate::SqrtW => ("SqrtW".to_string(), None, None, None, None),
        Gate::ISWAP => ("ISWAP".to_string(), None, None, None, None),
        Gate::Phase(theta) => ("Phase".to_string(), Some(vec![*theta]), None, None, None),
        Gate::Rx(theta) => ("Rx".to_string(), Some(vec![*theta]), None, None, None),
        Gate::Ry(theta) => ("Ry".to_string(), Some(vec![*theta]), None, None, None),
        Gate::Rz(theta) => ("Rz".to_string(), Some(vec![*theta]), None, None, None),
        Gate::FSim(theta, phi) => ("FSim".to_string(), Some(vec![*theta, *phi]), None, None, None),
        Gate::Custom { matrix, is_diagonal, label } => {
            let nrows = matrix.nrows();
            let ncols = matrix.ncols();
            let mat: Vec<Vec<[f64; 2]>> = (0..nrows).map(|i| {
                (0..ncols).map(|j| {
                    [matrix[[i, j]].re, matrix[[i, j]].im]
                }).collect()
            }).collect();
            ("Custom".to_string(), None, Some(label.clone()), Some(mat), Some(*is_diagonal))
        }
    };

    let controls = if pg.control_locs.is_empty() {
        None
    } else {
        Some(pg.control_locs.clone())
    };

    let control_configs = if pg.control_configs.is_empty() {
        None
    } else {
        Some(pg.control_configs.clone())
    };

    GateJson {
        gate: gate_name,
        params,
        targets: pg.target_locs.clone(),
        controls,
        control_configs,
        label,
        matrix,
        is_diagonal,
    }
}

/// Serialize a Circuit to a pretty-printed JSON string.
pub fn circuit_to_json(circuit: &Circuit) -> String {
    let num_qubits = circuit.num_sites();
    let elements: Vec<ElementJson> = circuit.elements.iter().map(|element| {
        match element {
            CircuitElement::Gate(pg) => ElementJson::Gate(positioned_gate_to_json(pg)),
            CircuitElement::Annotation(pa) => {
                match &pa.annotation {
                    Annotation::Label(text) => ElementJson::Label {
                        text: text.clone(),
                        loc: pa.loc,
                    },
                }
            }
        }
    }).collect();

    let circuit_json = CircuitJson { num_qubits, elements };
    serde_json::to_string_pretty(&circuit_json).unwrap()
}

/// Helper function to convert GateJson to a CircuitElement
fn gate_json_to_element(gj: GateJson) -> Result<CircuitElement, String> {
    let gate = match gj.gate.as_str() {
        "X" => Gate::X,
        "Y" => Gate::Y,
        "Z" => Gate::Z,
        "H" => Gate::H,
        "S" => Gate::S,
        "T" => Gate::T,
        "SWAP" => Gate::SWAP,
        "SqrtX" => Gate::SqrtX,
        "SqrtY" => Gate::SqrtY,
        "SqrtW" => Gate::SqrtW,
        "ISWAP" => Gate::ISWAP,
        "Phase" => {
            let params = gj.params.ok_or("Phase gate requires params")?;
            if params.is_empty() {
                return Err("Phase gate requires at least 1 parameter".to_string());
            }
            Gate::Phase(params[0])
        }
        "Rx" => {
            let params = gj.params.ok_or("Rx gate requires params")?;
            if params.is_empty() {
                return Err("Rx gate requires at least 1 parameter".to_string());
            }
            Gate::Rx(params[0])
        }
        "Ry" => {
            let params = gj.params.ok_or("Ry gate requires params")?;
            if params.is_empty() {
                return Err("Ry gate requires at least 1 parameter".to_string());
            }
            Gate::Ry(params[0])
        }
        "Rz" => {
            let params = gj.params.ok_or("Rz gate requires params")?;
            if params.is_empty() {
                return Err("Rz gate requires at least 1 parameter".to_string());
            }
            Gate::Rz(params[0])
        }
        "FSim" => {
            let params = gj.params.ok_or("FSim gate requires params")?;
            if params.len() < 2 {
                return Err("FSim gate requires 2 parameters".to_string());
            }
            Gate::FSim(params[0], params[1])
        }
        "Custom" => {
            let mat_data = gj.matrix.ok_or("Custom gate requires matrix")?;
            let nrows = mat_data.len();
            if nrows == 0 {
                return Err("Custom gate matrix cannot be empty".to_string());
            }
            let ncols = mat_data[0].len();
            let mut elements = Vec::with_capacity(nrows * ncols);
            for row in &mat_data {
                if row.len() != ncols {
                    return Err("Custom gate matrix rows must have equal length".to_string());
                }
                for &[re, im] in row {
                    elements.push(Complex64::new(re, im));
                }
            }
            let matrix = Array2::from_shape_vec((nrows, ncols), elements)
                .map_err(|e| format!("Failed to construct matrix: {}", e))?;
            let is_diagonal = gj.is_diagonal.unwrap_or(false);
            let label = gj.label.unwrap_or_default();
            Gate::Custom { matrix, is_diagonal, label }
        }
        other => return Err(format!("Unknown gate type: {}", other)),
    };

    let control_locs = gj.controls.unwrap_or_default();
    let control_configs = gj.control_configs.unwrap_or_else(|| vec![true; control_locs.len()]);

    let pg = PositionedGate::new(gate, gj.targets, control_locs, control_configs);
    Ok(CircuitElement::Gate(pg))
}

/// Deserialize a Circuit from a JSON string.
pub fn circuit_from_json(json: &str) -> Result<Circuit, String> {
    let circuit_json: CircuitJson = serde_json::from_str(json)
        .map_err(|e| format!("JSON parse error: {}", e))?;

    let num_qubits = circuit_json.num_qubits;
    let dims = vec![2; num_qubits];

    let mut elements = Vec::new();
    for ej in circuit_json.elements {
        let element = match ej {
            ElementJson::Gate(gj) => gate_json_to_element(gj)?,
            ElementJson::Label { text, loc } => {
                CircuitElement::Annotation(PositionedAnnotation {
                    annotation: Annotation::Label(text),
                    loc,
                })
            }
        };
        elements.push(element);
    }

    Circuit::new(dims, elements).map_err(|e| format!("Circuit validation error: {}", e))
}
