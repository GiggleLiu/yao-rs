use crate::circuit::{
    Annotation, Circuit, CircuitElement, PositionedAnnotation, PositionedChannel, PositionedGate,
};
use crate::gate::Gate;
use crate::noise::NoiseChannel;
use ndarray::Array2;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

/// JSON encoding of a complex matrix: rows of `[re, im]` pairs (matches `Custom` gates).
type MatrixJson = Vec<Vec<[f64; 2]>>;

/// Encode a complex matrix as nested `[re, im]` rows.
fn complex_matrix_to_json(matrix: &Array2<Complex64>) -> MatrixJson {
    (0..matrix.nrows())
        .map(|i| {
            (0..matrix.ncols())
                .map(|j| [matrix[[i, j]].re, matrix[[i, j]].im])
                .collect()
        })
        .collect()
}

/// Decode a complex matrix from nested `[re, im]` rows.
fn complex_matrix_from_json(data: &[Vec<[f64; 2]>]) -> Result<Array2<Complex64>, String> {
    let nrows = data.len();
    if nrows == 0 {
        return Err("matrix cannot be empty".to_string());
    }
    let ncols = data[0].len();
    let mut elements = Vec::with_capacity(nrows * ncols);
    for row in data {
        if row.len() != ncols {
            return Err("matrix rows must have equal length".to_string());
        }
        for &[re, im] in row {
            elements.push(Complex64::new(re, im));
        }
    }
    Array2::from_shape_vec((nrows, ncols), elements)
        .map_err(|e| format!("Failed to construct matrix: {e}"))
}

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
    #[serde(rename = "channel")]
    Channel {
        locs: Vec<usize>,
        #[serde(flatten)]
        kind: ChannelKind,
    },
}

/// Serde mirror of [`NoiseChannel`]: scalar params map directly; matrix-bearing
/// variants reuse the `[re, im]` matrix encoding used by `Custom` gates.
#[derive(Serialize, Deserialize)]
#[serde(tag = "channel")]
enum ChannelKind {
    BitFlip {
        p: f64,
    },
    PhaseFlip {
        p: f64,
    },
    Depolarizing {
        n: usize,
        p: f64,
    },
    PauliChannel {
        px: f64,
        py: f64,
        pz: f64,
    },
    Reset {
        p0: f64,
        p1: f64,
    },
    AmplitudeDamping {
        gamma: f64,
        excited_population: f64,
    },
    PhaseDamping {
        gamma: f64,
    },
    PhaseAmplitudeDamping {
        amplitude: f64,
        phase: f64,
        excited_population: f64,
    },
    ThermalRelaxation {
        t1: f64,
        t2: f64,
        time: f64,
        excited_population: f64,
    },
    Coherent {
        matrix: MatrixJson,
    },
    Custom {
        kraus_ops: Vec<MatrixJson>,
    },
}

impl ChannelKind {
    fn from_channel(c: &NoiseChannel) -> Self {
        match c {
            NoiseChannel::BitFlip { p } => ChannelKind::BitFlip { p: *p },
            NoiseChannel::PhaseFlip { p } => ChannelKind::PhaseFlip { p: *p },
            NoiseChannel::Depolarizing { n, p } => ChannelKind::Depolarizing { n: *n, p: *p },
            NoiseChannel::PauliChannel { px, py, pz } => ChannelKind::PauliChannel {
                px: *px,
                py: *py,
                pz: *pz,
            },
            NoiseChannel::Reset { p0, p1 } => ChannelKind::Reset { p0: *p0, p1: *p1 },
            NoiseChannel::AmplitudeDamping {
                gamma,
                excited_population,
            } => ChannelKind::AmplitudeDamping {
                gamma: *gamma,
                excited_population: *excited_population,
            },
            NoiseChannel::PhaseDamping { gamma } => ChannelKind::PhaseDamping { gamma: *gamma },
            NoiseChannel::PhaseAmplitudeDamping {
                amplitude,
                phase,
                excited_population,
            } => ChannelKind::PhaseAmplitudeDamping {
                amplitude: *amplitude,
                phase: *phase,
                excited_population: *excited_population,
            },
            NoiseChannel::ThermalRelaxation {
                t1,
                t2,
                time,
                excited_population,
            } => ChannelKind::ThermalRelaxation {
                t1: *t1,
                t2: *t2,
                time: *time,
                excited_population: *excited_population,
            },
            NoiseChannel::Coherent { matrix } => ChannelKind::Coherent {
                matrix: complex_matrix_to_json(matrix),
            },
            NoiseChannel::Custom { kraus_ops } => ChannelKind::Custom {
                kraus_ops: kraus_ops.iter().map(complex_matrix_to_json).collect(),
            },
        }
    }

    fn into_channel(self) -> Result<NoiseChannel, String> {
        Ok(match self {
            ChannelKind::BitFlip { p } => NoiseChannel::BitFlip { p },
            ChannelKind::PhaseFlip { p } => NoiseChannel::PhaseFlip { p },
            ChannelKind::Depolarizing { n, p } => NoiseChannel::Depolarizing { n, p },
            ChannelKind::PauliChannel { px, py, pz } => NoiseChannel::PauliChannel { px, py, pz },
            ChannelKind::Reset { p0, p1 } => NoiseChannel::Reset { p0, p1 },
            ChannelKind::AmplitudeDamping {
                gamma,
                excited_population,
            } => NoiseChannel::AmplitudeDamping {
                gamma,
                excited_population,
            },
            ChannelKind::PhaseDamping { gamma } => NoiseChannel::PhaseDamping { gamma },
            ChannelKind::PhaseAmplitudeDamping {
                amplitude,
                phase,
                excited_population,
            } => NoiseChannel::PhaseAmplitudeDamping {
                amplitude,
                phase,
                excited_population,
            },
            ChannelKind::ThermalRelaxation {
                t1,
                t2,
                time,
                excited_population,
            } => NoiseChannel::ThermalRelaxation {
                t1,
                t2,
                time,
                excited_population,
            },
            ChannelKind::Coherent { matrix } => NoiseChannel::Coherent {
                matrix: complex_matrix_from_json(&matrix)?,
            },
            ChannelKind::Custom { kraus_ops } => {
                if kraus_ops.is_empty() {
                    return Err("Custom channel requires at least one Kraus operator".to_string());
                }
                NoiseChannel::Custom {
                    kraus_ops: kraus_ops
                        .iter()
                        .map(|m| complex_matrix_from_json(m))
                        .collect::<Result<Vec<_>, _>>()?,
                }
            }
        })
    }
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
        Gate::FSim(theta, phi) => (
            "FSim".to_string(),
            Some(vec![*theta, *phi]),
            None,
            None,
            None,
        ),
        Gate::Custom {
            matrix,
            is_diagonal,
            label,
        } => (
            "Custom".to_string(),
            None,
            Some(label.clone()),
            Some(complex_matrix_to_json(matrix)),
            Some(*is_diagonal),
        ),
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
///
/// All element kinds round-trip through `circuit_to_json`/`circuit_from_json`,
/// including noise/reset channels (`CircuitElement::Channel`).
pub fn circuit_to_json(circuit: &Circuit) -> String {
    let num_qubits = circuit.num_sites();
    let elements: Vec<ElementJson> = circuit
        .elements
        .iter()
        .map(|element| match element {
            CircuitElement::Gate(pg) => ElementJson::Gate(positioned_gate_to_json(pg)),
            CircuitElement::Annotation(pa) => match &pa.annotation {
                Annotation::Label(text) => ElementJson::Label {
                    text: text.clone(),
                    loc: pa.loc,
                },
            },
            CircuitElement::Channel(pc) => ElementJson::Channel {
                locs: pc.locs.clone(),
                kind: ChannelKind::from_channel(&pc.channel),
            },
        })
        .collect();

    let circuit_json = CircuitJson {
        num_qubits,
        elements,
    };
    serde_json::to_string_pretty(&circuit_json).unwrap()
}

/// Helper function to convert GateJson to a CircuitElement
fn gate_json_to_element(gj: GateJson) -> Result<CircuitElement, String> {
    let gate = match gj.gate.as_str() {
        "X" | "CNOT" | "CX" => Gate::X,
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
            let matrix =
                complex_matrix_from_json(&mat_data).map_err(|e| format!("Custom gate {e}"))?;
            let is_diagonal = gj.is_diagonal.unwrap_or(false);
            let label = gj.label.unwrap_or_default();
            Gate::Custom {
                matrix,
                is_diagonal,
                label,
            }
        }
        other => return Err(format!("Unknown gate type: {}", other)),
    };

    let control_locs = gj.controls.unwrap_or_default();
    let control_configs = gj
        .control_configs
        .unwrap_or_else(|| vec![true; control_locs.len()]);

    let pg = PositionedGate::new(gate, gj.targets, control_locs, control_configs);
    Ok(CircuitElement::Gate(pg))
}

/// Deserialize a Circuit from a JSON string.
pub fn circuit_from_json(json: &str) -> Result<Circuit, String> {
    let circuit_json: CircuitJson =
        serde_json::from_str(json).map_err(|e| format!("JSON parse error: {}", e))?;

    let num_qubits = circuit_json.num_qubits;

    let mut elements = Vec::new();
    for ej in circuit_json.elements {
        let element = match ej {
            ElementJson::Gate(gj) => gate_json_to_element(gj)?,
            ElementJson::Label { text, loc } => CircuitElement::Annotation(PositionedAnnotation {
                annotation: Annotation::Label(text),
                loc,
            }),
            ElementJson::Channel { locs, kind } => CircuitElement::Channel(PositionedChannel {
                channel: kind.into_channel()?,
                locs,
            }),
        };
        elements.push(element);
    }

    Circuit::qubits(num_qubits, elements).map_err(|e| format!("Circuit validation error: {}", e))
}

#[cfg(test)]
#[path = "unit_tests/json.rs"]
mod tests;
