use std::collections::HashSet;
use std::fmt;

use crate::gate::Gate;

/// Annotation variants for circuit visualization.
///
/// Annotations are no-ops in execution but render as visual markers
/// in circuit diagrams.
#[derive(Debug, Clone, PartialEq)]
pub enum Annotation {
    /// A text label displayed on the circuit diagram
    Label(String),
}

/// An annotation placed at a specific qubit location.
#[derive(Debug, Clone)]
pub struct PositionedAnnotation {
    pub annotation: Annotation,
    pub loc: usize, // single qubit only
}

/// Elements that can appear in a circuit sequence.
#[derive(Debug, Clone)]
pub enum CircuitElement {
    Gate(PositionedGate),
    Annotation(PositionedAnnotation),
}

/// Error types for circuit validation.
#[derive(Debug, Clone, PartialEq)]
pub enum CircuitError {
    /// control_configs length does not match control_locs length
    ControlConfigLengthMismatch {
        control_locs_len: usize,
        control_configs_len: usize,
    },
    /// A location index is out of range
    LocOutOfRange { loc: usize, num_sites: usize },
    /// Overlap between target_locs and control_locs
    OverlappingLocs { overlapping: Vec<usize> },
    /// Control site does not have dimension 2
    ControlSiteNotQubit { loc: usize, dim: usize },
    /// Named gate target site does not have dimension 2
    NamedGateTargetNotQubit { loc: usize, dim: usize },
    /// Gate matrix size does not match the product of target site dimensions
    MatrixSizeMismatch { expected: usize, actual: usize },
}

impl fmt::Display for CircuitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CircuitError::ControlConfigLengthMismatch {
                control_locs_len,
                control_configs_len,
            } => write!(
                f,
                "control_configs length ({}) does not match control_locs length ({})",
                control_configs_len, control_locs_len
            ),
            CircuitError::LocOutOfRange { loc, num_sites } => write!(
                f,
                "location {} is out of range (num_sites = {})",
                loc, num_sites
            ),
            CircuitError::OverlappingLocs { overlapping } => write!(
                f,
                "target_locs and control_locs overlap at locations: {:?}",
                overlapping
            ),
            CircuitError::ControlSiteNotQubit { loc, dim } => write!(
                f,
                "control site at location {} has dimension {} (must be 2)",
                loc, dim
            ),
            CircuitError::NamedGateTargetNotQubit { loc, dim } => write!(
                f,
                "named gate target site at location {} has dimension {} (must be 2)",
                loc, dim
            ),
            CircuitError::MatrixSizeMismatch { expected, actual } => write!(
                f,
                "gate matrix size {} does not match product of target site dimensions {}",
                actual, expected
            ),
        }
    }
}

impl std::error::Error for CircuitError {}

/// A gate placed at specific locations in a circuit.
#[derive(Debug, Clone)]
pub struct PositionedGate {
    pub gate: Gate,
    pub target_locs: Vec<usize>,
    pub control_locs: Vec<usize>,
    pub control_configs: Vec<bool>,
}

impl PositionedGate {
    /// Creates a new PositionedGate.
    pub fn new(
        gate: Gate,
        target_locs: Vec<usize>,
        control_locs: Vec<usize>,
        control_configs: Vec<bool>,
    ) -> Self {
        PositionedGate {
            gate,
            target_locs,
            control_locs,
            control_configs,
        }
    }

    /// Returns all locations (control locations followed by target locations).
    pub fn all_locs(&self) -> Vec<usize> {
        let mut locs = self.control_locs.clone();
        locs.extend(&self.target_locs);
        locs
    }
}

/// A quantum circuit consisting of positioned gates and annotations on a register of qudits.
#[derive(Debug, Clone)]
pub struct Circuit {
    /// The local dimension of each site (e.g., [2, 2, 2] for 3 qubits).
    pub dims: Vec<usize>,
    /// The sequence of elements (gates and annotations) in the circuit.
    pub elements: Vec<CircuitElement>,
}

impl Circuit {
    /// Creates a new Circuit with validation.
    ///
    /// # Errors
    /// Returns a `CircuitError` if any validation rule is violated.
    pub fn new(dims: Vec<usize>, elements: Vec<CircuitElement>) -> Result<Self, CircuitError> {
        let num_sites = dims.len();

        for element in &elements {
            match element {
                CircuitElement::Gate(pg) => {
                    // 1. control_configs.len() == control_locs.len()
                    if pg.control_configs.len() != pg.control_locs.len() {
                        return Err(CircuitError::ControlConfigLengthMismatch {
                            control_locs_len: pg.control_locs.len(),
                            control_configs_len: pg.control_configs.len(),
                        });
                    }

                    // 2. All locs are in range (< dims.len())
                    for &loc in pg.target_locs.iter().chain(pg.control_locs.iter()) {
                        if loc >= num_sites {
                            return Err(CircuitError::LocOutOfRange { loc, num_sites });
                        }
                    }

                    // 3. No overlap between target_locs and control_locs
                    let target_set: HashSet<usize> = pg.target_locs.iter().copied().collect();
                    let control_set: HashSet<usize> = pg.control_locs.iter().copied().collect();
                    let overlapping: Vec<usize> =
                        target_set.intersection(&control_set).copied().collect();
                    if !overlapping.is_empty() {
                        return Err(CircuitError::OverlappingLocs { overlapping });
                    }

                    // 4. Control sites must have d=2
                    for &loc in &pg.control_locs {
                        if dims[loc] != 2 {
                            return Err(CircuitError::ControlSiteNotQubit {
                                loc,
                                dim: dims[loc],
                            });
                        }
                    }

                    // 5. Named gates (non-Custom) target sites must have d=2
                    let is_named = !matches!(pg.gate, Gate::Custom { .. });
                    if is_named {
                        for &loc in &pg.target_locs {
                            if dims[loc] != 2 {
                                return Err(CircuitError::NamedGateTargetNotQubit {
                                    loc,
                                    dim: dims[loc],
                                });
                            }
                        }
                    }

                    // 6. Gate matrix size must match product of target site dimensions
                    let target_dim_product: usize =
                        pg.target_locs.iter().map(|&loc| dims[loc]).product();
                    let matrix = pg.gate.matrix(dims[pg.target_locs[0]]);
                    let matrix_size = matrix.nrows();
                    if matrix_size != target_dim_product {
                        return Err(CircuitError::MatrixSizeMismatch {
                            expected: target_dim_product,
                            actual: matrix_size,
                        });
                    }
                }
                CircuitElement::Annotation(pa) => {
                    // Annotations only require loc < dims.len()
                    if pa.loc >= num_sites {
                        return Err(CircuitError::LocOutOfRange {
                            loc: pa.loc,
                            num_sites,
                        });
                    }
                }
            }
        }

        Ok(Circuit { dims, elements })
    }

    /// Returns the number of sites in the circuit.
    pub fn num_sites(&self) -> usize {
        self.dims.len()
    }

    /// Returns the total Hilbert space dimension (product of all site dimensions).
    pub fn total_dim(&self) -> usize {
        self.dims.iter().product()
    }

    /// Return the adjoint circuit U†.
    ///
    /// The dagger of a circuit has:
    /// - Elements in reverse order
    /// - Each gate replaced with its adjoint
    /// - Annotations preserved as-is
    ///
    /// For a unitary circuit U, U† U = I.
    pub fn dagger(&self) -> Result<Self, CircuitError> {
        let dagger_elements: Vec<CircuitElement> = self
            .elements
            .iter()
            .rev()
            .map(|element| match element {
                CircuitElement::Gate(pg) => CircuitElement::Gate(PositionedGate {
                    gate: pg.gate.dagger(),
                    target_locs: pg.target_locs.clone(),
                    control_locs: pg.control_locs.clone(),
                    control_configs: pg.control_configs.clone(),
                }),
                CircuitElement::Annotation(pa) => CircuitElement::Annotation(pa.clone()),
            })
            .collect();

        Circuit::new(self.dims.clone(), dagger_elements)
    }

    /// Compile the circuit to PDF bytes.
    ///
    /// This embeds the Typst compiler to render a visual diagram of the circuit.
    /// On first call, this will download the required Typst packages (~1MB).
    ///
    /// # Example
    /// ```ignore
    /// let circuit = Circuit::new(vec![2, 2], vec![
    ///     put(vec![0], Gate::H),
    ///     control(vec![0], vec![1], Gate::X),
    /// ]).unwrap();
    ///
    /// let pdf = circuit.to_pdf()?;
    /// std::fs::write("circuit.pdf", pdf)?;
    /// ```
    #[cfg(feature = "typst")]
    pub fn to_pdf(&self) -> Result<Vec<u8>, crate::typst::PdfError> {
        crate::typst::to_pdf(self)
    }
}

impl fmt::Display for Circuit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n = self.num_sites();
        writeln!(f, "nqubits: {}", n)?;
        for element in &self.elements {
            match element {
                CircuitElement::Gate(pg) => {
                    if pg.control_locs.is_empty() {
                        writeln!(f, "  {} @ q[{}]", pg.gate, format_locs(&pg.target_locs))?;
                    } else {
                        writeln!(
                            f,
                            "  C(q[{}]) {} @ q[{}]",
                            format_locs(&pg.control_locs),
                            pg.gate,
                            format_locs(&pg.target_locs)
                        )?;
                    }
                }
                CircuitElement::Annotation(pa) => match &pa.annotation {
                    Annotation::Label(text) => {
                        writeln!(f, "  \"{}\" @ q[{}]", text, pa.loc)?;
                    }
                },
            }
        }
        Ok(())
    }
}

fn format_locs(locs: &[usize]) -> String {
    locs.iter()
        .map(|l| l.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Place a gate on target locations (no controls).
///
/// Equivalent to Yao.jl's `put(n, locs => gate)`.
///
/// # Example
/// ```
/// use yao_rs::circuit::{put, CircuitElement};
/// use yao_rs::gate::Gate;
/// let elem = put(vec![0], Gate::H);
/// if let CircuitElement::Gate(pg) = elem {
///     assert_eq!(pg.target_locs, vec![0]);
///     assert!(pg.control_locs.is_empty());
/// }
/// ```
pub fn put(target_locs: Vec<usize>, gate: Gate) -> CircuitElement {
    CircuitElement::Gate(PositionedGate::new(gate, target_locs, vec![], vec![]))
}

/// Place a controlled gate with active-high control (all controls trigger on |1⟩).
///
/// Equivalent to Yao.jl's `control(n, ctrl_locs, target_locs => gate)`.
///
/// # Example
/// ```
/// use yao_rs::circuit::{control, CircuitElement};
/// use yao_rs::gate::Gate;
/// let elem = control(vec![0], vec![1], Gate::X);
/// if let CircuitElement::Gate(cnot) = elem {
///     assert_eq!(cnot.control_locs, vec![0]);
///     assert_eq!(cnot.target_locs, vec![1]);
///     assert_eq!(cnot.control_configs, vec![true]);
/// }
/// ```
pub fn control(ctrl_locs: Vec<usize>, target_locs: Vec<usize>, gate: Gate) -> CircuitElement {
    let configs = vec![true; ctrl_locs.len()];
    CircuitElement::Gate(PositionedGate::new(gate, target_locs, ctrl_locs, configs))
}

/// Place a text label annotation on a qubit wire.
///
/// Labels are no-ops in execution but render as floating text on the
/// circuit diagram at the specified qubit location.
///
/// # Example
/// ```
/// use yao_rs::circuit::{label, CircuitElement, Annotation};
/// let elem = label(0, "Bell prep");
/// if let CircuitElement::Annotation(pa) = elem {
///     assert_eq!(pa.loc, 0);
///     assert!(matches!(pa.annotation, Annotation::Label(ref s) if s == "Bell prep"));
/// }
/// ```
pub fn label(loc: usize, text: impl Into<String>) -> CircuitElement {
    CircuitElement::Annotation(PositionedAnnotation {
        annotation: Annotation::Label(text.into()),
        loc,
    })
}
