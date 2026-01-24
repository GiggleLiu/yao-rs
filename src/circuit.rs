use std::collections::HashSet;
use std::fmt;

use crate::gate::Gate;

/// Error types for circuit validation.
#[derive(Debug, Clone, PartialEq)]
pub enum CircuitError {
    /// control_configs length does not match control_locs length
    ControlConfigLengthMismatch {
        control_locs_len: usize,
        control_configs_len: usize,
    },
    /// A location index is out of range
    LocOutOfRange {
        loc: usize,
        num_sites: usize,
    },
    /// Overlap between target_locs and control_locs
    OverlappingLocs {
        overlapping: Vec<usize>,
    },
    /// Control site does not have dimension 2
    ControlSiteNotQubit {
        loc: usize,
        dim: usize,
    },
    /// Named gate target site does not have dimension 2
    NamedGateTargetNotQubit {
        loc: usize,
        dim: usize,
    },
    /// Gate matrix size does not match the product of target site dimensions
    MatrixSizeMismatch {
        expected: usize,
        actual: usize,
    },
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

/// A quantum circuit consisting of positioned gates on a register of qudits.
#[derive(Debug, Clone)]
pub struct Circuit {
    /// The local dimension of each site (e.g., [2, 2, 2] for 3 qubits).
    pub dims: Vec<usize>,
    /// The sequence of gates applied in the circuit.
    pub gates: Vec<PositionedGate>,
}

impl Circuit {
    /// Creates a new Circuit with validation.
    ///
    /// # Errors
    /// Returns a `CircuitError` if any validation rule is violated.
    pub fn new(dims: Vec<usize>, gates: Vec<PositionedGate>) -> Result<Self, CircuitError> {
        let num_sites = dims.len();

        for pg in &gates {
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

        Ok(Circuit { dims, gates })
    }

    /// Returns the number of sites in the circuit.
    pub fn num_sites(&self) -> usize {
        self.dims.len()
    }

    /// Returns the total Hilbert space dimension (product of all site dimensions).
    pub fn total_dim(&self) -> usize {
        self.dims.iter().product()
    }
}
