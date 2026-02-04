use std::f64::consts::PI;

use ndarray::Array2;
use num_complex::Complex64;
use rand::Rng;

use crate::circuit::{Circuit, CircuitElement, control, put};
use crate::gate::Gate;

// =============================================================================
// Entanglement Layouts
// =============================================================================

/// Ring entanglement layout: [(0,1), (1,2), ..., (n-2,n-1), (n-1,0)]
pub fn pair_ring(n: usize) -> Vec<(usize, usize)> {
    (0..n).map(|i| (i, (i + 1) % n)).collect()
}

/// Square lattice on m x n grid. Returns pairs of (row-major index, row-major index).
///
/// For a periodic lattice, wraps around edges (torus topology).
pub fn pair_square(m: usize, n: usize, periodic: bool) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for row in 0..m {
        for col in 0..n {
            let idx = row * n + col;
            // Horizontal neighbor (right)
            if col + 1 < n {
                pairs.push((idx, idx + 1));
            } else if periodic {
                pairs.push((idx, row * n));
            }
            // Vertical neighbor (down)
            if row + 1 < m {
                pairs.push((idx, idx + n));
            } else if periodic {
                pairs.push((idx, col));
            }
        }
    }
    pairs
}

// =============================================================================
// Circuit Builders
// =============================================================================

/// Build an n-qubit QFT circuit.
///
/// For each qubit i (0-indexed): apply H, then for j in 1..(n-i):
/// controlled-Phase(2pi/2^(j+1)) with control=i+j, target=i.
/// Finally SWAP pairs to reverse bit order.
pub fn qft_circuit(n: usize) -> Circuit {
    let mut elements: Vec<CircuitElement> = Vec::new();

    for i in 0..n {
        // H gate on qubit i
        elements.push(put(vec![i], Gate::H));

        // Controlled phase rotations
        for j in 1..(n - i) {
            let theta = 2.0 * PI / (1u64 << (j + 1)) as f64;
            elements.push(control(vec![i + j], vec![i], Gate::Phase(theta)));
        }
    }

    // Reverse qubit order with SWAPs
    for i in 0..(n / 2) {
        elements.push(put(vec![i, n - 1 - i], Gate::SWAP));
    }

    Circuit::new(vec![2; n], elements).unwrap()
}

/// General single-qubit gate: Rz(theta3) * Ry(theta2) * Rz(theta1), positioned on `qubit`.
///
/// Returns a vector of 3 CircuitElements representing the decomposition.
pub fn general_u2(qubit: usize, theta1: f64, theta2: f64, theta3: f64) -> Vec<CircuitElement> {
    vec![
        put(vec![qubit], Gate::Rz(theta1)),
        put(vec![qubit], Gate::Ry(theta2)),
        put(vec![qubit], Gate::Rz(theta3)),
    ]
}

/// General two-qubit SU(4) decomposition (15 params), on qubits qubit0, qubit0+1.
///
/// Structure:
/// - general_u2(q0, p[0..3])
/// - general_u2(q1, p[3..6])
/// - CNOT(control=q1, target=q0)
/// - Rz(p[6]) on q0
/// - Ry(p[7]) on q1
/// - CNOT(control=q0, target=q1)
/// - Ry(p[8]) on q1
/// - CNOT(control=q1, target=q0)
/// - general_u2(q0, p[9..12])
/// - general_u2(q1, p[12..15])
pub fn general_u4(qubit0: usize, params: &[f64; 15]) -> Vec<CircuitElement> {
    let q0 = qubit0;
    let q1 = qubit0 + 1;
    let mut elements = Vec::new();

    // general_u2(q0, p[0], p[1], p[2])
    elements.extend(general_u2(q0, params[0], params[1], params[2]));
    // general_u2(q1, p[3], p[4], p[5])
    elements.extend(general_u2(q1, params[3], params[4], params[5]));
    // CNOT(control=q1, target=q0)
    elements.push(control(vec![q1], vec![q0], Gate::X));
    // Rz(p[6]) on q0
    elements.push(put(vec![q0], Gate::Rz(params[6])));
    // Ry(p[7]) on q1
    elements.push(put(vec![q1], Gate::Ry(params[7])));
    // CNOT(control=q0, target=q1)
    elements.push(control(vec![q0], vec![q1], Gate::X));
    // Ry(p[8]) on q1
    elements.push(put(vec![q1], Gate::Ry(params[8])));
    // CNOT(control=q1, target=q0)
    elements.push(control(vec![q1], vec![q0], Gate::X));
    // general_u2(q0, p[9], p[10], p[11])
    elements.extend(general_u2(q0, params[9], params[10], params[11]));
    // general_u2(q1, p[12], p[13], p[14])
    elements.extend(general_u2(q1, params[12], params[13], params[14]));

    elements
}

/// Hardware-efficient variational circuit. All rotation angles = 0.
///
/// Structure: [rotor_noleading] + nlayer * [CNOT_entangler + rotor_full] + [rotor_notrailing]
/// where:
/// - rotor_noleading = Rx(0), Rz(0) per qubit
/// - rotor_full = Rz(0), Rx(0), Rz(0) per qubit
/// - rotor_notrailing = Rz(0), Rx(0) per qubit
pub fn variational_circuit(n: usize, nlayer: usize, pairs: &[(usize, usize)]) -> Circuit {
    let mut elements: Vec<CircuitElement> = Vec::new();

    for layer in 0..=nlayer {
        // CNOT entangler (not on the first layer)
        if layer > 0 {
            for &(ctrl, tgt) in pairs {
                elements.push(control(vec![ctrl], vec![tgt], Gate::X));
            }
        }

        // Rotor block
        for qubit in 0..n {
            if layer == 0 {
                // noleading: skip leading Rz, just Rx(0), Rz(0)
                elements.push(put(vec![qubit], Gate::Rx(0.0)));
                elements.push(put(vec![qubit], Gate::Rz(0.0)));
            } else if layer == nlayer {
                // notrailing: Rz(0), Rx(0), skip trailing Rz
                elements.push(put(vec![qubit], Gate::Rz(0.0)));
                elements.push(put(vec![qubit], Gate::Rx(0.0)));
            } else {
                // full: Rz(0), Rx(0), Rz(0)
                elements.push(put(vec![qubit], Gate::Rz(0.0)));
                elements.push(put(vec![qubit], Gate::Rx(0.0)));
                elements.push(put(vec![qubit], Gate::Rz(0.0)));
            }
        }
    }

    Circuit::new(vec![2; n], elements).unwrap()
}

/// Hadamard test circuit. N+1 qubits (qubit 0 = ancilla).
///
/// Takes a Custom gate as the unitary.
/// Structure: H(0) -> Rz(phi, 0) -> Controlled-U(0 -> 1..N) -> H(0)
pub fn hadamard_test_circuit(unitary: Gate, phi: f64) -> Circuit {
    let n_u = unitary.num_sites(2);
    let n_total = n_u + 1;
    let mut elements: Vec<CircuitElement> = Vec::new();

    // H on ancilla (qubit 0)
    elements.push(put(vec![0], Gate::H));

    // Rz(phi) on ancilla
    elements.push(put(vec![0], Gate::Rz(phi)));

    // Controlled-U: control=0, targets=1..n_total
    let target_locs: Vec<usize> = (1..n_total).collect();
    elements.push(control(vec![0], target_locs, unitary));

    // H on ancilla
    elements.push(put(vec![0], Gate::H));

    Circuit::new(vec![2; n_total], elements).unwrap()
}

/// Swap test circuit. nstate*nbit+1 qubits (qubit 0 = ancilla).
///
/// Structure: H(0) -> Rz(phi, 0) -> Controlled-SWAP between consecutive registers -> H(0)
pub fn swap_test_circuit(nbit: usize, nstate: usize, phi: f64) -> Circuit {
    let n_total = nstate * nbit + 1;
    let mut elements: Vec<CircuitElement> = Vec::new();

    // H on ancilla (qubit 0)
    elements.push(put(vec![0], Gate::H));

    // Rz(phi) on ancilla
    elements.push(put(vec![0], Gate::Rz(phi)));

    // Controlled-SWAP between consecutive registers
    // Registers are: [1..1+nbit], [1+nbit..1+2*nbit], ..., [1+(nstate-1)*nbit..1+nstate*nbit]
    for s in 0..(nstate - 1) {
        for b in 0..nbit {
            let q1 = 1 + s * nbit + b;
            let q2 = 1 + (s + 1) * nbit + b;
            // Controlled-SWAP with control=0
            elements.push(control(vec![0], vec![q1, q2], Gate::SWAP));
        }
    }

    // H on ancilla
    elements.push(put(vec![0], Gate::H));

    Circuit::new(vec![2; n_total], elements).unwrap()
}

/// Phase estimation circuit. n_reg + n_b qubits.
///
/// Structure:
/// - H on register qubits 0..n_reg
/// - For i in 0..n_reg: controlled-U^(2^i) with control=qubit i, targets=n_reg..n_reg+n_b
/// - Inverse QFT on register
pub fn phase_estimation_circuit(unitary: Gate, n_reg: usize, n_b: usize) -> Circuit {
    let n_total = n_reg + n_b;
    let mut elements: Vec<CircuitElement> = Vec::new();

    // H on register qubits
    for i in 0..n_reg {
        elements.push(put(vec![i], Gate::H));
    }

    // Controlled-U^(2^i) for each register qubit
    let target_locs: Vec<usize> = (n_reg..n_total).collect();

    // Compute U^(2^i) by repeated squaring
    let u_matrix = unitary.matrix(2);
    let dim = u_matrix.nrows();
    let mut current_matrix = u_matrix;

    for i in 0..n_reg {
        let power = 1u64 << i;
        let label = format!("U^{}", power);

        let gate = Gate::Custom {
            matrix: current_matrix.clone(),
            is_diagonal: false,
            label,
        };

        elements.push(control(vec![i], target_locs.clone(), gate));

        // Square the matrix for next iteration
        current_matrix = mat_mul(&current_matrix, &current_matrix, dim);
    }

    // Inverse QFT on register qubits 0..n_reg
    // Inverse QFT: iterate in reverse order, apply negative-phase controlled rotations, then H.
    // Finish with SWAP for bit reversal.
    for i in (0..n_reg).rev() {
        // Controlled phase rotations (negative phase for inverse)
        for j in 1..(n_reg - i) {
            let theta = -2.0 * PI / (1u64 << (j + 1)) as f64;
            elements.push(control(vec![i + j], vec![i], Gate::Phase(theta)));
        }
        // H gate on qubit i
        elements.push(put(vec![i], Gate::H));
    }

    // SWAP for bit reversal on register
    for i in 0..(n_reg / 2) {
        elements.push(put(vec![i, n_reg - 1 - i], Gate::SWAP));
    }

    Circuit::new(vec![2; n_total], elements).unwrap()
}

/// Helper: multiply two complex matrices of given dimension.
fn mat_mul(a: &Array2<Complex64>, b: &Array2<Complex64>, dim: usize) -> Array2<Complex64> {
    let mut result = Array2::zeros((dim, dim));
    for i in 0..dim {
        for j in 0..dim {
            let mut sum = Complex64::new(0.0, 0.0);
            for k in 0..dim {
                sum += a[[i, k]] * b[[k, j]];
            }
            result[[i, j]] = sum;
        }
    }
    result
}

// =============================================================================
// Random Circuit Builders
// =============================================================================

/// Generate entanglement patterns for the supremacy circuit on an nx x ny grid.
///
/// Returns 4 patterns of CZ pairs:
/// - Horizontal even columns
/// - Horizontal odd columns
/// - Vertical even rows
/// - Vertical odd rows
fn pair_supremacy(nx: usize, ny: usize) -> Vec<Vec<(usize, usize)>> {
    let mut patterns: Vec<Vec<(usize, usize)>> = Vec::new();

    // Pattern 0: Horizontal, even columns (col % 2 == 0)
    let mut p = Vec::new();
    for row in 0..ny {
        for col in (0..nx - 1).step_by(2) {
            let idx1 = row * nx + col;
            let idx2 = row * nx + col + 1;
            p.push((idx1, idx2));
        }
    }
    patterns.push(p);

    // Pattern 1: Horizontal, odd columns (col % 2 == 1)
    let mut p = Vec::new();
    for row in 0..ny {
        for col in (1..nx - 1).step_by(2) {
            let idx1 = row * nx + col;
            let idx2 = row * nx + col + 1;
            p.push((idx1, idx2));
        }
    }
    patterns.push(p);

    // Pattern 2: Vertical, even rows (row % 2 == 0)
    let mut p = Vec::new();
    for row in (0..ny - 1).step_by(2) {
        for col in 0..nx {
            let idx1 = row * nx + col;
            let idx2 = (row + 1) * nx + col;
            p.push((idx1, idx2));
        }
    }
    patterns.push(p);

    // Pattern 3: Vertical, odd rows (row % 2 == 1)
    let mut p = Vec::new();
    for row in (1..ny - 1).step_by(2) {
        for col in 0..nx {
            let idx1 = row * nx + col;
            let idx2 = (row + 1) * nx + col;
            p.push((idx1, idx2));
        }
    }
    patterns.push(p);

    patterns
}

/// Build a random supremacy circuit on an nx x ny qubit grid with given depth.
///
/// Structure:
/// 1. Initial layer: H on all qubits
/// 2. For each of depth-2 middle layers:
///    - CZ entangler on cycling pattern from pair_supremacy
///    - Random single-qubit gate from {T, SqrtX, SqrtY} on non-entangled qubits
///    - T must be applied first before other gates on a qubit; don't repeat the same gate
/// 3. Final layer: H on all qubits (if depth > 1)
pub fn rand_supremacy2d(nx: usize, ny: usize, depth: usize, rng: &mut impl Rng) -> Circuit {
    let n = nx * ny;
    let mut elements: Vec<CircuitElement> = Vec::new();
    let patterns = pair_supremacy(nx, ny);
    let num_patterns = patterns.len();

    // Track last gate applied to each qubit (0=none, 1=T, 2=SqrtX, 3=SqrtY)
    let mut last_gate: Vec<u8> = vec![0; n];

    // Layer 0: H on all qubits
    for q in 0..n {
        elements.push(put(vec![q], Gate::H));
    }

    // Middle layers
    if depth > 2 {
        for layer in 0..(depth - 2) {
            let pattern_idx = layer % num_patterns;
            let pattern = &patterns[pattern_idx];

            // Track which qubits are entangled in this layer
            let mut entangled = vec![false; n];
            for &(q1, q2) in pattern.iter() {
                entangled[q1] = true;
                entangled[q2] = true;
                // CZ = control(vec![i], vec![j], Gate::Z)
                elements.push(control(vec![q1], vec![q2], Gate::Z));
            }

            // Apply random single-qubit gates on non-entangled qubits
            for q in 0..n {
                if !entangled[q] {
                    let gate = pick_supremacy_gate(&mut last_gate[q], rng);
                    elements.push(put(vec![q], gate));
                }
            }
        }
    }

    // Final layer: H on all qubits (if depth > 1)
    if depth > 1 {
        for q in 0..n {
            elements.push(put(vec![q], Gate::H));
        }
    }

    Circuit::new(vec![2; n], elements).unwrap()
}

/// Pick a random single-qubit gate for the supremacy circuit.
/// Rules: T must be applied first. Don't repeat the same gate.
/// Options: T=1, SqrtX=2, SqrtY=3
fn pick_supremacy_gate(last: &mut u8, rng: &mut impl Rng) -> Gate {
    if *last == 0 {
        // First gate must be T
        *last = 1;
        Gate::T
    } else {
        // Pick from {T, SqrtX, SqrtY} but not the same as last
        let choices: Vec<u8> = vec![1, 2, 3].into_iter().filter(|&x| x != *last).collect();
        let idx = rng.gen_range(0..choices.len());
        let chosen = choices[idx];
        *last = chosen;
        match chosen {
            1 => Gate::T,
            2 => Gate::SqrtX,
            _ => Gate::SqrtY,
        }
    }
}

/// Lattice53 represents the Sycamore-style 53-qubit topology.
/// 5x12 grid with specific holes, column-major filling.
struct Lattice53 {
    /// Maps grid position (row, col) to qubit index, or None if hole
    grid: Vec<Vec<Option<usize>>>,
    /// Number of qubits actually used
    nbits: usize,
    /// Number of rows
    nrows: usize,
    /// Number of cols
    ncols: usize,
}

impl Lattice53 {
    /// Create a new Lattice53 with up to nbits qubits.
    /// 5x12 grid, column-major filling with holes:
    /// - row 4 (0-indexed) at odd columns (1-indexed: 1,3,5,7,9,11 => 0-indexed: 0,2,4,6,8,10)
    /// - position [0, 6] (row=0, col=6)
    fn new(nbits: usize) -> Self {
        let nrows = 5;
        let ncols = 12;
        let mut grid = vec![vec![None; ncols]; nrows];

        // Determine which positions are holes
        let mut is_hole = vec![vec![false; ncols]; nrows];

        // row 4 at even columns (0-indexed columns: 0, 2, 4, 6, 8, 10)
        for col in (0..ncols).step_by(2) {
            is_hole[4][col] = true;
        }
        // position [0, 6]
        is_hole[0][6] = true;

        // Fill column-major up to nbits
        let mut count = 0;
        for col in 0..ncols {
            for row in 0..nrows {
                if count >= nbits {
                    break;
                }
                if !is_hole[row][col] {
                    grid[row][col] = Some(count);
                    count += 1;
                }
            }
            if count >= nbits {
                break;
            }
        }

        Lattice53 {
            grid,
            nbits: count.min(nbits),
            nrows,
            ncols,
        }
    }

    /// Get qubit pairs for a given pattern character.
    /// Patterns A-H connect specific pairs based on direction and offset.
    fn pattern(&self, chr: char) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();
        match chr {
            // A: vertical connections at even columns
            'A' => {
                for col in (0..self.ncols).step_by(2) {
                    for row in (0..self.nrows - 1).step_by(2) {
                        self.add_pair(&mut pairs, row, col, row + 1, col);
                    }
                }
            }
            // B: vertical connections at odd columns
            'B' => {
                for col in (1..self.ncols).step_by(2) {
                    for row in (0..self.nrows - 1).step_by(2) {
                        self.add_pair(&mut pairs, row, col, row + 1, col);
                    }
                }
            }
            // C: horizontal connections at even rows
            'C' => {
                for row in (0..self.nrows).step_by(2) {
                    for col in (0..self.ncols - 1).step_by(2) {
                        self.add_pair(&mut pairs, row, col, row, col + 1);
                    }
                }
            }
            // D: horizontal connections at odd rows
            'D' => {
                for row in (1..self.nrows).step_by(2) {
                    for col in (0..self.ncols - 1).step_by(2) {
                        self.add_pair(&mut pairs, row, col, row, col + 1);
                    }
                }
            }
            // E: vertical connections at even columns, offset by 1
            'E' => {
                for col in (0..self.ncols).step_by(2) {
                    for row in (1..self.nrows - 1).step_by(2) {
                        self.add_pair(&mut pairs, row, col, row + 1, col);
                    }
                }
            }
            // F: vertical connections at odd columns, offset by 1
            'F' => {
                for col in (1..self.ncols).step_by(2) {
                    for row in (1..self.nrows - 1).step_by(2) {
                        self.add_pair(&mut pairs, row, col, row + 1, col);
                    }
                }
            }
            // G: horizontal connections at even rows, offset by 1
            'G' => {
                for row in (0..self.nrows).step_by(2) {
                    for col in (1..self.ncols - 1).step_by(2) {
                        self.add_pair(&mut pairs, row, col, row, col + 1);
                    }
                }
            }
            // H: horizontal connections at odd rows, offset by 1
            'H' => {
                for row in (1..self.nrows).step_by(2) {
                    for col in (1..self.ncols - 1).step_by(2) {
                        self.add_pair(&mut pairs, row, col, row, col + 1);
                    }
                }
            }
            _ => {}
        }
        pairs
    }

    /// Helper to add a qubit pair if both positions have valid qubits.
    fn add_pair(
        &self,
        pairs: &mut Vec<(usize, usize)>,
        r1: usize,
        c1: usize,
        r2: usize,
        c2: usize,
    ) {
        if r1 < self.nrows && c1 < self.ncols && r2 < self.nrows && c2 < self.ncols
            && let (Some(q1), Some(q2)) = (self.grid[r1][c1], self.grid[r2][c2])
            && q1 < self.nbits && q2 < self.nbits
        {
            pairs.push((q1, q2));
        }
    }
}

/// Build a random Google-53 (Sycamore-style) circuit.
///
/// For each layer in 0..depth:
/// - Random single-qubit gate from {SqrtX, SqrtY, SqrtW} on each qubit
/// - FSim(pi/2, pi/6) entanglers on pattern pairs
///
/// Pattern cycles through: ['A', 'B', 'C', 'D', 'C', 'D', 'A', 'B']
pub fn rand_google53(depth: usize, nbits: usize, rng: &mut impl Rng) -> Circuit {
    let lattice = Lattice53::new(nbits);
    let n = lattice.nbits;
    let mut elements: Vec<CircuitElement> = Vec::new();

    let pattern_cycle = ['A', 'B', 'C', 'D', 'C', 'D', 'A', 'B'];
    let single_gates = [Gate::SqrtX, Gate::SqrtY, Gate::SqrtW];

    for layer in 0..depth {
        // Random single-qubit gates on each qubit
        for q in 0..n {
            let idx = rng.gen_range(0..single_gates.len());
            elements.push(put(vec![q], single_gates[idx].clone()));
        }

        // FSim entanglers on pattern pairs
        let pattern_char = pattern_cycle[layer % pattern_cycle.len()];
        let pairs = lattice.pattern(pattern_char);
        for (q1, q2) in pairs {
            elements.push(put(vec![q1, q2], Gate::FSim(PI / 2.0, PI / 6.0)));
        }
    }

    Circuit::new(vec![2; n], elements).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;
    use ndarray::Array2;

    #[test]
    fn test_mat_mul_identity() {
        let one = Complex64::new(1.0, 0.0);
        let zero = Complex64::new(0.0, 0.0);
        let id = Array2::from_shape_vec((2, 2), vec![one, zero, zero, one]).unwrap();
        let a = Array2::from_shape_vec((2, 2), vec![
            Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0),
            Complex64::new(5.0, 6.0), Complex64::new(7.0, 8.0),
        ]).unwrap();

        let result = mat_mul(&a, &id, 2);
        for i in 0..2 {
            for j in 0..2 {
                assert!((result[[i, j]] - a[[i, j]]).norm() < 1e-10);
            }
        }
    }

    #[test]
    fn test_mat_mul_known_product() {
        // [[1, 2], [3, 4]] * [[5, 6], [7, 8]] = [[19, 22], [43, 50]]
        let a = Array2::from_shape_vec((2, 2), vec![
            Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0),
        ]).unwrap();
        let b = Array2::from_shape_vec((2, 2), vec![
            Complex64::new(5.0, 0.0), Complex64::new(6.0, 0.0),
            Complex64::new(7.0, 0.0), Complex64::new(8.0, 0.0),
        ]).unwrap();
        let result = mat_mul(&a, &b, 2);
        assert!((result[[0, 0]] - Complex64::new(19.0, 0.0)).norm() < 1e-10);
        assert!((result[[0, 1]] - Complex64::new(22.0, 0.0)).norm() < 1e-10);
        assert!((result[[1, 0]] - Complex64::new(43.0, 0.0)).norm() < 1e-10);
        assert!((result[[1, 1]] - Complex64::new(50.0, 0.0)).norm() < 1e-10);
    }

    #[test]
    fn test_mat_mul_complex() {
        // Pauli Y = [[0, -i], [i, 0]], Y^2 = I
        let i = Complex64::new(0.0, 1.0);
        let one = Complex64::new(1.0, 0.0);
        let zero = Complex64::new(0.0, 0.0);
        let y = Array2::from_shape_vec((2, 2), vec![zero, -i, i, zero]).unwrap();
        let result = mat_mul(&y, &y, 2);
        assert!((result[[0, 0]] - one).norm() < 1e-10);
        assert!((result[[0, 1]] - zero).norm() < 1e-10);
        assert!((result[[1, 0]] - zero).norm() < 1e-10);
        assert!((result[[1, 1]] - one).norm() < 1e-10);
    }
}
