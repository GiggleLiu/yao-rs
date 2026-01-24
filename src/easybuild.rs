use std::f64::consts::PI;

use ndarray::Array2;
use num_complex::Complex64;

use crate::circuit::{Circuit, PositionedGate, control, put};
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
    let mut gates: Vec<PositionedGate> = Vec::new();

    for i in 0..n {
        // H gate on qubit i
        gates.push(put(vec![i], Gate::H));

        // Controlled phase rotations
        for j in 1..(n - i) {
            let theta = 2.0 * PI / (1u64 << (j + 1)) as f64;
            gates.push(control(vec![i + j], vec![i], Gate::Phase(theta)));
        }
    }

    // Reverse qubit order with SWAPs
    for i in 0..(n / 2) {
        gates.push(PositionedGate::new(
            Gate::SWAP,
            vec![i, n - 1 - i],
            vec![],
            vec![],
        ));
    }

    Circuit::new(vec![2; n], gates).unwrap()
}

/// General single-qubit gate: Rz(theta3) * Ry(theta2) * Rz(theta1), positioned on `qubit`.
///
/// Returns a vector of 3 PositionedGates representing the decomposition.
pub fn general_u2(qubit: usize, theta1: f64, theta2: f64, theta3: f64) -> Vec<PositionedGate> {
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
pub fn general_u4(qubit0: usize, params: &[f64; 15]) -> Vec<PositionedGate> {
    let q0 = qubit0;
    let q1 = qubit0 + 1;
    let mut gates = Vec::new();

    // general_u2(q0, p[0], p[1], p[2])
    gates.extend(general_u2(q0, params[0], params[1], params[2]));
    // general_u2(q1, p[3], p[4], p[5])
    gates.extend(general_u2(q1, params[3], params[4], params[5]));
    // CNOT(control=q1, target=q0)
    gates.push(control(vec![q1], vec![q0], Gate::X));
    // Rz(p[6]) on q0
    gates.push(put(vec![q0], Gate::Rz(params[6])));
    // Ry(p[7]) on q1
    gates.push(put(vec![q1], Gate::Ry(params[7])));
    // CNOT(control=q0, target=q1)
    gates.push(control(vec![q0], vec![q1], Gate::X));
    // Ry(p[8]) on q1
    gates.push(put(vec![q1], Gate::Ry(params[8])));
    // CNOT(control=q1, target=q0)
    gates.push(control(vec![q1], vec![q0], Gate::X));
    // general_u2(q0, p[9], p[10], p[11])
    gates.extend(general_u2(q0, params[9], params[10], params[11]));
    // general_u2(q1, p[12], p[13], p[14])
    gates.extend(general_u2(q1, params[12], params[13], params[14]));

    gates
}

/// Hardware-efficient variational circuit. All rotation angles = 0.
///
/// Structure: [rotor_noleading] + nlayer * [CNOT_entangler + rotor_full] + [rotor_notrailing]
/// where:
/// - rotor_noleading = Rx(0), Rz(0) per qubit
/// - rotor_full = Rz(0), Rx(0), Rz(0) per qubit
/// - rotor_notrailing = Rz(0), Rx(0) per qubit
pub fn variational_circuit(n: usize, nlayer: usize, pairs: &[(usize, usize)]) -> Circuit {
    let mut gates: Vec<PositionedGate> = Vec::new();

    for layer in 0..=nlayer {
        // CNOT entangler (not on the first layer)
        if layer > 0 {
            for &(ctrl, tgt) in pairs {
                gates.push(control(vec![ctrl], vec![tgt], Gate::X));
            }
        }

        // Rotor block
        for qubit in 0..n {
            if layer == 0 {
                // noleading: skip leading Rz, just Rx(0), Rz(0)
                gates.push(put(vec![qubit], Gate::Rx(0.0)));
                gates.push(put(vec![qubit], Gate::Rz(0.0)));
            } else if layer == nlayer {
                // notrailing: Rz(0), Rx(0), skip trailing Rz
                gates.push(put(vec![qubit], Gate::Rz(0.0)));
                gates.push(put(vec![qubit], Gate::Rx(0.0)));
            } else {
                // full: Rz(0), Rx(0), Rz(0)
                gates.push(put(vec![qubit], Gate::Rz(0.0)));
                gates.push(put(vec![qubit], Gate::Rx(0.0)));
                gates.push(put(vec![qubit], Gate::Rz(0.0)));
            }
        }
    }

    Circuit::new(vec![2; n], gates).unwrap()
}

/// Hadamard test circuit. N+1 qubits (qubit 0 = ancilla).
///
/// Takes a Custom gate as the unitary.
/// Structure: H(0) -> Rz(phi, 0) -> Controlled-U(0 -> 1..N) -> H(0)
pub fn hadamard_test_circuit(unitary: Gate, phi: f64) -> Circuit {
    let n_u = unitary.num_sites(2);
    let n_total = n_u + 1;
    let mut gates: Vec<PositionedGate> = Vec::new();

    // H on ancilla (qubit 0)
    gates.push(put(vec![0], Gate::H));

    // Rz(phi) on ancilla
    gates.push(put(vec![0], Gate::Rz(phi)));

    // Controlled-U: control=0, targets=1..n_total
    let target_locs: Vec<usize> = (1..n_total).collect();
    gates.push(control(vec![0], target_locs, unitary));

    // H on ancilla
    gates.push(put(vec![0], Gate::H));

    Circuit::new(vec![2; n_total], gates).unwrap()
}

/// Swap test circuit. nstate*nbit+1 qubits (qubit 0 = ancilla).
///
/// Structure: H(0) -> Rz(phi, 0) -> Controlled-SWAP between consecutive registers -> H(0)
pub fn swap_test_circuit(nbit: usize, nstate: usize, phi: f64) -> Circuit {
    let n_total = nstate * nbit + 1;
    let mut gates: Vec<PositionedGate> = Vec::new();

    // H on ancilla (qubit 0)
    gates.push(put(vec![0], Gate::H));

    // Rz(phi) on ancilla
    gates.push(put(vec![0], Gate::Rz(phi)));

    // Controlled-SWAP between consecutive registers
    // Registers are: [1..1+nbit], [1+nbit..1+2*nbit], ..., [1+(nstate-1)*nbit..1+nstate*nbit]
    for s in 0..(nstate - 1) {
        for b in 0..nbit {
            let q1 = 1 + s * nbit + b;
            let q2 = 1 + (s + 1) * nbit + b;
            // Controlled-SWAP with control=0
            gates.push(control(vec![0], vec![q1, q2], Gate::SWAP));
        }
    }

    // H on ancilla
    gates.push(put(vec![0], Gate::H));

    Circuit::new(vec![2; n_total], gates).unwrap()
}

/// Phase estimation circuit. n_reg + n_b qubits.
///
/// Structure:
/// - H on register qubits 0..n_reg
/// - For i in 0..n_reg: controlled-U^(2^i) with control=qubit i, targets=n_reg..n_reg+n_b
/// - Inverse QFT on register
pub fn phase_estimation_circuit(unitary: Gate, n_reg: usize, n_b: usize) -> Circuit {
    let n_total = n_reg + n_b;
    let mut gates: Vec<PositionedGate> = Vec::new();

    // H on register qubits
    for i in 0..n_reg {
        gates.push(put(vec![i], Gate::H));
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

        gates.push(control(vec![i], target_locs.clone(), gate));

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
            gates.push(control(vec![i + j], vec![i], Gate::Phase(theta)));
        }
        // H gate on qubit i
        gates.push(put(vec![i], Gate::H));
    }

    // SWAP for bit reversal on register
    for i in 0..(n_reg / 2) {
        gates.push(PositionedGate::new(
            Gate::SWAP,
            vec![i, n_reg - 1 - i],
            vec![],
            vec![],
        ));
    }

    Circuit::new(vec![2; n_total], gates).unwrap()
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
