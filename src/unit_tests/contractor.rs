use crate::apply::apply;
use crate::circuit::{Circuit, control, put};
use crate::contractor::{contract, contract_dm, contract_dm_with_tree, contract_with_tree};
use crate::einsum::{circuit_to_einsum_dm, circuit_to_einsum_with_boundary, circuit_to_overlap};
use crate::gate::Gate;
use crate::register::ArrayReg;
use num_complex::Complex64;
use omeco::{GreedyMethod, optimize_code};

fn assert_scalar_close(result: &ndarray::ArrayD<Complex64>, expected: Complex64) {
    let val = result.iter().next().unwrap();
    assert!(
        (val - expected).norm() < 1e-10,
        "expected {expected}, got {val}"
    );
}

/// Cross-validate: contract the TN state vector and compare against apply().
fn cross_validate(circuit: &Circuit) {
    let tn = circuit_to_einsum_with_boundary(circuit, &[]);
    let tn_result = contract(&tn);
    let apply_result = apply(circuit, &ArrayReg::zero_state(circuit.nbits));
    let state = apply_result.state_vec();
    // Flatten TN result and compare element-wise
    for (i, (tn_val, apply_val)) in tn_result.iter().zip(state.iter()).enumerate() {
        assert!(
            (tn_val - apply_val).norm() < 1e-10,
            "mismatch at index {i}: tn={tn_val}, apply={apply_val}"
        );
    }
}

#[test]
fn test_contract_identity() {
    let circuit = Circuit::new(vec![2, 2], vec![]).unwrap();
    let tn = circuit_to_einsum_with_boundary(&circuit, &[0, 1]);
    let result = contract(&tn);
    assert_scalar_close(&result, Complex64::new(1.0, 0.0));
}

#[test]
fn test_contract_h_gate() {
    let circuit = Circuit::new(vec![2], vec![put(vec![0], Gate::H)]).unwrap();
    let tn = circuit_to_einsum_with_boundary(&circuit, &[0]);
    let result = contract(&tn);
    let expected = 1.0 / 2.0_f64.sqrt();
    assert_scalar_close(&result, Complex64::new(expected, 0.0));
}

#[test]
fn test_contract_bell_state() {
    let circuit = Circuit::new(
        vec![2, 2],
        vec![put(vec![0], Gate::H), control(vec![0], vec![1], Gate::X)],
    )
    .unwrap();
    let tn = circuit_to_einsum_with_boundary(&circuit, &[0, 1]);
    let result = contract(&tn);
    let expected = 1.0 / 2.0_f64.sqrt();
    assert_scalar_close(&result, Complex64::new(expected, 0.0));
}

#[test]
fn test_contract_overlap() {
    let circuit = Circuit::new(
        vec![2, 2],
        vec![put(vec![0], Gate::H), control(vec![0], vec![1], Gate::X)],
    )
    .unwrap();
    let tn = circuit_to_overlap(&circuit);
    let result = contract(&tn);
    let expected = 1.0 / 2.0_f64.sqrt();
    assert_scalar_close(&result, Complex64::new(expected, 0.0));
}

#[test]
fn test_contract_state_vector() {
    // H|0⟩ = (|0⟩+|1⟩)/√2 — use boundary with no pinned outputs
    let circuit = Circuit::new(vec![2], vec![put(vec![0], Gate::H)]).unwrap();
    let tn = circuit_to_einsum_with_boundary(&circuit, &[]);
    let result = contract(&tn);
    let s = 1.0 / 2.0_f64.sqrt();
    assert!((result[[0]] - Complex64::new(s, 0.0)).norm() < 1e-10);
    assert!((result[[1]] - Complex64::new(s, 0.0)).norm() < 1e-10);
}

#[test]
fn test_contract_two_qubit_state() {
    // Bell state (|00⟩+|11⟩)/√2 — use boundary with no pinned outputs
    let circuit = Circuit::new(
        vec![2, 2],
        vec![put(vec![0], Gate::H), control(vec![0], vec![1], Gate::X)],
    )
    .unwrap();
    let tn = circuit_to_einsum_with_boundary(&circuit, &[]);
    let result = contract(&tn);
    let s = 1.0 / 2.0_f64.sqrt();
    assert!((result[[0, 0]] - Complex64::new(s, 0.0)).norm() < 1e-10);
    assert!((result[[0, 1]] - Complex64::new(0.0, 0.0)).norm() < 1e-10);
    assert!((result[[1, 0]] - Complex64::new(0.0, 0.0)).norm() < 1e-10);
    assert!((result[[1, 1]] - Complex64::new(s, 0.0)).norm() < 1e-10);
}

// --- Asymmetric gate tests (catch transpose bugs) ---

#[test]
fn test_contract_y_gate() {
    // Y|0⟩ = i|1⟩ — asymmetric matrix, catches column-major bugs
    let circuit = Circuit::new(vec![2], vec![put(vec![0], Gate::Y)]).unwrap();
    cross_validate(&circuit);
}

#[test]
fn test_contract_rx_gate() {
    // Rx(π/3)|0⟩ — asymmetric complex entries
    let circuit = Circuit::new(
        vec![2],
        vec![put(vec![0], Gate::Rx(std::f64::consts::FRAC_PI_3))],
    )
    .unwrap();
    cross_validate(&circuit);
}

#[test]
fn test_contract_ry_gate() {
    let circuit = Circuit::new(vec![2], vec![put(vec![0], Gate::Ry(1.23))]).unwrap();
    cross_validate(&circuit);
}

// --- Diagonal gate tests ---

#[test]
fn test_contract_z_gate() {
    // Z is diagonal — exercises the diagonal gate path in einsum
    let circuit =
        Circuit::new(vec![2], vec![put(vec![0], Gate::H), put(vec![0], Gate::Z)]).unwrap();
    cross_validate(&circuit);
}

#[test]
fn test_contract_t_gate() {
    let circuit =
        Circuit::new(vec![2], vec![put(vec![0], Gate::H), put(vec![0], Gate::T)]).unwrap();
    cross_validate(&circuit);
}

#[test]
fn test_contract_s_gate() {
    let circuit =
        Circuit::new(vec![2], vec![put(vec![0], Gate::H), put(vec![0], Gate::S)]).unwrap();
    cross_validate(&circuit);
}

#[test]
fn test_contract_phase_gate() {
    let circuit = Circuit::new(
        vec![2],
        vec![put(vec![0], Gate::H), put(vec![0], Gate::Phase(0.7))],
    )
    .unwrap();
    cross_validate(&circuit);
}

// --- Multi-gate / deeper circuits ---

#[test]
fn test_contract_qft_3qubit() {
    let circuit = crate::easybuild::qft_circuit(3);
    cross_validate(&circuit);
}

#[test]
fn test_contract_ghz_4qubit() {
    let n = 4;
    let mut elements = vec![put(vec![0], Gate::H)];
    for i in 0..n - 1 {
        elements.push(control(vec![i], vec![i + 1], Gate::X));
    }
    let circuit = Circuit::new(vec![2; n], elements).unwrap();
    cross_validate(&circuit);
}

#[test]
fn test_contract_mixed_gates() {
    // Circuit with both diagonal and non-diagonal gates
    let circuit = Circuit::new(
        vec![2, 2],
        vec![
            put(vec![0], Gate::H),
            put(vec![0], Gate::T), // diagonal
            control(vec![0], vec![1], Gate::X),
            put(vec![1], Gate::S), // diagonal
            put(vec![1], Gate::Ry(0.5)),
        ],
    )
    .unwrap();
    cross_validate(&circuit);
}

// --- Layout correctness test ---

#[test]
fn test_contract_result_follows_input_convention() {
    // Output should be column-major (Fortran layout), matching the input convention
    let circuit = Circuit::new(
        vec![2, 2],
        vec![put(vec![0], Gate::H), control(vec![0], vec![1], Gate::X)],
    )
    .unwrap();
    let tn = circuit_to_einsum_with_boundary(&circuit, &[]);
    let result = contract(&tn);
    assert!(
        !result.is_standard_layout(),
        "result should be column-major (Fortran layout)"
    );
}

// --- Pre-computed contraction tree test ---

#[test]
fn test_contract_with_tree() {
    let circuit = Circuit::new(
        vec![2, 2],
        vec![put(vec![0], Gate::H), control(vec![0], vec![1], Gate::X)],
    )
    .unwrap();
    let tn = circuit_to_einsum_with_boundary(&circuit, &[]);

    // Optimize externally
    let tree = optimize_code(&tn.code, &tn.size_dict, &GreedyMethod::default()).unwrap();

    // Contract with pre-computed tree
    let result = contract_with_tree(&tn, tree);
    let s = 1.0 / 2.0_f64.sqrt();
    assert!((result[[0, 0]] - Complex64::new(s, 0.0)).norm() < 1e-10);
    assert!((result[[0, 1]] - Complex64::new(0.0, 0.0)).norm() < 1e-10);
    assert!((result[[1, 0]] - Complex64::new(0.0, 0.0)).norm() < 1e-10);
    assert!((result[[1, 1]] - Complex64::new(s, 0.0)).norm() < 1e-10);
}

// --- Density matrix contraction tests ---

/// Cross-validate DM contraction: contract the DM TN and compare against
/// the pure-state density matrix |psi><psi|.
fn cross_validate_dm(circuit: &Circuit) {
    let tn_dm = circuit_to_einsum_dm(circuit);
    let dm_result = contract_dm(&tn_dm);

    // Get pure state for reference
    let reg = apply(circuit, &ArrayReg::zero_state(circuit.nbits));
    let psi = reg.state_vec();
    let n = circuit.num_sites();

    // dm_result shape should be [d0, d1, ..., d0, d1, ...] (ket then bra)
    // Verify tr(rho) = 1
    let mut trace = Complex64::new(0.0, 0.0);
    for idx in 0..psi.len() {
        // Build multi-index for diagonal entry: ket_i = bra_i for all sites
        let mut multi_idx = vec![0usize; 2 * n];
        let mut rem = idx;
        for site in (0..n).rev() {
            let d = circuit.dims[site];
            multi_idx[site] = rem % d;
            multi_idx[n + site] = rem % d;
            rem /= d;
        }
        trace += dm_result[ndarray::IxDyn(&multi_idx)];
    }
    assert!(
        (trace - Complex64::new(1.0, 0.0)).norm() < 1e-10,
        "tr(rho) should be 1, got {trace}"
    );

    // Verify rho[i,j] = psi[i] * conj(psi[j])
    for i in 0..psi.len() {
        for j in 0..psi.len() {
            let expected = psi[i] * psi[j].conj();
            let mut multi_idx = vec![0usize; 2 * n];
            let mut rem_i = i;
            let mut rem_j = j;
            for site in (0..n).rev() {
                let d = circuit.dims[site];
                multi_idx[site] = rem_i % d;
                multi_idx[n + site] = rem_j % d;
                rem_i /= d;
                rem_j /= d;
            }
            let actual = dm_result[ndarray::IxDyn(&multi_idx)];
            assert!(
                (actual - expected).norm() < 1e-10,
                "rho[{i},{j}] mismatch: expected {expected}, got {actual}"
            );
        }
    }
}

#[test]
fn test_contract_dm_identity() {
    let circuit = Circuit::new(vec![2], vec![]).unwrap();
    cross_validate_dm(&circuit);
}

#[test]
fn test_contract_dm_h_gate() {
    let circuit = Circuit::new(vec![2], vec![put(vec![0], Gate::H)]).unwrap();
    cross_validate_dm(&circuit);
}

#[test]
fn test_contract_dm_bell_state() {
    let circuit = Circuit::new(
        vec![2, 2],
        vec![put(vec![0], Gate::H), control(vec![0], vec![1], Gate::X)],
    )
    .unwrap();
    cross_validate_dm(&circuit);
}

#[test]
fn test_contract_dm_with_tree() {
    let circuit = Circuit::new(vec![2], vec![put(vec![0], Gate::H)]).unwrap();
    let tn_dm = circuit_to_einsum_dm(&circuit);

    let tree = optimize_code(&tn_dm.code, &tn_dm.size_dict, &GreedyMethod::default()).unwrap();
    let dm_result = contract_dm_with_tree(&tn_dm, tree);

    // H|0⟩ = (|0⟩+|1⟩)/sqrt(2), so rho = [[0.5, 0.5], [0.5, 0.5]]
    let half = Complex64::new(0.5, 0.0);
    assert!((dm_result[ndarray::IxDyn(&[0, 0])] - half).norm() < 1e-10);
    assert!((dm_result[ndarray::IxDyn(&[0, 1])] - half).norm() < 1e-10);
    assert!((dm_result[ndarray::IxDyn(&[1, 0])] - half).norm() < 1e-10);
    assert!((dm_result[ndarray::IxDyn(&[1, 1])] - half).norm() < 1e-10);
}
