use ndarray::Array2;
use num_complex::Complex64;

use approx::assert_abs_diff_eq;
use rand::SeedableRng;
use rand::rngs::StdRng;

use yao_rs::apply::apply;
use yao_rs::easybuild::{
    general_u2, general_u4, hadamard_test_circuit, pair_ring, pair_square,
    phase_estimation_circuit, qft_circuit, rand_google53, rand_supremacy2d,
    swap_test_circuit, variational_circuit,
};
use yao_rs::gate::Gate;
use yao_rs::state::State;

const ATOL: f64 = 1e-10;

// =============================================================================
// Entanglement Layout Tests
// =============================================================================

#[test]
fn test_pair_ring() {
    let pairs = pair_ring(4);
    assert_eq!(pairs, vec![(0, 1), (1, 2), (2, 3), (3, 0)]);
}

#[test]
fn test_pair_square_non_periodic() {
    let pairs = pair_square(2, 2, false);
    // 2x2 grid, non-periodic: 4 edges
    // (0,1) horizontal, (0,2) vertical, (1,3) vertical, (2,3) horizontal
    assert_eq!(pairs.len(), 4);
    assert!(pairs.contains(&(0, 1)));
    assert!(pairs.contains(&(0, 2)));
    assert!(pairs.contains(&(1, 3)));
    assert!(pairs.contains(&(2, 3)));
}

#[test]
fn test_pair_square_periodic() {
    let pairs = pair_square(2, 2, true);
    // 2x2 grid, periodic (torus): 8 edges
    assert_eq!(pairs.len(), 8);
}

// =============================================================================
// QFT Tests
// =============================================================================

#[test]
fn test_qft_uniform_superposition() {
    // QFT|000> gives uniform superposition with amplitude 1/sqrt(8)
    let n = 3;
    let circuit = qft_circuit(n);
    let state = State::zero_state(&vec![2; n]);
    let result = apply(&circuit, &state);

    let expected_amp = 1.0 / (8.0f64).sqrt();
    for i in 0..result.total_dim() {
        assert!(
            (result.data[i].norm() - expected_amp).abs() < ATOL,
            "Amplitude at index {} should be {}, got {}",
            i,
            expected_amp,
            result.data[i].norm()
        );
    }
}

#[test]
fn test_qft_norm_preserved() {
    // QFT on an arbitrary state preserves norm
    let n = 3;
    let circuit = qft_circuit(n);

    // Create a state |010>
    let state = State::product_state(&vec![2; n], &[0, 1, 0]);
    let result = apply(&circuit, &state);

    assert!(
        (result.norm() - 1.0).abs() < ATOL,
        "Norm should be preserved: got {}",
        result.norm()
    );
}

// =============================================================================
// General U2/U4 Tests
// =============================================================================

#[test]
fn test_general_u2_unitary() {
    // Apply general_u2 to |0>, check output norm = 1
    let gates = general_u2(0, 0.3, 0.7, 1.2);
    let circuit = yao_rs::Circuit::new(vec![2], gates).unwrap();
    let state = State::zero_state(&vec![2]);
    let result = apply(&circuit, &state);
    assert!(
        (result.norm() - 1.0).abs() < ATOL,
        "Output norm should be 1, got {}",
        result.norm()
    );
}

#[test]
fn test_general_u4_unitary() {
    // Apply general_u4 to |00>, check output norm = 1
    let params: [f64; 15] = [
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
    ];
    let gates = general_u4(0, &params);
    let circuit = yao_rs::Circuit::new(vec![2, 2], gates).unwrap();
    let state = State::zero_state(&vec![2, 2]);
    let result = apply(&circuit, &state);
    assert!(
        (result.norm() - 1.0).abs() < ATOL,
        "Output norm should be 1, got {}",
        result.norm()
    );
}

// =============================================================================
// Variational Circuit Test
// =============================================================================

#[test]
fn test_variational_circuit_structure() {
    let n = 4;
    let nlayer = 2;
    let pairs = pair_ring(n);
    let circuit = variational_circuit(n, nlayer, &pairs);

    // Check correct num_sites
    assert_eq!(circuit.num_sites(), n);

    // Apply to zero state, check output normalized
    let state = State::zero_state(&vec![2; n]);
    let result = apply(&circuit, &state);
    assert!(
        (result.norm() - 1.0).abs() < ATOL,
        "Output should be normalized, got norm = {}",
        result.norm()
    );
}

// =============================================================================
// Hadamard Test Circuit
// =============================================================================

#[test]
fn test_hadamard_test_circuit() {
    // Create a 2-qubit unitary (Custom gate)
    let n_u = 2;
    let dim = 1 << n_u; // 4
    let mut matrix = Array2::zeros((dim, dim));
    // Use identity as unitary
    for i in 0..dim {
        matrix[[i, i]] = Complex64::new(1.0, 0.0);
    }
    let unitary = Gate::Custom {
        matrix,
        is_diagonal: false,
        label: "I4".to_string(),
    };

    let phi = 0.5;
    let circuit = hadamard_test_circuit(unitary, phi);

    // Check correct num_sites (n_u + 1)
    assert_eq!(circuit.num_sites(), n_u + 1);

    // Apply to zero state, check output normalized
    let state = State::zero_state(&vec![2; n_u + 1]);
    let result = apply(&circuit, &state);
    assert!(
        (result.norm() - 1.0).abs() < ATOL,
        "Output should be normalized, got norm = {}",
        result.norm()
    );
}

// =============================================================================
// Swap Test Circuit
// =============================================================================

#[test]
fn test_swap_test_circuit() {
    let nbit = 2;
    let nstate = 2;
    let phi = 0.3;
    let circuit = swap_test_circuit(nbit, nstate, phi);

    // Check correct num_sites (nstate*nbit + 1)
    let expected_sites = nstate * nbit + 1;
    assert_eq!(circuit.num_sites(), expected_sites);

    // Apply to zero state, check output normalized
    let state = State::zero_state(&vec![2; expected_sites]);
    let result = apply(&circuit, &state);
    assert!(
        (result.norm() - 1.0).abs() < ATOL,
        "Output should be normalized, got norm = {}",
        result.norm()
    );
}

// =============================================================================
// Phase Estimation Circuit
// =============================================================================

#[test]
fn test_phase_estimation_circuit() {
    let n_reg = 3;
    let n_b = 1;

    // Create a 1-qubit unitary (Phase gate as Custom)
    let theta = std::f64::consts::PI / 4.0;
    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    let phase = Complex64::from_polar(1.0, theta);
    let matrix = Array2::from_shape_vec((2, 2), vec![one, zero, zero, phase]).unwrap();
    let unitary = Gate::Custom {
        matrix,
        is_diagonal: false,
        label: "U".to_string(),
    };

    let circuit = phase_estimation_circuit(unitary, n_reg, n_b);

    // Check correct num_sites (n_reg + n_b)
    assert_eq!(circuit.num_sites(), n_reg + n_b);

    // Apply to zero state, check output normalized
    let state = State::zero_state(&vec![2; n_reg + n_b]);
    let result = apply(&circuit, &state);
    assert!(
        (result.norm() - 1.0).abs() < ATOL,
        "Output should be normalized, got norm = {}",
        result.norm()
    );
}

// =============================================================================
// Random Circuit Builder Tests
// =============================================================================

#[test]
fn test_rand_supremacy2d_structure() {
    let mut rng = StdRng::seed_from_u64(42);
    let circuit = rand_supremacy2d(3, 3, 5, &mut rng);
    assert_eq!(circuit.num_sites(), 9);
    let state = State::zero_state(&vec![2; 9]);
    let result = apply(&circuit, &state);
    assert_abs_diff_eq!(result.norm(), 1.0, epsilon = 1e-10);
}

#[test]
fn test_rand_google53_structure() {
    let mut rng = StdRng::seed_from_u64(42);
    let circuit = rand_google53(4, 10, &mut rng);
    assert_eq!(circuit.num_sites(), 10);
    let state = State::zero_state(&vec![2; 10]);
    let result = apply(&circuit, &state);
    assert_abs_diff_eq!(result.norm(), 1.0, epsilon = 1e-10);
}

#[test]
fn test_rand_supremacy2d_deterministic_with_seed() {
    let mut rng1 = StdRng::seed_from_u64(123);
    let mut rng2 = StdRng::seed_from_u64(123);
    let c1 = rand_supremacy2d(2, 2, 3, &mut rng1);
    let c2 = rand_supremacy2d(2, 2, 3, &mut rng2);
    assert_eq!(c1.elements.len(), c2.elements.len());
}
