use crate::common;

use ndarray::array;
use num_complex::Complex64;
use rand::SeedableRng;

use yao_rs::apply::apply;
use yao_rs::circuit::{Circuit, control, put};
use yao_rs::gate::Gate;
use yao_rs::measure::{
    MeasureResult, PostProcess, collapse_to, measure, measure_and_collapse, measure_remove,
    measure_reset, measure_with_postprocess, probs,
};
use yao_rs::state::State;
use yao_rs::{ArrayReg, DensityMatrix};

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-10
}

#[test]
fn test_probs_zero_state() {
    let state = State::zero_state(&[2, 2]);
    let p = probs(&state, None);
    assert_eq!(p.len(), 4);
    assert!(approx_eq(p[0], 1.0)); // |00>
    assert!(approx_eq(p[1], 0.0));
    assert!(approx_eq(p[2], 0.0));
    assert!(approx_eq(p[3], 0.0));
}

#[test]
fn test_probs_superposition() {
    // |+> = (|0> + |1>)/sqrt(2)
    let state = State::new(
        vec![2],
        array![
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        ],
    );
    let p = probs(&state, None);
    assert!(approx_eq(p[0], 0.5));
    assert!(approx_eq(p[1], 0.5));
}

#[test]
fn test_probs_bell_state() {
    // Bell state (|00> + |11>)/sqrt(2)
    let circuit = Circuit::new(
        vec![2, 2],
        vec![put(vec![0], Gate::H), control(vec![0], vec![1], Gate::X)],
    )
    .unwrap();
    let state = apply(&circuit, &State::zero_state(&[2, 2]));
    let p = probs(&state, None);
    assert!(approx_eq(p[0], 0.5)); // |00>
    assert!(approx_eq(p[1], 0.0)); // |01>
    assert!(approx_eq(p[2], 0.0)); // |10>
    assert!(approx_eq(p[3], 0.5)); // |11>
}

#[test]
fn test_probs_marginal() {
    // Bell state (|00> + |11>)/sqrt(2)
    let circuit = Circuit::new(
        vec![2, 2],
        vec![put(vec![0], Gate::H), control(vec![0], vec![1], Gate::X)],
    )
    .unwrap();
    let state = apply(&circuit, &State::zero_state(&[2, 2]));

    // Marginal on first qubit
    let p0 = probs(&state, Some(&[0]));
    assert!(approx_eq(p0[0], 0.5)); // P(q0=0)
    assert!(approx_eq(p0[1], 0.5)); // P(q0=1)

    // Marginal on second qubit
    let p1 = probs(&state, Some(&[1]));
    assert!(approx_eq(p1[0], 0.5)); // P(q1=0)
    assert!(approx_eq(p1[1], 0.5)); // P(q1=1)
}

#[test]
fn test_probs_sum_to_one() {
    let circuit = Circuit::new(
        vec![2, 2, 2],
        vec![
            put(vec![0], Gate::H),
            put(vec![1], Gate::H),
            control(vec![0], vec![2], Gate::X),
        ],
    )
    .unwrap();
    let state = apply(&circuit, &State::zero_state(&[2, 2, 2]));
    let p = probs(&state, None);
    let sum: f64 = p.iter().sum();
    assert!(approx_eq(sum, 1.0));
}

#[test]
fn test_measure_deterministic() {
    let state = State::zero_state(&[2, 2]); // |00>
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let results = measure(&state, None, 100, &mut rng);

    for result in results {
        assert_eq!(result, vec![0, 0]);
    }
}

#[test]
fn test_measure_superposition_statistics() {
    // |+> state
    let state = State::new(
        vec![2],
        array![
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        ],
    );

    let mut rng = rand::rngs::StdRng::seed_from_u64(12345);
    let results = measure(&state, None, 1000, &mut rng);

    let count_0 = results.iter().filter(|r| r[0] == 0).count();
    let count_1 = results.iter().filter(|r| r[0] == 1).count();

    // Should be roughly 50-50 (within statistical fluctuation)
    assert!(count_0 > 400 && count_0 < 600);
    assert!(count_1 > 400 && count_1 < 600);
}

#[test]
fn test_measure_subset() {
    // Bell state, measure only first qubit
    let circuit = Circuit::new(
        vec![2, 2],
        vec![put(vec![0], Gate::H), control(vec![0], vec![1], Gate::X)],
    )
    .unwrap();
    let state = apply(&circuit, &State::zero_state(&[2, 2]));

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let results = measure(&state, Some(&[0]), 100, &mut rng);

    // Each result should be a single value (0 or 1)
    for result in &results {
        assert_eq!(result.len(), 1);
        assert!(result[0] == 0 || result[0] == 1);
    }
}

#[test]
fn test_measure_and_collapse_basic() {
    let mut state = State::zero_state(&[2]);
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    let result = measure_and_collapse(&mut state, None, &mut rng);
    assert_eq!(result, vec![0]);

    // State should still be |0>
    let p = probs(&state, None);
    assert!(approx_eq(p[0], 1.0));
}

#[test]
fn test_measure_and_collapse_superposition() {
    // Create |+>
    let circuit = Circuit::new(vec![2], vec![put(vec![0], Gate::H)]).unwrap();
    let mut state = apply(&circuit, &State::zero_state(&[2]));

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let result = measure_and_collapse(&mut state, None, &mut rng);

    // Result should be 0 or 1
    assert!(result == vec![0] || result == vec![1]);

    // State should be collapsed
    let p = probs(&state, None);
    if result[0] == 0 {
        assert!(approx_eq(p[0], 1.0));
        assert!(approx_eq(p[1], 0.0));
    } else {
        assert!(approx_eq(p[0], 0.0));
        assert!(approx_eq(p[1], 1.0));
    }
}

#[test]
fn test_measure_and_collapse_partial() {
    // Bell state
    let circuit = Circuit::new(
        vec![2, 2],
        vec![put(vec![0], Gate::H), control(vec![0], vec![1], Gate::X)],
    )
    .unwrap();
    let mut state = apply(&circuit, &State::zero_state(&[2, 2]));

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let result = measure_and_collapse(&mut state, Some(&[0]), &mut rng);

    // Measuring first qubit collapses both due to entanglement
    let p = probs(&state, None);
    if result[0] == 0 {
        assert!(approx_eq(p[0], 1.0)); // |00>
    } else {
        assert!(approx_eq(p[3], 1.0)); // |11>
    }
}

#[test]
fn test_collapse_to_basic() {
    // Bell state
    let circuit = Circuit::new(
        vec![2, 2],
        vec![put(vec![0], Gate::H), control(vec![0], vec![1], Gate::X)],
    )
    .unwrap();
    let mut state = apply(&circuit, &State::zero_state(&[2, 2]));

    // Collapse to |00>
    collapse_to(&mut state, &[0, 1], &[0, 0]);

    let p = probs(&state, None);
    assert!(approx_eq(p[0], 1.0));
    assert!(approx_eq(p[1], 0.0));
    assert!(approx_eq(p[2], 0.0));
    assert!(approx_eq(p[3], 0.0));
}

#[test]
fn test_collapse_to_partial() {
    // GHZ state (|000> + |111>)/sqrt(2)
    let circuit = Circuit::new(
        vec![2, 2, 2],
        vec![
            put(vec![0], Gate::H),
            control(vec![0], vec![1], Gate::X),
            control(vec![1], vec![2], Gate::X),
        ],
    )
    .unwrap();
    let mut state = apply(&circuit, &State::zero_state(&[2, 2, 2]));

    // Collapse first qubit to 1
    collapse_to(&mut state, &[0], &[1]);

    // Should be |111>
    let p = probs(&state, None);
    assert!(approx_eq(p[7], 1.0)); // |111> = index 7
}

#[test]
fn test_collapse_to_preserves_normalization() {
    let circuit = Circuit::new(
        vec![2, 2],
        vec![put(vec![0], Gate::H), put(vec![1], Gate::H)],
    )
    .unwrap();
    let mut state = apply(&circuit, &State::zero_state(&[2, 2]));

    collapse_to(&mut state, &[0], &[0]);

    let p = probs(&state, None);
    let sum: f64 = p.iter().sum();
    assert!(approx_eq(sum, 1.0));
}

#[test]
fn test_probs_qutrit() {
    // Equal superposition of |0>, |1>, |2>
    let state = State::new(
        vec![3],
        array![
            Complex64::new(1.0 / 3.0_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 3.0_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 3.0_f64.sqrt(), 0.0),
        ],
    );
    let p = probs(&state, None);
    assert_eq!(p.len(), 3);
    for &prob in &p {
        assert!(approx_eq(prob, 1.0 / 3.0));
    }
}

#[test]
fn test_measure_qutrit() {
    let state = State::new(
        vec![3],
        array![
            Complex64::new(1.0 / 3.0_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 3.0_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 3.0_f64.sqrt(), 0.0),
        ],
    );

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let results = measure(&state, None, 100, &mut rng);

    // All results should be 0, 1, or 2
    for result in results {
        assert!(result[0] <= 2);
    }
}

#[test]
fn test_measure_mixed_dims() {
    // Qubit-qutrit system
    let state = State::zero_state(&[2, 3]);
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let results = measure(&state, None, 10, &mut rng);

    for result in results {
        assert_eq!(result, vec![0, 0]);
    }
}

#[test]
fn test_collapse_to_qutrit() {
    let mut state = State::new(
        vec![3],
        array![
            Complex64::new(1.0 / 3.0_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 3.0_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 3.0_f64.sqrt(), 0.0),
        ],
    );

    collapse_to(&mut state, &[0], &[2]);

    let p = probs(&state, None);
    assert!(approx_eq(p[0], 0.0));
    assert!(approx_eq(p[1], 0.0));
    assert!(approx_eq(p[2], 1.0));
}

// ==================== Tests from Yao.jl ====================

// From register.jl: test probs matches abs2 of state
#[test]
fn test_probs_equals_abs2_state() {
    // Random-ish state
    let state = State::new(
        vec![2, 2],
        array![
            Complex64::new(0.5, 0.1),
            Complex64::new(0.3, -0.2),
            Complex64::new(0.4, 0.3),
            Complex64::new(0.1, -0.4),
        ],
    );
    let p = probs(&state, None);

    // probs should equal |amplitude|^2
    for (i, &prob) in p.iter().enumerate() {
        let expected = state.data[i].norm_sqr();
        assert!(
            approx_eq(prob, expected),
            "probs[{}] = {} != |state[{}]|^2 = {}",
            i,
            prob,
            i,
            expected
        );
    }
}

// From measure.jl: measure nshots returns correct count
#[test]
fn test_measure_nshots_count() {
    let state = State::zero_state(&[2, 2, 2]);
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    for nshots in [1, 10, 100] {
        let results = measure(&state, None, nshots, &mut rng);
        assert_eq!(results.len(), nshots);
    }
}

// From measure.jl: measure product state is deterministic
#[test]
fn test_measure_product_state_deterministic() {
    // |101> state
    let state = State::product_state(&[2, 2, 2], &[1, 0, 1]);
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    let results = measure(&state, None, 100, &mut rng);
    for result in results {
        assert_eq!(result, vec![1, 0, 1]);
    }
}

// From measure.jl: test isnormalized after measure
#[test]
fn test_normalized_after_collapse() {
    // Create superposition
    let circuit = Circuit::new(
        vec![2, 2, 2],
        vec![
            put(vec![0], Gate::H),
            put(vec![1], Gate::H),
            put(vec![2], Gate::H),
        ],
    )
    .unwrap();
    let mut state = apply(&circuit, &State::zero_state(&[2, 2, 2]));

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let _result = measure_and_collapse(&mut state, None, &mut rng);

    // State should be normalized
    let norm = state.norm();
    assert!(
        approx_eq(norm, 1.0),
        "State norm after collapse: {} != 1.0",
        norm
    );
}

// From measure.jl: measure and reset to specific value
#[test]
fn test_measure_and_reset() {
    // Create superposition, measure, then reset to |0>
    let circuit = Circuit::new(
        vec![2, 2],
        vec![put(vec![0], Gate::H), put(vec![1], Gate::H)],
    )
    .unwrap();
    let mut state = apply(&circuit, &State::zero_state(&[2, 2]));

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let _result = measure_and_collapse(&mut state, None, &mut rng);

    // Now reset to |01>
    // First create a fresh |01> state and copy amplitudes
    let target = State::product_state(&[2, 2], &[0, 1]);
    state.data.assign(&target.data);

    // Verify state is now |01>
    let p = probs(&state, None);
    assert!(approx_eq(p[1], 1.0)); // |01> = index 1
}

// From measure.jl: measure qudit product state (like dit"121;3")
#[test]
fn test_measure_qudit_product_state() {
    // Qutrit product state |1,2,1> with dims [3,3,3]
    let state = State::product_state(&[3, 3, 3], &[1, 2, 1]);
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    let results = measure(&state, None, 10, &mut rng);
    for result in results {
        assert_eq!(result, vec![1, 2, 1]);
    }
}

// From density_matrix.jl: measure on subset of qubits
#[test]
fn test_measure_subset_deterministic() {
    // |101> state, measure only qubits 0 and 1
    let state = State::product_state(&[2, 2, 2], &[1, 0, 1]);
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    let results = measure(&state, Some(&[0, 1]), 10, &mut rng);
    for result in results {
        assert_eq!(result, vec![1, 0]); // First two qubits are |10>
    }
}

// From register.jl: collapseto test
#[test]
fn test_collapse_to_like_yao() {
    // Create random-ish 4-qubit state
    let circuit = Circuit::new(
        vec![2, 2, 2, 2],
        vec![
            put(vec![0], Gate::H),
            put(vec![1], Gate::H),
            put(vec![2], Gate::H),
            put(vec![3], Gate::H),
        ],
    )
    .unwrap();
    let mut state = apply(&circuit, &State::zero_state(&[2, 2, 2, 2]));

    // Collapse qubits 1 and 3 to |01> (like focus!(reg, (4,2)) then collapseto!(reg, bit"01"))
    collapse_to(&mut state, &[1, 3], &[0, 1]);

    // After collapse, measuring qubits 1 and 3 should always give [0, 1]
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let results = measure(&state, Some(&[1, 3]), 10, &mut rng);
    for result in results {
        assert_eq!(result, vec![0, 1]);
    }
}

// Test probability distribution matches measurements
#[test]
fn test_measurement_matches_probs() {
    // Create state with known probabilities
    // |psi> = sqrt(0.7)|0> + sqrt(0.3)|1>
    let state = State::new(
        vec![2],
        array![
            Complex64::new(0.7_f64.sqrt(), 0.0),
            Complex64::new(0.3_f64.sqrt(), 0.0),
        ],
    );

    let p = probs(&state, None);
    assert!(approx_eq(p[0], 0.7));
    assert!(approx_eq(p[1], 0.3));

    // Measure many times and check distribution
    let mut rng = rand::rngs::StdRng::seed_from_u64(12345);
    let results = measure(&state, None, 10000, &mut rng);

    let count_0 = results.iter().filter(|r| r[0] == 0).count();
    let freq_0 = count_0 as f64 / 10000.0;

    // Should be within 5% of expected (0.7)
    assert!(
        (freq_0 - 0.7).abs() < 0.05,
        "Frequency of 0: {} not close to 0.7",
        freq_0
    );
}

// Test multi-qubit marginal probabilities
#[test]
fn test_marginal_probs_multi_qubit() {
    // Create |GHZ> = (|000> + |111>)/sqrt(2)
    let circuit = Circuit::new(
        vec![2, 2, 2],
        vec![
            put(vec![0], Gate::H),
            control(vec![0], vec![1], Gate::X),
            control(vec![1], vec![2], Gate::X),
        ],
    )
    .unwrap();
    let state = apply(&circuit, &State::zero_state(&[2, 2, 2]));

    // Marginal on qubits 0 and 2
    let p = probs(&state, Some(&[0, 2]));
    assert_eq!(p.len(), 4); // 2x2 = 4 outcomes

    // For GHZ, only |00> and |11> have probability
    assert!(approx_eq(p[0], 0.5)); // |00>
    assert!(approx_eq(p[1], 0.0)); // |01>
    assert!(approx_eq(p[2], 0.0)); // |10>
    assert!(approx_eq(p[3], 0.5)); // |11>
}

// Test measure with all zeros probability edge case
#[test]
fn test_collapse_to_zero_probability() {
    // |00> state
    let mut state = State::zero_state(&[2, 2]);

    // Try to collapse to |11> which has zero probability
    collapse_to(&mut state, &[0, 1], &[1, 1]);

    // State should be zero (not normalized since it had zero probability)
    let norm = state.norm();
    assert!(
        norm < 1e-10,
        "State norm should be ~0 for zero-probability collapse"
    );
}

// Test partial collapse preserves other amplitudes
#[test]
fn test_partial_collapse_preserves_structure() {
    // |psi> = (|00> + |01> + |10> + |11>)/2
    let state = State::new(
        vec![2, 2],
        array![
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
        ],
    );
    let mut collapsed = state.clone();

    // Collapse first qubit to 0
    collapse_to(&mut collapsed, &[0], &[0]);

    let p = probs(&collapsed, None);
    // Should have |00> and |01> with equal probability
    assert!(approx_eq(p[0], 0.5)); // |00>
    assert!(approx_eq(p[1], 0.5)); // |01>
    assert!(approx_eq(p[2], 0.0)); // |10>
    assert!(approx_eq(p[3], 0.0)); // |11>
}

// Test measurement on large system
#[test]
fn test_measure_large_system() {
    // 10-qubit system (like rand_state(10) in Yao tests)
    let dims: Vec<usize> = vec![2; 10];
    let state = State::zero_state(&dims);

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let results = measure(&state, None, 10, &mut rng);

    assert_eq!(results.len(), 10);
    for result in results {
        assert_eq!(result.len(), 10);
        assert_eq!(result, vec![0; 10]); // All zeros
    }
}

// Test measure subset with non-contiguous qubits
#[test]
fn test_measure_non_contiguous_subset() {
    // |1010> state
    let state = State::product_state(&[2, 2, 2, 2], &[1, 0, 1, 0]);
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    // Measure qubits 0 and 2 (non-contiguous)
    let results = measure(&state, Some(&[0, 2]), 10, &mut rng);
    for result in results {
        assert_eq!(result, vec![1, 1]); // Qubits 0 and 2 are both |1>
    }

    // Measure qubits 1 and 3
    let results = measure(&state, Some(&[1, 3]), 10, &mut rng);
    for result in results {
        assert_eq!(result, vec![0, 0]); // Qubits 1 and 3 are both |0>
    }
}

// Test repeated measurements don't change state (for non-collapsing measure)
#[test]
fn test_measure_does_not_collapse() {
    let state = State::new(
        vec![2],
        array![
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        ],
    );

    let p_before = probs(&state, None);

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let _results = measure(&state, None, 1000, &mut rng);

    let p_after = probs(&state, None);

    // Probabilities should be unchanged
    assert!(approx_eq(p_before[0], p_after[0]));
    assert!(approx_eq(p_before[1], p_after[1]));
}

// Test ququart (d=4) measurement
#[test]
fn test_measure_ququart() {
    // Ququart in state |3>
    let state = State::product_state(&[4], &[3]);
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    let results = measure(&state, None, 10, &mut rng);
    for result in results {
        assert_eq!(result, vec![3]);
    }
}

// Test mixed qudit dimensions
#[test]
fn test_measure_mixed_qudit_dims() {
    // Qubit(2) - Qutrit(3) - Ququart(4) system in |1,2,3>
    let state = State::product_state(&[2, 3, 4], &[1, 2, 3]);
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    let results = measure(&state, None, 10, &mut rng);
    for result in results {
        assert_eq!(result, vec![1, 2, 3]);
    }
}

// Test collapse then measure gives consistent results
#[test]
fn test_collapse_then_measure_consistent() {
    let circuit = Circuit::new(
        vec![2, 2],
        vec![put(vec![0], Gate::H), put(vec![1], Gate::H)],
    )
    .unwrap();
    let mut state = apply(&circuit, &State::zero_state(&[2, 2]));

    // Collapse first qubit to 1
    collapse_to(&mut state, &[0], &[1]);

    // Now measure - should always get 1 for first qubit
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let results = measure(&state, Some(&[0]), 100, &mut rng);
    for result in results {
        assert_eq!(result[0], 1);
    }
}

#[test]
fn test_probs_arrayreg_supports_marginals() {
    let reg = ArrayReg::ghz_state(3);
    let p = probs(&reg, Some(&[0, 2]));

    assert_eq!(p.len(), 4);
    assert!(approx_eq(p[0], 0.5));
    assert!(approx_eq(p[1], 0.0));
    assert!(approx_eq(p[2], 0.0));
    assert!(approx_eq(p[3], 0.5));
}

#[test]
fn test_probs_arrayreg_uses_circuit_qubit_ordering() {
    let reg = ArrayReg::from_vec(
        3,
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
    );

    let p = probs(&reg, Some(&[0]));
    assert_eq!(p.len(), 2);
    assert!(approx_eq(p[0], 0.0));
    assert!(approx_eq(p[1], 1.0));
}

#[test]
fn test_probs_density_matrix_uses_diagonal_distribution() {
    let r0 = ArrayReg::from_vec(1, vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]);
    let r1 = ArrayReg::from_vec(1, vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)]);
    let dm = DensityMatrix::mixed(&[0.25, 0.75], &[r0, r1]);

    let p = probs(&dm, None);
    assert_eq!(p.len(), 2);
    assert!(approx_eq(p[0], 0.25));
    assert!(approx_eq(p[1], 0.75));
}

#[test]
fn test_measure_with_postprocess_no_postprocess_preserves_arrayreg() {
    let mut reg = ArrayReg::uniform_state(1);
    let before = reg.clone();
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    let result = measure_with_postprocess(&mut reg, &[0], PostProcess::NoPostProcess, &mut rng);

    match result {
        MeasureResult::Value(bits) => assert!(bits == vec![0] || bits == vec![1]),
        MeasureResult::Removed(_, _) => panic!("unexpected register removal"),
    }
    assert!(approx_eq(reg.fidelity(&before), 1.0));
}

#[test]
fn test_measure_with_postprocess_reset_to_sets_target_value() {
    let circuit = Circuit::new(
        vec![2, 2],
        vec![put(vec![0], Gate::H), control(vec![0], vec![1], Gate::X)],
    )
    .unwrap();
    let mut reg = apply(&circuit, &ArrayReg::zero_state(2));
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    let result = measure_with_postprocess(&mut reg, &[0], PostProcess::ResetTo(0), &mut rng);

    match result {
        MeasureResult::Value(bits) => assert!(bits == vec![0] || bits == vec![1]),
        MeasureResult::Removed(_, _) => panic!("unexpected register removal"),
    }
    assert!(approx_eq(reg.norm(), 1.0));
    let p = probs(&reg, Some(&[0]));
    assert!(approx_eq(p[0], 1.0));
}

#[test]
fn test_measure_with_postprocess_remove_measured_returns_smaller_arrayreg() {
    let mut reg = ArrayReg::ghz_state(2);
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    let result = measure_with_postprocess(&mut reg, &[0], PostProcess::RemoveMeasured, &mut rng);

    match result {
        MeasureResult::Removed(bits, new_reg) => {
            assert!(bits == vec![0] || bits == vec![1]);
            assert_eq!(new_reg.nqubits(), 1);
            assert!(approx_eq(new_reg.norm(), 1.0));
            let p = probs(&new_reg, None);
            assert!(approx_eq(p[bits[0]], 1.0));
        }
        MeasureResult::Value(_) => panic!("expected measured qubit removal"),
    }
}

// ============================================================
// Ground truth validation against Yao.jl
// ============================================================

#[test]
fn test_measure_ground_truth() {
    let data = common::load_measure_data();
    let mut tested = 0;
    for case in &data.cases {
        let dims = vec![2; case.num_qubits];
        let circuit = common::circuit_from_measure_case(case);
        let state = State::zero_state(&dims);
        let result_state = apply(&circuit, &state);
        let result_probs = probs(&result_state, None);

        for (i, (&got, &expected)) in result_probs
            .iter()
            .zip(case.probabilities.iter())
            .enumerate()
        {
            assert!(
                (got - expected).abs() < 1e-10,
                "Case '{}': probability mismatch at index {}: got {}, expected {}",
                case.label,
                i,
                got,
                expected
            );
        }
        tested += 1;
    }
    assert_eq!(tested, 11, "Expected 11 measure cases in ground truth data");
}

// ============================================================
// measure_reset and measure_remove tests
// ============================================================

#[test]
fn test_measure_reset_to_zero() {
    // Create superposition state, measure and reset to 0
    let circuit = Circuit::new(vec![2, 2], vec![put(vec![0], Gate::H)]).unwrap();
    let mut state = apply(&circuit, &State::zero_state(&[2, 2]));

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let result = measure_reset(&mut state, &[0], 0, &mut rng);

    // Measurement result should be a valid qubit outcome
    assert!(
        result[0] == 0 || result[0] == 1,
        "Invalid measurement result"
    );

    // State should be normalized after reset
    assert!(
        (state.norm() - 1.0).abs() < 1e-10,
        "State not normalized after reset"
    );

    // Qubit 0 should be in |0> state
    let p = probs(&state, Some(&[0]));
    assert!(
        (p[0] - 1.0).abs() < 1e-10,
        "Qubit 0 should be |0> after reset to 0"
    );
}

#[test]
fn test_measure_reset_to_one() {
    // Create superposition, measure and reset to 1
    let circuit = Circuit::new(vec![2, 2], vec![put(vec![0], Gate::H)]).unwrap();
    let mut state = apply(&circuit, &State::zero_state(&[2, 2]));

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let result = measure_reset(&mut state, &[0], 1, &mut rng);

    // Measurement result should be a valid qubit outcome
    assert!(
        result[0] == 0 || result[0] == 1,
        "Invalid measurement result"
    );

    assert!(
        (state.norm() - 1.0).abs() < 1e-10,
        "State not normalized after reset"
    );

    let p = probs(&state, Some(&[0]));
    assert!(
        (p[1] - 1.0).abs() < 1e-10,
        "Qubit 0 should be |1> after reset to 1"
    );
}

#[test]
fn test_measure_remove_product_state() {
    let state = State::product_state(&[2, 2, 2], &[1, 0, 1]); // |101>
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    let (result, new_state) = measure_remove(&state, &[1], &mut rng);

    // Measured qubit 1 should be 0
    assert_eq!(result, vec![0]);
    // Remaining state should have 2 qubits
    assert_eq!(new_state.dims.len(), 2);
    assert_eq!(new_state.dims, vec![2, 2]);
    // Remaining state should be |11> (qubits 0 and 2, which were |1> and |1>)
    assert!((new_state.data[3].norm() - 1.0).abs() < 1e-10);
}

#[test]
fn test_measure_remove_bell_state() {
    // Bell state
    let circuit = Circuit::new(
        vec![2, 2],
        vec![put(vec![0], Gate::H), control(vec![0], vec![1], Gate::X)],
    )
    .unwrap();
    let state = apply(&circuit, &State::zero_state(&[2, 2]));

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let (result, new_state) = measure_remove(&state, &[0], &mut rng);

    // Should get a single-qubit state
    assert_eq!(new_state.dims, vec![2]);
    assert!((new_state.norm() - 1.0).abs() < 1e-10);

    // Due to entanglement, remaining qubit should match measured result
    let p = probs(&new_state, None);
    assert!((p[result[0]] - 1.0).abs() < 1e-10);
}
