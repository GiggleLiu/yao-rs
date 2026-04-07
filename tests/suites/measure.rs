use crate::common;

use num_complex::Complex64;
use rand::SeedableRng;

use yao_rs::apply::apply;
use yao_rs::circuit::{Circuit, control, put};
use yao_rs::gate::Gate;
use yao_rs::measure::{MeasureResult, PostProcess, measure_with_postprocess, probs};
use yao_rs::{ArrayReg, DensityMatrix};

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-10
}

/// Helper to create a product state |bits[0], bits[1], ...> for qubits.
fn product_state(nbits: usize, bits: &[usize]) -> ArrayReg {
    let dim = 1usize << nbits;
    let mut state = vec![Complex64::new(0.0, 0.0); dim];
    let mut idx = 0usize;
    for (i, &b) in bits.iter().enumerate() {
        idx |= b << (nbits - 1 - i);
    }
    state[idx] = Complex64::new(1.0, 0.0);
    ArrayReg::from_vec(nbits, state)
}

#[test]
fn test_probs_zero_state() {
    let reg = ArrayReg::zero_state(2);
    let p = probs(&reg, None);
    assert_eq!(p.len(), 4);
    assert!(approx_eq(p[0], 1.0)); // |00>
    assert!(approx_eq(p[1], 0.0));
    assert!(approx_eq(p[2], 0.0));
    assert!(approx_eq(p[3], 0.0));
}

#[test]
fn test_probs_superposition() {
    // |+> = (|0> + |1>)/sqrt(2)
    let reg = ArrayReg::from_vec(
        1,
        vec![
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        ],
    );
    let p = probs(&reg, None);
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
    let reg = apply(&circuit, &ArrayReg::zero_state(2));
    let p = probs(&reg, None);
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
    let reg = apply(&circuit, &ArrayReg::zero_state(2));

    // Marginal on first qubit
    let p0 = probs(&reg, Some(&[0]));
    assert!(approx_eq(p0[0], 0.5)); // P(q0=0)
    assert!(approx_eq(p0[1], 0.5)); // P(q0=1)

    // Marginal on second qubit
    let p1 = probs(&reg, Some(&[1]));
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
    let reg = apply(&circuit, &ArrayReg::zero_state(3));
    let p = probs(&reg, None);
    let sum: f64 = p.iter().sum();
    assert!(approx_eq(sum, 1.0));
}

#[test]
fn test_measure_deterministic() {
    // |00> is deterministic — probs should be all on index 0
    let reg = ArrayReg::zero_state(2);
    let p = probs(&reg, None);
    assert!(approx_eq(p[0], 1.0));
    for pi in p.iter().skip(1) {
        assert!(approx_eq(*pi, 0.0));
    }
}

#[test]
fn test_measure_superposition_statistics() {
    // |+> state — sample many times via measure_with_postprocess
    let reg = ArrayReg::from_vec(
        1,
        vec![
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        ],
    );

    let mut rng = rand::rngs::StdRng::seed_from_u64(12345);
    let mut count_0 = 0usize;
    let mut count_1 = 0usize;
    for _ in 0..1000 {
        let mut reg_copy = reg.clone();
        let result =
            measure_with_postprocess(&mut reg_copy, &[0], PostProcess::NoPostProcess, &mut rng);
        match result {
            MeasureResult::Value(bits) => {
                if bits[0] == 0 {
                    count_0 += 1;
                } else {
                    count_1 += 1;
                }
            }
            _ => panic!("unexpected result"),
        }
    }

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
    let reg = apply(&circuit, &ArrayReg::zero_state(2));

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    for _ in 0..100 {
        let mut reg_copy = reg.clone();
        let result =
            measure_with_postprocess(&mut reg_copy, &[0], PostProcess::NoPostProcess, &mut rng);
        match result {
            MeasureResult::Value(bits) => {
                assert_eq!(bits.len(), 1);
                assert!(bits[0] == 0 || bits[0] == 1);
            }
            _ => panic!("unexpected result"),
        }
    }
}

#[test]
fn test_measure_and_collapse_basic() {
    // |0> — measure with ResetTo(0) to simulate collapse
    let mut reg = ArrayReg::zero_state(1);
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    let result = measure_with_postprocess(&mut reg, &[0], PostProcess::ResetTo(0), &mut rng);
    match result {
        MeasureResult::Value(bits) => assert_eq!(bits, vec![0]),
        _ => panic!("unexpected result"),
    }

    // State should still be |0>
    let p = probs(&reg, None);
    assert!(approx_eq(p[0], 1.0));
}

#[test]
fn test_measure_and_collapse_superposition() {
    // Create |+>
    let circuit = Circuit::new(vec![2], vec![put(vec![0], Gate::H)]).unwrap();
    let mut reg = apply(&circuit, &ArrayReg::zero_state(1));

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    // Use ResetTo to collapse: measure and reset to the measured value
    // We don't know which value, so we use NoPostProcess and manually check probs
    let result = measure_with_postprocess(&mut reg, &[0], PostProcess::NoPostProcess, &mut rng);

    // Result should be 0 or 1
    match result {
        MeasureResult::Value(bits) => {
            assert!(bits == vec![0] || bits == vec![1]);
            // NoPostProcess does NOT collapse the state, so we verify sampling is valid
        }
        _ => panic!("unexpected result"),
    }
}

#[test]
fn test_measure_and_collapse_partial() {
    // Bell state — measure first qubit with ResetTo to collapse
    let circuit = Circuit::new(
        vec![2, 2],
        vec![put(vec![0], Gate::H), control(vec![0], vec![1], Gate::X)],
    )
    .unwrap();

    // Run many times to see both outcomes
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    for _ in 0..20 {
        let mut reg = apply(&circuit, &ArrayReg::zero_state(2));
        let result = measure_with_postprocess(&mut reg, &[0], PostProcess::ResetTo(0), &mut rng);
        match result {
            MeasureResult::Value(bits) => {
                // Measuring first qubit of Bell state should give 0 or 1
                assert!(bits[0] == 0 || bits[0] == 1);
            }
            _ => panic!("unexpected result"),
        }
        // After ResetTo(0), qubit 0 should be deterministically 0
        let p = probs(&reg, Some(&[0]));
        assert!(approx_eq(p[0], 1.0));
    }
}

// ==================== Tests from Yao.jl ====================

// From register.jl: test probs matches abs2 of state
#[test]
fn test_probs_equals_abs2_state() {
    // Random-ish state
    let reg = ArrayReg::from_vec(
        2,
        vec![
            Complex64::new(0.5, 0.1),
            Complex64::new(0.3, -0.2),
            Complex64::new(0.4, 0.3),
            Complex64::new(0.1, -0.4),
        ],
    );
    let p = probs(&reg, None);

    // probs should equal |amplitude|^2
    for (i, &prob) in p.iter().enumerate() {
        let expected = reg.state_vec()[i].norm_sqr();
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
    let reg = ArrayReg::zero_state(3);
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    for nshots in [1, 10, 100] {
        let mut results = Vec::new();
        for _ in 0..nshots {
            let mut reg_copy = reg.clone();
            let result = measure_with_postprocess(
                &mut reg_copy,
                &[0, 1, 2],
                PostProcess::NoPostProcess,
                &mut rng,
            );
            results.push(result);
        }
        assert_eq!(results.len(), nshots);
    }
}

// From measure.jl: measure product state is deterministic
#[test]
fn test_measure_product_state_deterministic() {
    // |101> state
    let reg = product_state(3, &[1, 0, 1]);

    // Verify probs are deterministic
    let p = probs(&reg, None);
    // |101> = index 5 (bit pattern: qubit0=1 at MSB)
    assert!(approx_eq(p[5], 1.0));
    for (i, &prob) in p.iter().enumerate() {
        if i != 5 {
            assert!(approx_eq(prob, 0.0));
        }
    }

    // Also verify via sampling
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    for _ in 0..100 {
        let mut reg_copy = reg.clone();
        let result = measure_with_postprocess(
            &mut reg_copy,
            &[0, 1, 2],
            PostProcess::NoPostProcess,
            &mut rng,
        );
        match result {
            MeasureResult::Value(bits) => assert_eq!(bits, vec![1, 0, 1]),
            _ => panic!("unexpected result"),
        }
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
    let mut reg = apply(&circuit, &ArrayReg::zero_state(3));

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let _result = measure_with_postprocess(&mut reg, &[0, 1, 2], PostProcess::ResetTo(0), &mut rng);

    // State should be normalized
    let norm = reg.norm();
    assert!(
        approx_eq(norm, 1.0),
        "State norm after collapse: {} != 1.0",
        norm
    );
}

// From measure.jl: measure and reset to specific value
#[test]
fn test_measure_and_reset() {
    // Create superposition, measure, then reset to |01>
    let circuit = Circuit::new(
        vec![2, 2],
        vec![put(vec![0], Gate::H), put(vec![1], Gate::H)],
    )
    .unwrap();
    let mut reg = apply(&circuit, &ArrayReg::zero_state(2));

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    // Measure all qubits and reset to value 1 (= |01> in 2-qubit encoding)
    let _result = measure_with_postprocess(&mut reg, &[0, 1], PostProcess::ResetTo(1), &mut rng);

    // Verify state is now |01>
    let p = probs(&reg, None);
    assert!(approx_eq(p[1], 1.0)); // |01> = index 1
}

// From density_matrix.jl: measure on subset of qubits
#[test]
fn test_measure_subset_deterministic() {
    // |101> state, measure only qubits 0 and 1
    let reg = product_state(3, &[1, 0, 1]);
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    for _ in 0..10 {
        let mut reg_copy = reg.clone();
        let result =
            measure_with_postprocess(&mut reg_copy, &[0, 1], PostProcess::NoPostProcess, &mut rng);
        match result {
            MeasureResult::Value(bits) => {
                assert_eq!(bits, vec![1, 0]); // First two qubits are |10>
            }
            _ => panic!("unexpected result"),
        }
    }
}

// Test probability distribution matches measurements
#[test]
fn test_measurement_matches_probs() {
    // Create state with known probabilities
    // |psi> = sqrt(0.7)|0> + sqrt(0.3)|1>
    let reg = ArrayReg::from_vec(
        1,
        vec![
            Complex64::new(0.7_f64.sqrt(), 0.0),
            Complex64::new(0.3_f64.sqrt(), 0.0),
        ],
    );

    let p = probs(&reg, None);
    assert!(approx_eq(p[0], 0.7));
    assert!(approx_eq(p[1], 0.3));

    // Measure many times and check distribution
    let mut rng = rand::rngs::StdRng::seed_from_u64(12345);
    let mut count_0 = 0usize;
    for _ in 0..10000 {
        let mut reg_copy = reg.clone();
        let result =
            measure_with_postprocess(&mut reg_copy, &[0], PostProcess::NoPostProcess, &mut rng);
        match result {
            MeasureResult::Value(bits) => {
                if bits[0] == 0 {
                    count_0 += 1;
                }
            }
            _ => panic!("unexpected result"),
        }
    }
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
    let reg = apply(&circuit, &ArrayReg::zero_state(3));

    // Marginal on qubits 0 and 2
    let p = probs(&reg, Some(&[0, 2]));
    assert_eq!(p.len(), 4); // 2x2 = 4 outcomes

    // For GHZ, only |00> and |11> have probability
    assert!(approx_eq(p[0], 0.5)); // |00>
    assert!(approx_eq(p[1], 0.0)); // |01>
    assert!(approx_eq(p[2], 0.0)); // |10>
    assert!(approx_eq(p[3], 0.5)); // |11>
}

// Test partial collapse preserves other amplitudes
#[test]
fn test_partial_collapse_preserves_structure() {
    // |psi> = (|00> + |01> + |10> + |11>)/2
    let reg = ArrayReg::from_vec(
        2,
        vec![
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
        ],
    );

    // Measure first qubit with ResetTo(0) — collapses q0 to 0
    let mut collapsed = reg.clone();
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let _result = measure_with_postprocess(&mut collapsed, &[0], PostProcess::ResetTo(0), &mut rng);

    let p = probs(&collapsed, None);
    // After reset to 0 for qubit 0, should have |00> and |01> with equal probability
    assert!(approx_eq(p[0], 0.5)); // |00>
    assert!(approx_eq(p[1], 0.5)); // |01>
    assert!(approx_eq(p[2], 0.0)); // |10>
    assert!(approx_eq(p[3], 0.0)); // |11>
}

// Test measurement on large system
#[test]
fn test_measure_large_system() {
    // 10-qubit zero state
    let reg = ArrayReg::zero_state(10);
    let p = probs(&reg, None);

    assert_eq!(p.len(), 1024);
    assert!(approx_eq(p[0], 1.0)); // All zeros
    for pi in p.iter().skip(1) {
        assert!(approx_eq(*pi, 0.0));
    }
}

// Test measure subset with non-contiguous qubits
#[test]
fn test_measure_non_contiguous_subset() {
    // |1010> state
    let reg = product_state(4, &[1, 0, 1, 0]);

    // Marginal probs on qubits 0 and 2 (non-contiguous)
    let p02 = probs(&reg, Some(&[0, 2]));
    assert!(approx_eq(p02[3], 1.0)); // Qubits 0 and 2 are both |1> → index |11>=3

    // Marginal probs on qubits 1 and 3
    let p13 = probs(&reg, Some(&[1, 3]));
    assert!(approx_eq(p13[0], 1.0)); // Qubits 1 and 3 are both |0> → index |00>=0
}

// Test repeated measurements don't change state (for non-collapsing measure)
#[test]
fn test_measure_does_not_collapse() {
    let reg = ArrayReg::from_vec(
        1,
        vec![
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        ],
    );

    let p_before = probs(&reg, None);

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    for _ in 0..1000 {
        let mut reg_copy = reg.clone();
        let _result =
            measure_with_postprocess(&mut reg_copy, &[0], PostProcess::NoPostProcess, &mut rng);
    }

    let p_after = probs(&reg, None);

    // Probabilities should be unchanged (we measured copies, not the original)
    assert!(approx_eq(p_before[0], p_after[0]));
    assert!(approx_eq(p_before[1], p_after[1]));
}

// Test collapse then measure gives consistent results
#[test]
fn test_collapse_then_measure_consistent() {
    let circuit = Circuit::new(
        vec![2, 2],
        vec![put(vec![0], Gate::H), put(vec![1], Gate::H)],
    )
    .unwrap();
    let mut reg = apply(&circuit, &ArrayReg::zero_state(2));

    // Measure and reset qubit 0 to 1
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let _result = measure_with_postprocess(&mut reg, &[0], PostProcess::ResetTo(1), &mut rng);

    // Now measure qubit 0 — should always get 1
    for _ in 0..100 {
        let mut reg_copy = reg.clone();
        let result =
            measure_with_postprocess(&mut reg_copy, &[0], PostProcess::NoPostProcess, &mut rng);
        match result {
            MeasureResult::Value(bits) => assert_eq!(bits[0], 1),
            _ => panic!("unexpected result"),
        }
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
        let circuit = common::circuit_from_measure_case(case);
        let reg = ArrayReg::zero_state(case.num_qubits);
        let result_reg = apply(&circuit, &reg);
        let result_probs = probs(&result_reg, None);

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
    let mut reg = apply(&circuit, &ArrayReg::zero_state(2));

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let result = measure_with_postprocess(&mut reg, &[0], PostProcess::ResetTo(0), &mut rng);

    // Measurement result should be a valid qubit outcome
    match result {
        MeasureResult::Value(bits) => {
            assert!(bits[0] == 0 || bits[0] == 1, "Invalid measurement result");
        }
        _ => panic!("unexpected result"),
    }

    // State should be normalized after reset
    assert!(
        (reg.norm() - 1.0).abs() < 1e-10,
        "State not normalized after reset"
    );

    // Qubit 0 should be in |0> state
    let p = probs(&reg, Some(&[0]));
    assert!(
        (p[0] - 1.0).abs() < 1e-10,
        "Qubit 0 should be |0> after reset to 0"
    );
}

#[test]
fn test_measure_reset_to_one() {
    // Create superposition, measure and reset to 1
    let circuit = Circuit::new(vec![2, 2], vec![put(vec![0], Gate::H)]).unwrap();
    let mut reg = apply(&circuit, &ArrayReg::zero_state(2));

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let result = measure_with_postprocess(&mut reg, &[0], PostProcess::ResetTo(1), &mut rng);

    // Measurement result should be a valid qubit outcome
    match result {
        MeasureResult::Value(bits) => {
            assert!(bits[0] == 0 || bits[0] == 1, "Invalid measurement result");
        }
        _ => panic!("unexpected result"),
    }

    assert!(
        (reg.norm() - 1.0).abs() < 1e-10,
        "State not normalized after reset"
    );

    let p = probs(&reg, Some(&[0]));
    assert!(
        (p[1] - 1.0).abs() < 1e-10,
        "Qubit 0 should be |1> after reset to 1"
    );
}

#[test]
fn test_measure_remove_product_state() {
    let mut reg = product_state(3, &[1, 0, 1]); // |101>
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    let result = measure_with_postprocess(&mut reg, &[1], PostProcess::RemoveMeasured, &mut rng);

    match result {
        MeasureResult::Removed(bits, new_reg) => {
            // Measured qubit 1 should be 0
            assert_eq!(bits, vec![0]);
            // Remaining state should have 2 qubits
            assert_eq!(new_reg.nqubits(), 2);
            // Remaining state should be |11> (qubits 0 and 2, which were |1> and |1>)
            assert!((new_reg.state_vec()[3].norm() - 1.0).abs() < 1e-10);
        }
        MeasureResult::Value(_) => panic!("expected measured qubit removal"),
    }
}

#[test]
fn test_measure_remove_bell_state() {
    // Bell state
    let circuit = Circuit::new(
        vec![2, 2],
        vec![put(vec![0], Gate::H), control(vec![0], vec![1], Gate::X)],
    )
    .unwrap();
    let mut reg = apply(&circuit, &ArrayReg::zero_state(2));

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let result = measure_with_postprocess(&mut reg, &[0], PostProcess::RemoveMeasured, &mut rng);

    match result {
        MeasureResult::Removed(bits, new_reg) => {
            // Should get a single-qubit state
            assert_eq!(new_reg.nqubits(), 1);
            assert!((new_reg.norm() - 1.0).abs() < 1e-10);

            // Due to entanglement, remaining qubit should match measured result
            let p = probs(&new_reg, None);
            assert!((p[bits[0]] - 1.0).abs() < 1e-10);
        }
        MeasureResult::Value(_) => panic!("expected measured qubit removal"),
    }
}
