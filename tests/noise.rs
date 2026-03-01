use ndarray::Array2;
use num_complex::Complex64;

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

fn assert_matrix_approx(a: &Array2<Complex64>, b: &Array2<Complex64>, tol: f64) {
    assert_eq!(a.shape(), b.shape(), "Shape mismatch");
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).norm() < tol, "Element mismatch: {} vs {}", x, y);
    }
}

#[test]
fn test_phase_amplitude_damping_kraus() {
    use yao_rs::noise::NoiseChannel;

    // PhaseAmplitudeDamping(a=0.3, b=0.2, p1=0.0)
    // Julia ref: errortypes.jl:271-296
    // Expected: A0, A1, A2 only (p1=0)
    let ch = NoiseChannel::PhaseAmplitudeDamping {
        amplitude: 0.3,
        phase: 0.2,
        excited_population: 0.0,
    };
    let kraus = ch.kraus_operators();

    // A0 = sqrt(1-p1) * [[1, 0], [0, sqrt(1-a-b)]] = [[1,0],[0,sqrt(0.5)]]
    let expected_a0 = Array2::from_shape_vec(
        (2, 2),
        vec![
            c(1.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.5_f64.sqrt(), 0.0),
        ],
    )
    .unwrap();
    assert_matrix_approx(&kraus[0], &expected_a0, 1e-10);

    // A1 = sqrt(1-p1) * [[0, sqrt(a)], [0, 0]] = [[0, sqrt(0.3)], [0, 0]]
    let expected_a1 = Array2::from_shape_vec(
        (2, 2),
        vec![
            c(0.0, 0.0),
            c(0.3_f64.sqrt(), 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
        ],
    )
    .unwrap();
    assert_matrix_approx(&kraus[1], &expected_a1, 1e-10);

    // A2 = sqrt(1-p1) * [[0, 0], [0, sqrt(b)]] = [[0, 0], [0, sqrt(0.2)]]
    let expected_a2 = Array2::from_shape_vec(
        (2, 2),
        vec![
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.2_f64.sqrt(), 0.0),
        ],
    )
    .unwrap();
    assert_matrix_approx(&kraus[2], &expected_a2, 1e-10);

    assert_eq!(kraus.len(), 3);
}

#[test]
fn test_phase_amplitude_damping_with_excited_pop() {
    use yao_rs::noise::NoiseChannel;

    // PhaseAmplitudeDamping(a=0.3, b=0.2, p1=0.4)
    // Should produce 6 Kraus operators (A0,A1,A2,B0,B1,B2)
    let ch = NoiseChannel::PhaseAmplitudeDamping {
        amplitude: 0.3,
        phase: 0.2,
        excited_population: 0.4,
    };
    let kraus = ch.kraus_operators();
    assert_eq!(kraus.len(), 6);

    // Verify completeness: sum_i K_i^dag K_i = I
    let mut sum = Array2::<Complex64>::zeros((2, 2));
    for k in &kraus {
        let kdag = k.t().mapv(|c| c.conj());
        sum = sum + kdag.dot(k);
    }
    let eye = Array2::from_diag(&ndarray::arr1(&[c(1.0, 0.0), c(1.0, 0.0)]));
    assert_matrix_approx(&sum, &eye, 1e-10);
}

#[test]
fn test_noise_channel_num_qubits() {
    use yao_rs::noise::NoiseChannel;

    assert_eq!(
        NoiseChannel::PhaseAmplitudeDamping {
            amplitude: 0.1,
            phase: 0.1,
            excited_population: 0.0
        }
        .num_qubits(),
        1
    );
}

#[test]
fn test_bit_flip_kraus() {
    use yao_rs::noise::NoiseChannel;

    let ch = NoiseChannel::BitFlip { p: 0.1 };
    let kraus = ch.kraus_operators();
    assert_eq!(kraus.len(), 2);

    // K0 = sqrt(0.9)*I
    assert!((kraus[0][[0, 0]].re - 0.9_f64.sqrt()).abs() < 1e-10);
    assert!((kraus[0][[1, 1]].re - 0.9_f64.sqrt()).abs() < 1e-10);
    // K1 = sqrt(0.1)*X
    assert!((kraus[1][[0, 1]].re - 0.1_f64.sqrt()).abs() < 1e-10);
    assert!((kraus[1][[1, 0]].re - 0.1_f64.sqrt()).abs() < 1e-10);

    verify_completeness(&kraus);
}

#[test]
fn test_phase_flip_kraus() {
    use yao_rs::noise::NoiseChannel;

    let ch = NoiseChannel::PhaseFlip { p: 0.2 };
    let kraus = ch.kraus_operators();
    assert_eq!(kraus.len(), 2);
    verify_completeness(&kraus);

    // K1 = sqrt(0.2)*Z → [[sqrt(0.2), 0], [0, -sqrt(0.2)]]
    assert!((kraus[1][[0, 0]].re - 0.2_f64.sqrt()).abs() < 1e-10);
    assert!((kraus[1][[1, 1]].re + 0.2_f64.sqrt()).abs() < 1e-10);
}

#[test]
fn test_pauli_channel_kraus() {
    use yao_rs::noise::NoiseChannel;

    let ch = NoiseChannel::PauliChannel {
        px: 0.1,
        py: 0.2,
        pz: 0.05,
    };
    let kraus = ch.kraus_operators();
    assert_eq!(kraus.len(), 4);
    verify_completeness(&kraus);
}

#[test]
fn test_depolarizing_single_qubit_kraus() {
    use yao_rs::noise::NoiseChannel;

    let ch = NoiseChannel::Depolarizing { n: 1, p: 0.1 };
    let kraus = ch.kraus_operators();
    assert_eq!(kraus.len(), 4);
    verify_completeness(&kraus);
}

#[test]
fn test_depolarizing_two_qubit_kraus() {
    use yao_rs::noise::NoiseChannel;

    let ch = NoiseChannel::Depolarizing { n: 2, p: 0.1 };
    let kraus = ch.kraus_operators();
    assert_eq!(kraus.len(), 16);

    // Verify each is 4x4
    for k in &kraus {
        assert_eq!(k.shape(), &[4, 4]);
    }
    verify_completeness(&kraus);
}

#[test]
fn test_reset_kraus() {
    use yao_rs::noise::NoiseChannel;

    let ch = NoiseChannel::Reset { p0: 0.1, p1: 0.05 };
    let kraus = ch.kraus_operators();
    // 1 (identity) + 2 (p0>0) + 2 (p1>0) = 5
    assert_eq!(kraus.len(), 5);
    verify_completeness(&kraus);
}

#[test]
fn test_reset_only_p0() {
    use yao_rs::noise::NoiseChannel;

    let ch = NoiseChannel::Reset { p0: 0.3, p1: 0.0 };
    let kraus = ch.kraus_operators();
    assert_eq!(kraus.len(), 3); // identity + P0 + Pd
    verify_completeness(&kraus);
}

#[test]
fn test_amplitude_damping_kraus() {
    use yao_rs::noise::NoiseChannel;

    let ch = NoiseChannel::AmplitudeDamping {
        gamma: 0.3,
        excited_population: 0.0,
    };
    let kraus = ch.kraus_operators();
    assert_eq!(kraus.len(), 2); // A0 and A1
    verify_completeness(&kraus);
}

#[test]
fn test_phase_damping_kraus() {
    use yao_rs::noise::NoiseChannel;

    let ch = NoiseChannel::PhaseDamping { gamma: 0.2 };
    let kraus = ch.kraus_operators();
    assert_eq!(kraus.len(), 2); // A0 and A2
    verify_completeness(&kraus);
}

#[test]
fn test_thermal_relaxation_kraus() {
    use yao_rs::noise::NoiseChannel;

    let ch = NoiseChannel::ThermalRelaxation {
        t1: 100.0,
        t2: 80.0,
        time: 10.0,
        excited_population: 0.0,
    };
    let kraus = ch.kraus_operators();
    assert!(kraus.len() >= 2);
    verify_completeness(&kraus);
}

#[test]
fn test_coherent_kraus() {
    use yao_rs::noise::NoiseChannel;

    // Coherent error = single unitary (e.g., small rotation)
    let theta = 0.01_f64;
    let matrix = Array2::from_shape_vec(
        (2, 2),
        vec![
            c(theta.cos(), 0.0),
            c(0.0, -theta.sin()),
            c(0.0, -theta.sin()),
            c(theta.cos(), 0.0),
        ],
    )
    .unwrap();
    let ch = NoiseChannel::Coherent {
        matrix: matrix.clone(),
    };
    let kraus = ch.kraus_operators();
    assert_eq!(kraus.len(), 1);
    assert_matrix_approx(&kraus[0], &matrix, 1e-10);
}

#[test]
fn test_superop_from_kraus() {
    use yao_rs::noise::NoiseChannel;

    let ch = NoiseChannel::BitFlip { p: 0.1 };
    let superop = ch.superop();
    assert_eq!(superop.shape(), &[4, 4]);

    // Verify: superop = sum_i kron(conj(K_i), K_i)
    let kraus = ch.kraus_operators();
    let mut expected = Array2::<Complex64>::zeros((4, 4));
    for k in &kraus {
        let kc = k.mapv(|c| c.conj());
        for i in 0..2 {
            for j in 0..2 {
                for m in 0..2 {
                    for n in 0..2 {
                        expected[[i * 2 + m, j * 2 + n]] += kc[[i, j]] * k[[m, n]];
                    }
                }
            }
        }
    }
    assert_matrix_approx(&superop, &expected, 1e-10);
}

/// Verify Kraus completeness: sum_i K_i^dag K_i = I
fn verify_completeness(kraus: &[Array2<Complex64>]) {
    let d = kraus[0].nrows();
    let mut sum = Array2::<Complex64>::zeros((d, d));
    for k in kraus {
        let kdag = k.t().mapv(|c| c.conj());
        sum = sum + kdag.dot(k);
    }
    let eye = Array2::from_diag(&ndarray::Array1::from_vec(vec![c(1.0, 0.0); d]));
    assert_matrix_approx(&sum, &eye, 1e-10);
}

// =========================================================================
// Julia fixture cross-validation tests
// =========================================================================

/// Load noise fixture data from tests/data/noise.json
fn load_noise_fixtures() -> serde_json::Value {
    let data = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/noise.json"
    ))
    .unwrap();
    serde_json::from_str(&data).unwrap()
}

/// Parse a 2D matrix from fixture JSON {re: [[...]], im: [[...]]}
fn parse_matrix(val: &serde_json::Value) -> Array2<Complex64> {
    let re = val["re"].as_array().unwrap();
    let im = val["im"].as_array().unwrap();
    let rows = re.len();
    let cols = re[0].as_array().unwrap().len();
    let mut mat = Array2::zeros((rows, cols));
    for i in 0..rows {
        for j in 0..cols {
            let r = re[i].as_array().unwrap()[j].as_f64().unwrap();
            let m = im[i].as_array().unwrap()[j].as_f64().unwrap();
            mat[[i, j]] = c(r, m);
        }
    }
    mat
}

#[test]
fn test_kraus_matches_julia_bit_flip() {
    use yao_rs::noise::NoiseChannel;
    let fixtures = load_noise_fixtures();
    let fixture = &fixtures["kraus"]["bit_flip_0.1"];

    let ch = NoiseChannel::BitFlip { p: 0.1 };
    let kraus = ch.kraus_operators();

    assert_eq!(kraus.len(), fixture["num_kraus"].as_u64().unwrap() as usize);
    for (i, k) in kraus.iter().enumerate() {
        let expected = parse_matrix(&fixture["kraus"][i]);
        assert_matrix_approx(k, &expected, 1e-10);
    }
}

#[test]
fn test_kraus_matches_julia_phase_flip() {
    use yao_rs::noise::NoiseChannel;
    let fixtures = load_noise_fixtures();
    let fixture = &fixtures["kraus"]["phase_flip_0.2"];

    let ch = NoiseChannel::PhaseFlip { p: 0.2 };
    let kraus = ch.kraus_operators();

    assert_eq!(kraus.len(), fixture["num_kraus"].as_u64().unwrap() as usize);
    for (i, k) in kraus.iter().enumerate() {
        let expected = parse_matrix(&fixture["kraus"][i]);
        assert_matrix_approx(k, &expected, 1e-10);
    }
}

#[test]
fn test_kraus_matches_julia_pauli_channel() {
    use yao_rs::noise::NoiseChannel;
    let fixtures = load_noise_fixtures();
    let fixture = &fixtures["kraus"]["pauli_0.1_0.2_0.05"];

    let ch = NoiseChannel::PauliChannel {
        px: 0.1,
        py: 0.2,
        pz: 0.05,
    };
    let kraus = ch.kraus_operators();

    assert_eq!(kraus.len(), fixture["num_kraus"].as_u64().unwrap() as usize);
    for (i, k) in kraus.iter().enumerate() {
        let expected = parse_matrix(&fixture["kraus"][i]);
        assert_matrix_approx(k, &expected, 1e-10);
    }
}

#[test]
fn test_kraus_matches_julia_amplitude_damping() {
    use yao_rs::noise::NoiseChannel;
    let fixtures = load_noise_fixtures();
    let fixture = &fixtures["kraus"]["amplitude_damping_0.3_0.0"];

    let ch = NoiseChannel::AmplitudeDamping {
        gamma: 0.3,
        excited_population: 0.0,
    };
    let kraus = ch.kraus_operators();

    assert_eq!(kraus.len(), fixture["num_kraus"].as_u64().unwrap() as usize);
    for (i, k) in kraus.iter().enumerate() {
        let expected = parse_matrix(&fixture["kraus"][i]);
        assert_matrix_approx(k, &expected, 1e-10);
    }
}

#[test]
fn test_kraus_matches_julia_phase_damping() {
    use yao_rs::noise::NoiseChannel;
    let fixtures = load_noise_fixtures();
    let fixture = &fixtures["kraus"]["phase_damping_0.2"];

    let ch = NoiseChannel::PhaseDamping { gamma: 0.2 };
    let kraus = ch.kraus_operators();

    assert_eq!(kraus.len(), fixture["num_kraus"].as_u64().unwrap() as usize);
    for (i, k) in kraus.iter().enumerate() {
        let expected = parse_matrix(&fixture["kraus"][i]);
        assert_matrix_approx(k, &expected, 1e-10);
    }
}

#[test]
fn test_kraus_matches_julia_phase_amp_damping() {
    use yao_rs::noise::NoiseChannel;
    let fixtures = load_noise_fixtures();
    let fixture = &fixtures["kraus"]["phase_amp_damping_0.3_0.2_0.0"];

    let ch = NoiseChannel::PhaseAmplitudeDamping {
        amplitude: 0.3,
        phase: 0.2,
        excited_population: 0.0,
    };
    let kraus = ch.kraus_operators();

    assert_eq!(kraus.len(), fixture["num_kraus"].as_u64().unwrap() as usize);
    for (i, k) in kraus.iter().enumerate() {
        let expected = parse_matrix(&fixture["kraus"][i]);
        assert_matrix_approx(k, &expected, 1e-10);
    }
}

#[test]
fn test_kraus_matches_julia_phase_amp_damping_excited() {
    use yao_rs::noise::NoiseChannel;
    let fixtures = load_noise_fixtures();
    let fixture = &fixtures["kraus"]["phase_amp_damping_0.3_0.2_0.4"];

    let ch = NoiseChannel::PhaseAmplitudeDamping {
        amplitude: 0.3,
        phase: 0.2,
        excited_population: 0.4,
    };
    let kraus = ch.kraus_operators();

    assert_eq!(kraus.len(), fixture["num_kraus"].as_u64().unwrap() as usize);
    for (i, k) in kraus.iter().enumerate() {
        let expected = parse_matrix(&fixture["kraus"][i]);
        assert_matrix_approx(k, &expected, 1e-10);
    }
}

#[test]
fn test_kraus_matches_julia_thermal_relaxation() {
    use yao_rs::noise::NoiseChannel;
    let fixtures = load_noise_fixtures();
    let fixture = &fixtures["kraus"]["thermal_relaxation_100_80_10_0.0"];

    let ch = NoiseChannel::ThermalRelaxation {
        t1: 100.0,
        t2: 80.0,
        time: 10.0,
        excited_population: 0.0,
    };
    let kraus = ch.kraus_operators();

    assert_eq!(kraus.len(), fixture["num_kraus"].as_u64().unwrap() as usize);
    for (i, k) in kraus.iter().enumerate() {
        let expected = parse_matrix(&fixture["kraus"][i]);
        assert_matrix_approx(k, &expected, 1e-10);
    }
}

#[test]
fn test_superop_matches_julia_bit_flip() {
    use yao_rs::noise::NoiseChannel;
    let fixtures = load_noise_fixtures();
    let expected = parse_matrix(&fixtures["kraus"]["bit_flip_0.1"]["superop"]);

    let ch = NoiseChannel::BitFlip { p: 0.1 };
    let superop = ch.superop();
    assert_matrix_approx(&superop, &expected, 1e-10);
}

#[test]
fn test_superop_matches_julia_depolarizing() {
    use yao_rs::noise::NoiseChannel;
    let fixtures = load_noise_fixtures();
    let expected = parse_matrix(&fixtures["kraus"]["depolarizing_1_0.1"]["superop"]);

    let ch = NoiseChannel::Depolarizing { n: 1, p: 0.1 };
    let superop = ch.superop();
    assert_matrix_approx(&superop, &expected, 1e-10);
}
