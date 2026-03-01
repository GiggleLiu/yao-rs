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
        vec![c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.5_f64.sqrt(), 0.0)],
    ).unwrap();
    assert_matrix_approx(&kraus[0], &expected_a0, 1e-10);

    // A1 = sqrt(1-p1) * [[0, sqrt(a)], [0, 0]] = [[0, sqrt(0.3)], [0, 0]]
    let expected_a1 = Array2::from_shape_vec(
        (2, 2),
        vec![c(0.0, 0.0), c(0.3_f64.sqrt(), 0.0), c(0.0, 0.0), c(0.0, 0.0)],
    ).unwrap();
    assert_matrix_approx(&kraus[1], &expected_a1, 1e-10);

    // A2 = sqrt(1-p1) * [[0, 0], [0, sqrt(b)]] = [[0, 0], [0, sqrt(0.2)]]
    let expected_a2 = Array2::from_shape_vec(
        (2, 2),
        vec![c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.2_f64.sqrt(), 0.0)],
    ).unwrap();
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
        NoiseChannel::PhaseAmplitudeDamping { amplitude: 0.1, phase: 0.1, excited_population: 0.0 }.num_qubits(),
        1
    );
}
