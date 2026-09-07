use crate::fusion::FusionError;
use crate::{
    ArrayReg, Circuit, CircuitElement, DensityMatrix, Gate, NoiseChannel, PositionedGate, Register,
    apply, channel, circuit_from_json, circuit_to_json, control, label, put,
};
use ndarray::Array2;
use num_complex::Complex64 as C;

fn input(n: usize) -> ArrayReg {
    let values: Vec<C> = (0..1usize << n)
        .map(|i| C::new((i as f64 * 0.17).sin(), (i as f64 * 0.31).cos()))
        .collect();
    let norm = values.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
    ArrayReg::from_vec(n, values.into_iter().map(|z| z / norm).collect())
}

fn equal(a: &[C], b: &[C]) {
    assert_eq!(a.len(), b.len());
    assert!(a.iter().zip(b).all(|(a, b)| (*a - *b).norm() < 1e-11));
}

fn fixture() -> Circuit {
    let mut matrix = Array2::zeros((4, 4));
    for (row, column, phase) in [(0, 0, 0.2), (1, 3, -0.3), (2, 2, 0.4), (3, 1, -0.5)] {
        matrix[[row, column]] = C::from_polar(1., phase);
    }
    Circuit::qubits(
        6,
        vec![
            put(vec![5], Gate::Rx(0.7)),
            CircuitElement::Gate(PositionedGate {
                gate: Gate::Ry(0.17),
                target_locs: vec![0],
                control_locs: vec![4, 1],
                control_configs: vec![false, true],
            }),
            put(
                vec![2, 5],
                Gate::Custom {
                    matrix: matrix.reversed_axes(),
                    is_diagonal: false,
                    label: "asymmetric".into(),
                },
            ),
            put(vec![3], Gate::H),
            put(vec![1, 4], Gate::FSim(0.21, -0.4)),
            put(vec![0], Gate::Phase(0.5)),
            put(vec![5, 3], Gate::SWAP),
            put(vec![0], Gate::Ry(-0.3)),
            control(vec![5], vec![2], Gate::X),
        ],
    )
    .unwrap()
}

#[test]
fn fused_state_matches_controls_nonadjacent_targets_and_strided_matrices() {
    let circuit = fixture();
    let input = input(6);
    let expected = apply(&circuit, &input);
    for limit in 1..=6 {
        let fused = circuit.fused(limit).unwrap();
        equal(apply(&fused, &input).state_vec(), expected.state_vec());
        assert_eq!(fused.num_params(), 0);
        let restored = circuit_from_json(&circuit_to_json(&fused)).unwrap();
        equal(apply(&restored, &input).state_vec(), expected.state_vec());
    }
}

#[test]
fn fused_parameters_are_an_independent_snapshot() {
    let mut circuit = fixture();
    let input = input(6);
    let fused = circuit.fused(2).unwrap();
    let before = apply(&fused, &input);
    let changed: Vec<f64> = circuit.parameters().iter().map(|x| x + 0.2).collect();
    circuit.dispatch(&changed);
    equal(apply(&fused, &input).state_vec(), before.state_vec());
    let new = apply(&circuit.fused(2).unwrap(), &input);
    equal(new.state_vec(), apply(&circuit, &input).state_vec());
    assert!(
        new.state_vec()
            .iter()
            .zip(before.state_vec())
            .any(|(a, b)| (*a - *b).norm() > 1e-4)
    );
}

#[test]
fn annotation_and_noise_boundaries_preserve_intermediate_states() {
    let circuit = Circuit::qubits(
        3,
        vec![
            put(vec![2], Gate::Ry(0.4)),
            control(vec![2], vec![0], Gate::X),
            label(0, "before noise"),
            put(vec![0], Gate::Rx(0.3)),
            put(vec![1], Gate::H),
            channel(vec![0], NoiseChannel::BitFlip { p: 0.23 }),
            put(vec![0], Gate::Rz(0.2)),
            put(vec![2], Gate::Ry(-0.3)),
        ],
    )
    .unwrap();
    let checkpoints = |circuit: &Circuit| {
        let mut state = DensityMatrix::from_reg(&input(3));
        let mut result = Vec::new();
        for element in &circuit.elements {
            state.apply(&Circuit::qubits(3, vec![element.clone()]).unwrap());
            if !matches!(element, CircuitElement::Gate(_)) {
                result.push(state.state.clone());
            }
        }
        result.push(state.state);
        result
    };
    let expected = checkpoints(&circuit);
    for limit in [1, 2, 3] {
        let fused = circuit.fused(limit).unwrap();
        let actual = checkpoints(&fused);
        assert_eq!(actual.len(), expected.len());
        for (a, b) in actual.iter().zip(&expected) {
            equal(a, b);
        }
    }
}

#[test]
fn fusion_handles_empty_circuits_and_rejects_invalid_requests() {
    let empty = Circuit::qubits(4, vec![]).unwrap();
    assert!(empty.fused(8).unwrap().elements.is_empty());
    for limit in [0, 9, usize::MAX] {
        assert!(matches!(empty.fused(limit), Err(FusionError::BlockSize(_))));
    }
    let qudit = Circuit::new(vec![3], vec![]).unwrap();
    assert!(matches!(qudit.fused(2), Err(FusionError::QubitsRequired)));
    let mut invalid = empty;
    invalid.elements.push(put(vec![5], Gate::X));
    assert!(matches!(
        invalid.fused(2),
        Err(FusionError::InvalidCircuit(_))
    ));
}

#[test]
fn fusion_bounds_new_matrices_and_keeps_large_gates_separate() {
    let circuit = fixture();
    for limit in [1, 2] {
        let fused = circuit.fused(limit).unwrap();
        for element in &fused.elements {
            if let CircuitElement::Gate(gate) = element {
                // Original gates in this fixture have at most two target sites.
                assert!(gate.gate.matrix().nrows() <= 1 << limit.max(2));
            }
        }
    }
}
