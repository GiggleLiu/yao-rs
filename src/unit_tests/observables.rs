use super::*;
use crate::{Circuit, Gate, Op, OperatorString, put};

fn observable() -> OperatorPolynomial {
    OperatorPolynomial::new(
        vec![
            Complex64::new(0.4, -0.2),
            Complex64::new(-0.7, 0.3),
            Complex64::new(0.8, 0.1),
        ],
        vec![
            OperatorString::identity(),
            OperatorString::new(vec![(0, Op::Y), (2, Op::X)]),
            OperatorString::new(vec![(1, Op::Pu)]),
        ],
    )
}

#[test]
fn polynomial_export_shares_circuit_and_zero_has_valid_shapes() {
    let circuit = Circuit::qubits(
        3,
        vec![put(vec![0], Gate::Ry(0.3)), put(vec![2], Gate::Rx(0.7))],
    )
    .unwrap();
    for operator in [
        observable(),
        OperatorPolynomial::zero(),
        OperatorPolynomial::identity(),
    ] {
        let tn = circuit_to_expectation(&circuit, &operator);
        assert_eq!(
            tn.tensors.len(),
            2 * circuit.elements.len() + 3 * circuit.num_sites()
        );
        assert!(tn.code.iy.is_empty());
        for (labels, tensor) in tn.code.ixs.iter().zip(&tn.tensors) {
            assert_eq!(
                labels.iter().map(|l| tn.size_dict[l]).collect::<Vec<_>>(),
                tensor.shape()
            );
        }
    }
    let empty = Circuit::qubits(0, vec![]).unwrap();
    let tn = circuit_to_expectation(&empty, &OperatorPolynomial::zero());
    assert_eq!(tn.tensors[0].shape(), &[] as &[usize]);
    assert_eq!(tn.tensors[0][IxDyn(&[])], Complex64::new(0., 0.));
}

#[test]
fn invalid_observables_and_channels_are_rejected() {
    let circuit = Circuit::qubits(3, vec![]).unwrap();
    for word in [
        OperatorString::new(vec![(3, Op::X)]),
        OperatorString::new(vec![(0, Op::X), (0, Op::Y)]),
    ] {
        let op = OperatorPolynomial::new(vec![Complex64::new(1., 0.)], vec![word]);
        assert!(std::panic::catch_unwind(|| circuit_to_expectation(&circuit, &op)).is_err());
    }
    let noisy = Circuit::qubits(
        1,
        vec![crate::channel(
            vec![0],
            crate::NoiseChannel::BitFlip { p: 0.2 },
        )],
    )
    .unwrap();
    assert!(
        std::panic::catch_unwind(|| circuit_to_expectation(
            &noisy,
            &OperatorPolynomial::identity()
        ))
        .is_err()
    );
}

#[cfg(feature = "tenferro")]
#[test]
fn polynomial_pure_and_noisy_dm_match_direct_expectations() {
    use crate::{ArrayReg, DensityMatrix, Register};
    let cpu = crate::tenferro::CpuContractor::new(1).unwrap();
    let circuit = Circuit::qubits(
        3,
        vec![
            put(vec![0], Gate::Ry(0.3)),
            put(vec![2], Gate::Rx(0.7)),
            put(vec![1], Gate::Ry(-0.5)),
            crate::control(vec![0], vec![2], Gate::X),
        ],
    )
    .unwrap();
    for operator in [
        observable(),
        &OperatorPolynomial::identity() * Complex64::new(0.3, -0.8),
        OperatorPolynomial::zero(),
    ] {
        let state = crate::apply(&circuit, &ArrayReg::zero_state(3));
        let expected = crate::expect::expect_arrayreg(&state, &operator);
        let tn = circuit_to_expectation(&circuit, &operator);
        let plan = cpu.prepare(&tn.code, &tn.size_dict, None).unwrap();
        let got = cpu.execute(&plan, &tn.tensors).unwrap()[IxDyn(&[])];
        assert!((got - expected).norm() < 1e-11, "{got} != {expected}");
        #[cfg(feature = "omeinsum")]
        assert!((crate::contractor::contract(&tn)[IxDyn(&[])] - expected).norm() < 1e-11);
        let mut elements = circuit.elements.clone();
        elements.push(crate::channel(
            vec![2],
            crate::NoiseChannel::AmplitudeDamping {
                gamma: 0.2,
                excited_population: 0.0,
            },
        ));
        let noisy = Circuit::qubits(3, elements).unwrap();
        let mut dm = DensityMatrix::zero_state(3);
        dm.apply(&noisy);
        let expected = crate::expect::expect_dm(&dm, &operator);
        let tn = circuit_to_expectation_dm(&noisy, &operator);
        let plan = cpu.prepare(&tn.code, &tn.size_dict, None).unwrap();
        assert!((cpu.execute(&plan, &tn.tensors).unwrap()[IxDyn(&[])] - expected).norm() < 1e-11);
        #[cfg(feature = "omeinsum")]
        assert!((crate::contractor::contract_dm(&tn)[IxDyn(&[])] - expected).norm() < 1e-11);
    }
}

#[cfg(feature = "tenferro")]
#[test]
fn weighted_identity_handles_qudits_and_empty_registers() {
    let cpu = crate::tenferro::CpuContractor::new(1).unwrap();
    let c = Complex64::new(-0.4, 0.7);
    let identity = &OperatorPolynomial::identity() * c;
    for dims in [vec![3, 2], vec![]] {
        let circuit = Circuit::new(dims, vec![]).unwrap();
        let tn = circuit_to_expectation(&circuit, &(&identity + &identity));
        let plan = cpu.prepare(&tn.code, &tn.size_dict, None).unwrap();
        assert!((cpu.execute(&plan, &tn.tensors).unwrap()[IxDyn(&[])] - 2. * c).norm() < 1e-12);
        let tn = circuit_to_expectation_dm(&circuit, &identity);
        let plan = cpu.prepare(&tn.code, &tn.size_dict, None).unwrap();
        assert!((cpu.execute(&plan, &tn.tensors).unwrap()[IxDyn(&[])] - c).norm() < 1e-12);
    }
}
