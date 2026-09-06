use super::*;
use crate::{Gate, put};

fn example() -> BoundCircuit {
    BoundCircuit::new(
        Circuit::qubits(
            1,
            vec![
                put(vec![0], Gate::Rx(0.)),
                put(vec![0], Gate::Ry(0.)),
                put(vec![0], Gate::Rz(0.)),
                put(vec![0], Gate::Phase(0.)),
            ],
        )
        .unwrap(),
        vec![0.3, -0.7],
        vec![
            ParameterBinding::Scaled {
                index: 0,
                scale: 2.,
            },
            ParameterBinding::Product {
                left: 0,
                right: 1,
                scale: -3.,
            },
            ParameterBinding::Product {
                left: 0,
                right: 0,
                scale: 0.5,
            },
            ParameterBinding::Fixed(0.8),
        ],
    )
    .unwrap()
}

#[test]
fn shared_product_jacobian_and_duality() {
    let bound = example();
    let angles = bound.circuit().parameters();
    for (a, b) in angles.iter().zip([0.6, 0.63, 0.045, 0.8]) {
        assert!((a - b).abs() < 1e-14);
    }
    let gradient = bound.pullback(&[1., 2., 3., 999.]).unwrap();
    assert!((gradient[0] - 7.1).abs() < 1e-14);
    assert!((gradient[1] + 1.8).abs() < 1e-14);
    let tangent = [0.13, -0.23];
    let pushed = bound.pushforward(&tangent).unwrap();
    let lhs: f64 = pushed
        .iter()
        .zip([1., 2., 3., 999.])
        .map(|(a, b)| a * b)
        .sum();
    let rhs: f64 = gradient.iter().zip(tangent).map(|(a, b)| a * b).sum();
    assert!((lhs - rhs).abs() < 1e-14);
    for eps in [1e-4, 1e-5, 1e-6] {
        let mut plus = bound.clone();
        let mut minus = bound.clone();
        let parameters = bound.parameters();
        plus.dispatch(
            &parameters
                .iter()
                .zip(tangent)
                .map(|(p, v)| p + eps * v)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        minus
            .dispatch(
                &parameters
                    .iter()
                    .zip(tangent)
                    .map(|(p, v)| p - eps * v)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        for ((p, m), want) in plus
            .circuit()
            .parameters()
            .iter()
            .zip(minus.circuit().parameters())
            .zip(&pushed)
        {
            assert!(((p - m) / (2. * eps) - want).abs() < 1e-9);
        }
    }
}

#[test]
fn rejected_dispatch_is_atomic_and_invalid_bindings_fail() {
    let mut bound = example();
    let before = bound.circuit().parameters();
    for bad in [
        vec![0.],
        vec![f64::NAN, 1.],
        vec![f64::INFINITY, 0.],
        vec![f64::MAX, f64::MAX],
    ] {
        assert!(bound.dispatch(&bad).is_err());
        assert_eq!(bound.parameters(), &[0.3, -0.7]);
        assert_eq!(bound.circuit().parameters(), before);
    }
    assert!(bound.pullback(&[1.]).is_err());
    assert!(bound.pullback(&[1., f64::NAN, 0., 0.]).is_err());
    assert!(bound.pushforward(&[]).is_err());
    assert!(bound.pushforward(&[1., f64::INFINITY]).is_err());
    let circuit = Circuit::qubits(1, vec![put(vec![0], Gate::Rx(0.))]).unwrap();
    for binding in [
        ParameterBinding::Fixed(f64::NAN),
        ParameterBinding::Scaled {
            index: 1,
            scale: 1.,
        },
        ParameterBinding::Product {
            left: 0,
            right: 1,
            scale: 1.,
        },
        ParameterBinding::Scaled {
            index: 0,
            scale: f64::INFINITY,
        },
    ] {
        assert!(BoundCircuit::new(circuit.clone(), vec![0.], vec![binding]).is_err());
    }
    assert!(BoundCircuit::new(circuit, vec![0.], vec![]).is_err());
    assert!(bound.apply(&ArrayReg::zero_state(2)).is_err());
    let mut malformed = ArrayReg::zero_state(1);
    malformed.state.pop();
    assert!(bound.apply(&malformed).is_err());
}

#[test]
fn fixed_only_and_empty_bindings() {
    let bound = BoundCircuit::new(
        Circuit::qubits(1, vec![put(vec![0], Gate::Rx(0.))]).unwrap(),
        vec![],
        vec![ParameterBinding::Fixed(0.3)],
    )
    .unwrap();
    assert!(bound.parameters().is_empty());
    assert_eq!(bound.pullback(&[8.]).unwrap(), Vec::<f64>::new());
    assert_eq!(bound.pushforward(&[]).unwrap(), vec![0.]);
    let empty = BoundCircuit::new(Circuit::qubits(1, vec![]).unwrap(), vec![0.1], vec![]).unwrap();
    assert_eq!(empty.pullback(&[]).unwrap(), vec![0.]);
    assert_eq!(
        empty.apply(&ArrayReg::zero_state(1)).unwrap().state,
        ArrayReg::zero_state(1).state
    );
}
