use crate::circuit::{Circuit, channel, put};
use crate::gate::Gate;
use crate::noise::NoiseChannel;

#[test]
fn test_circuit_with_noise_channel() {
    let elements = vec![
        put(vec![0], Gate::H),
        channel(vec![0], NoiseChannel::BitFlip { p: 0.1 }),
    ];
    let circuit = Circuit::new(vec![2, 2], elements).unwrap();
    assert_eq!(circuit.elements.len(), 2);
}

#[test]
fn test_circuit_channel_loc_validation() {
    let elements = vec![
        channel(vec![5], NoiseChannel::BitFlip { p: 0.1 }), // loc out of range
    ];
    assert!(Circuit::new(vec![2, 2], elements).is_err());
}

#[test]
fn test_apply_skips_channels() {
    use crate::circuit::channel;
    use crate::{ArrayReg, apply};

    let elements = vec![
        put(vec![0], Gate::X),
        channel(vec![0], NoiseChannel::BitFlip { p: 0.1 }),
    ];
    let circuit = Circuit::new(vec![2], elements).unwrap();
    let reg = ArrayReg::zero_state(1);
    let result = apply(&circuit, &reg);
    // Channel is skipped; only X is applied: |0⟩ → |1⟩
    assert!((result.state_vec()[0].norm() - 0.0).abs() < 1e-10);
    assert!((result.state_vec()[1].norm() - 1.0).abs() < 1e-10);
}
