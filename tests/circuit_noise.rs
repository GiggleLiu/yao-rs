use yao_rs::circuit::{Circuit, channel, put};
use yao_rs::gate::Gate;
use yao_rs::noise::NoiseChannel;

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
    use yao_rs::circuit::channel;
    use yao_rs::{State, apply};

    let elements = vec![
        put(vec![0], Gate::X),
        channel(vec![0], NoiseChannel::BitFlip { p: 0.1 }),
    ];
    let circuit = Circuit::new(vec![2], elements).unwrap();
    let state = State::zero_state(&[2]);
    let result = apply(&circuit, &state);
    // Channel is skipped; only X is applied: |0⟩ → |1⟩
    assert!((result.data[0].norm() - 0.0).abs() < 1e-10);
    assert!((result.data[1].norm() - 1.0).abs() < 1e-10);
}
