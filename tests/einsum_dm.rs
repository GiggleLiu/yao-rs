use num_complex::Complex64;
use yao_rs::circuit::{Circuit, channel, control, put};
use yao_rs::einsum::{circuit_to_einsum_dm, circuit_to_einsum_with_boundary};
use yao_rs::gate::Gate;
use yao_rs::noise::NoiseChannel;

mod common;
use common::{contract_tn, contract_tn_dm};

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

#[test]
fn test_dm_single_hadamard() {
    // H|0⟩ = (|0⟩+|1⟩)/sqrt(2)
    // rho = |psi><psi| = [[0.5, 0.5], [0.5, 0.5]]
    let elements = vec![put(vec![0], Gate::H)];
    let circuit = Circuit::new(vec![2], elements).unwrap();

    let tn = circuit_to_einsum_dm(&circuit);
    let rho = contract_tn_dm(&tn);

    // Expected: 2x2 density matrix
    assert_eq!(rho.shape(), &[2, 2]);
    assert!((rho[[0, 0]].re - 0.5).abs() < 1e-10);
    assert!((rho[[0, 1]].re - 0.5).abs() < 1e-10);
    assert!((rho[[1, 0]].re - 0.5).abs() < 1e-10);
    assert!((rho[[1, 1]].re - 0.5).abs() < 1e-10);

    // All imaginary parts should be 0
    assert!(rho[[0, 0]].im.abs() < 1e-10);
    assert!(rho[[0, 1]].im.abs() < 1e-10);
    assert!(rho[[1, 0]].im.abs() < 1e-10);
    assert!(rho[[1, 1]].im.abs() < 1e-10);
}

#[test]
fn test_dm_identity() {
    // No gates: rho = |0><0| = [[1, 0], [0, 0]]
    let circuit = Circuit::new(vec![2], vec![]).unwrap();
    let tn = circuit_to_einsum_dm(&circuit);
    let rho = contract_tn_dm(&tn);

    assert_eq!(rho.shape(), &[2, 2]);
    assert!((rho[[0, 0]] - c(1.0, 0.0)).norm() < 1e-10);
    assert!((rho[[0, 1]] - c(0.0, 0.0)).norm() < 1e-10);
    assert!((rho[[1, 0]] - c(0.0, 0.0)).norm() < 1e-10);
    assert!((rho[[1, 1]] - c(0.0, 0.0)).norm() < 1e-10);
}

#[test]
fn test_dm_with_phase_flip_noise() {
    // H|0⟩ then PhaseFlip(p=0.1)
    // E(rho) = (1-p)*rho + p*Z*rho*Z
    // Z|+><+|Z = |-><-| = [[0.5,-0.5],[-0.5,0.5]]
    // rho_noisy = (1-p)*[[0.5,0.5],[0.5,0.5]] + p*[[0.5,-0.5],[-0.5,0.5]]
    //           = [[0.5, 0.5*(1-2p)], [0.5*(1-2p), 0.5]]
    //           = for p=0.1: [[0.5, 0.4], [0.4, 0.5]]
    let elements = vec![
        put(vec![0], Gate::H),
        channel(vec![0], NoiseChannel::PhaseFlip { p: 0.1 }),
    ];
    let circuit = Circuit::new(vec![2], elements).unwrap();
    let tn = circuit_to_einsum_dm(&circuit);
    let rho = contract_tn_dm(&tn);

    assert_eq!(rho.shape(), &[2, 2]);
    assert!((rho[[0, 0]].re - 0.5).abs() < 1e-10);
    assert!((rho[[0, 1]].re - 0.4).abs() < 1e-10);
    assert!((rho[[1, 0]].re - 0.4).abs() < 1e-10);
    assert!((rho[[1, 1]].re - 0.5).abs() < 1e-10);
}

#[test]
fn test_dm_with_bit_flip_on_zero() {
    // BitFlip(p=0.1) on |0⟩
    // E(|0><0|) = (1-p)|0><0| + p X|0><0|X = (1-p)|0><0| + p|1><1|
    // = [[0.9, 0], [0, 0.1]]
    let elements = vec![channel(vec![0], NoiseChannel::BitFlip { p: 0.1 })];
    let circuit = Circuit::new(vec![2], elements).unwrap();
    let tn = circuit_to_einsum_dm(&circuit);
    let rho = contract_tn_dm(&tn);

    assert_eq!(rho.shape(), &[2, 2]);
    assert!((rho[[0, 0]].re - 0.9).abs() < 1e-10);
    assert!((rho[[0, 1]].re).abs() < 1e-10);
    assert!((rho[[1, 0]].re).abs() < 1e-10);
    assert!((rho[[1, 1]].re - 0.1).abs() < 1e-10);
}

#[test]
fn test_dm_bell_state_with_depolarizing() {
    // Bell state: H on q0, CNOT(q0, q1)
    // Then depolarizing(p=0.1) on q0
    let elements = vec![
        put(vec![0], Gate::H),
        control(vec![0], vec![1], Gate::X),
        channel(vec![0], NoiseChannel::Depolarizing { n: 1, p: 0.1 }),
    ];
    let circuit = Circuit::new(vec![2, 2], elements).unwrap();
    let tn = circuit_to_einsum_dm(&circuit);
    let rho = contract_tn_dm(&tn);

    // Output is [ket0, ket1, bra0, bra1] = [2, 2, 2, 2]
    assert_eq!(rho.shape(), &[2, 2, 2, 2]);

    // Verify trace = 1: sum rho[i, j, i, j] over all i, j
    let mut trace = Complex64::new(0.0, 0.0);
    for i in 0..2 {
        for j in 0..2 {
            trace += rho[[i, j, i, j]];
        }
    }
    assert!((trace.re - 1.0).abs() < 1e-10);
    assert!(trace.im.abs() < 1e-10);

    // Verify rho is Hermitian: rho[i,j,k,l] = conj(rho[k,l,i,j])
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                for l in 0..2 {
                    assert!(
                        (rho[[i, j, k, l]] - rho[[k, l, i, j]].conj()).norm() < 1e-10,
                        "Not Hermitian at [{},{},{},{}]",
                        i,
                        j,
                        k,
                        l
                    );
                }
            }
        }
    }
}

#[test]
fn test_dm_pure_matches_vector_mode() {
    // Pure circuit (no noise): DM mode should give |psi><psi|
    let elements = vec![put(vec![0], Gate::H), control(vec![0], vec![1], Gate::X)];
    let circuit = Circuit::new(vec![2, 2], elements).unwrap();

    // Vector mode (with |0⟩ boundary, no final state pinning)
    let tn_vec = circuit_to_einsum_with_boundary(&circuit, &[]);
    let psi = contract_tn(&tn_vec); // shape [2, 2]

    // DM mode
    let tn_dm = circuit_to_einsum_dm(&circuit);
    let rho = contract_tn_dm(&tn_dm);

    // Verify: rho[i,j,k,l] = psi[i,j] * conj(psi[k,l])
    assert_eq!(rho.shape(), &[2, 2, 2, 2]);
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                for l in 0..2 {
                    let expected = psi[[i, j]] * psi[[k, l]].conj();
                    assert!(
                        (rho[[i, j, k, l]] - expected).norm() < 1e-10,
                        "Mismatch at [{},{},{},{}]: {} vs {}",
                        i,
                        j,
                        k,
                        l,
                        rho[[i, j, k, l]],
                        expected
                    );
                }
            }
        }
    }
}
