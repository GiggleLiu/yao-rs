//! Profiling harness for benchmark workloads.
//! Usage: samply record -- cargo run --example profile_bench --release -- [gate|qft|dm]

use num_complex::Complex64;
use std::hint::black_box;
use yao_rs::circuit::CircuitElement;
use yao_rs::register::Register;
use yao_rs::{ArrayReg, Circuit, DensityMatrix, Gate, NoiseChannel, apply, channel, control, put};

fn bench_single_gate() {
    let nq = 20;
    let reg = ArrayReg::deterministic_state(nq);
    let circuits: Vec<(&str, Circuit)> = vec![
        (
            "X",
            Circuit::qubits(nq, vec![put(vec![2], Gate::X)]).unwrap(),
        ),
        (
            "H",
            Circuit::qubits(nq, vec![put(vec![2], Gate::H)]).unwrap(),
        ),
        (
            "Rz",
            Circuit::qubits(nq, vec![put(vec![2], Gate::Rz(0.5))]).unwrap(),
        ),
        (
            "CNOT",
            Circuit::qubits(nq, vec![control(vec![2], vec![3], Gate::X)]).unwrap(),
        ),
        (
            "CRx",
            Circuit::qubits(nq, vec![control(vec![2], vec![3], Gate::Rx(0.5))]).unwrap(),
        ),
        (
            "Toffoli",
            Circuit::qubits(nq, vec![control(vec![2, 3], vec![1], Gate::X)]).unwrap(),
        ),
    ];

    for (name, circuit) in &circuits {
        for _ in 0..200 {
            let result = apply(black_box(circuit), black_box(&reg));
            black_box(&result);
        }
        eprintln!("{name} done");
    }
}

fn bench_qft() {
    // Focus on 20 qubits where Julia wins
    let nq = 20;
    let circuit = yao_rs::easybuild::qft_circuit(nq);
    let mut init = vec![Complex64::new(0.0, 0.0); 1 << nq];
    init[1] = Complex64::new(1.0, 0.0);
    let reg = ArrayReg::from_vec(nq, init);

    for _ in 0..5 {
        let result = apply(black_box(&circuit), black_box(&reg));
        black_box(&result);
    }
    eprintln!("QFT {nq}q done");
}

fn bench_dm() {
    let nq = 8;
    let mut elements: Vec<CircuitElement> = Vec::new();
    for q in 0..nq {
        elements.push(put(vec![q], Gate::H));
    }
    for q in 0..(nq - 1) {
        elements.push(control(vec![q], vec![q + 1], Gate::X));
    }
    for q in 0..nq {
        elements.push(channel(
            vec![q],
            NoiseChannel::Depolarizing { n: 1, p: 0.01 },
        ));
    }
    for q in 0..nq {
        elements.push(put(vec![q], Gate::Rz(0.3)));
    }
    for q in 0..nq {
        elements.push(channel(
            vec![q],
            NoiseChannel::AmplitudeDamping {
                gamma: 0.05,
                excited_population: 0.0,
            },
        ));
    }
    let circuit = Circuit::qubits(nq, elements).unwrap();
    let template_dm = DensityMatrix::zero_state(nq);

    for _ in 0..5 {
        let mut dm = template_dm.clone();
        dm.apply(black_box(&circuit));
        black_box(&dm);
    }
    eprintln!("DM {nq}q done");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("all");

    match mode {
        "gate" => bench_single_gate(),
        "qft" => bench_qft(),
        "dm" => bench_dm(),
        _ => {
            bench_single_gate();
            bench_qft();
            bench_dm();
        }
    }
}
