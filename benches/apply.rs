//! Benchmarks for the apply function comparing different circuit sizes.
//!
//! Run with: cargo bench
//! For HTML reports: cargo bench -- --verbose

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::f64::consts::PI;
use yao_rs::circuit::PositionedGate;
use yao_rs::{apply, control, put, Circuit, Gate, State};

/// Build a circuit with H gates on all qubits.
fn h_all_circuit(n: usize) -> Circuit {
    let gates: Vec<PositionedGate> = (0..n).map(|i| put(vec![i], Gate::H)).collect();
    Circuit::new(vec![2; n], gates).unwrap()
}

/// Build a QFT circuit on n qubits (without final SWAP).
fn qft_circuit(n: usize) -> Circuit {
    let mut gates: Vec<PositionedGate> = Vec::new();

    for i in 0..n {
        // H gate on qubit i
        gates.push(put(vec![i], Gate::H));

        // Controlled phase rotations
        for j in 1..(n - i) {
            let theta = 2.0 * PI / (1 << (j + 1)) as f64;
            gates.push(control(vec![i + j], vec![i], Gate::Phase(theta)));
        }
    }

    // Reverse qubit order with SWAPs
    for i in 0..(n / 2) {
        gates.push(PositionedGate::new(Gate::SWAP, vec![i, n - 1 - i], vec![], vec![]));
    }

    Circuit::new(vec![2; n], gates).unwrap()
}

/// Build a random-ish circuit with H, X, CNOT gates.
fn mixed_circuit(n: usize) -> Circuit {
    let mut gates: Vec<PositionedGate> = Vec::new();

    // Layer of H gates
    for i in 0..n {
        gates.push(put(vec![i], Gate::H));
    }

    // Layer of CNOT gates
    for i in 0..(n - 1) {
        gates.push(control(vec![i], vec![i + 1], Gate::X));
    }

    // Layer of rotation gates
    for i in 0..n {
        gates.push(put(vec![i], Gate::Rz(PI / (i as f64 + 1.0))));
    }

    // Another layer of CNOT gates in reverse
    for i in (0..(n - 1)).rev() {
        gates.push(control(vec![i + 1], vec![i], Gate::X));
    }

    Circuit::new(vec![2; n], gates).unwrap()
}

fn bench_apply_h_all(c: &mut Criterion) {
    let mut group = c.benchmark_group("apply_h_all");

    for n_qubits in [8, 10, 12, 14, 16] {
        let circuit = h_all_circuit(n_qubits);
        let state = State::zero_state(&vec![2; n_qubits]);

        group.bench_with_input(BenchmarkId::new("new", n_qubits), &n_qubits, |b, _| {
            b.iter(|| apply(black_box(&circuit), black_box(&state)))
        });
    }

    group.finish();
}

fn bench_apply_qft(c: &mut Criterion) {
    let mut group = c.benchmark_group("apply_qft");

    for n_qubits in [4, 6, 8, 10, 12] {
        let circuit = qft_circuit(n_qubits);
        let state = State::zero_state(&vec![2; n_qubits]);

        group.bench_with_input(BenchmarkId::new("new", n_qubits), &n_qubits, |b, _| {
            b.iter(|| apply(black_box(&circuit), black_box(&state)))
        });
    }

    group.finish();
}

fn bench_apply_mixed(c: &mut Criterion) {
    let mut group = c.benchmark_group("apply_mixed");

    for n_qubits in [8, 10, 12, 14, 16] {
        let circuit = mixed_circuit(n_qubits);
        let state = State::zero_state(&vec![2; n_qubits]);

        group.bench_with_input(BenchmarkId::new("new", n_qubits), &n_qubits, |b, _| {
            b.iter(|| apply(black_box(&circuit), black_box(&state)))
        });
    }

    group.finish();
}

fn bench_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling");
    group.sample_size(50); // Reduce sample size for larger circuits

    // Test scaling from 8 to 18 qubits with H gates
    for n_qubits in [8, 10, 12, 14, 16, 18] {
        let circuit = h_all_circuit(n_qubits);
        let state = State::zero_state(&vec![2; n_qubits]);

        group.bench_with_input(
            BenchmarkId::new("h_gates", n_qubits),
            &n_qubits,
            |b, _| b.iter(|| apply(black_box(&circuit), black_box(&state))),
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_apply_h_all,
    bench_apply_qft,
    bench_apply_mixed,
    bench_scaling,
);
criterion_main!(benches);
