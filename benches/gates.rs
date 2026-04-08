use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use yao_rs::{ArrayReg, Circuit, Gate, apply, control, put};

fn bench_gates_1q(c: &mut Criterion) {
    let mut group = c.benchmark_group("gates_1q");
    let gates = vec![
        ("X", Gate::X),
        ("H", Gate::H),
        ("T", Gate::T),
        ("Rx_0.5", Gate::Rx(0.5)),
        ("Rz_0.5", Gate::Rz(0.5)),
    ];

    for (name, gate) in gates {
        for nq in [4, 8, 12, 16, 20, 25] {
            let circuit = Circuit::qubits(nq, vec![put(vec![2], gate.clone())]).unwrap();
            let reg = ArrayReg::deterministic_state(nq);
            group.bench_with_input(BenchmarkId::new(name, nq), &nq, |b, _| {
                b.iter(|| apply(black_box(&circuit), black_box(&reg)))
            });
        }
    }

    group.finish();
}

fn bench_gates_2q(c: &mut Criterion) {
    let mut group = c.benchmark_group("gates_2q");
    let gates: Vec<(&str, Box<dyn Fn(usize) -> Circuit>)> = vec![
        (
            "CNOT",
            Box::new(|nq| Circuit::qubits(nq, vec![control(vec![2], vec![3], Gate::X)]).unwrap()),
        ),
        (
            "CRx_0.5",
            Box::new(|nq| {
                Circuit::qubits(nq, vec![control(vec![2], vec![3], Gate::Rx(0.5))]).unwrap()
            }),
        ),
    ];

    for (name, build_circuit) in gates {
        for nq in [4, 8, 12, 16, 20, 25] {
            let circuit = build_circuit(nq);
            let reg = ArrayReg::deterministic_state(nq);
            group.bench_with_input(BenchmarkId::new(name, nq), &nq, |b, _| {
                b.iter(|| apply(black_box(&circuit), black_box(&reg)))
            });
        }
    }

    group.finish();
}

fn bench_gates_multi(c: &mut Criterion) {
    let mut group = c.benchmark_group("gates_multi");

    for nq in [4, 8, 12, 16, 20, 25] {
        let circuit = Circuit::qubits(nq, vec![control(vec![2, 3], vec![1], Gate::X)]).unwrap();
        let reg = ArrayReg::deterministic_state(nq);
        group.bench_with_input(BenchmarkId::new("Toffoli", nq), &nq, |b, _| {
            b.iter(|| apply(black_box(&circuit), black_box(&reg)))
        });
    }

    group.finish();
}

criterion_group!(benches, bench_gates_1q, bench_gates_2q, bench_gates_multi);
criterion_main!(benches);
