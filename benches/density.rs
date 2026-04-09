use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use yao_rs::circuit::CircuitElement;
use yao_rs::{Circuit, DensityMatrix, Gate, NoiseChannel, Register, channel, control, put};

fn build_noisy_circuit(nq: usize) -> Circuit {
    let mut elements: Vec<CircuitElement> = Vec::new();

    for qubit in 0..nq {
        elements.push(put(vec![qubit], Gate::H));
    }
    for qubit in 0..(nq - 1) {
        elements.push(control(vec![qubit], vec![qubit + 1], Gate::X));
    }
    for qubit in 0..nq {
        elements.push(channel(
            vec![qubit],
            NoiseChannel::Depolarizing { n: 1, p: 0.01 },
        ));
    }
    for qubit in 0..nq {
        elements.push(put(vec![qubit], Gate::Rz(0.3)));
    }
    for qubit in 0..nq {
        elements.push(channel(
            vec![qubit],
            NoiseChannel::AmplitudeDamping {
                gamma: 0.05,
                excited_population: 0.0,
            },
        ));
    }

    Circuit::qubits(nq, elements).unwrap()
}

fn bench_noisy_dm(c: &mut Criterion) {
    let mut group = c.benchmark_group("noisy_dm");
    group.sample_size(20);

    for nq in [4, 6, 8, 10] {
        let circuit = build_noisy_circuit(nq);
        let template_dm = DensityMatrix::zero_state(nq);
        group.bench_with_input(BenchmarkId::new("noisy_dm", nq), &nq, |b, _| {
            b.iter(|| {
                let mut dm = template_dm.clone();
                dm.apply(black_box(&circuit));
                dm
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_noisy_dm);
criterion_main!(benches);
