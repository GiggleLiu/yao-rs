use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use num_complex::Complex64;
use yao_rs::{ArrayReg, apply};

fn bench_qft(c: &mut Criterion) {
    let mut group = c.benchmark_group("qft");

    for nq in [4, 8, 12, 16, 20, 25] {
        let circuit = yao_rs::easybuild::qft_circuit(nq);
        let mut init = vec![Complex64::new(0.0, 0.0); 1 << nq];
        init[1] = Complex64::new(1.0, 0.0);
        let reg = ArrayReg::from_vec(nq, init);

        group.bench_with_input(BenchmarkId::new("QFT", nq), &nq, |b, _| {
            b.iter(|| apply(black_box(&circuit), black_box(&reg)))
        });
    }

    group.finish();
}

criterion_group!(benches, bench_qft);
criterion_main!(benches);
