use criterion::{Criterion, criterion_group, criterion_main};
use std::{hint::black_box, time::Duration};
use yao_rs::cuda::CudaSimulator;
use yao_tenferro_probe::{
    cases,
    cuda::{PreparedCase, check, reference},
};

fn bench(c: &mut Criterion) {
    let gpu = CudaSimulator::new(0).unwrap();
    for case in cases().unwrap() {
        if case.cuda_diagnostic_only {
            continue;
        }
        let mut group = c.benchmark_group(&case.id);
        group
            .sample_size(10)
            .warm_up_time(Duration::from_secs(1))
            .measurement_time(Duration::from_secs(2));
        let prepared = PreparedCase::new(&case, &gpu).unwrap();
        let expected = reference(&case).unwrap();
        let got = prepared.resident(&gpu).unwrap();
        check(&prepared.download(&gpu, &got).unwrap(), &expected).unwrap();
        check(&prepared.end_to_end(&gpu).unwrap(), &expected).unwrap();
        drop(got);
        drop(expected);
        group.bench_function("cuda_resident", |b| {
            b.iter(|| {
                gpu.synchronize().unwrap();
                let result = prepared.resident(&gpu).unwrap();
                gpu.synchronize().unwrap();
                black_box(result);
            })
        });
        group.bench_function("cuda_transfer_inclusive", |b| {
            b.iter(|| {
                gpu.synchronize().unwrap();
                let result = prepared.end_to_end(&gpu).unwrap();
                gpu.synchronize().unwrap();
                black_box(result);
            })
        });
        group.finish();
    }
}
criterion_group!(benches, bench);
criterion_main!(benches);
