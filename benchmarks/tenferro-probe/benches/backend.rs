use criterion::{Criterion, criterion_group, criterion_main};
use num_complex::Complex64 as C;
use std::{hint::black_box, time::Duration};
use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_cpu::CpuBackend;
use tenferro_tensor::Tensor;
use yao_rs::einsum::{circuit_to_einsum_dm, circuit_to_einsum_with_boundary};
use yao_rs::{DensityMatrix, Op, OperatorPolynomial, Register, apply, expect_grad};
use yao_tenferro_probe::{
    cases, convert, execute, extension::prepared_x, output_array, prepare, subscripts,
};

fn backend(c: &mut Criterion) {
    let threads = std::env::var("YAO_BENCH_THREADS")
        .unwrap_or("1".into())
        .parse()
        .unwrap();
    let mut backend = CpuBackend::with_threads(threads).unwrap();
    let cpu = yao_rs::tenferro::CpuContractor::new(threads).unwrap();
    for case in cases().unwrap() {
        let circuit = case.circuit().unwrap();
        let state = case.initial(circuit.nbits);
        let mut group = c.benchmark_group(&case.id);
        match case.mode.as_str() {
            "state" => {
                group.bench_function("native", |b| {
                    b.iter(|| apply(black_box(&circuit), black_box(&state)))
                });
            }
            "density" => {
                let dm = DensityMatrix::from_reg(&state);
                group.bench_function("native", |b| {
                    b.iter(|| {
                        let mut out = black_box(&dm).clone();
                        out.apply(black_box(&circuit));
                        black_box(out)
                    })
                });
            }
            "expectation" | "expectation_dm" => {
                let op = case.operator.as_ref().unwrap();
                let noisy = case.mode == "expectation_dm";
                let evaluate = || {
                    if noisy {
                        let mut dm = DensityMatrix::from_reg(&state);
                        dm.apply(&circuit);
                        yao_rs::expect::expect_dm(&dm, op)
                    } else {
                        yao_rs::expect::expect_arrayreg(&apply(&circuit, &state), op)
                    }
                };
                let expected = evaluate();
                group.bench_function("native", |b| b.iter(|| black_box(evaluate())));
                use yao_tenferro_probe::tensor_memory as tm;
                let tn = tm::network(&circuit, op, noisy);
                group.bench_function("observable_export", |b| {
                    b.iter(|| tm::network(black_box(&circuit), black_box(op), noisy))
                });
                let tree = yao_rs::contraction_plan::optimize_code(
                    &tn.code,
                    &tn.size_dict,
                    &omeco::GreedyMethod::default(),
                )
                .unwrap();
                let selector = tm::term_label(&tn, op.len()).unwrap();
                for (name, labels) in [("unsliced", vec![]), ("term_sliced", vec![selector])] {
                    let plan = yao_rs::slicing::SlicedPlan::new(
                        &tn.code,
                        &tn.size_dict,
                        &tree,
                        &labels,
                        yao_rs::slicing::SliceBudget::default(),
                    )
                    .unwrap();
                    tm::record_plan(&case.id, name, &plan).unwrap();
                    let prepared = cpu.prepare_sliced(&plan).unwrap();
                    let got =
                        cpu.execute_sliced(&prepared, &tn.tensors).unwrap()[ndarray::IxDyn(&[])];
                    assert!((got - expected).norm() < 1e-10);
                    let got = yao_rs::contractor::contract_sliced(&plan, &tn.tensors).unwrap()
                        [ndarray::IxDyn(&[])];
                    assert!((got - expected).norm() < 1e-10);
                    group.bench_function(format!("tenferro_{name}_prepare"), |b| {
                        b.iter(|| cpu.prepare_sliced(black_box(&plan)).unwrap())
                    });
                    group.bench_function(format!("tenferro_{name}_warm"), |b| {
                        b.iter(|| {
                            cpu.execute_sliced(black_box(&prepared), black_box(&tn.tensors))
                                .unwrap()
                        })
                    });
                    group.bench_function(format!("omeinsum_{name}"), |b| {
                        b.iter(|| {
                            yao_rs::contractor::contract_sliced(
                                black_box(&plan),
                                black_box(&tn.tensors),
                            )
                            .unwrap()
                        })
                    });
                }
            }
            "gradient" => {
                let op = OperatorPolynomial::single(0, Op::Z, 1.0.into());
                group.bench_function("native", |b| {
                    b.iter(|| expect_grad(black_box(&op), black_box(&circuit), black_box(&state)))
                });
            }
            "custom_gradient" => {
                use yao_tenferro_probe::circuit_ad as ad;
                let dc = std::sync::Arc::new(
                    yao_rs::differentiable::DifferentiableCircuit::from_circuit(circuit.clone())
                        .unwrap(),
                );
                let target = ad::target(circuit.nbits);
                let expected = ad::native(&dc, &state, &target).unwrap();
                group.bench_function("native", |b| {
                    b.iter(|| {
                        ad::native(black_box(&dc), black_box(&state), black_box(&target)).unwrap()
                    })
                });
                for (name, composed) in [
                    ("tenferro_circuit_ad", false),
                    ("tenferro_composed_ad", true),
                ] {
                    // A separate CPU-time-bounded process records the 100-layer
                    // composition limit. Keep it out of repeated timing runs.
                    if composed && dc.num_parameters() > 30 {
                        continue;
                    }
                    let ctx = yao_rs::tenferro_ad::eager_cpu_runtime(threads).unwrap();
                    let got = ad::finish(
                        &ad::prepare(dc.clone(), &state, &target, ctx.clone(), composed).unwrap(),
                    )
                    .unwrap();
                    assert_eq!(got.len(), expected.len());
                    assert!(
                        got.iter()
                            .zip(&expected)
                            .all(|(a, b)| (a - b).norm() < 1e-9)
                    );
                    group.bench_function(name, |b| {
                        b.iter(|| {
                            ad::finish(
                                &ad::prepare(
                                    dc.clone(),
                                    black_box(&state),
                                    black_box(&target),
                                    ctx.clone(),
                                    composed,
                                )
                                .unwrap(),
                            )
                            .unwrap()
                        })
                    });
                }
            }
            _ => panic!("unknown mode"),
        }
        if case.tensor {
            let (arrays, code, old) = if case.mode == "density" {
                let tn = circuit_to_einsum_dm(&circuit);
                let code = subscripts(&tn.code).unwrap();
                let old = yao_rs::contractor::contract_dm(&tn);
                group.bench_function("omeinsum", |b| {
                    b.iter(|| yao_rs::contractor::contract_dm(black_box(&tn)))
                });
                (tn.tensors, code, old)
            } else {
                let tn = circuit_to_einsum_with_boundary(&circuit, &[]);
                let code = subscripts(&tn.code).unwrap();
                let old = yao_rs::contractor::contract(&tn);
                group.bench_function("omeinsum", |b| {
                    b.iter(|| yao_rs::contractor::contract(black_box(&tn)))
                });
                (tn.tensors, code, old)
            };
            supported(&mut group, &cpu, &arrays, &code, &old);
            let tensors = convert(&arrays).unwrap();
            let plan = prepare(&tensors, &code).unwrap();
            let got = output_array(&execute(&plan, &tensors, &mut backend).unwrap()).unwrap();
            assert_eq!(old.shape(), got.shape());
            assert!(
                old.iter()
                    .zip(got.iter())
                    .all(|(a, b)| (a - b).norm() < 1e-10)
            );
            group.bench_function("conversion", |b| {
                b.iter(|| convert(black_box(&arrays)).unwrap())
            });
            group.bench_function("planning", |b| {
                b.iter(|| prepare(black_box(&tensors), black_box(&code)).unwrap())
            });
            group.bench_function("tenferro_warm", |b| {
                b.iter(|| execute(black_box(&plan), black_box(&tensors), &mut backend).unwrap())
            });
            group.bench_function("tenferro_from_arrays", |b| {
                b.iter(|| {
                    let tensors = convert(black_box(&arrays)).unwrap();
                    let plan = prepare(&tensors, &code).unwrap();
                    output_array(&execute(&plan, &tensors, &mut backend).unwrap()).unwrap()
                })
            });
        }
        group.finish();
    }
    if std::env::var("YAO_BENCH_SUITE").as_deref() == Ok("tensor-memory") {
        matrix_slicing(c, &cpu);
        return;
    }
    for n in [8, 12, 16] {
        let batch = 1usize << (n - 1);
        let input =
            Tensor::from_vec_col_major(vec![2, batch], vec![C::new(0.3, 0.4); 2 * batch]).unwrap();
        let x = Tensor::from_vec_col_major(
            vec![2, 2],
            vec![
                C::new(0., 0.),
                C::new(1., 0.),
                C::new(1., 0.),
                C::new(0., 0.),
            ],
        )
        .unwrap();
        let tensors = vec![x, input];
        let input = &tensors[1];
        let plan = prepare(
            &tensors,
            &tenferro_einsum::EinsumSubscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]),
        )
        .unwrap();
        let (runtime, program) = prepared_x(batch, threads).unwrap();
        let mut group = c.benchmark_group(format!("extension_{n}"));
        group.bench_function("composed", |b| {
            b.iter(|| execute(&plan, &tensors, &mut backend).unwrap())
        });
        group.bench_function("custom", |b| {
            b.iter(|| runtime.run_compiled(&program, &[black_box(input)]).unwrap())
        });
        group.bench_function("custom_prepare", |b| {
            b.iter(|| prepared_x(black_box(batch), threads).unwrap())
        });
        let ctx =
            EagerRuntime::with_cpu_backend(CpuBackend::with_threads(threads).unwrap()).unwrap();
        // Includes graph construction, forward and backward; never retains one graph per iteration.
        group.bench_function("complex_ad", |b| {
            b.iter(|| {
                let x = EagerTensor::requires_grad_in(
                    Tensor::from_vec_col_major(
                        input.shape().to_vec(),
                        input.as_slice::<C>().unwrap().to_vec(),
                    )
                    .unwrap(),
                    ctx.clone(),
                )
                .unwrap();
                let magnitude = x.abs().unwrap();
                let loss = magnitude
                    .mul(&magnitude)
                    .unwrap()
                    .reduce_sum(Some(&[0, 1]))
                    .unwrap();
                ctx.grad(&loss, &x).unwrap()
            })
        });
        group.finish();
    }
}
fn configuration() -> Criterion {
    Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_millis(200))
        .measurement_time(Duration::from_millis(500))
        .without_plots()
}
criterion_group! {name=benches;config=configuration();targets=backend}
criterion_main!(benches);

// Fixed-tree comparisons include ndarray input/output adaptation in both providers.
fn supported(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    cpu: &yao_rs::tenferro::CpuContractor,
    arrays: &[ndarray::ArrayD<C>],
    subs: &tenferro_einsum::EinsumSubscripts,
    reference: &ndarray::ArrayD<C>,
) {
    let code = omeco::EinCode::new(
        subs.inputs
            .iter()
            .map(|xs| xs.iter().map(|&x| i32::try_from(x).unwrap()).collect())
            .collect(),
        subs.output
            .iter()
            .map(|&x| i32::try_from(x).unwrap())
            .collect(),
    );
    let sizes = code
        .ixs
        .iter()
        .zip(arrays)
        .flat_map(|(xs, a)| xs.iter().copied().zip(a.shape().iter().copied()))
        .collect();
    let tree =
        yao_rs::contraction_plan::optimize_code(&code, &sizes, &omeco::GreedyMethod::default())
            .unwrap();
    let plan = cpu.prepare(&code, &sizes, Some(&tree)).unwrap();
    let got = cpu.execute(&plan, arrays).unwrap();
    assert_eq!(got.shape(), reference.shape());
    assert!(
        got.iter()
            .zip(reference)
            .all(|(a, b)| (a - b).norm() < 1e-10)
    );
    group.bench_function("supported_planning", |b| {
        b.iter(|| {
            cpu.prepare(black_box(&code), black_box(&sizes), Some(black_box(&tree)))
                .unwrap()
        })
    });
    group.bench_function("supported_warm", |b| {
        b.iter(|| cpu.execute(black_box(&plan), black_box(arrays)).unwrap())
    });
    group.bench_function("supported_from_arrays", |b| {
        b.iter(|| {
            let plan = cpu.prepare(&code, &sizes, Some(&tree)).unwrap();
            cpu.execute(&plan, black_box(arrays)).unwrap()
        })
    });
    let tn = yao_rs::TensorNetworkDM {
        code,
        size_dict: sizes,
        tensors: arrays.to_vec(),
    };
    let got = yao_rs::contractor::contract_dm_with_tree(&tn, tree.clone());
    assert!(
        got.iter()
            .zip(reference)
            .all(|(a, b)| (a - b).norm() < 1e-10)
    );
    group.bench_function("omeinsum_fixed_tree", |b| {
        b.iter(|| {
            yao_rs::contractor::contract_dm_with_tree(black_box(&tn), black_box(&tree).clone())
        })
    });
}

fn matrix_slicing(c: &mut Criterion, cpu: &yao_rs::tenferro::CpuContractor) {
    use yao_tenferro_probe::tensor_memory as tm;
    for (kind, n, modes) in tm::matrix_workloads() {
        let tn = if kind == "outer" {
            tm::outer_network(n)
        } else {
            tm::matrix_network(n)
        };
        let baseline = tm::matrix_plan(&tn, "unsliced").unwrap();
        let reference = yao_rs::contractor::contract_sliced(&baseline, &tn.tensors).unwrap();
        let mut group = c.benchmark_group(format!("matrix_{kind}_{n}"));
        for mode in modes {
            let plan = tm::matrix_plan(&tn, mode).unwrap();
            tm::record_plan(&format!("matrix_{kind}_{n}"), mode, &plan).unwrap();
            let prepared = cpu.prepare_sliced(&plan).unwrap();
            for result in [
                cpu.execute_sliced(&prepared, &tn.tensors).unwrap(),
                yao_rs::contractor::contract_sliced(&plan, &tn.tensors).unwrap(),
            ] {
                assert!(
                    result
                        .iter()
                        .zip(&reference)
                        .all(|(a, b)| (a - b).norm() < 1e-9)
                );
            }
            group.bench_function(format!("plan_{mode}"), |b| {
                b.iter(|| tm::matrix_plan(black_box(&tn), mode).unwrap())
            });
            group.bench_function(format!("tenferro_{mode}_prepare"), |b| {
                b.iter(|| cpu.prepare_sliced(black_box(&plan)).unwrap())
            });
            group.bench_function(format!("tenferro_{mode}_warm"), |b| {
                b.iter(|| {
                    cpu.execute_sliced(black_box(&prepared), black_box(&tn.tensors))
                        .unwrap()
                })
            });
            group.bench_function(format!("omeinsum_{mode}"), |b| {
                b.iter(|| {
                    yao_rs::contractor::contract_sliced(black_box(&plan), black_box(&tn.tensors))
                        .unwrap()
                })
            });
        }
        group.finish();
    }
}
