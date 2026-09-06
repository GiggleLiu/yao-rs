use super::*;
use crate::{ArrayReg, Circuit, control, parameters::BoundCircuit, put};

type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

fn state(n: usize, shift: f64) -> ArrayReg {
    ArrayReg::from_vec(
        n,
        (0..1usize << n)
            .map(|i| {
                C::new(
                    (i as f64 * 0.4 + shift).sin(),
                    (i as f64 * 0.7 - shift).cos(),
                )
            })
            .collect(),
    )
}
fn upload_state(gpu: &CudaSimulator, state: &ArrayReg, track: bool) -> Result<EagerTensor> {
    Ok(gpu.upload(
        &Tensor::from_vec_col_major(vec![state.state.len()], state.state.clone())?,
        track,
    )?)
}
fn upload_parameters(gpu: &CudaSimulator, values: &[f64], track: bool) -> Result<EagerTensor> {
    Ok(gpu.upload(
        &Tensor::from_vec_col_major(vec![values.len()], values.to_vec())?,
        track,
    )?)
}
fn close(a: &[C], b: &[C], tol: f64) {
    assert_eq!(a.len(), b.len());
    for (i, (a, b)) in a.iter().zip(b).enumerate() {
        assert!((a - b).norm() < tol, "amplitude {i}: {a} vs {b}");
    }
}
fn read(gpu: &CudaSimulator, tensor: &EagerTensor) -> Result<Vec<C>> {
    assert!(
        tensor.to_tensor()?.as_slice::<C>().is_err(),
        "device result became host-accessible"
    );
    Ok(gpu.download(tensor)?.as_slice::<C>()?.to_vec())
}
fn pairing(a: &[C], b: &[C]) -> f64 {
    a.iter().zip(b).map(|(a, b)| (a.conj() * b).re).sum()
}

#[test]
#[ignore = "requires CUDA hardware; run explicitly on the GPU host"]
fn cuda_all_gate_families_match_native_with_ordered_targets_and_controls() -> Result {
    let gpu = CudaSimulator::new(0)?;
    let x = state(4, 0.31);
    let input = upload_state(&gpu, &x, false)?;
    let mut custom = Array2::<C>::zeros((4, 4));
    for col in 0..4 {
        custom[[(col + 1) % 4, col]] = C::from_polar(1., 0.2 + 0.37 * col as f64);
    }
    let mut gates = vec![
        Gate::X,
        Gate::Y,
        Gate::Z,
        Gate::H,
        Gate::S,
        Gate::T,
        Gate::SWAP,
        Gate::SqrtX,
        Gate::SqrtY,
        Gate::SqrtW,
        Gate::ISWAP,
        Gate::Custom {
            matrix: custom,
            is_diagonal: false,
            label: "phase-cycle".into(),
        },
    ];
    for angle in [0., 0.37, -1.2] {
        gates.extend([
            Gate::Rx(angle),
            Gate::Ry(angle),
            Gate::Rz(angle),
            Gate::Phase(angle),
            Gate::FSim(angle, -0.43),
        ]);
    }
    for gate in gates {
        let targets = if gate.matrix().nrows() == 4 {
            vec![3, 0]
        } else {
            vec![2]
        };
        for controlled in [false, true] {
            let mut element = if controlled {
                control(vec![1], targets.clone(), gate.clone())
            } else {
                put(targets.clone(), gate.clone())
            };
            if let CircuitElement::Gate(pg) = &mut element {
                pg.control_configs.fill(false);
            }
            let circuit = DifferentiableCircuit::from_circuit(Circuit::qubits(4, vec![element])?)?;
            let plan = gpu.prepare(&circuit)?;
            let p = upload_parameters(&gpu, circuit.template().parameters(), false)?;
            let output = plan.apply(&p, &input)?;
            close(
                &read(&gpu, &output)?,
                &circuit.forward(circuit.template().parameters(), &x)?.state,
                2e-12,
            );
        }
    }
    gpu.synchronize()?;
    Ok(())
}

fn model() -> Result<DifferentiableCircuit> {
    let mut low = control(vec![1], vec![3, 0], Gate::FSim(0., 0.));
    if let CircuitElement::Gate(pg) = &mut low {
        pg.control_configs.fill(false);
    }
    let circuit = Circuit::qubits(
        4,
        vec![
            put(vec![0], Gate::H),
            put(vec![1], Gate::Rx(0.)),
            control(vec![0], vec![2], Gate::Ry(0.)),
            low,
            put(vec![3], Gate::Phase(0.)),
            put(vec![0], Gate::Rz(0.)),
            put(vec![1], Gate::Ry(0.)),
        ],
    )?;
    Ok(DifferentiableCircuit::new(BoundCircuit::new(
        circuit,
        vec![0.3, -0.7, 0.4, 1.2],
        vec![
            ParameterBinding::Scaled {
                index: 0,
                scale: 0.7,
            },
            ParameterBinding::Product {
                left: 0,
                right: 1,
                scale: -0.4,
            },
            ParameterBinding::Product {
                left: 1,
                right: 1,
                scale: 0.2,
            },
            ParameterBinding::Scaled {
                index: 2,
                scale: 1.3,
            },
            ParameterBinding::Scaled {
                index: 0,
                scale: -0.5,
            },
            ParameterBinding::Fixed(0.37),
            ParameterBinding::Scaled {
                index: 1,
                scale: 0.,
            },
        ],
    )?)?)
}

#[test]
#[ignore = "requires CUDA hardware; run explicitly on the GPU host"]
fn cuda_physical_and_complex_input_derivatives_match_native_and_differences() -> Result {
    let gpu = CudaSimulator::new(0)?;
    let circuit = model()?;
    let plan = gpu.prepare(&circuit)?;
    let x = state(4, 0.4);
    let bar = state(4, -0.7);
    let seed = upload_state(&gpu, &bar, false)?;
    for parameters in [
        [0.3, -0.7, 0.4, 1.2],
        [0., -0.2, 0.7, 2.],
        [0.5, 0., -0.4, 3.],
    ] {
        let p = upload_parameters(&gpu, &parameters, true)?;
        let input = upload_state(&gpu, &x, true)?;
        let y = plan.apply(&p, &input)?;
        let dp = gpu.runtime().vjp(&y, &p, &seed)?;
        let dx = gpu.runtime().vjp(&y, &input, &seed)?;
        assert!(dp.to_tensor()?.as_slice::<f64>().is_err());
        let dp = gpu.download(&dp)?.as_slice::<f64>()?.to_vec();
        let dx = read(&gpu, &dx)?;
        let expected = circuit.vjp(&parameters, &x, &bar)?;
        close(&dx, &expected.input.state, 2e-10);
        for (a, b) in dp.iter().zip(&expected.parameters) {
            assert!((a - b).abs() < 2e-10, "{a} vs {b}");
        }
        assert_eq!(dp[3], 0.);
        for eps in [1e-4, 1e-5] {
            for i in 0..parameters.len() {
                let mut plus = parameters;
                let mut minus = parameters;
                plus[i] += eps;
                minus[i] -= eps;
                let fd = (pairing(&bar.state, &circuit.forward(&plus, &x)?.state)
                    - pairing(&bar.state, &circuit.forward(&minus, &x)?.state))
                    / (2. * eps);
                assert!((fd - dp[i]).abs() < 2e-8);
            }
        }
        let direction = [0.2, -0.3, 0.5, -0.7];
        let input_direction = state(4, 0.8);
        let dy_p = gpu
            .runtime()
            .jvp(&y, &p, &upload_parameters(&gpu, &direction, false)?)?;
        let dy_x = gpu
            .runtime()
            .jvp(&y, &input, &upload_state(&gpu, &input_direction, false)?)?;
        let dy = read(&gpu, &dy_p.add(&dy_x)?)?;
        let rhs = pairing(&dx, &input_direction.state)
            + dp.iter().zip(direction).map(|(a, b)| a * b).sum::<f64>();
        assert!((pairing(&bar.state, &dy) - rhs).abs() < 2e-10);
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA hardware; run explicitly on the GPU host"]
fn cuda_custom_loss_and_gpu_parameter_update_preserve_residency() -> Result {
    let gpu = CudaSimulator::new(0)?;
    let circuit = model()?;
    let plan = gpu.prepare(&circuit)?;
    let x = state(4, 0.3);
    let initial = circuit.template().parameters();
    let p = upload_parameters(&gpu, initial, true)?;
    let input = upload_state(&gpu, &x, true)?;
    let target = upload_state(&gpu, &state(4, -0.5), false)?;
    let y = plan.apply(&p, &input)?;
    let delta = y.sub(&target)?;
    // abs has an unsupported C128 sign VJP upstream. This smooth expression
    // also gives the correct derivative when an element of delta is zero.
    let loss = delta
        .conj()?
        .mul(&delta)?
        .cast(DType::F64)?
        .reduce_sum(Some(&[0]))?;
    let dp = gpu.runtime().grad(&loss, &p)?;
    let native = circuit.forward(initial, &x)?;
    let target_native = state(4, -0.5);
    let seed = ArrayReg::from_vec(
        4,
        native
            .state
            .iter()
            .zip(&target_native.state)
            .map(|(a, b)| 2. * (a - b))
            .collect(),
    );
    let expected = circuit.vjp(initial, &x, &seed)?;
    let observed = gpu.download(&dp)?.as_slice::<f64>()?.to_vec();
    for (a, b) in observed.iter().zip(&expected.parameters) {
        assert!((a - b).abs() < 1e-9);
    }
    let step = upload_parameters(&gpu, &[0.01; 4], false)?;
    let updated = p.sub(&dp.mul(&step)?)?;
    let y = plan.apply(&updated, &input)?;
    let updated_native: Vec<_> = initial
        .iter()
        .zip(&expected.parameters)
        .map(|(p, g)| p - 0.01 * g)
        .collect();
    close(
        &read(&gpu, &y)?,
        &circuit.forward(&updated_native, &x)?.state,
        1e-10,
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA hardware; run explicitly on the GPU host"]
fn cuda_empty_parameters_scalar_phase_and_many_controls() -> Result {
    let gpu = CudaSimulator::new(0)?;
    let p = upload_parameters(&gpu, &[], false)?;
    for n in [0, 3] {
        let phase = C::from_polar(1., 0.47);
        let scalar = Gate::Custom {
            matrix: Array2::from_elem((1, 1), phase),
            is_diagonal: true,
            label: "global phase".into(),
        };
        let circuit =
            DifferentiableCircuit::from_circuit(Circuit::qubits(n, vec![put(vec![], scalar)])?)?;
        let x = state(n, 0.4);
        let output = gpu
            .prepare(&circuit)?
            .apply(&p, &upload_state(&gpu, &x, true)?)?;
        close(
            &read(&gpu, &output)?,
            &x.state.iter().map(|x| phase * x).collect::<Vec<_>>(),
            1e-12,
        );
    }
    let mut element = control((0..10).collect(), vec![11], Gate::Y);
    if let CircuitElement::Gate(pg) = &mut element {
        for (i, value) in pg.control_configs.iter_mut().enumerate() {
            *value = i % 2 == 0;
        }
    }
    let circuit = DifferentiableCircuit::from_circuit(Circuit::qubits(12, vec![element])?)?;
    let x = state(12, 0.31);
    let output = gpu
        .prepare(&circuit)?
        .apply(&p, &upload_state(&gpu, &x, false)?)?;
    close(
        &read(&gpu, &output)?,
        &circuit.forward(&[], &x)?.state,
        1e-12,
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA hardware; run explicitly on the GPU host"]
fn cuda_rejects_host_tensors_bad_shapes_and_foreign_contexts() -> Result {
    let gpu = CudaSimulator::new(0)?;
    let circuit = model()?;
    let plan = gpu.prepare(&circuit)?;
    let x = state(4, 0.4);
    let input = upload_state(&gpu, &x, false)?;
    let p = upload_parameters(&gpu, circuit.template().parameters(), false)?;
    assert!(
        crate::tenferro_ad::circuit_apply_eager(std::sync::Arc::new(circuit), &p, &input).is_err()
    );
    assert!(
        EagerTensor::from_tensor_in(
            Tensor::from_vec_col_major(vec![16], x.state.clone())?,
            gpu.runtime().clone(),
        )
        .is_err()
    );
    assert!(
        plan.apply(&upload_parameters(&gpu, &[0.], false)?, &input)
            .is_err()
    );
    assert!(plan.apply(&input, &input).is_err());
    assert!(
        gpu.upload(&Tensor::from_vec_col_major(vec![1], vec![f64::NAN])?, false)
            .is_err()
    );
    assert!(
        gpu.upload(&Tensor::from_vec_col_major(vec![1], vec![1_i64])?, false)
            .is_err()
    );
    let other = CudaSimulator::new(0)?;
    let foreign = upload_state(&other, &x, false)?;
    assert!(
        plan.apply(&p, &foreign)
            .unwrap_err()
            .contains("different runtime")
    );
    assert!(gpu.download(&foreign).is_err());
    Ok(())
}

fn upload_array(
    gpu: &CudaSimulator,
    array: &ndarray::ArrayD<C>,
    track: bool,
) -> Result<EagerTensor> {
    Ok(gpu.upload(
        &Tensor::from_vec_col_major(array.shape().to_vec(), array.t().iter().copied().collect())?,
        track,
    )?)
}

#[test]
fn cuda_plan_shape_and_tree_validation_does_not_initialize_cuda() {
    use omeco::{EinCode, NestedEinsum};
    use std::collections::HashMap;
    let code = EinCode::new(vec![vec![0, 1], vec![1, 2]], vec![2, 0]);
    let sizes = HashMap::from([(0, 2), (1, 3), (2, 4)]);
    let plan = PreparedCudaContraction::new(&code, &sizes, None).unwrap();
    assert_eq!(plan.input_shapes(), &[vec![2, 3], vec![3, 4]]);
    assert_eq!(plan.output_shape(), &[4, 2]);
    assert!(
        PreparedCudaContraction::new(&code, &HashMap::from([(0, 2), (1, 0), (2, 4)]), None)
            .is_err()
    );
    assert!(
        PreparedCudaContraction::new(
            &code,
            &HashMap::from([(0, usize::MAX), (1, 3), (2, 4)]),
            None
        )
        .is_err()
    );
    assert!(PreparedCudaContraction::new(&code, &sizes, Some(&NestedEinsum::leaf(0))).is_err());
    let duplicate = EinCode::new(vec![vec![0]], vec![0, 0]);
    assert!(PreparedCudaContraction::new(&duplicate, &sizes, None).is_err());
}

#[test]
#[ignore = "requires CUDA hardware; run explicitly on the GPU host"]
fn cuda_contraction_circuits_qudits_and_exact_noise_match_cpu() -> Result {
    use crate::{
        NoiseChannel, channel, circuit_to_einsum_dm, circuit_to_einsum_with_boundary,
        circuit_to_overlap,
    };
    let gpu = CudaSimulator::new(0)?;
    let cpu = crate::tenferro::CpuContractor::new(1)?;
    let mut low = control(vec![0, 1], vec![3], Gate::Ry(0.37));
    if let CircuitElement::Gate(pg) = &mut low {
        pg.control_configs = vec![false, true];
    }
    let circuit = Circuit::qubits(
        4,
        vec![
            put(vec![1], Gate::H),
            put(vec![2], Gate::Rx(-0.4)),
            low,
            put(vec![3, 0], Gate::FSim(0.2, -0.7)),
            put(vec![2, 0], Gate::SWAP),
            put(vec![0], Gate::Phase(0.3)),
        ],
    )?;
    let noisy = Circuit::qubits(
        2,
        vec![
            put(vec![0], Gate::H),
            control(vec![0], vec![1], Gate::X),
            channel(
                vec![1],
                NoiseChannel::AmplitudeDamping {
                    gamma: 0.23,
                    excited_population: 0.2,
                },
            ),
            channel(vec![0], NoiseChannel::Depolarizing { n: 1, p: 0.17 }),
        ],
    )?;
    let qudit = Circuit::new(
        vec![3, 2],
        vec![put(
            vec![0],
            Gate::Custom {
                matrix: ndarray::array![
                    [C::default(), C::default(), C::new(0., 1.)],
                    [C::new(1., 0.), C::default(), C::default()],
                    [C::default(), C::new(0., -1.), C::default()]
                ],
                is_diagonal: false,
                label: "qutrit-cycle".into(),
            },
        )],
    )?;
    for tn in [
        circuit_to_einsum_with_boundary(&circuit, &[]),
        circuit_to_overlap(&circuit),
        circuit_to_einsum_with_boundary(&qudit, &[]),
    ] {
        check_network(&gpu, &cpu, &tn.code, &tn.size_dict, &tn.tensors)?;
    }
    let tn = circuit_to_einsum_dm(&noisy);
    check_network(&gpu, &cpu, &tn.code, &tn.size_dict, &tn.tensors)?;
    Ok(())
}

#[test]
#[ignore = "requires CUDA hardware; run explicitly on the GPU host"]
fn cuda_contraction_unary_hyperedges_gradients_and_explicit_order() -> Result {
    use omeco::{EinCode, NestedEinsum};
    use std::collections::HashMap;
    let gpu = CudaSimulator::new(0)?;
    let cpu = crate::tenferro::CpuContractor::new(1)?;
    let data: Vec<_> = (0..9)
        .map(|i| C::new(0.2 * i as f64, 0.3 - 0.1 * i as f64))
        .collect();
    let array = ndarray::Array2::from_shape_vec((3, 3), data)?.into_dyn();
    for code in [
        EinCode::new(vec![vec![0, 0]], vec![]),
        EinCode::new(vec![vec![0, 0]], vec![0]),
        EinCode::new(vec![vec![0, 1]], vec![1, 0]),
        EinCode::new(vec![vec![0, 1]], vec![1]),
    ] {
        let sizes = HashMap::from([(0, 3), (1, 3)]);
        let p = PreparedCudaContraction::new(&code, &sizes, None)?;
        let x = upload_array(&gpu, &array, true)?;
        let diagonal = code.ixs[0][0] == code.ixs[0][1];
        let x = if diagonal {
            assert!(
                gpu.contract(&p, &[x])
                    .unwrap_err()
                    .contains("trace/diagonal gradients")
            );
            upload_array(&gpu, &array, false)?
        } else {
            x
        };
        let output = gpu.contract(&p, std::slice::from_ref(&x))?;
        let reference = cpu.execute(
            &cpu.prepare(&code, &sizes, None)?,
            std::slice::from_ref(&array),
        )?;
        close(
            &read(&gpu, &output)?,
            &reference.t().iter().copied().collect::<Vec<_>>(),
            1e-11,
        );
        if diagonal {
            continue;
        }
        let loss = output
            .conj()?
            .mul(&output)?
            .cast(DType::F64)?
            .reduce_sum(None)?;
        let dx = gpu.runtime().grad(&loss, &x)?;
        assert_eq!(dx.shape(), array.shape());
        let expected = if code.iy.len() == 2 {
            array.mapv(|z| 2. * z)
        } else {
            ndarray::Array2::from_shape_fn((3, 3), |(_, j)| {
                2. * (0..3).map(|i| array[[i, j]]).sum::<C>()
            })
            .into_dyn()
        };
        close(
            &read(&gpu, &dx)?,
            &expected.t().iter().copied().collect::<Vec<_>>(),
            1e-11,
        );
    }
    let code = EinCode::new(vec![vec![0], vec![0], vec![0]], vec![0]);
    let sizes = HashMap::from([(0, 3)]);
    let vectors: Vec<_> = (0..3)
        .map(|i| Tensor::from_vec_col_major(vec![3], vec![C::new(0.3 + i as f64, 0.2); 3]))
        .collect::<std::result::Result<_, _>>()?;
    let tensors = vectors
        .iter()
        .enumerate()
        .map(|(i, x)| gpu.upload(x, i == 0))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let output = gpu.contract(
        &PreparedCudaContraction::new(&code, &sizes, None)?,
        &tensors,
    )?;
    close(
        &read(&gpu, &output)?,
        &[C::new(0.3, 0.2) * C::new(1.3, 0.2) * C::new(2.3, 0.2); 3],
        1e-12,
    );
    let loss = output
        .conj()?
        .mul(&output)?
        .cast(DType::F64)?
        .reduce_sum(None)?;
    let gradient = gpu.runtime().grad(&loss, &tensors[0])?;
    let other = C::new(1.3, 0.2) * C::new(2.3, 0.2);
    close(
        &read(&gpu, &gradient)?,
        &[2. * C::new(0.3, 0.2) * other.norm_sqr(); 3],
        1e-11,
    );
    let empty = PreparedCudaContraction::new(
        &EinCode::<usize>::new(vec![], vec![]),
        &HashMap::new(),
        None,
    )?;
    close(
        &read(&gpu, &gpu.contract(&empty, &[])?)?,
        &[C::new(1., 0.)],
        1e-15,
    );

    // This underflow fixture distinguishes left-to-right from right-first
    // multiplication. A supplied n-ary tree must not be silently reoptimized.
    let code = EinCode::<usize>::new(vec![vec![], vec![], vec![]], vec![]);
    let left = NestedEinsum::node((0..3).map(NestedEinsum::leaf).collect(), code.clone());
    let pair = EinCode::new(vec![vec![], vec![]], vec![]);
    let right = NestedEinsum::node(
        vec![
            NestedEinsum::leaf(0),
            NestedEinsum::node(
                vec![NestedEinsum::leaf(1), NestedEinsum::leaf(2)],
                pair.clone(),
            ),
        ],
        pair,
    );
    let tensors = [1e-200, 1e-200, 1e200]
        .iter()
        .map(|&x| {
            gpu.upload(
                &Tensor::from_vec_col_major(vec![], vec![C::new(x, 0.)]).unwrap(),
                false,
            )
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let left = gpu.contract(
        &PreparedCudaContraction::new(&code, &HashMap::new(), Some(&left))?,
        &tensors,
    )?;
    let right = gpu.contract(
        &PreparedCudaContraction::new(&code, &HashMap::new(), Some(&right))?,
        &tensors,
    )?;
    assert_eq!(read(&gpu, &left)?[0], C::default());
    assert!((read(&gpu, &right)?[0].re / 1e-200 - 1.).abs() < 1e-12);
    Ok(())
}

fn check_network<L: omeco::Label>(
    gpu: &CudaSimulator,
    cpu: &crate::tenferro::CpuContractor,
    code: &omeco::EinCode<L>,
    sizes: &std::collections::HashMap<L, usize>,
    tensors: &[ndarray::ArrayD<C>],
) -> Result {
    use omeco::{GreedyMethod, NestedEinsum};
    let tree =
        crate::contraction_plan::optimize_code(code, sizes, &GreedyMethod::default()).unwrap();
    let nary = NestedEinsum::node(
        (0..tensors.len()).map(NestedEinsum::leaf).collect(),
        code.clone(),
    );
    let inputs = tensors
        .iter()
        .map(|x| upload_array(gpu, x, false))
        .collect::<Result<Vec<_>>>()?;
    for tree in [None, Some(&tree), Some(&nary)] {
        let plan = PreparedCudaContraction::new(code, sizes, tree)?;
        let reference = cpu.execute(&cpu.prepare(code, sizes, tree)?, tensors)?;
        for scale in [C::new(1., 0.), C::new(0.7, -0.2)] {
            let mut changed = inputs.clone();
            changed[0] = upload_array(gpu, &tensors[0].mapv(|x| x * scale), false)?;
            let output = gpu.contract(&plan, &changed)?;
            close(
                &read(gpu, &output)?,
                &reference.t().iter().map(|x| x * scale).collect::<Vec<_>>(),
                1e-10,
            );
        }
    }
    Ok(())
}
