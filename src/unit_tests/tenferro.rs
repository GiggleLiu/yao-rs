use super::*;
use crate::{
    ArrayReg, Circuit, CircuitElement, DensityMatrix, Gate, NoiseChannel, Register, apply, channel,
    circuit_to_einsum_dm, circuit_to_einsum_with_boundary, circuit_to_overlap, control, put,
};
use ndarray::{array, s};
type C = Complex64;

fn close(got: &ArrayD<C>, want: &[C]) {
    assert_eq!(got.len(), want.len());
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        assert!((a - b).norm() < 1e-11, "entry {i}: {a} != {b}");
    }
}

#[test]
fn circuits_match_direct_simulation_with_auto_and_explicit_trees() {
    let mut active_low = control(vec![0, 1], vec![3], Gate::Ry(0.37));
    if let CircuitElement::Gate(g) = &mut active_low {
        g.control_configs = vec![false, true];
    }
    let mut elements = vec![
        put(vec![0], Gate::Ry(0.6)),
        put(vec![1], Gate::H),
        put(vec![2], Gate::Rx(-0.7)),
        put(vec![3], Gate::Y),
    ];
    let gates = vec![
        put(vec![0], Gate::Rz(0.5)),
        put(vec![1], Gate::Phase(-0.2)),
        control(vec![1], vec![3], Gate::Z),
        active_low,
        put(vec![3, 0], Gate::FSim(0.4, 0.8)),
        put(vec![2, 0], Gate::SWAP),
        put(
            vec![1],
            Gate::Custom {
                matrix: Gate::Ry(-0.1).matrix(),
                is_diagonal: false,
                label: "custom".into(),
            },
        ),
    ];
    for threads in [1, 2] {
        let cpu = CpuContractor::new(threads).unwrap();
        assert_eq!(cpu.threads(), threads);
        for gate in &gates {
            elements.push(gate.clone());
            let circuit = Circuit::qubits(4, elements.clone()).unwrap();
            let state = apply(&circuit, &ArrayReg::zero_state(4));
            for tn in [
                circuit_to_einsum_with_boundary(&circuit, &[]),
                circuit_to_overlap(&circuit),
            ] {
                let tree = omeco::optimize_code(&tn.code, &tn.size_dict, &GreedyMethod::default())
                    .unwrap();
                for tree in [None, Some(&tree)] {
                    let plan = cpu.prepare(&tn.code, &tn.size_dict, tree).unwrap();
                    let got = cpu.execute(&plan, &tn.tensors).unwrap();
                    close(
                        &got,
                        if tn.code.iy.is_empty() {
                            &state.state_vec()[..1]
                        } else {
                            state.state_vec()
                        },
                    );
                    #[cfg(feature = "omeinsum")]
                    close(
                        &got,
                        &crate::contractor::contract(&tn)
                            .iter()
                            .copied()
                            .collect::<Vec<_>>(),
                    );
                }
            }
        }
    }
}

#[test]
fn negative_labels_and_noisy_density_matrix() {
    let circuit = Circuit::qubits(
        2,
        vec![
            put(vec![0], Gate::Ry(0.7)),
            put(vec![1], Gate::Rx(-0.3)),
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
    )
    .unwrap();
    let mut dm = DensityMatrix::zero_state(2);
    dm.apply(&circuit);
    let tn = circuit_to_einsum_dm(&circuit);
    assert!(tn.code.ixs.iter().flatten().any(|&x| x < 0));
    let cpu = CpuContractor::new(1).unwrap();
    let tree = omeco::optimize_code(&tn.code, &tn.size_dict, &GreedyMethod::default()).unwrap();
    for tree in [None, Some(&tree)] {
        let got = cpu
            .execute(
                &cpu.prepare(&tn.code, &tn.size_dict, tree).unwrap(),
                &tn.tensors,
            )
            .unwrap();
        close(&got, &dm.state);
    }
}

#[test]
fn empty_network_identity_and_mixed_qudit_dimensions() {
    let cpu = CpuContractor::new(1).unwrap();
    let empty = EinCode::<i32>::new(vec![], vec![]);
    let tree = NestedEinsum::node(vec![], empty.clone());
    for tree in [None, Some(&tree)] {
        let plan = cpu.prepare(&empty, &HashMap::new(), tree).unwrap();
        let got = cpu.execute(&plan, &[]).unwrap();
        assert_eq!(got.ndim(), 0);
        close(&got, &[C::new(1., 0.)]);
    }
    for dims in [vec![], vec![3, 2], vec![2, 4]] {
        let circuit = Circuit::new(dims.clone(), vec![]).unwrap();
        let tn = circuit_to_einsum_with_boundary(&circuit, &[]);
        let plan = cpu.prepare(&tn.code, &tn.size_dict, None).unwrap();
        let got = cpu.execute(&plan, &tn.tensors).unwrap();
        let mut want = vec![C::new(0., 0.); dims.iter().product()];
        want[0] = C::new(1., 0.);
        close(&got, &want);
    }
    let matrix = array![
        [C::new(0., 0.), C::new(0., 0.), C::new(1., 0.)],
        [C::new(0., 1.), C::new(0., 0.), C::new(0., 0.)],
        [C::new(0., 0.), C::new(1., 0.), C::new(0., 0.)]
    ];
    let circuit = Circuit::new(
        vec![3, 2],
        vec![
            put(
                vec![0],
                Gate::Custom {
                    matrix,
                    is_diagonal: false,
                    label: "Q".into(),
                },
            ),
            put(vec![1], Gate::X),
        ],
    )
    .unwrap();
    let tn = circuit_to_einsum_with_boundary(&circuit, &[]);
    let got = cpu
        .execute(
            &cpu.prepare(&tn.code, &tn.size_dict, None).unwrap(),
            &tn.tensors,
        )
        .unwrap();
    close(
        &got,
        &[
            C::new(0., 0.),
            C::new(0., 0.),
            C::new(0., 0.),
            C::new(0., 1.),
            C::new(0., 0.),
            C::new(0., 0.),
        ],
    );
}

#[test]
fn layouts_identity_transpose_and_reusing_new_values() {
    let cpu = CpuContractor::new(1).unwrap();
    let a = ArrayD::from_shape_vec(
        IxDyn(&[3, 4]),
        (0..12).map(|x| C::new(x as f64, 1. - x as f64)).collect(),
    )
    .unwrap();
    for a in [
        a.clone(),
        a.clone().reversed_axes(),
        a.clone().slice_move(s![..;-1,..]).into_dyn(),
        a.clone().slice_move(s![1..,..;2]).into_dyn(),
    ] {
        let storage = InputStorage::new(&a);
        assert_eq!(
            matches!(storage.data, Cow::Borrowed(_)),
            a.as_slice_memory_order().is_some() && a.strides().iter().all(|&s| s >= 0)
        );
        if let Cow::Borrowed(slice) = &storage.data {
            assert_eq!(slice.as_ptr(), a.as_slice_memory_order().unwrap().as_ptr());
        }
        let sizes = HashMap::from([(u64::MAX, a.shape()[0]), (7, a.shape()[1])]);
        for output in [vec![u64::MAX, 7], vec![7, u64::MAX]] {
            let code = EinCode::new(vec![vec![u64::MAX, 7]], output.clone());
            let plan = cpu.prepare(&code, &sizes, None).unwrap();
            for factor in [1., 2.] {
                // Pass the original sliced storage itself: ndarray::clone
                // may materialize noncontiguous layouts before the adapter.
                let got = if factor == 1. {
                    cpu.execute(&plan, std::slice::from_ref(&a)).unwrap()
                } else {
                    let input = a.as_standard_layout().mapv(|x| x * factor);
                    cpu.execute(&plan, &[input]).unwrap()
                };
                let want = if output[0] == 7 { a.t() } else { a.view() };
                close(&got, &want.iter().map(|x| x * factor).collect::<Vec<_>>());
            }
            assert!(cpu.execute(&plan, &[]).unwrap_err().contains("count"));
            assert!(
                cpu.execute(&plan, &[ArrayD::zeros(IxDyn(&[a.len()]))])
                    .unwrap_err()
                    .contains("shape")
            );
        }
    }
}

#[test]
fn repeated_labels_trace_diagonal_and_hyperedge() {
    let cpu = CpuContractor::new(1).unwrap();
    let a = array![
        [C::new(1., 1.), C::new(9., 0.)],
        [C::new(7., 0.), C::new(2., -1.)]
    ]
    .into_dyn();
    let sizes = HashMap::from([('i', 2)]);
    for output in [vec![], vec!['i']] {
        let code = EinCode::new(vec![vec!['i', 'i']], output.clone());
        let got = cpu
            .execute(
                &cpu.prepare(&code, &sizes, None).unwrap(),
                std::slice::from_ref(&a),
            )
            .unwrap();
        close(
            &got,
            &if output.is_empty() {
                vec![C::new(3., 0.)]
            } else {
                vec![C::new(1., 1.), C::new(2., -1.)]
            },
        );
    }
    let code = EinCode::new(vec![vec!['i'], vec!['i'], vec!['i']], vec![]);
    let a = array![C::new(1., 1.), C::new(2., 0.)].into_dyn();
    let got = cpu
        .execute(
            &cpu.prepare(&code, &sizes, None).unwrap(),
            &[a.clone(), a.clone(), a],
        )
        .unwrap();
    close(&got, &[C::new(6., 2.)]);
}

#[test]
fn explicit_tree_retains_grouping_and_noncanonical_intermediate_axes() {
    let cpu = CpuContractor::new(1).unwrap();
    // The selected association is observable in floating point: (a*b)*c is 0,
    // whereas a*(b*c) is 1e-200. Greedy planning must never replace this tree.
    let code = EinCode::<i32>::new(vec![vec![], vec![], vec![]], vec![]);
    let tree = NestedEinsum::node(
        vec![
            NestedEinsum::node(
                vec![NestedEinsum::leaf(0), NestedEinsum::leaf(1)],
                EinCode::new(vec![vec![], vec![]], vec![]),
            ),
            NestedEinsum::leaf(2),
        ],
        EinCode::new(vec![vec![], vec![]], vec![]),
    );
    let values = [1e-200, 1e-200, 1e200].map(|x| ArrayD::from_elem(IxDyn(&[]), C::new(x, 0.)));
    let got = cpu
        .execute(
            &cpu.prepare(&code, &HashMap::new(), Some(&tree)).unwrap(),
            &values,
        )
        .unwrap();
    assert_eq!(got[[]], C::new(0., 0.));
    let right = NestedEinsum::node(
        vec![
            NestedEinsum::leaf(0),
            NestedEinsum::node(
                vec![NestedEinsum::leaf(1), NestedEinsum::leaf(2)],
                EinCode::new(vec![vec![], vec![]], vec![]),
            ),
        ],
        EinCode::new(vec![vec![], vec![]], vec![]),
    );
    let got = cpu
        .execute(
            &cpu.prepare(&code, &HashMap::new(), Some(&right)).unwrap(),
            &values,
        )
        .unwrap();
    assert_eq!(got[[]], C::new(1e-200, 0.));
    let code = EinCode::new(vec![vec![0, 1], vec![1, 2], vec![2, 3]], vec![3, 0]);
    let tree = NestedEinsum::node(
        vec![
            NestedEinsum::node(
                vec![NestedEinsum::leaf(0), NestedEinsum::leaf(1)],
                EinCode::new(vec![vec![0, 1], vec![1, 2]], vec![2, 0]),
            ),
            NestedEinsum::leaf(2),
        ],
        EinCode::new(vec![vec![2, 0], vec![2, 3]], vec![3, 0]),
    );
    let a = array![
        [C::new(1., 1.), C::new(2., -1.)],
        [C::new(3., 0.), C::new(4., 2.)]
    ];
    let want = a.dot(&a).dot(&a).t().iter().copied().collect::<Vec<_>>();
    let a = a.into_dyn();
    let plan = cpu
        .prepare(&code, &(0..4).map(|i| (i, 2)).collect(), Some(&tree))
        .unwrap();
    close(
        &cpu.execute(&plan, &[a.clone(), a.clone(), a]).unwrap(),
        &want,
    );
}

#[test]
fn invalid_metadata_and_trees_return_errors_before_execution() {
    assert!(CpuContractor::new(0).is_err());
    let cpu = CpuContractor::new(1).unwrap();
    let code = EinCode::new(vec![vec![0], vec![0]], vec![]);
    for sizes in [
        HashMap::new(),
        HashMap::from([(0, 0)]),
        HashMap::from([(0, usize::MAX)]),
    ] {
        assert!(cpu.prepare(&code, &sizes, None).is_err());
    }
    let sizes = HashMap::from([(0, 2)]);
    for code in [
        EinCode::new(vec![vec![0]], vec![1]),
        EinCode::new(vec![vec![0]], vec![0, 0]),
        EinCode::new(vec![], vec![0]),
    ] {
        assert!(cpu.prepare(&code, &sizes, None).is_err());
    }
    let valid = |args| NestedEinsum::node(args, code.clone());
    let bad = [
        NestedEinsum::leaf(0),
        NestedEinsum::leaf(2),
        valid(vec![NestedEinsum::leaf(0), NestedEinsum::leaf(0)]),
        valid(vec![NestedEinsum::leaf(0)]),
        NestedEinsum::node(
            vec![NestedEinsum::leaf(0), NestedEinsum::leaf(1)],
            EinCode::new(vec![vec![1], vec![0]], vec![]),
        ),
        NestedEinsum::node(
            vec![NestedEinsum::leaf(0), NestedEinsum::leaf(1)],
            EinCode::new(vec![vec![0], vec![0]], vec![1]),
        ),
        // Premature reduction loses the hyperedge needed by input 1.
        NestedEinsum::node(
            vec![
                NestedEinsum::node(
                    vec![NestedEinsum::leaf(0)],
                    EinCode::new(vec![vec![0]], vec![]),
                ),
                NestedEinsum::leaf(1),
            ],
            EinCode::new(vec![vec![], vec![0]], vec![]),
        ),
    ];
    for tree in bad {
        assert!(cpu.prepare(&code, &sizes, Some(&tree)).is_err());
    }
    let overflow = EinCode::new(vec![vec![0], vec![1]], vec![0, 1]);
    assert!(
        cpu.prepare(
            &overflow,
            &HashMap::from([(0, 1usize << 32), (1, 1usize << 32)]),
            None
        )
        .is_err()
    );
    let tensor = array![C::new(1., 0.), C::new(2., 0.)].into_dyn();
    let plan = cpu.prepare(&code, &sizes, None).unwrap();
    close(
        &cpu.execute(&plan, &[tensor.clone(), tensor]).unwrap(),
        &[C::new(5., 0.)],
    );
}
