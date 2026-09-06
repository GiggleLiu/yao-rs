use super::*;
use crate::contraction_plan::optimize_code;
use omeco::GreedyMethod;

fn naive(code: &EinCode<i32>, sizes: &HashMap<i32, usize>, tensors: &[ArrayD<C>]) -> ArrayD<C> {
    let mut labels = Vec::new();
    for l in code.ixs.iter().flatten() {
        if !labels.contains(l) {
            labels.push(*l);
        }
    }
    let dims: Vec<_> = labels.iter().map(|l| sizes[l]).collect();
    let output: Vec<_> = code.iy.iter().map(|l| sizes[l]).collect();
    let mut result = ArrayD::zeros(IxDyn(&output));
    for index in ndarray::indices(IxDyn(&dims)) {
        let mut product = C::new(1., 0.);
        for (tensor, legs) in tensors.iter().zip(&code.ixs) {
            let coordinates: Vec<_> = legs
                .iter()
                .map(|l| index[labels.iter().position(|v| v == l).unwrap()])
                .collect();
            product *= tensor[IxDyn(&coordinates)];
        }
        let coordinates: Vec<_> = code
            .iy
            .iter()
            .map(|l| index[labels.iter().position(|v| v == l).unwrap()])
            .collect();
        result[IxDyn(&coordinates)] += product;
    }
    result
}
fn tensor(shape: &[usize], offset: f64) -> ArrayD<C> {
    ArrayD::from_shape_vec(
        IxDyn(shape),
        (0..shape.iter().product())
            .map(|i| C::new((i as f64 + offset).cos(), (0.3 * i as f64 - offset).sin()))
            .collect(),
    )
    .unwrap()
}
fn close(got: &ArrayD<C>, want: &ArrayD<C>) {
    assert_eq!(got.shape(), want.shape());
    for (a, b) in got.iter().zip(want) {
        assert!((a - b).norm() < 1e-10, "{a} != {b}");
    }
}
fn check(
    code: EinCode<i32>,
    sizes: HashMap<i32, usize>,
    tensors: Vec<ArrayD<C>>,
    selections: &[Vec<i32>],
) {
    let tree = optimize_code(&code, &sizes, &GreedyMethod::default()).unwrap();
    let want = naive(&code, &sizes, &tensors);
    for selection in selections {
        let plan =
            SlicedPlan::new(&code, &sizes, &tree, selection, SliceBudget::default()).unwrap();
        let got = plan
            .execute_with(&tensors, |inputs| {
                Ok(naive(&code, plan.slice_sizes(), inputs))
            })
            .unwrap();
        close(&got, &want);
        let again = plan
            .execute_with(&tensors, |inputs| {
                Ok(naive(&code, plan.slice_sizes(), inputs))
            })
            .unwrap();
        assert_eq!(again, got);
        #[cfg(feature = "tenferro")]
        {
            let cpu = crate::tenferro::CpuContractor::new(1).unwrap();
            let prepared = cpu.prepare_sliced(&plan).unwrap();
            close(&cpu.execute_sliced(&prepared, &tensors).unwrap(), &want);
            let changed: Vec<_> = tensors
                .iter()
                .map(|t| t.mapv(|z| z * C::new(0.7, -0.2)))
                .collect();
            close(
                &cpu.execute_sliced(&prepared, &changed).unwrap(),
                &naive(&code, &sizes, &changed),
            );
        }
        #[cfg(feature = "omeinsum")]
        close(
            &crate::contractor::contract_sliced(&plan, &tensors).unwrap(),
            &want,
        );
    }
}

#[test]
fn internal_output_mixed_slices_and_complex_strided_inputs() {
    let code = EinCode::new(vec![vec![-2, 7], vec![7, -4]], vec![-4, -2]);
    let sizes = HashMap::from([(-2, 2), (7, 3), (-4, 4)]);
    let a = tensor(&[2, 3], 0.1)
        .slice_move(ndarray::s![.., ..;-1])
        .into_dyn();
    let b = tensor(&[4, 3], -0.8).reversed_axes();
    check(
        code,
        sizes,
        vec![a, b],
        &[vec![], vec![7], vec![-2], vec![-4, 7], vec![-4, -2, 7]],
    );
}

#[test]
fn unary_trace_diagonal_output_scalar_and_empty_network() {
    for output in [vec![], vec![1]] {
        check(
            EinCode::new(vec![vec![1, 1]], output),
            HashMap::from([(1, 3)]),
            vec![tensor(&[3, 3], 0.2)],
            &[vec![], vec![1]],
        );
    }
    check(
        EinCode::new(vec![vec![], vec![]], vec![]),
        HashMap::new(),
        vec![tensor(&[], 0.1), tensor(&[], 0.7)],
        &[vec![]],
    );
    check(
        EinCode::new(vec![], vec![]),
        HashMap::new(),
        vec![],
        &[vec![]],
    );
}

#[test]
fn budgets_account_for_inputs_output_and_one_active_slice() {
    let code = EinCode::new(vec![vec![0, 1], vec![1, 2]], vec![0, 2]);
    let sizes = HashMap::from([(0, 8), (1, 16), (2, 8)]);
    let tree = optimize_code(&code, &sizes, &GreedyMethod::default()).unwrap();
    let unsliced = SlicedPlan::new(&code, &sizes, &tree, &[], SliceBudget::default()).unwrap();
    let sliced = SlicedPlan::new(
        &code,
        &sizes,
        &tree,
        &[0, 1, 2],
        SliceBudget {
            workspace_bytes: 123,
            ..SliceBudget::default()
        },
    )
    .unwrap();
    let e = sliced.estimate();
    assert_eq!(e.input_bytes, 2 * 8 * 16 * 16);
    assert_eq!(e.output_bytes, 8 * 8 * 16);
    assert_eq!(e.slices, 8 * 16 * 8);
    assert_eq!(e.concurrent_slices, 1);
    assert_eq!(e.workspace_reserved_bytes, 123);
    assert_eq!(e.omeco_peak_bytes, 3 * 16);
    assert_eq!(e.worker_buffer_bytes, 7 * 16);
    assert_eq!(
        e.estimated_total_bytes,
        e.input_bytes + e.output_bytes + e.worker_buffer_bytes + 123
    );
    assert!(e.estimated_total_bytes < unsliced.estimate().estimated_total_bytes);
    let exact = SliceBudget {
        max_bytes: Some(e.estimated_total_bytes),
        workspace_bytes: 123,
        ..SliceBudget::default()
    };
    assert!(SlicedPlan::new(&code, &sizes, &tree, &[0, 1, 2], exact).is_ok());
    assert!(
        SlicedPlan::new(
            &code,
            &sizes,
            &tree,
            &[0, 1, 2],
            SliceBudget {
                max_bytes: Some(e.estimated_total_bytes - 1),
                ..exact
            }
        )
        .is_err()
    );
    assert!(
        SlicedPlan::new(
            &code,
            &sizes,
            &tree,
            &[0, 1, 2],
            SliceBudget {
                max_slices: 1023,
                ..exact
            }
        )
        .is_err()
    );
}

#[test]
fn automatic_omeco_slicing_fits_checked_budget_and_matches_unsliced() {
    let code = EinCode::new(vec![vec![0, 1], vec![1, 2], vec![2, 3]], vec![0, 3]);
    let sizes = HashMap::from([(0, 3), (1, 5), (2, 5), (3, 3)]);
    let tree = optimize_code(&code, &sizes, &GreedyMethod::default()).unwrap();
    let full = SlicedPlan::new(&code, &sizes, &tree, &[], SliceBudget::default()).unwrap();
    let limit = full.estimate().input_bytes + full.estimate().output_bytes + 512;
    let budget = SliceBudget {
        max_bytes: Some(limit),
        ..SliceBudget::default()
    };
    let plan = SlicedPlan::auto(&code, &sizes, &tree, budget, &TreeSASlicer::fast()).unwrap();
    assert!(!plan.slicing().is_empty());
    assert!(plan.estimate().estimated_total_bytes <= limit);
    let inputs = vec![
        tensor(&[3, 5], 0.2),
        tensor(&[5, 5], 0.4),
        tensor(&[5, 3], 0.7),
    ];
    close(
        &plan
            .execute_with(&inputs, |t| Ok(naive(&code, plan.slice_sizes(), t)))
            .unwrap(),
        &naive(&code, &sizes, &inputs),
    );
    assert!(
        SlicedPlan::auto(
            &code,
            &sizes,
            &tree,
            SliceBudget {
                max_bytes: Some(full.estimate().input_bytes),
                ..budget
            },
            &TreeSASlicer::fast()
        )
        .unwrap_err()
        .contains("cannot hold")
    );
    assert!(
        SlicedPlan::auto(
            &code,
            &sizes,
            &tree,
            budget,
            &TreeSASlicer::fast().with_ntrials(0)
        )
        .is_err()
    );
}

#[test]
fn invalid_shapes_labels_trees_overflow_and_callback_output_rejected() {
    let code = EinCode::new(vec![vec![0, 1], vec![1, 2]], vec![0, 2]);
    let sizes = HashMap::from([(0, 2), (1, 3), (2, 2)]);
    let tree = optimize_code(&code, &sizes, &GreedyMethod::default()).unwrap();
    for selection in [vec![4], vec![1, 1]] {
        assert!(SlicedPlan::new(&code, &sizes, &tree, &selection, SliceBudget::default()).is_err());
    }
    assert!(
        SlicedPlan::new(
            &code,
            &HashMap::from([(0, usize::MAX), (1, 2), (2, 2)]),
            &tree,
            &[],
            SliceBudget::default()
        )
        .is_err()
    );
    assert!(
        SlicedPlan::new(
            &code,
            &HashMap::from([(0, 0), (1, 3), (2, 2)]),
            &tree,
            &[],
            SliceBudget::default()
        )
        .is_err()
    );
    assert!(
        SlicedPlan::new(
            &code,
            &sizes,
            &NestedEinsum::leaf(0),
            &[],
            SliceBudget::default()
        )
        .is_err()
    );
    assert!(
        SlicedPlan::new(
            &code,
            &sizes,
            &tree,
            &[],
            SliceBudget {
                workspace_bytes: usize::MAX,
                ..SliceBudget::default()
            }
        )
        .is_err()
    );
    let plan = SlicedPlan::new(&code, &sizes, &tree, &[1], SliceBudget::default()).unwrap();
    assert!(
        plan.execute_with(&[], |_| panic!("must validate first"))
            .is_err()
    );
    assert!(
        plan.execute_with(&[tensor(&[2, 3], 0.1), tensor(&[3, 2], 0.1)], |_| Ok(
            tensor(&[], 0.1)
        ))
        .is_err()
    );
    assert_eq!(
        SlicedPlan::new(&code, &sizes, &tree, &[2, 0, 1], SliceBudget::default())
            .unwrap()
            .slicing(),
        &[0, 1, 2]
    );
}
