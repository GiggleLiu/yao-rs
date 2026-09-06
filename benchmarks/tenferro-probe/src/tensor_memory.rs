//! Shared polynomial and fixed-path memory fixtures. Matrices are synthetic
//! contraction workloads, separately labelled from the Yao circuit comparison.
use crate::Result;
use ndarray::{ArrayD, IxDyn};
use num_complex::Complex64 as C;
use omeco::{EinCode, NestedEinsum};
use std::collections::HashMap;
use yao_rs::slicing::{SliceBudget, SlicedPlan};
use yao_rs::{
    Circuit, Gate, NoiseChannel, Op, OperatorPolynomial, OperatorString, TensorNetworkDM, channel,
    control, put,
};

pub fn observable(n: usize, terms: usize) -> OperatorPolynomial {
    let mut coefficients = vec![C::new(0.2, 0.1)];
    let mut words = vec![OperatorString::identity()];
    for k in 1..terms {
        coefficients.push(C::new(
            (k as f64 * 0.17).cos() / terms as f64,
            (k as f64 * 0.23).sin() / terms as f64,
        ));
        words.push(OperatorString::new(vec![
            (k % n, Op::Y),
            ((k + 1) % n, Op::Z),
        ]));
    }
    OperatorPolynomial::new(coefficients, words)
}
pub fn circuit(n: usize, noisy: bool) -> Result<Circuit> {
    let mut gates = Vec::new();
    for layer in 0..3 {
        for site in 0..n {
            gates.push(put(
                vec![site],
                Gate::Ry(0.1 + site as f64 * 0.13 + layer as f64 * 0.2),
            ));
            gates.push(put(vec![site], Gate::Rz(-0.3 + site as f64 * 0.07)));
            if site + 1 < n {
                gates.push(control(vec![site], vec![site + 1], Gate::X));
            }
        }
    }
    if noisy {
        gates.push(channel(vec![n - 1], NoiseChannel::BitFlip { p: 0.2 }));
    }
    Ok(Circuit::qubits(n, gates)?)
}
pub fn network(circuit: &Circuit, operator: &OperatorPolynomial, noisy: bool) -> TensorNetworkDM {
    if noisy {
        yao_rs::circuit_to_expectation_dm(circuit, operator)
    } else {
        let tn = yao_rs::circuit_to_expectation(circuit, operator);
        TensorNetworkDM {
            code: EinCode::new(
                tn.code
                    .ixs
                    .iter()
                    .map(|v| v.iter().map(|&l| l as i32).collect())
                    .collect(),
                vec![],
            ),
            tensors: tn.tensors,
            size_dict: tn.size_dict.iter().map(|(&l, &d)| (l as i32, d)).collect(),
        }
    }
}
pub fn term_label(tn: &TensorNetworkDM, terms: usize) -> Result<i32> {
    tn.size_dict
        .iter()
        .find(|(_, d)| **d == terms)
        .map(|(&l, _)| l)
        .ok_or_else(|| "term selector missing".into())
}

/// A B C -> i,l with a fixed ((A B) C) tree, dense complex128 matrices.
/// Data depends only on row/column, not storage layout or process seeds.
pub fn matrix_network(n: usize) -> TensorNetworkDM {
    let tensors = (0..3)
        .map(|k| {
            ArrayD::from_shape_fn(IxDyn(&[n, n]), |i| {
                let phase = (i[0] * 3 + i[1] * 7 + k * 11) as f64 * 0.013;
                C::new(phase.cos(), (phase * 0.7).sin()) / (n as f64).sqrt()
            })
        })
        .collect();
    TensorNetworkDM {
        code: EinCode::new(vec![vec![0, 1], vec![1, 2], vec![2, 3]], vec![0, 3]),
        tensors,
        size_dict: HashMap::from([(0, n), (1, n), (2, n), (3, n)]),
    }
}
/// Deliberately memory-intensive supplied order: first form A[i,j] B[k,l],
/// then contract C[j,k]. A greedy plan avoids the n^4 intermediate entirely.
pub fn outer_network(n: usize) -> TensorNetworkDM {
    let mut tn = matrix_network(n);
    tn.code = EinCode::new(vec![vec![0, 1], vec![2, 3], vec![1, 2]], vec![0, 3]);
    tn
}
fn outer_tree() -> NestedEinsum<i32> {
    NestedEinsum::node(
        vec![
            NestedEinsum::node(
                vec![NestedEinsum::leaf(0), NestedEinsum::leaf(1)],
                EinCode::new(vec![vec![0, 1], vec![2, 3]], vec![0, 1, 2, 3]),
            ),
            NestedEinsum::leaf(2),
        ],
        EinCode::new(vec![vec![0, 1, 2, 3], vec![1, 2]], vec![0, 3]),
    )
}
pub fn matrix_workloads() -> Vec<(&'static str, usize, Vec<&'static str>)> {
    vec![
        ("chain", 32, vec!["unsliced", "fixed_output", "auto"]),
        ("chain", 128, vec!["unsliced", "fixed_output"]),
        ("chain", 256, vec!["unsliced", "fixed_output"]),
        ("outer", 32, vec!["unsliced", "fixed_output", "greedy"]),
        ("outer", 64, vec!["unsliced", "fixed_output", "greedy"]),
    ]
}

pub fn matrix_tree() -> NestedEinsum<i32> {
    NestedEinsum::node(
        vec![
            NestedEinsum::node(
                vec![NestedEinsum::leaf(0), NestedEinsum::leaf(1)],
                EinCode::new(vec![vec![0, 1], vec![1, 2]], vec![0, 2]),
            ),
            NestedEinsum::leaf(2),
        ],
        EinCode::new(vec![vec![0, 2], vec![2, 3]], vec![0, 3]),
    )
}
pub fn matrix_plan(tn: &TensorNetworkDM, mode: &str) -> Result<SlicedPlan<i32>> {
    let tree = if tn.code.ixs[1] == [2, 3] {
        outer_tree()
    } else {
        matrix_tree()
    };
    let base = SlicedPlan::new(&tn.code, &tn.size_dict, &tree, &[], SliceBudget::default())?;
    Ok(match mode {
        "unsliced" => base,
        "greedy" => {
            let tree = yao_rs::contraction_plan::optimize_code(
                &tn.code,
                &tn.size_dict,
                &omeco::GreedyMethod::default(),
            )
            .ok_or("greedy planning failed")?;
            SlicedPlan::new(&tn.code, &tn.size_dict, &tree, &[], SliceBudget::default())?
        }
        "fixed_output" => {
            SlicedPlan::new(&tn.code, &tn.size_dict, &tree, &[0], SliceBudget::default())?
        }
        "auto" => {
            let target = base.estimate().input_bytes
                + base.estimate().output_bytes
                + base.estimate().worker_buffer_bytes / 2;
            SlicedPlan::auto(
                &tn.code,
                &tn.size_dict,
                &tree,
                SliceBudget {
                    max_bytes: Some(target),
                    ..SliceBudget::default()
                },
                &omeco::TreeSASlicer::fast(),
            )?
        }
        _ => return Err("unknown matrix slicing mode".into()),
    })
}

/// Record the actual chosen tree/slices for each independent timing process.
/// TreeSA heuristics may choose different equivalent plans across processes.
pub fn record_plan<L: omeco::Label + serde::Serialize>(
    id: &str,
    mode: &str,
    plan: &SlicedPlan<L>,
) -> Result<()> {
    use std::io::Write;
    if let Ok(path) = std::env::var("YAO_BENCH_PLAN_LOG") {
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        let value = serde_json::json!({"id":id,"mode":mode,"slicing":plan.slicing(),"estimate":plan.estimate(),"tree":omeco::json::NestedEinsumTree::from(plan.tree())});
        writeln!(file, "{value}")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn outer_product_stress_path_and_greedy_match_dense_product() {
        let tn = outer_network(8);
        let m: Vec<_> = tn
            .tensors
            .iter()
            .map(|t| t.view().into_dimensionality::<ndarray::Ix2>().unwrap())
            .collect();
        let expected = m[0].dot(&m[2]).dot(&m[1]).into_dyn();
        let cpu = yao_rs::tenferro::CpuContractor::new(1).unwrap();
        for mode in ["unsliced", "fixed_output", "greedy"] {
            let plan = matrix_plan(&tn, mode).unwrap();
            let prepared = cpu.prepare_sliced(&plan).unwrap();
            for result in [
                cpu.execute_sliced(&prepared, &tn.tensors).unwrap(),
                yao_rs::contractor::contract_sliced(&plan, &tn.tensors).unwrap(),
            ] {
                assert!(
                    result
                        .iter()
                        .zip(&expected)
                        .all(|(a, b)| (a - b).norm() < 1e-11)
                );
            }
        }
    }
    #[test]
    fn matrix_plans_match_independent_dense_multiplication() {
        let tn = matrix_network(8);
        let matrices: Vec<_> = tn
            .tensors
            .iter()
            .map(|t| t.view().into_dimensionality::<ndarray::Ix2>().unwrap())
            .collect();
        let expected = matrices[0].dot(&matrices[1]).dot(&matrices[2]).into_dyn();
        let cpu = yao_rs::tenferro::CpuContractor::new(1).unwrap();
        for mode in ["unsliced", "fixed_output", "auto"] {
            let plan = matrix_plan(&tn, mode).unwrap();
            let prepared = cpu.prepare_sliced(&plan).unwrap();
            for result in [
                cpu.execute_sliced(&prepared, &tn.tensors).unwrap(),
                yao_rs::contractor::contract_sliced(&plan, &tn.tensors).unwrap(),
            ] {
                assert!(
                    result
                        .iter()
                        .zip(&expected)
                        .all(|(a, b)| (a - b).norm() < 1e-11)
                );
            }
        }
    }
}
