//! Shared test utilities for yao-rs integration tests.

use std::collections::HashMap;

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use num_complex::Complex64;
use serde::Deserialize;

use yao_rs::einsum::TensorNetwork;
use yao_rs::state::State;

// ==================== JSON Data Structures ====================

#[derive(Deserialize)]
#[allow(dead_code)]
pub struct GatesData {
    pub gates: Vec<GateEntry>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
pub struct GateEntry {
    pub name: String,
    pub params: Vec<f64>,
    pub matrix_re: Vec<Vec<f64>>,
    pub matrix_im: Vec<Vec<f64>>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
pub struct ApplyData {
    pub cases: Vec<ApplyCase>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
pub struct ApplyCase {
    pub label: String,
    pub dims: Vec<usize>,
    pub gates: Vec<GateSpec>,
    #[serde(default)]
    pub input_state_re: Option<Vec<f64>>,
    #[serde(default)]
    pub input_state_im: Option<Vec<f64>>,
    pub output_state_re: Vec<f64>,
    pub output_state_im: Vec<f64>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
pub struct GateSpec {
    pub name: String,
    pub params: Vec<f64>,
    pub targets: Vec<usize>,
    pub controls: Vec<usize>,
    pub control_configs: Vec<bool>,
    #[serde(default)]
    pub matrix_re: Option<Vec<Vec<f64>>>,
    #[serde(default)]
    pub matrix_im: Option<Vec<Vec<f64>>>,
    #[serde(default)]
    pub label: Option<String>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
pub struct MeasureData {
    pub cases: Vec<MeasureCase>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
pub struct MeasureCase {
    pub label: String,
    pub num_qubits: usize,
    pub gates: Vec<GateSpec>,
    pub probabilities: Vec<f64>,
}

// ==================== JSON Loading Functions ====================

#[allow(dead_code)]
pub fn load_gates_data() -> GatesData {
    let data = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/gates.json"
    ))
    .unwrap();
    serde_json::from_str(&data).unwrap()
}

#[allow(dead_code)]
pub fn load_apply_data() -> ApplyData {
    let data = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/apply.json"
    ))
    .unwrap();
    serde_json::from_str(&data).unwrap()
}

#[allow(dead_code)]
pub fn load_einsum_data() -> ApplyData {
    // Same format as apply
    let data = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/einsum.json"
    ))
    .unwrap();
    serde_json::from_str(&data).unwrap()
}

#[allow(dead_code)]
pub fn load_measure_data() -> MeasureData {
    let data = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/measure.json"
    ))
    .unwrap();
    serde_json::from_str(&data).unwrap()
}

// ==================== JSON Conversion Utilities ====================

/// Convert JSON matrix (re/im arrays) to ndarray Array2<Complex64>
#[allow(dead_code)]
pub fn matrix_from_json(re: &[Vec<f64>], im: &[Vec<f64>]) -> Array2<Complex64> {
    let rows = re.len();
    let cols = re[0].len();
    let mut data = Vec::with_capacity(rows * cols);
    for i in 0..rows {
        for j in 0..cols {
            data.push(Complex64::new(re[i][j], im[i][j]));
        }
    }
    Array2::from_shape_vec((rows, cols), data).unwrap()
}

/// Convert JSON state (re/im vectors) to Array1<Complex64>
#[allow(dead_code)]
pub fn state_from_json(re: &[f64], im: &[f64]) -> Array1<Complex64> {
    let data: Vec<Complex64> = re
        .iter()
        .zip(im.iter())
        .map(|(&r, &i)| Complex64::new(r, i))
        .collect();
    Array1::from(data)
}

/// Build a yao-rs Gate from a GateSpec
#[allow(dead_code)]
pub fn gate_from_spec(spec: &GateSpec) -> yao_rs::Gate {
    match spec.name.as_str() {
        "X" => yao_rs::Gate::X,
        "Y" => yao_rs::Gate::Y,
        "Z" => yao_rs::Gate::Z,
        "H" => yao_rs::Gate::H,
        "S" => yao_rs::Gate::S,
        "T" => yao_rs::Gate::T,
        "SWAP" => yao_rs::Gate::SWAP,
        "SqrtX" => yao_rs::Gate::SqrtX,
        "SqrtY" => yao_rs::Gate::SqrtY,
        "SqrtW" => yao_rs::Gate::SqrtW,
        "ISWAP" => yao_rs::Gate::ISWAP,
        "Phase" => yao_rs::Gate::Phase(spec.params[0]),
        "Rx" => yao_rs::Gate::Rx(spec.params[0]),
        "Ry" => yao_rs::Gate::Ry(spec.params[0]),
        "Rz" => yao_rs::Gate::Rz(spec.params[0]),
        "FSim" => yao_rs::Gate::FSim(spec.params[0], spec.params[1]),
        "Custom" => {
            let re = spec.matrix_re.as_ref().unwrap();
            let im = spec.matrix_im.as_ref().unwrap();
            let matrix = matrix_from_json(re, im);
            // Determine if diagonal by checking off-diagonal elements
            let rows = matrix.nrows();
            let is_diagonal =
                (0..rows).all(|i| (0..rows).all(|j| i == j || matrix[[i, j]].norm() < 1e-15));
            yao_rs::Gate::Custom {
                matrix,
                is_diagonal,
                label: spec.label.clone().unwrap_or_default(),
            }
        }
        other => panic!("Unknown gate type in test data: {}", other),
    }
}

/// Build a Gate from a GateEntry (for gates.json)
#[allow(dead_code)]
pub fn gate_from_entry(entry: &GateEntry) -> yao_rs::Gate {
    match entry.name.as_str() {
        "X" => yao_rs::Gate::X,
        "Y" => yao_rs::Gate::Y,
        "Z" => yao_rs::Gate::Z,
        "H" => yao_rs::Gate::H,
        "S" => yao_rs::Gate::S,
        "T" => yao_rs::Gate::T,
        "SWAP" => yao_rs::Gate::SWAP,
        "SqrtX" => yao_rs::Gate::SqrtX,
        "SqrtY" => yao_rs::Gate::SqrtY,
        "SqrtW" => yao_rs::Gate::SqrtW,
        "ISWAP" => yao_rs::Gate::ISWAP,
        "Phase" => yao_rs::Gate::Phase(entry.params[0]),
        "Rx" => yao_rs::Gate::Rx(entry.params[0]),
        "Ry" => yao_rs::Gate::Ry(entry.params[0]),
        "Rz" => yao_rs::Gate::Rz(entry.params[0]),
        "FSim" => yao_rs::Gate::FSim(entry.params[0], entry.params[1]),
        other => panic!("Unknown gate type in gates data: {}", other),
    }
}

/// Build a Circuit from an ApplyCase's gate specs
#[allow(dead_code)]
pub fn circuit_from_case(case: &ApplyCase) -> yao_rs::Circuit {
    let elements: Vec<yao_rs::CircuitElement> = case
        .gates
        .iter()
        .map(|spec| {
            let gate = gate_from_spec(spec);
            if spec.controls.is_empty() {
                yao_rs::circuit::put(spec.targets.clone(), gate)
            } else {
                let configs = if spec.control_configs.is_empty() {
                    vec![true; spec.controls.len()]
                } else {
                    spec.control_configs.clone()
                };
                yao_rs::CircuitElement::Gate(yao_rs::PositionedGate::new(
                    gate,
                    spec.targets.clone(),
                    spec.controls.clone(),
                    configs,
                ))
            }
        })
        .collect();
    yao_rs::Circuit::new(case.dims.clone(), elements).unwrap()
}

/// Build a Circuit from a MeasureCase's gate specs (uses num_qubits for dims)
#[allow(dead_code)]
pub fn circuit_from_measure_case(case: &MeasureCase) -> yao_rs::Circuit {
    let dims = vec![2; case.num_qubits];
    let elements: Vec<yao_rs::CircuitElement> = case
        .gates
        .iter()
        .map(|spec| {
            let gate = gate_from_spec(spec);
            if spec.controls.is_empty() {
                yao_rs::circuit::put(spec.targets.clone(), gate)
            } else {
                let configs = if spec.control_configs.is_empty() {
                    vec![true; spec.controls.len()]
                } else {
                    spec.control_configs.clone()
                };
                yao_rs::CircuitElement::Gate(yao_rs::PositionedGate::new(
                    gate,
                    spec.targets.clone(),
                    spec.controls.clone(),
                    configs,
                ))
            }
        })
        .collect();
    yao_rs::Circuit::new(dims, elements).unwrap()
}

// ==================== State Comparison ====================

/// Assert that two state vectors (Array1<Complex64>) are close element-wise.
#[allow(dead_code)]
pub fn assert_states_close(a: &Array1<Complex64>, b: &Array1<Complex64>) {
    const ATOL: f64 = 1e-10;
    assert_eq!(
        a.len(),
        b.len(),
        "State vectors have different lengths: {} vs {}",
        a.len(),
        b.len()
    );
    for (i, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (av - bv).norm();
        assert!(
            diff < ATOL,
            "States differ at index {}: got {:?}, expected {:?}, diff = {}",
            i,
            av,
            bv,
            diff
        );
    }
}

/// Assert that two matrices (Array2<Complex64>) are close element-wise.
#[allow(dead_code)]
pub fn assert_matrices_close(a: &Array2<Complex64>, b: &Array2<Complex64>, tol: f64, msg: &str) {
    assert_eq!(
        a.dim(),
        b.dim(),
        "{}: matrices have different dimensions",
        msg
    );
    let (rows, cols) = a.dim();
    for i in 0..rows {
        for j in 0..cols {
            let diff = (a[[i, j]] - b[[i, j]]).norm();
            assert!(
                diff < tol,
                "{}: matrices differ at [{},{}]: got {:?}, expected {:?}, diff = {}",
                msg,
                i,
                j,
                a[[i, j]],
                b[[i, j]],
                diff
            );
        }
    }
}

/// Check approximate equality of two complex numbers.
#[allow(dead_code)]
pub fn approx_eq(a: Complex64, b: Complex64, tol: f64) -> bool {
    (a - b).norm() < tol
}

// ==================== Index Utilities ====================

/// Decompose a flat index into a multi-index given dimensions (row-major order).
#[allow(dead_code)]
pub fn flat_to_multi(mut index: usize, shape: &[usize]) -> Vec<usize> {
    let n = shape.len();
    let mut multi = vec![0usize; n];
    for i in (0..n).rev() {
        multi[i] = index % shape[i];
        index /= shape[i];
    }
    multi
}

/// Compose a multi-index into a flat index given dimensions (row-major order).
#[allow(dead_code)]
pub fn multi_to_flat(indices: &[usize], shape: &[usize]) -> usize {
    let mut index = 0usize;
    for (i, &idx) in indices.iter().enumerate() {
        let stride: usize = shape[i + 1..].iter().product();
        index += idx * stride;
    }
    index
}

// ==================== Tensor Contraction ====================

/// Contract two tensors along shared (contracted) indices.
///
/// a: tensor A with indices a_indices
/// b: tensor B with indices b_indices
/// contracted: the set of index labels that are summed over and removed
/// size_dict: maps index labels to their dimensions
///
/// Indices that appear in both a_indices and b_indices but NOT in contracted
/// are treated as "batch" dimensions (element-wise, kept in result once).
///
/// Returns (result_tensor, result_indices)
#[allow(dead_code)]
pub fn contract_tensors(
    a: &ArrayD<Complex64>,
    a_indices: &[usize],
    b: &ArrayD<Complex64>,
    b_indices: &[usize],
    contracted: &[usize],
    size_dict: &HashMap<usize, usize>,
) -> (ArrayD<Complex64>, Vec<usize>) {
    // Batch indices: appear in both a and b but are NOT contracted
    let batch: Vec<usize> = a_indices
        .iter()
        .filter(|idx| b_indices.contains(idx) && !contracted.contains(idx))
        .copied()
        .collect();

    // Result indices: all indices from a and b except contracted ones,
    // with batch indices appearing only once
    let mut result_indices: Vec<usize> = Vec::new();
    for &idx in a_indices {
        if !contracted.contains(&idx) {
            result_indices.push(idx);
        }
    }
    for &idx in b_indices {
        if !contracted.contains(&idx) && !batch.contains(&idx) {
            // Only add b's non-batch, non-contracted indices
            if !result_indices.contains(&idx) {
                result_indices.push(idx);
            }
        }
    }

    // Build result shape
    let result_shape: Vec<usize> = result_indices.iter().map(|idx| size_dict[idx]).collect();
    let result_total: usize = if result_shape.is_empty() {
        1
    } else {
        result_shape.iter().product()
    };

    // Build shapes for a and b
    let a_shape: Vec<usize> = a_indices.iter().map(|idx| size_dict[idx]).collect();
    let b_shape: Vec<usize> = b_indices.iter().map(|idx| size_dict[idx]).collect();
    let a_total: usize = if a_shape.is_empty() {
        1
    } else {
        a_shape.iter().product()
    };
    let b_total: usize = if b_shape.is_empty() {
        1
    } else {
        b_shape.iter().product()
    };

    let mut result_data = vec![Complex64::new(0.0, 0.0); result_total];

    // Iterate over all index combinations of a and b
    for ai in 0..a_total {
        let a_multi = flat_to_multi(ai, &a_shape);
        let a_val = a[IxDyn(&a_multi)];

        if a_val == Complex64::new(0.0, 0.0) {
            continue;
        }

        for bi in 0..b_total {
            let b_multi = flat_to_multi(bi, &b_shape);

            // Check that contracted indices match between a and b
            let mut match_ok = true;
            for &s in contracted {
                let a_pos = a_indices.iter().position(|&x| x == s);
                let b_pos = b_indices.iter().position(|&x| x == s);
                match (a_pos, b_pos) {
                    (Some(ap), Some(bp)) => {
                        if a_multi[ap] != b_multi[bp] {
                            match_ok = false;
                            break;
                        }
                    }
                    _ => {
                        match_ok = false;
                        break;
                    }
                }
            }
            if !match_ok {
                continue;
            }

            // Check that batch indices match between a and b
            for &s in &batch {
                let a_pos = a_indices.iter().position(|&x| x == s);
                let b_pos = b_indices.iter().position(|&x| x == s);
                match (a_pos, b_pos) {
                    (Some(ap), Some(bp)) => {
                        if a_multi[ap] != b_multi[bp] {
                            match_ok = false;
                            break;
                        }
                    }
                    _ => {
                        match_ok = false;
                        break;
                    }
                }
            }
            if !match_ok {
                continue;
            }

            let b_val = b[IxDyn(&b_multi)];
            let product = a_val * b_val;

            if product == Complex64::new(0.0, 0.0) {
                continue;
            }

            // Determine result multi-index
            let mut res_multi = vec![0usize; result_indices.len()];
            for (ri, &ridx) in result_indices.iter().enumerate() {
                if let Some(pos) = a_indices.iter().position(|&x| x == ridx) {
                    res_multi[ri] = a_multi[pos];
                } else if let Some(pos) = b_indices.iter().position(|&x| x == ridx) {
                    res_multi[ri] = b_multi[pos];
                }
            }

            let flat_idx = if result_shape.is_empty() {
                0
            } else {
                multi_to_flat(&res_multi, &result_shape)
            };
            result_data[flat_idx] += product;
        }
    }

    let result_tensor = if result_shape.is_empty() {
        ArrayD::from_shape_vec(IxDyn(&[]), result_data).unwrap()
    } else {
        ArrayD::from_shape_vec(IxDyn(&result_shape), result_data).unwrap()
    };

    (result_tensor, result_indices)
}

/// Naive contraction of a tensor network with an input state.
///
/// 1. Convert state to a tensor with shape (d0, d1, ..., d_{n-1}) and indices [0, 1, ..., n-1]
/// 2. For each gate tensor, contract with the current result
/// 3. Permute the result to match tn.code.iy order
/// 4. Flatten to Array1
#[allow(dead_code)]
pub fn naive_contract(tn: &TensorNetwork, state: &State) -> Array1<Complex64> {
    let n = state.dims.len();

    // Convert state to a multi-dimensional tensor with initial indices 0..n-1
    let state_shape: Vec<usize> = state.dims.clone();
    let state_tensor = ArrayD::from_shape_vec(IxDyn(&state_shape), state.data.to_vec()).unwrap();

    let mut current_tensor = state_tensor;
    let mut current_indices: Vec<usize> = (0..n).collect();

    // Collect all indices that appear in the output or in future gate tensors
    // to determine which shared indices should be contracted at each step.
    let output_set: std::collections::HashSet<usize> = tn.code.iy.iter().copied().collect();

    // Contract each gate tensor sequentially
    for (i, gate_tensor) in tn.tensors.iter().enumerate() {
        let gate_indices = &tn.code.ixs[i];

        // Determine which indices are needed in the future (output or later gates)
        let mut future_needed: std::collections::HashSet<usize> = output_set.clone();
        for j in (i + 1)..tn.tensors.len() {
            for &idx in &tn.code.ixs[j] {
                future_needed.insert(idx);
            }
        }

        // Shared indices are those appearing in both current tensor and gate tensor
        // Only contract (sum over) shared indices that are NOT needed in the future
        let shared: Vec<usize> = current_indices
            .iter()
            .filter(|idx| gate_indices.contains(idx) && !future_needed.contains(idx))
            .copied()
            .collect();

        let (result, result_indices) = contract_tensors(
            &current_tensor,
            &current_indices,
            gate_tensor,
            gate_indices,
            &shared,
            &tn.size_dict,
        );

        current_tensor = result;
        current_indices = result_indices;
    }

    // Permute result to match tn.code.iy order
    let output_indices = &tn.code.iy;

    // Build permutation: for each position in iy, find its position in current_indices
    let perm: Vec<usize> = output_indices
        .iter()
        .map(|idx| {
            current_indices
                .iter()
                .position(|x| x == idx)
                .unwrap_or_else(|| {
                    panic!(
                        "Output index {} not found in current indices {:?}",
                        idx, current_indices
                    )
                })
        })
        .collect();

    // Permute by building a new tensor
    let output_shape: Vec<usize> = output_indices.iter().map(|idx| tn.size_dict[idx]).collect();
    let total: usize = output_shape.iter().product();
    let mut result_data = vec![Complex64::new(0.0, 0.0); total];

    let current_shape: Vec<usize> = current_indices
        .iter()
        .map(|idx| tn.size_dict[idx])
        .collect();
    let current_total: usize = current_shape.iter().product();

    for flat in 0..current_total {
        let multi = flat_to_multi(flat, &current_shape);
        let val = current_tensor[IxDyn(&multi)];
        if val == Complex64::new(0.0, 0.0) {
            continue;
        }

        // Apply permutation
        let mut out_multi = vec![0usize; output_indices.len()];
        for (out_pos, &src_pos) in perm.iter().enumerate() {
            out_multi[out_pos] = multi[src_pos];
        }

        let out_flat = multi_to_flat(&out_multi, &output_shape);
        result_data[out_flat] = val;
    }

    Array1::from_vec(result_data)
}

/// Contract a TensorNetwork by naive summation (for testing small networks).
#[allow(dead_code)]
pub fn contract_tn(tn: &TensorNetwork) -> ArrayD<Complex64> {
    let size_dict = &tn.size_dict;

    // Collect all unique labels
    let mut all_labels: Vec<usize> = Vec::new();
    for ixs in &tn.code.ixs {
        for &label in ixs {
            if !all_labels.contains(&label) {
                all_labels.push(label);
            }
        }
    }
    for &label in &tn.code.iy {
        if !all_labels.contains(&label) {
            all_labels.push(label);
        }
    }

    let output_labels = &tn.code.iy;

    let label_to_idx: HashMap<usize, usize> = all_labels
        .iter()
        .enumerate()
        .map(|(i, &l)| (l, i))
        .collect();

    let all_dims: Vec<usize> = all_labels
        .iter()
        .map(|l| *size_dict.get(l).unwrap())
        .collect();

    let total: usize = all_dims.iter().product();

    let out_dims: Vec<usize> = output_labels
        .iter()
        .map(|l| *size_dict.get(l).unwrap())
        .collect();
    let out_total: usize = if out_dims.is_empty() {
        1
    } else {
        out_dims.iter().product()
    };

    let mut result_data = vec![Complex64::new(0.0, 0.0); out_total];

    for flat_idx in 0..total {
        let mut multi_idx = vec![0usize; all_labels.len()];
        let mut remainder = flat_idx;
        for i in (0..all_labels.len()).rev() {
            multi_idx[i] = remainder % all_dims[i];
            remainder /= all_dims[i];
        }

        let mut product = Complex64::new(1.0, 0.0);
        for (t_idx, tensor) in tn.tensors.iter().enumerate() {
            let ixs = &tn.code.ixs[t_idx];
            let t_indices: Vec<usize> = ixs.iter().map(|l| multi_idx[label_to_idx[l]]).collect();
            let ix_dyn = IxDyn(&t_indices);
            product *= tensor[ix_dyn];
        }

        let mut out_flat = 0usize;
        let mut out_stride = 1usize;
        for i in (0..output_labels.len()).rev() {
            let label = output_labels[i];
            out_flat += multi_idx[label_to_idx[&label]] * out_stride;
            out_stride *= out_dims[i];
        }

        result_data[out_flat] += product;
    }

    if out_dims.is_empty() {
        ArrayD::from_shape_vec(IxDyn(&[]), result_data).unwrap()
    } else {
        ArrayD::from_shape_vec(IxDyn(&out_dims), result_data).unwrap()
    }
}

// ==================== Apply Old (Reference Implementation) ====================

/// Decompose a flat index into a multi-index given dimensions (row-major order).
/// Used by apply_old reference implementation.
#[allow(dead_code)]
pub fn linear_to_multi(mut index: usize, dims: &[usize]) -> Vec<usize> {
    let n = dims.len();
    let mut multi = vec![0usize; n];
    for i in (0..n).rev() {
        multi[i] = index % dims[i];
        index /= dims[i];
    }
    multi
}

/// Compose a multi-index into a flat index given dimensions (row-major order).
/// Used by apply_old reference implementation.
#[allow(dead_code)]
pub fn multi_to_linear(indices: &[usize], dims: &[usize]) -> usize {
    let mut index = 0usize;
    for (i, &idx) in indices.iter().enumerate() {
        let stride: usize = dims[i + 1..].iter().product();
        index += idx * stride;
    }
    index
}

/// Simple matrix-vector multiplication for complex matrices and vectors.
#[allow(dead_code)]
pub fn matrix_vector_mul(mat: &Array2<Complex64>, vec: &Array1<Complex64>) -> Array1<Complex64> {
    let n = mat.nrows();
    let mut result = Array1::zeros(n);
    for i in 0..n {
        let mut sum = Complex64::new(0.0, 0.0);
        for j in 0..mat.ncols() {
            sum += mat[[i, j]] * vec[j];
        }
        result[i] = sum;
    }
    result
}

/// Build the controlled local matrix on all involved sites (controls + targets).
///
/// all_dims: dimensions for all involved sites (control_locs ++ target_locs order)
/// gate_matrix: the gate's matrix on target sites only
/// control_configs: which control configuration triggers the gate
/// num_controls: number of control sites
#[allow(dead_code)]
pub fn build_controlled_matrix(
    all_dims: &[usize],
    gate_matrix: &Array2<Complex64>,
    control_configs: &[bool],
    num_controls: usize,
) -> Array2<Complex64> {
    if num_controls == 0 {
        return gate_matrix.clone();
    }

    let involved_dim: usize = all_dims.iter().product();
    let control_dims = &all_dims[..num_controls];
    let target_dims = &all_dims[num_controls..];
    let target_dim: usize = target_dims.iter().product();

    let mut mat = Array2::zeros((involved_dim, involved_dim));

    // Compute the trigger index in the control subspace
    let trigger_index: usize = control_configs
        .iter()
        .enumerate()
        .map(|(i, &cfg)| {
            let val = if cfg { 1usize } else { 0usize };
            let stride: usize = control_dims[i + 1..].iter().product();
            val * stride
        })
        .sum();

    let control_dim: usize = control_dims.iter().product();

    for ctrl_idx in 0..control_dim {
        for t_row in 0..target_dim {
            let row = ctrl_idx * target_dim + t_row;
            if ctrl_idx == trigger_index {
                // Apply the gate matrix on the target portion
                for t_col in 0..target_dim {
                    let col = ctrl_idx * target_dim + t_col;
                    mat[[row, col]] = gate_matrix[[t_row, t_col]];
                }
            } else {
                // Identity on the target portion
                let col = ctrl_idx * target_dim + t_row;
                mat[[row, col]] = Complex64::new(1.0, 0.0);
            }
        }
    }

    mat
}

/// Apply a circuit to a quantum state by building and multiplying full-space matrices.
///
/// This is the original implementation kept for comparison testing.
/// It builds full matrices in the Hilbert space which is O(4^n) memory.
///
/// For each gate in the circuit:
/// 1. Build the controlled local matrix on all involved sites
/// 2. Embed it into the full Hilbert space
/// 3. Multiply by the state vector
#[allow(dead_code)]
pub fn apply_old(circuit: &yao_rs::Circuit, state: &State) -> State {
    use yao_rs::circuit::CircuitElement;

    let dims = &circuit.dims;
    let total_dim = circuit.total_dim();
    let mut current_data = state.data.clone();

    for element in &circuit.elements {
        let pg = match element {
            CircuitElement::Gate(pg) => pg,
            CircuitElement::Annotation(_) => continue, // Skip annotations
        };

        // Get the gate's local matrix on target sites
        let d = dims[pg.target_locs[0]];
        let gate_matrix = pg.gate.matrix(d);

        // Build the controlled local matrix on all involved sites
        let all_locs = pg.all_locs(); // control_locs ++ target_locs
        let all_dims: Vec<usize> = all_locs.iter().map(|&loc| dims[loc]).collect();
        let num_controls = pg.control_locs.len();

        let local_matrix =
            build_controlled_matrix(&all_dims, &gate_matrix, &pg.control_configs, num_controls);

        // Embed into full Hilbert space and multiply
        let mut full_matrix = Array2::zeros((total_dim, total_dim));

        for row in 0..total_dim {
            let row_multi = linear_to_multi(row, dims);
            for col in 0..total_dim {
                let col_multi = linear_to_multi(col, dims);

                // Check that non-involved sites are the same between row and col
                let mut non_involved_match = true;
                for site in 0..dims.len() {
                    if !all_locs.contains(&site) && row_multi[site] != col_multi[site] {
                        non_involved_match = false;
                        break;
                    }
                }

                if !non_involved_match {
                    // entry is 0 (already initialized)
                    continue;
                }

                // Extract involved-site indices for row and col
                let row_involved: Vec<usize> = all_locs.iter().map(|&loc| row_multi[loc]).collect();
                let col_involved: Vec<usize> = all_locs.iter().map(|&loc| col_multi[loc]).collect();

                let local_row = multi_to_linear(&row_involved, &all_dims);
                let local_col = multi_to_linear(&col_involved, &all_dims);

                full_matrix[[row, col]] = local_matrix[[local_row, local_col]];
            }
        }

        current_data = matrix_vector_mul(&full_matrix, &current_data);
    }

    State {
        dims: dims.clone(),
        data: current_data,
    }
}
