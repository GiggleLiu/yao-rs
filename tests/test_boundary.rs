use ndarray::{ArrayD, IxDyn};
use num_complex::Complex64;
use std::collections::HashMap;

use yao_rs::{
    Circuit, Gate, apply,
    circuit_to_einsum_with_boundary, TensorNetwork,
    state::State, easybuild::qft_circuit,
    circuit::put, circuit::control,
};

fn approx_eq(a: Complex64, b: Complex64, tol: f64) -> bool {
    (a - b).norm() < tol
}

/// Contract a TensorNetwork by naive summation (for testing small networks).
fn contract_tn(tn: &TensorNetwork) -> ArrayD<Complex64> {
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

    let label_to_idx: HashMap<usize, usize> = all_labels.iter()
        .enumerate()
        .map(|(i, &l)| (l, i))
        .collect();

    let all_dims: Vec<usize> = all_labels.iter()
        .map(|l| *size_dict.get(l).unwrap())
        .collect();

    let total: usize = all_dims.iter().product();

    let out_dims: Vec<usize> = output_labels.iter()
        .map(|l| *size_dict.get(l).unwrap())
        .collect();
    let out_total: usize = if out_dims.is_empty() { 1 } else { out_dims.iter().product() };

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
            let t_indices: Vec<usize> = ixs.iter()
                .map(|l| multi_idx[label_to_idx[l]])
                .collect();
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

#[test]
fn test_boundary_all_pinned_identity() {
    // Empty circuit (identity): ⟨0|I|0⟩ = 1
    let circuit = Circuit::new(vec![2, 2], vec![]).unwrap();
    let tn = circuit_to_einsum_with_boundary(&circuit, &[0, 1]);
    let result = contract_tn(&tn);
    assert_eq!(result.ndim(), 0);
    assert!(approx_eq(result[[]], Complex64::new(1.0, 0.0), 1e-10));
}

#[test]
fn test_boundary_x_gate_amplitude() {
    // X on qubit 0: ⟨0|X|0⟩ = 0
    let circuit = Circuit::new(
        vec![2],
        vec![put(vec![0], Gate::X)],
    ).unwrap();
    let tn = circuit_to_einsum_with_boundary(&circuit, &[0]);
    let result = contract_tn(&tn);
    assert!(approx_eq(result[[]], Complex64::new(0.0, 0.0), 1e-10));
}

#[test]
fn test_boundary_h_gate_amplitude() {
    // H on qubit 0: ⟨0|H|0⟩ = 1/√2
    let circuit = Circuit::new(
        vec![2],
        vec![put(vec![0], Gate::H)],
    ).unwrap();
    let tn = circuit_to_einsum_with_boundary(&circuit, &[0]);
    let result = contract_tn(&tn);
    let expected = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);
    assert!(approx_eq(result[[]], expected, 1e-10));
}

#[test]
fn test_boundary_cnot_amplitude() {
    // CNOT on |00⟩: output is |00⟩, so ⟨00|CNOT|00⟩ = 1
    let circuit = Circuit::new(
        vec![2, 2],
        vec![control(vec![0], vec![1], Gate::X)],
    ).unwrap();
    let tn = circuit_to_einsum_with_boundary(&circuit, &[0, 1]);
    let result = contract_tn(&tn);
    assert!(approx_eq(result[[]], Complex64::new(1.0, 0.0), 1e-10));
}

#[test]
fn test_boundary_matches_apply() {
    // H on qubit 0, CNOT(0→1): Bell state
    // ⟨00|C|00⟩ = 1/√2
    let circuit = Circuit::new(
        vec![2, 2],
        vec![
            put(vec![0], Gate::H),
            control(vec![0], vec![1], Gate::X),
        ],
    ).unwrap();

    let state = State::zero_state(&[2, 2]);
    let result_state = apply(&circuit, &state);

    let tn = circuit_to_einsum_with_boundary(&circuit, &[0, 1]);
    let result = contract_tn(&tn);

    // ⟨00|Bell⟩ = amplitude of |00⟩ = 1/√2
    let expected = result_state.data[0]; // first element of flat state
    assert!(approx_eq(result[[]], expected, 1e-10));
}

#[test]
fn test_boundary_open_legs() {
    // H on qubit 0 only, no pinning → full state vector
    let circuit = Circuit::new(
        vec![2, 2],
        vec![put(vec![0], Gate::H)],
    ).unwrap();

    let tn = circuit_to_einsum_with_boundary(&circuit, &[]);
    let result = contract_tn(&tn);

    let state = State::zero_state(&[2, 2]);
    let expected = apply(&circuit, &state);

    // result shape is [2, 2], expected.data is flat [4]
    assert_eq!(result.len(), expected.data.len());
    for flat_idx in 0..4 {
        let i = flat_idx / 2;
        let j = flat_idx % 2;
        assert!(approx_eq(result[[i, j]], expected.data[flat_idx], 1e-10),
            "Mismatch at flat_idx {}: got {:?}, expected {:?}",
            flat_idx, result[[i, j]], expected.data[flat_idx]);
    }
}

#[test]
fn test_boundary_partial_pinning() {
    // H on qubit 0, CNOT(0→1): Bell state
    // Pin qubit 0 only → result over qubit 1
    let circuit = Circuit::new(
        vec![2, 2],
        vec![
            put(vec![0], Gate::H),
            control(vec![0], vec![1], Gate::X),
        ],
    ).unwrap();

    let tn = circuit_to_einsum_with_boundary(&circuit, &[0]);
    let result = contract_tn(&tn);

    // Bell = (|00⟩ + |11⟩)/√2
    // Pin qubit 0 to |0⟩: project → 1/√2 * |0⟩_1
    assert_eq!(result.shape(), &[2]);
    assert!(approx_eq(result[[0]], Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0), 1e-10));
    assert!(approx_eq(result[[1]], Complex64::new(0.0, 0.0), 1e-10));
}

#[test]
fn test_boundary_qft_amplitude() {
    // QFT on |0⟩: ⟨0|QFT|0⟩ = 1/√(2^n)
    let n = 3;
    let circuit = qft_circuit(n);
    let pinned: Vec<usize> = (0..n).collect();
    let tn = circuit_to_einsum_with_boundary(&circuit, &pinned);
    let result = contract_tn(&tn);

    let expected = Complex64::new(1.0 / (2.0_f64.powi(n as i32)).sqrt(), 0.0);
    assert!(approx_eq(result[[]], expected, 1e-10),
        "Got {:?}, expected {:?}", result[[]], expected);
}

#[test]
fn test_boundary_diagonal_gate() {
    // Z gate (diagonal) on |0⟩: Z|0⟩ = |0⟩, so ⟨0|Z|0⟩ = 1
    let circuit = Circuit::new(
        vec![2],
        vec![put(vec![0], Gate::Z)],
    ).unwrap();
    let tn = circuit_to_einsum_with_boundary(&circuit, &[0]);
    let result = contract_tn(&tn);
    assert!(approx_eq(result[[]], Complex64::new(1.0, 0.0), 1e-10));
}

#[test]
fn test_boundary_phase_gate() {
    // Phase(π) ≈ Z on |0⟩: ⟨0|Z|0⟩ = 1
    let circuit = Circuit::new(
        vec![2],
        vec![put(vec![0], Gate::Phase(std::f64::consts::PI))],
    ).unwrap();
    let tn = circuit_to_einsum_with_boundary(&circuit, &[0]);
    let result = contract_tn(&tn);
    assert!(approx_eq(result[[]], Complex64::new(1.0, 0.0), 1e-10));
}

#[test]
fn test_boundary_controlled_phase() {
    // Controlled-Phase(π/2) on |00⟩: no effect since control is |0⟩
    // ⟨00|C-P(π/2)|00⟩ = 1
    let circuit = Circuit::new(
        vec![2, 2],
        vec![control(vec![0], vec![1], Gate::Phase(std::f64::consts::FRAC_PI_2))],
    ).unwrap();
    let tn = circuit_to_einsum_with_boundary(&circuit, &[0, 1]);
    let result = contract_tn(&tn);
    assert!(approx_eq(result[[]], Complex64::new(1.0, 0.0), 1e-10));
}
