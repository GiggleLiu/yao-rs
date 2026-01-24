use ndarray::{Array2, IxDyn};
use num_complex::Complex64;
use yao_rs::circuit::PositionedGate;
use yao_rs::gate::Gate;
use yao_rs::tensors::{gate_to_tensor, Leg};

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

fn approx_eq(a: Complex64, b: Complex64) -> bool {
    (a - b).norm() < 1e-10
}

#[test]
fn test_single_qubit_x_gate() {
    // X gate is non-diagonal, shape should be (2, 2) with legs [Out(0), In(0)]
    let pg = PositionedGate::new(Gate::X, vec![0], vec![], vec![]);
    let dims = vec![2];

    let (tensor, legs) = gate_to_tensor(&pg, &dims);

    assert_eq!(tensor.shape(), &[2, 2]);
    assert_eq!(legs, vec![Leg::Out(0), Leg::In(0)]);

    // X = [[0, 1], [1, 0]]
    // tensor[out, in] = M[out, in]
    assert!(approx_eq(tensor[[0, 0].as_ref()], c(0.0, 0.0)));
    assert!(approx_eq(tensor[[0, 1].as_ref()], c(1.0, 0.0)));
    assert!(approx_eq(tensor[[1, 0].as_ref()], c(1.0, 0.0)));
    assert!(approx_eq(tensor[[1, 1].as_ref()], c(0.0, 0.0)));
}

#[test]
fn test_single_qubit_z_gate_diagonal() {
    // Z gate is diagonal, shape should be (2,) with legs [Diag(0)]
    let pg = PositionedGate::new(Gate::Z, vec![0], vec![], vec![]);
    let dims = vec![2];

    let (tensor, legs) = gate_to_tensor(&pg, &dims);

    assert_eq!(tensor.shape(), &[2]);
    assert_eq!(legs, vec![Leg::Diag(0)]);

    // Z diagonal: [1, -1]
    assert!(approx_eq(tensor[[0].as_ref()], c(1.0, 0.0)));
    assert!(approx_eq(tensor[[1].as_ref()], c(-1.0, 0.0)));
}

#[test]
fn test_cnot_gate() {
    // CNOT = control on site 0, X on site 1
    // Shape: (2, 2, 2, 2) with legs [Out(0), Out(1), In(0), In(1)]
    let pg = PositionedGate::new(
        Gate::X,
        vec![1],
        vec![0],
        vec![true], // trigger when control is |1>
    );
    let dims = vec![2, 2];

    let (tensor, legs) = gate_to_tensor(&pg, &dims);

    assert_eq!(tensor.shape(), &[2, 2, 2, 2]);
    assert_eq!(
        legs,
        vec![Leg::Out(0), Leg::Out(1), Leg::In(0), Leg::In(1)]
    );

    // CNOT matrix (in |ctrl, tgt> basis):
    // |00> -> |00>  : M[0,0] = 1
    // |01> -> |01>  : M[1,1] = 1
    // |10> -> |11>  : M[2,3] = 1... wait, let me reconsider
    // Actually the trigger is ctrl=1, so:
    // ctrl=0: identity on target => M[0,0]=1, M[1,1]=1
    // ctrl=1: X on target => M[2,3]=1, M[3,2]=1
    //
    // tensor[out_ctrl, out_tgt, in_ctrl, in_tgt] = M[out_ctrl*2+out_tgt, in_ctrl*2+in_tgt]

    // |00> -> |00>: tensor[0,0,0,0] = 1
    assert!(approx_eq(tensor[IxDyn(&[0, 0, 0, 0])], c(1.0, 0.0)));
    // |01> -> |01>: tensor[0,1,0,1] = 1
    assert!(approx_eq(tensor[IxDyn(&[0, 1, 0, 1])], c(1.0, 0.0)));
    // |10> -> |11>: tensor[1,1,1,0] = 1
    assert!(approx_eq(tensor[IxDyn(&[1, 1, 1, 0])], c(1.0, 0.0)));
    // |11> -> |10>: tensor[1,0,1,1] = 1
    assert!(approx_eq(tensor[IxDyn(&[1, 0, 1, 1])], c(1.0, 0.0)));

    // Some zero entries
    assert!(approx_eq(tensor[IxDyn(&[0, 0, 1, 0])], c(0.0, 0.0)));
    assert!(approx_eq(tensor[IxDyn(&[1, 0, 0, 0])], c(0.0, 0.0)));
}

#[test]
fn test_cz_gate() {
    // CZ = control on site 0, Z on site 1
    // Even though Z is diagonal, with controls it becomes non-diagonal
    // Shape: (2, 2, 2, 2)
    let pg = PositionedGate::new(
        Gate::Z,
        vec![1],
        vec![0],
        vec![true], // trigger when control is |1>
    );
    let dims = vec![2, 2];

    let (tensor, legs) = gate_to_tensor(&pg, &dims);

    assert_eq!(tensor.shape(), &[2, 2, 2, 2]);
    assert_eq!(
        legs,
        vec![Leg::Out(0), Leg::Out(1), Leg::In(0), Leg::In(1)]
    );

    // CZ matrix:
    // ctrl=0: identity => M[0,0]=1, M[1,1]=1
    // ctrl=1: Z => M[2,2]=1, M[3,3]=-1
    //
    // tensor[0,0,0,0] = M[0,0] = 1
    assert!(approx_eq(tensor[IxDyn(&[0, 0, 0, 0])], c(1.0, 0.0)));
    // tensor[0,1,0,1] = M[1,1] = 1
    assert!(approx_eq(tensor[IxDyn(&[0, 1, 0, 1])], c(1.0, 0.0)));
    // tensor[1,0,1,0] = M[2,2] = 1
    assert!(approx_eq(tensor[IxDyn(&[1, 0, 1, 0])], c(1.0, 0.0)));
    // tensor[1,1,1,1] = M[3,3] = -1
    assert!(approx_eq(tensor[IxDyn(&[1, 1, 1, 1])], c(-1.0, 0.0)));

    // Off-diagonal should be zero
    assert!(approx_eq(tensor[IxDyn(&[0, 0, 0, 1])], c(0.0, 0.0)));
    assert!(approx_eq(tensor[IxDyn(&[1, 0, 1, 1])], c(0.0, 0.0)));
}

#[test]
fn test_swap_gate() {
    // SWAP acts on 2 sites, non-diagonal
    // Shape: (2, 2, 2, 2) with legs [Out(0), Out(1), In(0), In(1)]
    let pg = PositionedGate::new(Gate::SWAP, vec![0, 1], vec![], vec![]);
    let dims = vec![2, 2];

    let (tensor, legs) = gate_to_tensor(&pg, &dims);

    assert_eq!(tensor.shape(), &[2, 2, 2, 2]);
    assert_eq!(
        legs,
        vec![Leg::Out(0), Leg::Out(1), Leg::In(0), Leg::In(1)]
    );

    // SWAP matrix: |00>->|00>, |01>->|10>, |10>->|01>, |11>->|11>
    // tensor[out0, out1, in0, in1] = M[out0*2+out1, in0*2+in1]

    // |00> -> |00>: tensor[0,0,0,0] = 1
    assert!(approx_eq(tensor[IxDyn(&[0, 0, 0, 0])], c(1.0, 0.0)));
    // |01> -> |10>: tensor[1,0,0,1] = 1
    assert!(approx_eq(tensor[IxDyn(&[1, 0, 0, 1])], c(1.0, 0.0)));
    // |10> -> |01>: tensor[0,1,1,0] = 1
    assert!(approx_eq(tensor[IxDyn(&[0, 1, 1, 0])], c(1.0, 0.0)));
    // |11> -> |11>: tensor[1,1,1,1] = 1
    assert!(approx_eq(tensor[IxDyn(&[1, 1, 1, 1])], c(1.0, 0.0)));

    // Non-SWAP entries should be zero
    assert!(approx_eq(tensor[IxDyn(&[0, 1, 0, 1])], c(0.0, 0.0)));
    assert!(approx_eq(tensor[IxDyn(&[1, 0, 1, 0])], c(0.0, 0.0)));
}

#[test]
fn test_custom_diagonal_qutrit() {
    // Custom diagonal gate on a qutrit (d=3)
    // Shape: (3,) with legs [Diag(0)]
    let diag_vals = [c(1.0, 0.0), c(0.0, 1.0), c(-1.0, 0.0)];
    let mut mat = Array2::zeros((3, 3));
    for i in 0..3 {
        mat[[i, i]] = diag_vals[i];
    }

    let gate = Gate::Custom {
        matrix: mat,
        is_diagonal: true,
        label: "qutrit_diagonal".to_string(),
    };
    let pg = PositionedGate::new(gate, vec![0], vec![], vec![]);
    let dims = vec![3];

    let (tensor, legs) = gate_to_tensor(&pg, &dims);

    assert_eq!(tensor.shape(), &[3]);
    assert_eq!(legs, vec![Leg::Diag(0)]);

    assert!(approx_eq(tensor[[0].as_ref()], c(1.0, 0.0)));
    assert!(approx_eq(tensor[[1].as_ref()], c(0.0, 1.0)));
    assert!(approx_eq(tensor[[2].as_ref()], c(-1.0, 0.0)));
}

#[test]
fn test_h_gate_non_diagonal() {
    // H gate is non-diagonal, shape (2, 2)
    let pg = PositionedGate::new(Gate::H, vec![0], vec![], vec![]);
    let dims = vec![2];

    let (tensor, legs) = gate_to_tensor(&pg, &dims);

    assert_eq!(tensor.shape(), &[2, 2]);
    assert_eq!(legs, vec![Leg::Out(0), Leg::In(0)]);

    let s = 1.0 / 2.0_f64.sqrt();
    // H = 1/sqrt(2) * [[1, 1], [1, -1]]
    assert!(approx_eq(tensor[[0, 0].as_ref()], c(s, 0.0)));
    assert!(approx_eq(tensor[[0, 1].as_ref()], c(s, 0.0)));
    assert!(approx_eq(tensor[[1, 0].as_ref()], c(s, 0.0)));
    assert!(approx_eq(tensor[[1, 1].as_ref()], c(-s, 0.0)));
}

#[test]
fn test_s_gate_diagonal() {
    // S gate is diagonal, shape (2,)
    let pg = PositionedGate::new(Gate::S, vec![0], vec![], vec![]);
    let dims = vec![2];

    let (tensor, legs) = gate_to_tensor(&pg, &dims);

    assert_eq!(tensor.shape(), &[2]);
    assert_eq!(legs, vec![Leg::Diag(0)]);

    // S diagonal: [1, i]
    assert!(approx_eq(tensor[[0].as_ref()], c(1.0, 0.0)));
    assert!(approx_eq(tensor[[1].as_ref()], c(0.0, 1.0)));
}

#[test]
fn test_t_gate_diagonal() {
    // T gate is diagonal, shape (2,)
    let pg = PositionedGate::new(Gate::T, vec![0], vec![], vec![]);
    let dims = vec![2];

    let (tensor, legs) = gate_to_tensor(&pg, &dims);

    assert_eq!(tensor.shape(), &[2]);
    assert_eq!(legs, vec![Leg::Diag(0)]);

    // T diagonal: [1, e^(i*pi/4)]
    let t_phase = Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_4);
    assert!(approx_eq(tensor[[0].as_ref()], c(1.0, 0.0)));
    assert!(approx_eq(tensor[[1].as_ref()], t_phase));
}

#[test]
fn test_rz_gate_diagonal() {
    // Rz gate is diagonal, shape (2,)
    let theta = 1.5;
    let pg = PositionedGate::new(Gate::Rz(theta), vec![0], vec![], vec![]);
    let dims = vec![2];

    let (tensor, legs) = gate_to_tensor(&pg, &dims);

    assert_eq!(tensor.shape(), &[2]);
    assert_eq!(legs, vec![Leg::Diag(0)]);

    let phase_neg = Complex64::from_polar(1.0, -theta / 2.0);
    let phase_pos = Complex64::from_polar(1.0, theta / 2.0);
    assert!(approx_eq(tensor[[0].as_ref()], phase_neg));
    assert!(approx_eq(tensor[[1].as_ref()], phase_pos));
}

#[test]
fn test_controlled_gate_trigger_on_zero() {
    // Control with trigger on |0> (control_configs = [false])
    // This means the gate is applied when control is 0
    let pg = PositionedGate::new(
        Gate::X,
        vec![1],
        vec![0],
        vec![false], // trigger when control is |0>
    );
    let dims = vec![2, 2];

    let (tensor, legs) = gate_to_tensor(&pg, &dims);

    assert_eq!(tensor.shape(), &[2, 2, 2, 2]);
    assert_eq!(
        legs,
        vec![Leg::Out(0), Leg::Out(1), Leg::In(0), Leg::In(1)]
    );

    // ctrl=0 (trigger): X on target => |00>->|01>, |01>->|00>
    // ctrl=1: identity => |10>->|10>, |11>->|11>

    // tensor[0,1,0,0] = M[1,0] for ctrl=0 block = X[1,0] = 1
    assert!(approx_eq(tensor[IxDyn(&[0, 1, 0, 0])], c(1.0, 0.0)));
    // tensor[0,0,0,1] = M[0,1] for ctrl=0 block = X[0,1] = 1
    assert!(approx_eq(tensor[IxDyn(&[0, 0, 0, 1])], c(1.0, 0.0)));
    // tensor[1,0,1,0] = identity
    assert!(approx_eq(tensor[IxDyn(&[1, 0, 1, 0])], c(1.0, 0.0)));
    // tensor[1,1,1,1] = identity
    assert!(approx_eq(tensor[IxDyn(&[1, 1, 1, 1])], c(1.0, 0.0)));
}

#[test]
fn test_gate_on_non_first_site() {
    // Gate on site 2 in a 3-site system
    let pg = PositionedGate::new(Gate::X, vec![2], vec![], vec![]);
    let dims = vec![2, 2, 2];

    let (tensor, legs) = gate_to_tensor(&pg, &dims);

    // Should still be (2, 2) since it only acts on one site
    assert_eq!(tensor.shape(), &[2, 2]);
    assert_eq!(legs, vec![Leg::Out(0), Leg::In(0)]);
}

#[test]
fn test_custom_non_diagonal_qubit() {
    // Custom non-diagonal 2x2 gate
    let mat = Array2::from_shape_vec(
        (2, 2),
        vec![c(0.5, 0.5), c(0.5, -0.5), c(0.5, -0.5), c(0.5, 0.5)],
    )
    .unwrap();
    let gate = Gate::Custom {
        matrix: mat.clone(),
        is_diagonal: false,
        label: "custom_2x2".to_string(),
    };
    let pg = PositionedGate::new(gate, vec![0], vec![], vec![]);
    let dims = vec![2];

    let (tensor, legs) = gate_to_tensor(&pg, &dims);

    assert_eq!(tensor.shape(), &[2, 2]);
    assert_eq!(legs, vec![Leg::Out(0), Leg::In(0)]);

    assert!(approx_eq(tensor[[0, 0].as_ref()], c(0.5, 0.5)));
    assert!(approx_eq(tensor[[0, 1].as_ref()], c(0.5, -0.5)));
    assert!(approx_eq(tensor[[1, 0].as_ref()], c(0.5, -0.5)));
    assert!(approx_eq(tensor[[1, 1].as_ref()], c(0.5, 0.5)));
}
