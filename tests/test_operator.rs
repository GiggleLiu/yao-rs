use yao_rs::operator::{Op, op_matrix, OperatorString};
use ndarray::array;
use num_complex::Complex64;

#[test]
fn test_op_x_matrix() {
    let mat = op_matrix(&Op::X);
    let expected = array![
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]
    ];
    assert_eq!(mat, expected);
}

#[test]
fn test_op_z_matrix() {
    let mat = op_matrix(&Op::Z);
    let expected = array![
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)]
    ];
    assert_eq!(mat, expected);
}

#[test]
fn test_operator_string_creation() {
    // Z(0) * Z(1)
    let ops = OperatorString::new(vec![(0, Op::Z), (1, Op::Z)]);
    assert_eq!(ops.len(), 2);
}

#[test]
fn test_operator_string_identity() {
    let identity = OperatorString::identity();
    assert_eq!(identity.len(), 0);
}
