use yao_rs::operator::{Op, op_matrix, OperatorString, OperatorPolynomial};
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

#[test]
fn test_operator_polynomial_single() {
    // 0.5 * Z(0)
    let poly = OperatorPolynomial::single(0, Op::Z, 0.5.into());
    assert_eq!(poly.len(), 1);
}

#[test]
fn test_operator_polynomial_add() {
    // 0.5 * Z(0) + 0.5 * Z(1)
    let p1 = OperatorPolynomial::single(0, Op::Z, 0.5.into());
    let p2 = OperatorPolynomial::single(1, Op::Z, 0.5.into());
    let sum = &p1 + &p2;
    assert_eq!(sum.len(), 2);
}

#[test]
fn test_operator_polynomial_json_roundtrip() {
    let p1 = OperatorPolynomial::single(0, Op::Z, 0.5.into());
    let p2 = OperatorPolynomial::single(1, Op::Z, 0.5.into());
    let poly = &p1 + &p2;

    let json = serde_json::to_string(&poly).unwrap();
    let restored: OperatorPolynomial = serde_json::from_str(&json).unwrap();

    assert_eq!(poly.len(), restored.len());
}
