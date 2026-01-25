use yao_rs::operator::{Op, op_matrix};
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
