use crate::operator_parser;
use crate::output::OutputConfig;
use crate::state_io;
use anyhow::Result;
use ndarray::Array2;
use num_complex::Complex64;
use std::collections::HashMap;
use yao_rs::{State, op_matrix};

pub fn expect(input: &str, op_str: &str, out: &OutputConfig) -> Result<()> {
    let state = state_io::read_state(input)?;
    let operator = operator_parser::parse_operator(op_str)?;
    let value = compute_expectation(&state, &operator);

    let json_value = serde_json::json!({
        "operator": op_str,
        "expectation_value": {
            "re": value.re,
            "im": value.im,
        },
    });

    let human = if value.im.abs() < 1e-10 {
        format!("<op> = {:.10}", value.re)
    } else {
        format!("<op> = {:.10} + {:.10}i", value.re, value.im)
    };

    out.emit(&human, &json_value)
}

fn compute_expectation(state: &State, operator: &yao_rs::OperatorPolynomial) -> Complex64 {
    let n = state.dims.len();
    let total_dim: usize = state.dims.iter().product();
    let mut result = Complex64::new(0.0, 0.0);

    for (coeff, opstring) in operator.iter() {
        let ops = opstring.ops();
        let mut term_val = Complex64::new(0.0, 0.0);

        let mut site_ops: Vec<Array2<Complex64>> = Vec::with_capacity(n);
        let mut op_map: HashMap<usize, Array2<Complex64>> = HashMap::new();
        for &(site, ref op) in ops {
            op_map.insert(site, op_matrix(op));
        }

        for site in 0..n {
            if let Some(matrix) = op_map.get(&site) {
                site_ops.push(matrix.clone());
            } else {
                site_ops.push(op_matrix(&yao_rs::Op::I));
            }
        }

        for i in 0..total_dim {
            let psi_i_conj = state.data[i].conj();
            if psi_i_conj.norm() < 1e-15 {
                continue;
            }

            let mut i_indices = vec![0usize; n];
            let mut idx = i;
            for site in (0..n).rev() {
                i_indices[site] = idx % state.dims[site];
                idx /= state.dims[site];
            }

            for j in 0..total_dim {
                let psi_j = state.data[j];
                if psi_j.norm() < 1e-15 {
                    continue;
                }

                let mut j_indices = vec![0usize; n];
                let mut jdx = j;
                for site in (0..n).rev() {
                    j_indices[site] = jdx % state.dims[site];
                    jdx /= state.dims[site];
                }

                let mut matrix_elem = Complex64::new(1.0, 0.0);
                for site in 0..n {
                    matrix_elem *= site_ops[site][[i_indices[site], j_indices[site]]];
                }

                term_val += psi_i_conj * matrix_elem * psi_j;
            }
        }

        result += *coeff * term_val;
    }

    result
}

#[cfg(test)]
mod tests {
    use ndarray::array;
    use num_complex::Complex64;
    use yao_rs::{Op, OperatorPolynomial, State};

    use super::compute_expectation;

    #[test]
    fn computes_z_expectation_for_one_state() {
        let state = State::product_state(&[2], &[1]);
        let operator = OperatorPolynomial::single(0, Op::Z, Complex64::new(1.0, 0.0));

        let value = compute_expectation(&state, &operator);

        assert!((value - Complex64::new(-1.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn computes_x_expectation_for_plus_state() {
        let amplitude = 1.0 / 2.0_f64.sqrt();
        let state = State::new(
            vec![2],
            array![
                Complex64::new(amplitude, 0.0),
                Complex64::new(amplitude, 0.0)
            ],
        );
        let operator = OperatorPolynomial::single(0, Op::X, Complex64::new(1.0, 0.0));

        let value = compute_expectation(&state, &operator);

        assert!((value - Complex64::new(1.0, 0.0)).norm() < 1e-12);
    }
}
