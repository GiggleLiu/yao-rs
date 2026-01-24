use ndarray::Array1;
use num_complex::Complex64;

#[derive(Debug, Clone)]
pub struct State {
    pub dims: Vec<usize>,
    pub data: Array1<Complex64>,
}

impl State {
    /// Creates |0,0,...,0> state (first basis element = 1, rest = 0)
    pub fn zero_state(dims: &[usize]) -> Self {
        let total: usize = dims.iter().product();
        let mut data = Array1::zeros(total);
        data[0] = Complex64::new(1.0, 0.0);
        State {
            dims: dims.to_vec(),
            data,
        }
    }

    /// Creates |i_0, i_1, ..., i_{n-1}> state.
    /// Index computation uses row-major ordering:
    /// index = levels[0]*d_1*d_2*... + levels[1]*d_2*... + ... + levels[n-1]
    pub fn product_state(dims: &[usize], levels: &[usize]) -> Self {
        assert_eq!(dims.len(), levels.len(), "dims and levels must have the same length");
        for (i, (&level, &dim)) in levels.iter().zip(dims.iter()).enumerate() {
            assert!(level < dim, "level[{}] = {} is out of range for dim = {}", i, level, dim);
        }

        let total: usize = dims.iter().product();
        let mut index = 0usize;
        for (i, &level) in levels.iter().enumerate() {
            let stride: usize = dims[i + 1..].iter().product();
            index += level * stride;
        }

        let mut data = Array1::zeros(total);
        data[index] = Complex64::new(1.0, 0.0);
        State {
            dims: dims.to_vec(),
            data,
        }
    }

    /// L2 norm of the state vector
    pub fn norm(&self) -> f64 {
        self.data.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt()
    }

    /// Length of the data vector
    pub fn total_dim(&self) -> usize {
        self.data.len()
    }
}
