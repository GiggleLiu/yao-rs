use ndarray::Array2;
use num_complex::Complex64;

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

/// Quantum noise channel types.
///
/// Each variant represents a different physical noise process.
/// All channels can produce their Kraus operator representation
/// and superoperator matrix for tensor network export.
///
/// Julia ref: `~/.julia/dev/Yao/lib/YaoBlocks/src/channel/errortypes.jl`
#[derive(Debug, Clone)]
pub enum NoiseChannel {
    BitFlip { p: f64 },
    PhaseFlip { p: f64 },
    Depolarizing { n: usize, p: f64 },
    PauliChannel { px: f64, py: f64, pz: f64 },
    Reset { p0: f64, p1: f64 },
    AmplitudeDamping { gamma: f64, excited_population: f64 },
    PhaseDamping { gamma: f64 },
    PhaseAmplitudeDamping { amplitude: f64, phase: f64, excited_population: f64 },
    ThermalRelaxation { t1: f64, t2: f64, time: f64, excited_population: f64 },
    Coherent { matrix: Array2<Complex64> },
    Custom { kraus_ops: Vec<Array2<Complex64>> },
}

impl NoiseChannel {
    /// Number of qubits this channel acts on.
    pub fn num_qubits(&self) -> usize {
        match self {
            NoiseChannel::Depolarizing { n, .. } => *n,
            NoiseChannel::Custom { kraus_ops } => {
                let d = kraus_ops[0].nrows();
                (d as f64).log2() as usize
            }
            NoiseChannel::Coherent { matrix } => {
                let d = matrix.nrows();
                (d as f64).log2() as usize
            }
            _ => 1, // All other single-qubit channels
        }
    }

    /// Construct Kraus operators for this channel.
    ///
    /// Returns matrices K_i such that E(rho) = sum_i K_i rho K_i^dag.
    ///
    /// Julia ref: errortypes.jl KrausChannel() conversions
    pub fn kraus_operators(&self) -> Vec<Array2<Complex64>> {
        match self {
            NoiseChannel::PhaseAmplitudeDamping { amplitude, phase, excited_population } => {
                phase_amplitude_damping_kraus(*amplitude, *phase, *excited_population)
            }
            NoiseChannel::AmplitudeDamping { gamma, excited_population } => {
                // Julia ref: errortypes.jl:362
                phase_amplitude_damping_kraus(*gamma, 0.0, *excited_population)
            }
            NoiseChannel::PhaseDamping { gamma } => {
                // Julia ref: errortypes.jl:327
                phase_amplitude_damping_kraus(0.0, *gamma, 0.0)
            }
            NoiseChannel::ThermalRelaxation { t1, t2, time, excited_population } => {
                // Julia ref: errortypes.jl:223-229
                let t_phi = (t1 * t2) / (2.0 * t1 - t2);
                let a = 1.0 - (-time / t1).exp();
                let b = 1.0 - (-time / t_phi).exp();
                phase_amplitude_damping_kraus(a, b, *excited_population)
            }
            NoiseChannel::BitFlip { p } => {
                // Julia ref: errortypes.jl:38 → MixedUnitaryChannel([I2, X], [1-p, p])
                // K0 = sqrt(1-p)*I, K1 = sqrt(p)*X
                let s0 = (1.0 - p).sqrt();
                let s1 = p.sqrt();
                vec![
                    Array2::from_shape_vec((2, 2), vec![c(s0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(s0, 0.0)]).unwrap(),
                    Array2::from_shape_vec((2, 2), vec![c(0.0, 0.0), c(s1, 0.0), c(s1, 0.0), c(0.0, 0.0)]).unwrap(),
                ]
            }
            NoiseChannel::PhaseFlip { p } => {
                // Julia ref: errortypes.jl:56 → MixedUnitaryChannel([I2, Z], [1-p, p])
                // K0 = sqrt(1-p)*I, K1 = sqrt(p)*Z
                let s0 = (1.0 - p).sqrt();
                let s1 = p.sqrt();
                vec![
                    Array2::from_shape_vec((2, 2), vec![c(s0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(s0, 0.0)]).unwrap(),
                    Array2::from_shape_vec((2, 2), vec![c(s1, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(-s1, 0.0)]).unwrap(),
                ]
            }
            NoiseChannel::PauliChannel { px, py, pz } => {
                // Julia ref: errortypes.jl:120
                // K0 = sqrt(1-px-py-pz)*I, K1 = sqrt(px)*X, K2 = sqrt(py)*Y, K3 = sqrt(pz)*Z
                let s0 = (1.0 - px - py - pz).sqrt();
                let sx = px.sqrt();
                let sy = py.sqrt();
                let sz = pz.sqrt();
                vec![
                    Array2::from_shape_vec((2, 2), vec![c(s0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(s0, 0.0)]).unwrap(),
                    Array2::from_shape_vec((2, 2), vec![c(0.0, 0.0), c(sx, 0.0), c(sx, 0.0), c(0.0, 0.0)]).unwrap(),
                    Array2::from_shape_vec((2, 2), vec![c(0.0, 0.0), c(0.0, -sy), c(0.0, sy), c(0.0, 0.0)]).unwrap(),
                    Array2::from_shape_vec((2, 2), vec![c(sz, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(-sz, 0.0)]).unwrap(),
                ]
            }
            NoiseChannel::Depolarizing { n, p } => {
                // Julia ref: errortypes.jl:82 single-qubit = PauliChannel(p/4, p/4, p/4)
                // Multi-qubit: all n-qubit Pauli products
                if *n == 1 {
                    let q = p / 4.0;
                    NoiseChannel::PauliChannel { px: q, py: q, pz: q }.kraus_operators()
                } else {
                    depolarizing_multi_qubit_kraus(*n, *p)
                }
            }
            NoiseChannel::Reset { p0, p1 } => {
                // Julia ref: errortypes.jl:168-181
                let s = (1.0 - p0 - p1).sqrt();
                let mut ops = vec![
                    // K0 = sqrt(1-p0-p1) * I
                    Array2::from_shape_vec((2, 2), vec![c(s, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(s, 0.0)]).unwrap(),
                ];
                if *p0 > 0.0 {
                    let sp0 = p0.sqrt();
                    // sqrt(p0) * P0 = sqrt(p0) * |0><0|
                    ops.push(Array2::from_shape_vec((2, 2), vec![c(sp0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)]).unwrap());
                    // sqrt(p0) * Pd = sqrt(p0) * |0><1|
                    ops.push(Array2::from_shape_vec((2, 2), vec![c(0.0, 0.0), c(sp0, 0.0), c(0.0, 0.0), c(0.0, 0.0)]).unwrap());
                }
                if *p1 > 0.0 {
                    let sp1 = p1.sqrt();
                    // sqrt(p1) * P1 = sqrt(p1) * |1><1|
                    ops.push(Array2::from_shape_vec((2, 2), vec![c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(sp1, 0.0)]).unwrap());
                    // sqrt(p1) * Pu = sqrt(p1) * |1><0|
                    ops.push(Array2::from_shape_vec((2, 2), vec![c(0.0, 0.0), c(0.0, 0.0), c(sp1, 0.0), c(0.0, 0.0)]).unwrap());
                }
                ops
            }
            NoiseChannel::Coherent { matrix } => {
                // Julia ref: errortypes.jl:19 — single Kraus op = the unitary
                vec![matrix.clone()]
            }
            NoiseChannel::Custom { kraus_ops } => {
                kraus_ops.clone()
            }
        }
    }

    /// Build the superoperator matrix for this channel.
    ///
    /// S = sum_i kron(conj(K_i), K_i)
    ///
    /// Julia ref: kraus.jl:73-78
    pub fn superop(&self) -> Array2<Complex64> {
        let kraus = self.kraus_operators();
        let d = kraus[0].nrows();
        let d2 = d * d;
        let mut superop = Array2::<Complex64>::zeros((d2, d2));
        for k in &kraus {
            let k_conj = k.mapv(|c| c.conj());
            // kron(conj(K), K)
            for i in 0..d {
                for j in 0..d {
                    for k_row in 0..d {
                        for k_col in 0..d {
                            superop[[i * d + k_row, j * d + k_col]] += k_conj[[i, j]] * k[[k_row, k_col]];
                        }
                    }
                }
            }
        }
        superop
    }
}

/// Julia ref: errortypes.jl:271-296 KrausChannel(err::PhaseAmplitudeDampingError)
fn phase_amplitude_damping_kraus(a: f64, b: f64, p1: f64) -> Vec<Array2<Complex64>> {
    let mut ops = Vec::new();
    let rest = (1.0 - a - b).sqrt();

    if (p1 - 1.0).abs() > f64::EPSILON {
        // Damping to ground state
        let s = (1.0 - p1).sqrt();
        // A0
        ops.push(Array2::from_shape_vec((2, 2), vec![c(s, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(s * rest, 0.0)]).unwrap());
        // A1 (if a > 0)
        if a > 0.0 {
            let sa = (a).sqrt() * s;
            ops.push(Array2::from_shape_vec((2, 2), vec![c(0.0, 0.0), c(sa, 0.0), c(0.0, 0.0), c(0.0, 0.0)]).unwrap());
        }
        // A2 (if b > 0)
        if b > 0.0 {
            let sb = (b).sqrt() * s;
            ops.push(Array2::from_shape_vec((2, 2), vec![c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(sb, 0.0)]).unwrap());
        }
    }

    if p1.abs() > f64::EPSILON {
        // Damping to excited state
        let s = p1.sqrt();
        // B0
        ops.push(Array2::from_shape_vec((2, 2), vec![c(s * rest, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(s, 0.0)]).unwrap());
        // B1 (if a > 0)
        if a > 0.0 {
            let sa = (a).sqrt() * s;
            ops.push(Array2::from_shape_vec((2, 2), vec![c(0.0, 0.0), c(0.0, 0.0), c(sa, 0.0), c(0.0, 0.0)]).unwrap());
        }
        // B2 (if b > 0)
        if b > 0.0 {
            let sb = (b).sqrt() * s;
            ops.push(Array2::from_shape_vec((2, 2), vec![c(sb, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)]).unwrap());
        }
    }

    ops
}

/// Multi-qubit depolarizing Kraus operators.
/// Julia ref: mixed_unitary_channel.jl:144-152
fn depolarizing_multi_qubit_kraus(n: usize, p: f64) -> Vec<Array2<Complex64>> {
    // Pauli matrices
    let eye = Array2::from_shape_vec((2, 2), vec![c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0)]).unwrap();
    let px = Array2::from_shape_vec((2, 2), vec![c(0.0, 0.0), c(1.0, 0.0), c(1.0, 0.0), c(0.0, 0.0)]).unwrap();
    let py = Array2::from_shape_vec((2, 2), vec![c(0.0, 0.0), c(0.0, -1.0), c(0.0, 1.0), c(0.0, 0.0)]).unwrap();
    let pz = Array2::from_shape_vec((2, 2), vec![c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(-1.0, 0.0)]).unwrap();
    let paulis = [eye, px, py, pz];

    let dim = 1usize << (2 * n); // 4^n
    let mut ops = Vec::with_capacity(dim);

    // Generate all n-qubit Pauli products
    for idx in 0..dim {
        let mut mat = Array2::from_shape_vec((1, 1), vec![c(1.0, 0.0)]).unwrap();
        let mut tmp = idx;
        for _ in 0..n {
            let pauli_idx = tmp % 4;
            tmp /= 4;
            mat = kron(&mat, &paulis[pauli_idx]);
        }
        // Weight: p/4^n for all, identity gets extra 1-p
        let weight = if idx == 0 { 1.0 - p + p / dim as f64 } else { p / dim as f64 };
        let s = weight.sqrt();
        ops.push(mat.mapv(|v| v * c(s, 0.0)));
    }
    ops
}

/// Kronecker product of two matrices.
fn kron(a: &Array2<Complex64>, b: &Array2<Complex64>) -> Array2<Complex64> {
    let (ar, ac) = (a.nrows(), a.ncols());
    let (br, bc) = (b.nrows(), b.ncols());
    let mut result = Array2::zeros((ar * br, ac * bc));
    for i in 0..ar {
        for j in 0..ac {
            for k in 0..br {
                for l in 0..bc {
                    result[[i * br + k, j * bc + l]] = a[[i, j]] * b[[k, l]];
                }
            }
        }
    }
    result
}
