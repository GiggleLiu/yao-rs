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
    BitFlip {
        p: f64,
    },
    PhaseFlip {
        p: f64,
    },
    Depolarizing {
        n: usize,
        p: f64,
    },
    PauliChannel {
        px: f64,
        py: f64,
        pz: f64,
    },
    Reset {
        p0: f64,
        p1: f64,
    },
    AmplitudeDamping {
        gamma: f64,
        excited_population: f64,
    },
    PhaseDamping {
        gamma: f64,
    },
    PhaseAmplitudeDamping {
        amplitude: f64,
        phase: f64,
        excited_population: f64,
    },
    ThermalRelaxation {
        t1: f64,
        t2: f64,
        time: f64,
        excited_population: f64,
    },
    Coherent {
        matrix: Array2<Complex64>,
    },
    Custom {
        kraus_ops: Vec<Array2<Complex64>>,
    },
}

impl NoiseChannel {
    /// Number of qubits this channel acts on.
    pub fn num_qubits(&self) -> usize {
        match self {
            NoiseChannel::Depolarizing { n, .. } => *n,
            NoiseChannel::Custom { kraus_ops } => {
                let d = kraus_ops[0].nrows();
                assert!(
                    d.is_power_of_two(),
                    "Custom channel dimension {d} is not a power of 2"
                );
                d.ilog2() as usize
            }
            NoiseChannel::Coherent { matrix } => {
                let d = matrix.nrows();
                assert!(
                    d.is_power_of_two(),
                    "Coherent channel dimension {d} is not a power of 2"
                );
                d.ilog2() as usize
            }
            _ => 1, // All other single-qubit channels
        }
    }

    /// Validate parameter ranges for this noise channel.
    ///
    /// Returns an error before constructing invalid matrices.
    fn validate_params(&self) -> Result<(), String> {
        macro_rules! ensure {
            ($condition:expr, $($message:tt)*) => {
                let valid: bool = $condition;
                if !valid { return Err(format!($($message)*)); }
            };
        }
        match self {
            NoiseChannel::BitFlip { p } | NoiseChannel::PhaseFlip { p } => {
                ensure!(
                    (0.0..=1.0).contains(p),
                    "probability p={p} must be in [0, 1]"
                );
            }
            NoiseChannel::PauliChannel { px, py, pz } => {
                ensure!(
                    *px >= 0.0 && *py >= 0.0 && *pz >= 0.0 && px + py + pz <= 1.0,
                    "Pauli probabilities (px={px}, py={py}, pz={pz}) must be non-negative and sum <= 1"
                );
            }
            NoiseChannel::Depolarizing { n, p } => {
                ensure!(
                    *n < usize::BITS as usize / 4 - 1,
                    "Depolarizing Kraus storage exceeds addressable size"
                );
                ensure!(
                    (0.0..=1.0).contains(p),
                    "probability p={p} must be in [0, 1]"
                );
            }
            NoiseChannel::Reset { p0, p1 } => {
                ensure!(
                    *p0 >= 0.0 && *p1 >= 0.0 && p0 + p1 <= 1.0,
                    "Reset probabilities (p0={p0}, p1={p1}) must be non-negative and sum <= 1"
                );
            }
            NoiseChannel::PhaseAmplitudeDamping {
                amplitude,
                phase,
                excited_population,
            } => {
                ensure!(
                    *amplitude >= 0.0 && *phase >= 0.0 && amplitude + phase <= 1.0,
                    "amplitude={amplitude} and phase={phase} must be non-negative and sum <= 1"
                );
                ensure!(
                    (0.0..=1.0).contains(excited_population),
                    "excited_population={excited_population} must be in [0, 1]"
                );
            }
            NoiseChannel::AmplitudeDamping {
                gamma,
                excited_population,
            } => {
                ensure!(
                    (0.0..=1.0).contains(gamma),
                    "gamma={gamma} must be in [0, 1]"
                );
                ensure!(
                    (0.0..=1.0).contains(excited_population),
                    "excited_population={excited_population} must be in [0, 1]"
                );
            }
            NoiseChannel::PhaseDamping { gamma } => {
                ensure!(
                    (0.0..=1.0).contains(gamma),
                    "gamma={gamma} must be in [0, 1]"
                );
            }
            NoiseChannel::ThermalRelaxation {
                t1,
                t2,
                time,
                excited_population,
            } => {
                ensure!(*t1 > 0.0, "t1={t1} must be positive");
                ensure!(*t2 > 0.0, "t2={t2} must be positive");
                ensure!(
                    *time >= 0.0 && time.is_finite(),
                    "time={time} must be finite and non-negative"
                );
                ensure!(
                    *t2 <= 2.0 * t1,
                    "t2={t2} must be <= 2*t1={} (physics constraint)",
                    2.0 * t1
                );
                ensure!(
                    (0.0..=1.0).contains(excited_population),
                    "excited_population={excited_population} must be in [0, 1]"
                );
            }
            NoiseChannel::Custom { kraus_ops } => {
                ensure!(
                    !kraus_ops.is_empty(),
                    "Custom channel must have at least one Kraus operator"
                );
            }
            NoiseChannel::Coherent { .. } => {}
        }
        Ok(())
    }

    /// Construct Kraus operators for this channel.
    ///
    /// Returns matrices K_i such that E(rho) = sum_i K_i rho K_i^dag.
    ///
    /// Julia ref: errortypes.jl KrausChannel() conversions
    pub fn kraus_operators(&self) -> Vec<Array2<Complex64>> {
        self.validate_params()
            .unwrap_or_else(|error| panic!("{error}"));
        self.kraus_unchecked()
    }

    /// Construct finite, square qubit Kraus operators and verify trace preservation.
    /// The completeness relation is checked to `1e-12` per matrix entry.
    /// This fallible entry point is suitable for stochastic simulation.
    pub fn try_kraus_operators(&self) -> Result<Vec<Array2<Complex64>>, String> {
        self.validate_params()?;
        let operators = self.kraus_unchecked();
        validate_kraus(&operators)?;
        Ok(operators)
    }

    fn kraus_unchecked(&self) -> Vec<Array2<Complex64>> {
        match self {
            NoiseChannel::PhaseAmplitudeDamping {
                amplitude,
                phase,
                excited_population,
            } => phase_amplitude_damping_kraus(*amplitude, *phase, *excited_population),
            NoiseChannel::AmplitudeDamping {
                gamma,
                excited_population,
            } => {
                // Julia ref: errortypes.jl:362
                phase_amplitude_damping_kraus(*gamma, 0.0, *excited_population)
            }
            NoiseChannel::PhaseDamping { gamma } => {
                // Julia ref: errortypes.jl:327
                phase_amplitude_damping_kraus(0.0, *gamma, 0.0)
            }
            NoiseChannel::ThermalRelaxation {
                t1,
                t2,
                time,
                excited_population,
            } => {
                // Match population exp(-t/T1) and coherence exp(-t/T2).
                // The old Julia-derived conversion omitted the survival factor
                // in b and could produce a+b>1. Keep coherence explicitly to
                // avoid cancellation in sqrt(1-a-b) at long times.
                let x1 = time / t1;
                let x2 = time / t2;
                let a = -(-x1).exp_m1();
                let survival = (-x1).exp();
                let b = if survival == 0.0 {
                    0.0
                } else {
                    survival * -(x1 - 2.0 * x2).min(0.0).exp_m1()
                };
                damping_kraus(a, b, *excited_population, (-x2).exp())
            }
            NoiseChannel::BitFlip { p } => {
                // Julia ref: errortypes.jl:38 → MixedUnitaryChannel([I2, X], [1-p, p])
                // K0 = sqrt(1-p)*I, K1 = sqrt(p)*X
                let s0 = (1.0 - p).sqrt();
                let s1 = p.sqrt();
                vec![
                    Array2::from_shape_vec(
                        (2, 2),
                        vec![c(s0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(s0, 0.0)],
                    )
                    .unwrap(),
                    Array2::from_shape_vec(
                        (2, 2),
                        vec![c(0.0, 0.0), c(s1, 0.0), c(s1, 0.0), c(0.0, 0.0)],
                    )
                    .unwrap(),
                ]
            }
            NoiseChannel::PhaseFlip { p } => {
                // Julia ref: errortypes.jl:56 → MixedUnitaryChannel([I2, Z], [1-p, p])
                // K0 = sqrt(1-p)*I, K1 = sqrt(p)*Z
                let s0 = (1.0 - p).sqrt();
                let s1 = p.sqrt();
                vec![
                    Array2::from_shape_vec(
                        (2, 2),
                        vec![c(s0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(s0, 0.0)],
                    )
                    .unwrap(),
                    Array2::from_shape_vec(
                        (2, 2),
                        vec![c(s1, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(-s1, 0.0)],
                    )
                    .unwrap(),
                ]
            }
            NoiseChannel::PauliChannel { px, py, pz } => {
                // Julia ref: errortypes.jl:120
                // K0 = sqrt(1-px-py-pz)*I, K1 = sqrt(px)*X, K2 = sqrt(py)*Y, K3 = sqrt(pz)*Z
                let s0 = (1.0 - (px + py + pz)).sqrt();
                let sx = px.sqrt();
                let sy = py.sqrt();
                let sz = pz.sqrt();
                vec![
                    Array2::from_shape_vec(
                        (2, 2),
                        vec![c(s0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(s0, 0.0)],
                    )
                    .unwrap(),
                    Array2::from_shape_vec(
                        (2, 2),
                        vec![c(0.0, 0.0), c(sx, 0.0), c(sx, 0.0), c(0.0, 0.0)],
                    )
                    .unwrap(),
                    Array2::from_shape_vec(
                        (2, 2),
                        vec![c(0.0, 0.0), c(0.0, -sy), c(0.0, sy), c(0.0, 0.0)],
                    )
                    .unwrap(),
                    Array2::from_shape_vec(
                        (2, 2),
                        vec![c(sz, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(-sz, 0.0)],
                    )
                    .unwrap(),
                ]
            }
            NoiseChannel::Depolarizing { n, p } => {
                // Julia ref: errortypes.jl:82 single-qubit = PauliChannel(p/4, p/4, p/4)
                // Multi-qubit: all n-qubit Pauli products
                if *n == 1 {
                    let q = p / 4.0;
                    NoiseChannel::PauliChannel {
                        px: q,
                        py: q,
                        pz: q,
                    }
                    .kraus_operators()
                } else {
                    depolarizing_multi_qubit_kraus(*n, *p)
                }
            }
            NoiseChannel::Reset { p0, p1 } => {
                // Julia ref: errortypes.jl:168-181
                let s = (1.0 - (p0 + p1)).sqrt();
                let mut ops = vec![
                    // K0 = sqrt(1-p0-p1) * I
                    Array2::from_shape_vec(
                        (2, 2),
                        vec![c(s, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(s, 0.0)],
                    )
                    .unwrap(),
                ];
                if *p0 > 0.0 {
                    let sp0 = p0.sqrt();
                    // sqrt(p0) * P0 = sqrt(p0) * |0><0|
                    ops.push(
                        Array2::from_shape_vec(
                            (2, 2),
                            vec![c(sp0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)],
                        )
                        .unwrap(),
                    );
                    // sqrt(p0) * Pd = sqrt(p0) * |0><1|
                    ops.push(
                        Array2::from_shape_vec(
                            (2, 2),
                            vec![c(0.0, 0.0), c(sp0, 0.0), c(0.0, 0.0), c(0.0, 0.0)],
                        )
                        .unwrap(),
                    );
                }
                if *p1 > 0.0 {
                    let sp1 = p1.sqrt();
                    // sqrt(p1) * P1 = sqrt(p1) * |1><1|
                    ops.push(
                        Array2::from_shape_vec(
                            (2, 2),
                            vec![c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(sp1, 0.0)],
                        )
                        .unwrap(),
                    );
                    // sqrt(p1) * Pu = sqrt(p1) * |1><0|
                    ops.push(
                        Array2::from_shape_vec(
                            (2, 2),
                            vec![c(0.0, 0.0), c(0.0, 0.0), c(sp1, 0.0), c(0.0, 0.0)],
                        )
                        .unwrap(),
                    );
                }
                ops
            }
            NoiseChannel::Coherent { matrix } => {
                // Julia ref: errortypes.jl:19 — single Kraus op = the unitary
                vec![matrix.clone()]
            }
            NoiseChannel::Custom { kraus_ops } => kraus_ops.clone(),
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
        for kraus_op in &kraus {
            let k_conj = kraus_op.mapv(|c| c.conj());
            // kron(conj(K), K)
            for i in 0..d {
                for j in 0..d {
                    for k_row in 0..d {
                        for k_col in 0..d {
                            superop[[i * d + k_row, j * d + k_col]] +=
                                k_conj[[i, j]] * kraus_op[[k_row, k_col]];
                        }
                    }
                }
            }
        }
        superop
    }
}

/// Validate one local CPTP map, without constructing a full-register operator.
pub(crate) fn validate_kraus(operators: &[Array2<Complex64>]) -> Result<(), String> {
    let d = operators
        .first()
        .ok_or("A channel requires at least one Kraus operator")?
        .nrows();
    if !d.is_power_of_two() || operators.iter().any(|k| k.dim() != (d, d)) {
        return Err("Kraus operators must be square with the same power-of-two dimension".into());
    }
    if operators
        .iter()
        .flat_map(|k| k.iter())
        .any(|z| !z.re.is_finite() || !z.im.is_finite())
    {
        return Err("Kraus operators must be finite".into());
    }
    for i in 0..d {
        for j in 0..d {
            let mut gram = Complex64::new(0., 0.);
            for matrix in operators {
                for row in 0..d {
                    gram += matrix[[row, i]].conj() * matrix[[row, j]];
                }
            }
            let error = (gram - Complex64::new(f64::from(i == j), 0.)).norm();
            if !error.is_finite() || error > 1e-12 {
                return Err("Kraus operators must preserve trace (sum K†K = I)".into());
            }
        }
    }
    Ok(())
}

/// Julia ref: errortypes.jl:271-296 KrausChannel(err::PhaseAmplitudeDampingError)
fn phase_amplitude_damping_kraus(a: f64, b: f64, p1: f64) -> Vec<Array2<Complex64>> {
    damping_kraus(a, b, p1, (1.0 - a - b).max(0.0).sqrt())
}

fn damping_kraus(a: f64, b: f64, p1: f64, rest: f64) -> Vec<Array2<Complex64>> {
    let mut ops = Vec::new();

    if p1 < 1.0 {
        // Damping to ground state
        let s = (1.0 - p1).sqrt();
        // A0
        ops.push(
            Array2::from_shape_vec(
                (2, 2),
                vec![c(s, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(s * rest, 0.0)],
            )
            .unwrap(),
        );
        // A1 (if a > 0)
        if a > 0.0 {
            let sa = (a).sqrt() * s;
            ops.push(
                Array2::from_shape_vec(
                    (2, 2),
                    vec![c(0.0, 0.0), c(sa, 0.0), c(0.0, 0.0), c(0.0, 0.0)],
                )
                .unwrap(),
            );
        }
        // A2 (if b > 0)
        if b > 0.0 {
            let sb = (b).sqrt() * s;
            ops.push(
                Array2::from_shape_vec(
                    (2, 2),
                    vec![c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(sb, 0.0)],
                )
                .unwrap(),
            );
        }
    }

    if p1 > 0.0 {
        // Damping to excited state
        let s = p1.sqrt();
        // B0
        ops.push(
            Array2::from_shape_vec(
                (2, 2),
                vec![c(s * rest, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(s, 0.0)],
            )
            .unwrap(),
        );
        // B1 (if a > 0)
        if a > 0.0 {
            let sa = (a).sqrt() * s;
            ops.push(
                Array2::from_shape_vec(
                    (2, 2),
                    vec![c(0.0, 0.0), c(0.0, 0.0), c(sa, 0.0), c(0.0, 0.0)],
                )
                .unwrap(),
            );
        }
        // B2 (if b > 0)
        if b > 0.0 {
            let sb = (b).sqrt() * s;
            ops.push(
                Array2::from_shape_vec(
                    (2, 2),
                    vec![c(sb, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)],
                )
                .unwrap(),
            );
        }
    }

    ops
}

/// Multi-qubit depolarizing Kraus operators.
/// Julia ref: mixed_unitary_channel.jl:144-152
fn depolarizing_multi_qubit_kraus(n: usize, p: f64) -> Vec<Array2<Complex64>> {
    // Pauli matrices
    let eye = Array2::from_shape_vec(
        (2, 2),
        vec![c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0)],
    )
    .unwrap();
    let px = Array2::from_shape_vec(
        (2, 2),
        vec![c(0.0, 0.0), c(1.0, 0.0), c(1.0, 0.0), c(0.0, 0.0)],
    )
    .unwrap();
    let py = Array2::from_shape_vec(
        (2, 2),
        vec![c(0.0, 0.0), c(0.0, -1.0), c(0.0, 1.0), c(0.0, 0.0)],
    )
    .unwrap();
    let pz = Array2::from_shape_vec(
        (2, 2),
        vec![c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(-1.0, 0.0)],
    )
    .unwrap();
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
        let weight = if idx == 0 {
            1.0 - p + p / dim as f64
        } else {
            p / dim as f64
        };
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

#[cfg(test)]
#[path = "unit_tests/noise.rs"]
mod tests;
