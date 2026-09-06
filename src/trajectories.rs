//! Seeded Monte Carlo simulation of qubit circuits with trace-preserving noise.
//!
//! A channel samples Kraus branch `k` with probability `||K_k psi||²`, then
//! normalizes that branch. Ensembles estimate a mixed-state expectation; an
//! individual returned state is only one realization. No derivative through
//! discrete branch selection is provided.
//!
//! The CPU implementation reuses native gate kernels and Rand's ChaCha8 streams.
//! Stream IDs are trajectory numbers, independent of worker scheduling. Ordered
//! reduction makes statistics reproducible across thread counts on the same
//! version/platform. Memory is O(threads × state size), independent of sample
//! count, plus the prepared local Kraus operators.

use crate::apply::{dispatch_arrayreg_gate, validate_unitary_gate};
use crate::{ArrayReg, Circuit, CircuitElement, OperatorPolynomial, PositionedGate};
use num_complex::Complex64 as C;
use rand::{
    RngExt, SeedableRng,
    distr::{Distribution, weighted::WeightedIndex},
    rngs::ChaCha8Rng,
};
use serde::Serialize;

/// Ensemble size, master seed and bounded worker concurrency.
#[derive(Debug, Clone, Copy)]
pub struct TrajectoryOptions {
    /// Number of independent trajectories, at least one.
    pub trajectories: usize,
    /// Master seed. Each trajectory uses its own ChaCha8 stream.
    pub seed: u64,
    /// Active workers. Values above one require the `parallel` feature.
    pub threads: usize,
}
impl Default for TrajectoryOptions {
    fn default() -> Self {
        Self {
            trajectories: 1024,
            seed: 0,
            threads: 1,
        }
    }
}

/// Streaming statistics of complex expectation values across trajectories.
///
/// Variance/standard-error components describe real and imaginary parts, not
/// complex squares. Covariance is the sample covariance between those parts.
/// All three uncertainty fields are unavailable with only one trajectory.
/// Standard errors measure Monte Carlo sampling uncertainty, not roundoff or
/// model error; they are estimates rather than confidence guarantees.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TrajectoryStatistics {
    pub trajectories: usize,
    pub seed: u64,
    pub threads: usize,
    pub mean: C,
    pub sample_variance: Option<C>,
    pub standard_error: Option<C>,
    pub covariance: Option<f64>,
}

#[derive(Debug, Clone)]
enum Step {
    Gate(PositionedGate),
    Depolarizing {
        locs: Vec<usize>,
        p: f64,
    },
    Channel {
        locs: Vec<usize>,
        matrices: Vec<Vec<C>>,
    },
}

/// Immutable validated circuit with local Kraus matrices prepared once.
#[derive(Debug, Clone)]
pub struct TrajectoryCircuit {
    nqubits: usize,
    steps: Vec<Step>,
}

fn validate_locs(n: usize, locs: &[usize]) -> Result<(), String> {
    let mut seen = std::collections::HashSet::new();
    if locs.iter().any(|&i| i >= n || !seen.insert(i)) {
        return Err("Trajectory targets must be distinct, in-range qubits".into());
    }
    Ok(())
}

impl TrajectoryCircuit {
    /// Validate unitary gates and finite CPTP channels before any sampling.
    /// Input circuits are revalidated because their fields are public.
    pub fn new(circuit: Circuit) -> Result<Self, String> {
        let n = circuit.nbits;
        if circuit.dims.len() != n
            || circuit.dims.iter().any(|&d| d != 2)
            || n >= usize::BITS as usize
            || (1usize << n)
                .checked_mul(size_of::<C>())
                .is_none_or(|b| b > isize::MAX as usize)
        {
            return Err("Trajectories require addressable qubit states".into());
        }
        let mut steps = Vec::new();
        for element in circuit.elements {
            match element {
                CircuitElement::Gate(pg) => {
                    validate_unitary_gate(n, &pg)?;
                    steps.push(Step::Gate(pg));
                }
                CircuitElement::Channel(pc) => {
                    validate_locs(n, &pc.locs)?;
                    // E(rho)=(1-p)rho+p I/2^n: sample a uniform Pauli word
                    // with probability p, using O(n) storage rather than 16^n
                    // Kraus matrix entries. This is the same channel/unraveling
                    // up to the duplicated identity choice.
                    if let crate::NoiseChannel::Depolarizing { n, p } = pc.channel {
                        if n != pc.locs.len() || !(0.0..=1.0).contains(&p) {
                            return Err("Invalid depolarizing probability or target count".into());
                        }
                        steps.push(Step::Depolarizing { locs: pc.locs, p });
                        continue;
                    }
                    let operators = pc.channel.try_kraus_operators()?;
                    if operators[0].nrows() != 1usize << pc.locs.len() {
                        return Err("Channel dimension does not match its targets".into());
                    }
                    steps.push(Step::Channel {
                        locs: pc.locs,
                        matrices: operators
                            .into_iter()
                            .map(|k| k.iter().copied().collect())
                            .collect(),
                    });
                }
                CircuitElement::Annotation(a) => {
                    validate_locs(n, &[a.loc])?;
                }
            }
        }
        Ok(Self { nqubits: n, steps })
    }

    pub fn nqubits(&self) -> usize {
        self.nqubits
    }

    fn validate_input(&self, input: &ArrayReg) -> Result<(), String> {
        if input.nqubits() != self.nqubits || input.state.len() != 1usize << self.nqubits {
            return Err("Trajectory input dimension mismatch".into());
        }
        let norm: f64 = input.state.iter().map(|z| z.norm_sqr()).sum();
        if !norm.is_finite() || (norm - 1.).abs() > 1e-10 {
            return Err("Trajectory input must be finite and normalized".into());
        }
        Ok(())
    }

    /// Draw one normalized realization, identified by seed and trajectory ID.
    /// The input is unchanged. Use ensemble statistics for a mixed-state result.
    pub fn sample(
        &self,
        input: &ArrayReg,
        seed: u64,
        trajectory_id: u64,
    ) -> Result<ArrayReg, String> {
        self.validate_input(input)?;
        let mut worker = Worker::new(input);
        self.execute(input, &mut worker, seed, trajectory_id)?;
        Ok(worker.state)
    }

    /// Estimate a polynomial expectation with bounded memory and ordered moments.
    /// Includes neither measurement shots nor stochastic differentiation.
    pub fn expectation(
        &self,
        input: &ArrayReg,
        operator: &OperatorPolynomial,
        options: TrajectoryOptions,
    ) -> Result<TrajectoryStatistics, String> {
        self.validate_input(input)?;
        if options.trajectories == 0
            || options.trajectories as u128 > 1u128 << 53
            || options.threads == 0
        {
            return Err("Require 1..=2^53 trajectories and at least one thread".into());
        }
        if operator.coeffs().len() != operator.opstrings().len()
            || operator
                .coeffs()
                .iter()
                .any(|c| !c.re.is_finite() || !c.im.is_finite())
        {
            return Err("Observable coefficients must be finite and match the term count".into());
        }
        for word in operator.opstrings() {
            validate_locs(
                self.nqubits,
                &word.ops().iter().map(|&(i, _)| i).collect::<Vec<_>>(),
            )?;
        }
        let mut moments = Moments::default();
        if options.threads == 1 {
            let mut worker = Worker::new(input);
            for id in 0..options.trajectories {
                self.execute(input, &mut worker, options.seed, id as u64)?;
                moments.push(crate::expect::expect_arrayreg_with_scratch(
                    &worker.state,
                    operator,
                    &mut worker.scratch,
                ))?;
            }
        } else {
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                let concurrency = options.threads.min(options.trajectories);
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(concurrency)
                    .build()
                    .map_err(|e| e.to_string())?;
                let mut workers: Vec<_> = (0..concurrency).map(|_| Worker::new(input)).collect();
                // Only one scalar result per worker is retained; consume them
                // in trajectory order, regardless of parallel completion order.
                for start in (0..options.trajectories).step_by(concurrency) {
                    let len = concurrency.min(options.trajectories - start);
                    let values: Vec<Result<C, String>> = pool.install(|| {
                        workers[..len]
                            .par_iter_mut()
                            .enumerate()
                            .map(|(offset, worker)| {
                                self.execute(input, worker, options.seed, (start + offset) as u64)?;
                                Ok(crate::expect::expect_arrayreg_with_scratch(
                                    &worker.state,
                                    operator,
                                    &mut worker.scratch,
                                ))
                            })
                            .collect()
                    });
                    for value in values {
                        moments.push(value?)?;
                    }
                }
            }
            #[cfg(not(feature = "parallel"))]
            return Err("Multiple trajectory threads require the parallel feature".into());
        }
        Ok(moments.finish(options))
    }

    fn execute(
        &self,
        input: &ArrayReg,
        worker: &mut Worker,
        seed: u64,
        id: u64,
    ) -> Result<(), String> {
        worker.state.state.copy_from_slice(&input.state);
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        rng.set_stream(id);
        for step in &self.steps {
            match step {
                Step::Gate(pg) => dispatch_arrayreg_gate(self.nqubits, &mut worker.state.state, pg),
                Step::Depolarizing { locs, p } => {
                    if rng.random::<f64>() < *p {
                        for &loc in locs {
                            let gate = match rng.random_range(0..4) {
                                0 => continue,
                                1 => crate::Gate::X,
                                2 => crate::Gate::Y,
                                _ => crate::Gate::Z,
                            };
                            let matrix = gate.matrix();
                            crate::instruct_qubit::instruct_1q(
                                &mut worker.state.state,
                                loc,
                                matrix[[0, 0]],
                                matrix[[0, 1]],
                                matrix[[1, 0]],
                                matrix[[1, 1]],
                            );
                        }
                    }
                }
                Step::Channel { locs, matrices } => {
                    worker.probabilities.clear();
                    for matrix in matrices {
                        worker.scratch.copy_from_slice(&worker.state.state);
                        apply_matrix(self.nqubits, &mut worker.scratch, locs, matrix);
                        worker
                            .probabilities
                            .push(worker.scratch.iter().map(|z| z.norm_sqr()).sum::<f64>());
                    }
                    let total: f64 = worker.probabilities.iter().sum();
                    if !total.is_finite() || (total - 1.).abs() > 1e-9 {
                        return Err("Kraus branch probabilities lost normalization".into());
                    }
                    let distribution = WeightedIndex::new(&worker.probabilities)
                        .map_err(|e| format!("Invalid Kraus probabilities: {e}"))?;
                    let selected = distribution.sample(&mut rng);
                    let probability = worker.probabilities[selected];
                    if probability <= 0. {
                        return Err("Selected a zero-probability Kraus branch".into());
                    }
                    apply_matrix(
                        self.nqubits,
                        &mut worker.state.state,
                        locs,
                        &matrices[selected],
                    );
                    let norm = probability.sqrt();
                    for z in &mut worker.state.state {
                        *z /= norm;
                    }
                }
            }
        }
        Ok(())
    }
}

fn apply_matrix(n: usize, state: &mut [C], locs: &[usize], matrix: &[C]) {
    match locs {
        &[loc] => crate::instruct_qubit::instruct_1q(
            state, loc, matrix[0], matrix[1], matrix[2], matrix[3],
        ),
        [_, _] => crate::instruct_qubit::instruct_2q(state, n, locs, matrix),
        _ => crate::instruct_qubit::instruct_nq(state, n, locs, matrix, &[], &[]),
    }
}
struct Worker {
    state: ArrayReg,
    scratch: Vec<C>,
    probabilities: Vec<f64>,
}
impl Worker {
    fn new(input: &ArrayReg) -> Self {
        Self {
            state: input.clone(),
            scratch: vec![C::new(0., 0.); input.state.len()],
            probabilities: Vec::new(),
        }
    }
}

#[derive(Default)]
struct Moments {
    count: usize,
    mean: C,
    m2: C,
    cross: f64,
}
impl Moments {
    fn push(&mut self, value: C) -> Result<(), String> {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let after = value - self.mean;
        self.m2 += C::new(delta.re * after.re, delta.im * after.im);
        self.cross += delta.re * after.im;
        if [
            self.mean.re,
            self.mean.im,
            self.m2.re,
            self.m2.im,
            self.cross,
        ]
        .iter()
        .any(|x| !x.is_finite())
        {
            return Err("Observable moments exceeded finite floating-point range".into());
        }
        Ok(())
    }
    fn finish(self, options: TrajectoryOptions) -> TrajectoryStatistics {
        let variance = (self.count > 1).then(|| self.m2 / (self.count - 1) as f64);
        TrajectoryStatistics {
            trajectories: self.count,
            seed: options.seed,
            threads: options.threads,
            mean: self.mean,
            sample_variance: variance,
            standard_error: variance.map(|v| {
                C::new(
                    (v.re.max(0.) / self.count as f64).sqrt(),
                    (v.im.max(0.) / self.count as f64).sqrt(),
                )
            }),
            covariance: (self.count > 1).then(|| self.cross / (self.count - 1) as f64),
        }
    }
}

#[cfg(test)]
#[path = "unit_tests/trajectories.rs"]
mod tests;
