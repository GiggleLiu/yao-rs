#[cfg(not(feature = "omeinsum"))]
fn main() {
    eprintln!("Run with: cargo run --example midterm_demo --features omeinsum");
}

#[cfg(feature = "omeinsum")]
mod midterm {
    use std::{
        env,
        error::Error,
        f64::consts::{FRAC_PI_2, FRAC_PI_4, PI},
        fs,
        io::{self, Write},
        path::{Path, PathBuf},
        process::Command,
        time::Instant,
    };

    use ndarray::Array2;
    use num_complex::Complex64;
    use omeco::{
        contraction_complexity, optimize_code, ContractionComplexity, GreedyMethod, TreeSA,
    };
    use yao_rs::{
        apply, channel, circuit_from_json, circuit_to_einsum_with_boundary,
        circuit_to_expectation_dm, circuit_to_overlap, contract_dm_with_tree, expect_arrayreg,
        expect_dm, put, ArrayReg, Circuit, CircuitElement, DensityMatrix, Gate, NoiseChannel, Op,
        OperatorPolynomial, OperatorString, Register,
    };

    const DEFAULT_DATA_DIR: &str = "/Users/liujinguo/jcode/QuantumSimulationDemos/tn/data/circuits";

    #[derive(Debug)]
    struct Config {
        data_dir: PathBuf,
        work_dir: PathBuf,
        converter: PathBuf,
        full: bool,
        ntrials: usize,
        theta: Option<f64>,
    }

    impl Config {
        fn from_args(args: impl IntoIterator<Item = String>) -> Result<Self, Box<dyn Error>> {
            let mut config = Self {
                data_dir: PathBuf::from(DEFAULT_DATA_DIR),
                work_dir: PathBuf::from("target/midterm-demo"),
                converter: PathBuf::from("scripts/qflex_txt_to_yao_json.py"),
                full: false,
                ntrials: 1,
                theta: None,
            };

            let mut args = args.into_iter();
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--data-dir" => config.data_dir = next_path(&mut args, "--data-dir")?,
                    "--work-dir" => config.work_dir = next_path(&mut args, "--work-dir")?,
                    "--converter" => config.converter = next_path(&mut args, "--converter")?,
                    "--full" => config.full = true,
                    "--ntrials" => {
                        let value = args.next().ok_or("--ntrials requires a value")?;
                        config.ntrials = value.parse()?;
                    }
                    "--theta" => {
                        let value = args.next().ok_or("--theta requires a value")?;
                        config.theta = Some(parse_theta(&value)?);
                    }
                    "--help" | "-h" => {
                        print_help();
                        std::process::exit(0);
                    }
                    other => return Err(format!("unknown argument: {other}").into()),
                }
            }

            Ok(config)
        }

        fn converted_path(&self, stem: &str) -> PathBuf {
            self.work_dir.join("circuits").join(format!("{stem}.json"))
        }
    }

    struct ScalarRun {
        value: Complex64,
        complexity: ContractionComplexity,
        optimize_secs: f64,
        contract_secs: f64,
    }

    pub fn run() -> Result<(), Box<dyn Error>> {
        let config = Config::from_args(env::args().skip(1))?;
        fs::create_dir_all(config.work_dir.join("circuits"))?;
        fs::create_dir_all(config.work_dir.join("results"))?;

        println!("=== Midterm tensor-network reproduction (yao-rs) ===");
        println!(
            "Data source: qflex .txt files in {}",
            config.data_dir.display()
        );
        println!("Converter: {}", config.converter.display());
        println!("Output dir: {}", config.work_dir.display());
        println!(
            "Mode: {}",
            if config.full {
                "full 60q kicked-Ising sweep"
            } else {
                "quick; pass --full for the 60q kicked-Ising sweep"
            }
        );

        let small = load_converted_circuit(&config, "rectangular_2x2_1-2-1_0")?;
        run_small_validation(&config, &small)?;

        let bristlecone = load_converted_circuit(&config, "bristlecone_70_1-12-1_0")?;
        run_bristlecone(&bristlecone, config.ntrials)?;

        run_small_expectation(&small, config.ntrials)?;
        run_kicked_ising(&config)?;

        println!(
            "\nSVG output: {}",
            config
                .work_dir
                .join("results/midterm_circuit_small.svg")
                .display()
        );
        Ok(())
    }

    pub fn rzz_gate(theta: f64) -> Gate {
        let neg_phase = Complex64::from_polar(1.0, -theta / 2.0);
        let pos_phase = Complex64::from_polar(1.0, theta / 2.0);
        let zero = Complex64::new(0.0, 0.0);
        let matrix = Array2::from_shape_vec(
            (4, 4),
            vec![
                neg_phase, zero, zero, zero, zero, pos_phase, zero, zero, zero, zero, pos_phase,
                zero, zero, zero, zero, neg_phase,
            ],
        )
        .unwrap();

        Gate::Custom {
            matrix,
            is_diagonal: true,
            label: "RZZ".to_string(),
        }
    }

    pub fn kicked_ising_chain(
        nqubits: usize,
        layers: usize,
        theta: f64,
        error_rate: f64,
    ) -> Circuit {
        let mut elements = Vec::new();
        for _ in 0..layers {
            for qubit in 0..(nqubits - 1) {
                elements.push(put(vec![qubit, qubit + 1], rzz_gate(FRAC_PI_2)));
            }
            for qubit in 0..nqubits {
                elements.push(put(vec![qubit], Gate::Rx(theta)));
            }
            if error_rate > 0.0 {
                for qubit in 0..nqubits {
                    elements.push(channel(
                        vec![qubit],
                        NoiseChannel::Depolarizing {
                            n: 1,
                            p: error_rate,
                        },
                    ));
                }
            }
        }
        Circuit::qubits(nqubits, elements).unwrap()
    }

    #[cfg(test)]
    pub fn zero_overlap_amplitude(circuit: &Circuit) -> Result<Complex64, Box<dyn Error>> {
        Ok(contract_zero_overlap(circuit, 1)?.value)
    }

    fn next_path(
        args: &mut impl Iterator<Item = String>,
        name: &str,
    ) -> Result<PathBuf, Box<dyn Error>> {
        Ok(PathBuf::from(
            args.next()
                .ok_or_else(|| format!("{name} requires a value"))?,
        ))
    }

    fn print_help() {
        println!(
            "Usage: cargo run --example midterm_demo --features omeinsum -- [OPTIONS]\n\
             \n\
             Options:\n\
               --data-dir PATH    qflex .txt circuit directory\n\
               --work-dir PATH    converted JSON and result output directory\n\
               --converter PATH   Python qflex-to-yao JSON converter\n\
               --ntrials N        TreeSA trial count for this demo (default: 1)\n\
               --theta VALUE      only run one 60q angle; accepts a float, pi/4, pi/2, 3pi/8\n\
               --full             run the 60-qubit kicked-Ising sweep"
        );
    }

    fn load_converted_circuit(config: &Config, stem: &str) -> Result<Circuit, Box<dyn Error>> {
        let input = config.data_dir.join(format!("{stem}.txt"));
        let output = config.converted_path(stem);
        convert_qflex_file(&config.converter, &input, &output)?;
        load_circuit_json(&output)
    }

    fn convert_qflex_file(
        converter: &Path,
        input: &Path,
        output: &Path,
    ) -> Result<(), Box<dyn Error>> {
        let status = Command::new("python3")
            .arg(converter)
            .arg(input)
            .arg("-o")
            .arg(output)
            .status()?;
        if !status.success() {
            return Err(format!(
                "converter failed for {} with status {status}",
                input.display()
            )
            .into());
        }
        Ok(())
    }

    fn load_circuit_json(path: &Path) -> Result<Circuit, Box<dyn Error>> {
        let json = fs::read_to_string(path)?;
        circuit_from_json(&json).map_err(|err| {
            format!("failed to load converted circuit {}: {err}", path.display()).into()
        })
    }

    fn run_small_validation(config: &Config, circuit: &Circuit) -> Result<(), Box<dyn Error>> {
        println!("\n=== Part 1: small circuit validation ===");
        println!(
            "Loaded: {} qubits, {} gates",
            circuit.num_sites(),
            gate_count(circuit)
        );

        let svg_path = config
            .work_dir
            .join("results")
            .join("midterm_circuit_small.svg");
        fs::write(&svg_path, circuit.to_svg())?;
        println!("Saved circuit diagram: {}", svg_path.display());

        let exact_reg = apply(circuit, &ArrayReg::zero_state(circuit.num_sites()));
        let exact_amp = exact_reg.state_vec()[0];
        let tn_amp = contract_zero_overlap(circuit, config.ntrials)?;
        println!("Exact <0|U|0> = {}", format_complex(exact_amp));
        println!("TN    <0|U|0> = {}", format_complex(tn_amp.value));
        println!("|Exact - TN|  = {:.2e}", (exact_amp - tn_amp.value).norm());
        println!(
            "Complexity: tc=2^{:.1}, sc=2^{:.1}, rwc=2^{:.1}",
            tn_amp.complexity.tc, tn_amp.complexity.sc, tn_amp.complexity.rwc
        );

        let t_full = Instant::now();
        let tn_state = contract_statevector_tn(circuit, config.ntrials)?;
        let exact_probs: Vec<f64> = exact_reg
            .state_vec()
            .iter()
            .map(|amp| amp.norm_sqr())
            .collect();
        let tn_probs: Vec<f64> = tn_state.iter().map(|amp| amp.norm_sqr()).collect();
        let max_prob_diff = exact_probs
            .iter()
            .zip(tn_probs.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        println!(
            "Full statevector via TN: {:.2}s",
            t_full.elapsed().as_secs_f64()
        );
        println!(
            "Max |P_exact - P_TN| = {:.2e} over all 2^{} states",
            max_prob_diff,
            circuit.num_sites()
        );

        Ok(())
    }

    fn run_bristlecone(circuit: &Circuit, ntrials: usize) -> Result<(), Box<dyn Error>> {
        println!("\n=== Part 2: Bristlecone 70 qubits, 12 layers ===");
        println!(
            "Loaded: {} qubits, {} gates",
            circuit.num_sites(),
            gate_count(circuit)
        );

        let run = contract_zero_overlap(circuit, ntrials)?;
        println!(
            "Optimization: {:.2}s; contraction: {:.2}s",
            run.optimize_secs, run.contract_secs
        );
        println!(
            "Complexity: tc=2^{:.1}, sc=2^{:.1}, rwc=2^{:.1}",
            run.complexity.tc, run.complexity.sc, run.complexity.rwc
        );
        println!("<0|U|0> = {}", format_complex(run.value));
        println!("|<0|U|0>|^2 = {:.6e}", run.value.norm_sqr());
        Ok(())
    }

    fn run_small_expectation(circuit: &Circuit, ntrials: usize) -> Result<(), Box<dyn Error>> {
        println!("\n=== Part 3a: observable expectation validation ===");
        let op = OperatorPolynomial::new(
            vec![Complex64::new(1.0, 0.0)],
            vec![OperatorString::new(vec![(0, Op::Z), (1, Op::Z)])],
        );
        let exact_reg = apply(circuit, &ArrayReg::zero_state(circuit.num_sites()));
        let exact = expect_arrayreg(&exact_reg, &op);
        let tn = contract_expectation_dm(circuit, &op, ntrials)?;

        println!("Observable: Z0 Z1");
        println!("Exact <Z0Z1> = {}", format_complex(exact));
        println!("TN    <Z0Z1> = {}", format_complex(tn.value));
        println!("|Exact - TN| = {:.2e}", (exact - tn.value).norm());
        Ok(())
    }

    fn run_kicked_ising(config: &Config) -> Result<(), Box<dyn Error>> {
        println!("\n=== Part 3b/3c: kicked-Ising validation ===");
        if config.full {
            let nqubits = 60;
            let layers = 10;
            let obs_site = 29;
            let noise = 0.005;
            let thetas = selected_thetas(config.theta);
            println!("60 qubits, 10 layers, observing <Z_{}>", obs_site + 1);
            println!("theta     clean       noisy       tc-clean  tc-noisy");
            for theta in thetas {
                let clean = kicked_ising_chain(nqubits, layers, theta, 0.0);
                let noisy = kicked_ising_chain(nqubits, layers, theta, noise);
                let op = z_operator(obs_site);
                print!("  theta={theta:.3}: optimizing clean... ");
                io::stdout().flush()?;
                let clean_result = contract_expectation_dm(&clean, &op, config.ntrials)?;
                println!(
                    "value={:+.4}, tc=2^{:.1}, opt={:.2}s, contract={:.2}s",
                    clean_result.value.re,
                    clean_result.complexity.tc,
                    clean_result.optimize_secs,
                    clean_result.contract_secs
                );
                print!("  theta={theta:.3}: optimizing noisy... ");
                io::stdout().flush()?;
                let noisy_result = contract_expectation_dm(&noisy, &op, config.ntrials)?;
                println!(
                    "value={:+.4}, tc=2^{:.1}, opt={:.2}s, contract={:.2}s",
                    noisy_result.value.re,
                    noisy_result.complexity.tc,
                    noisy_result.optimize_secs,
                    noisy_result.contract_secs
                );
                println!(
                    "{theta:<9.3} {:+.4}     {:+.4}     2^{:<6.1} 2^{:<6.1}",
                    clean_result.value.re,
                    noisy_result.value.re,
                    clean_result.complexity.tc,
                    noisy_result.complexity.tc
                );
            }
        } else {
            println!("Skipping 60q sweep in quick mode; pass --full to run it.");
        }

        let nqubits = 10;
        let layers = 6;
        let obs_site = 4;
        let theta = FRAC_PI_4;
        let noise = 0.005;
        let op = z_operator(obs_site);

        let clean = kicked_ising_chain(nqubits, layers, theta, 0.0);
        let clean_tn = contract_expectation_dm(&clean, &op, config.ntrials)?.value;
        let clean_exact = expect_arrayreg(&apply(&clean, &ArrayReg::zero_state(nqubits)), &op);
        println!(
            "10q clean: TN={}, exact={}, |diff|={:.2e}",
            format_complex(clean_tn),
            format_complex(clean_exact),
            (clean_tn - clean_exact).norm()
        );

        let noisy = kicked_ising_chain(nqubits, layers, theta, noise);
        let noisy_tn = contract_expectation_dm(&noisy, &op, config.ntrials)?.value;
        let mut dm = DensityMatrix::from_reg(&ArrayReg::zero_state(nqubits));
        dm.apply(&noisy);
        let noisy_exact = expect_dm(&dm, &op);
        println!(
            "10q noisy: TN={}, density-matrix={}, |diff|={:.2e}",
            format_complex(noisy_tn),
            format_complex(noisy_exact),
            (noisy_tn - noisy_exact).norm()
        );

        Ok(())
    }

    fn contract_zero_overlap(
        circuit: &Circuit,
        ntrials: usize,
    ) -> Result<ScalarRun, Box<dyn Error>> {
        let tn = circuit_to_overlap(circuit);
        let optimizer = TreeSA::fast().with_ntrials(ntrials);
        let opt_start = Instant::now();
        let tree = optimize_code(&tn.code, &tn.size_dict, &optimizer)
            .ok_or("failed to optimize zero-overlap tensor network")?;
        let optimize_secs = opt_start.elapsed().as_secs_f64();
        let complexity = contraction_complexity(&tree, &tn.size_dict, &tn.code.ixs);

        let contract_start = Instant::now();
        let result = yao_rs::contractor::contract_with_tree(&tn, tree);
        let contract_secs = contract_start.elapsed().as_secs_f64();

        Ok(ScalarRun {
            value: result[[]],
            complexity,
            optimize_secs,
            contract_secs,
        })
    }

    fn contract_statevector_tn(
        circuit: &Circuit,
        ntrials: usize,
    ) -> Result<Vec<Complex64>, Box<dyn Error>> {
        let tn = circuit_to_einsum_with_boundary(circuit, &[]);
        let optimizer = TreeSA::fast().with_ntrials(ntrials);
        let tree = optimize_code(&tn.code, &tn.size_dict, &optimizer)
            .ok_or("failed to optimize statevector tensor network")?;
        let result = yao_rs::contractor::contract_with_tree(&tn, tree);
        Ok(result.iter().copied().collect())
    }

    fn contract_expectation_dm(
        circuit: &Circuit,
        operator: &OperatorPolynomial,
        _ntrials: usize,
    ) -> Result<ScalarRun, Box<dyn Error>> {
        let tn = circuit_to_expectation_dm(circuit, operator);
        let optimizer = GreedyMethod::default();
        let opt_start = Instant::now();
        let tree = optimize_code(&tn.code, &tn.size_dict, &optimizer)
            .ok_or("failed to optimize expectation tensor network")?;
        let optimize_secs = opt_start.elapsed().as_secs_f64();
        let complexity = contraction_complexity(&tree, &tn.size_dict, &tn.code.ixs);

        let contract_start = Instant::now();
        let result = contract_dm_with_tree(&tn, tree);
        let contract_secs = contract_start.elapsed().as_secs_f64();

        Ok(ScalarRun {
            value: result[[]],
            complexity,
            optimize_secs,
            contract_secs,
        })
    }

    fn z_operator(site: usize) -> OperatorPolynomial {
        OperatorPolynomial::single(site, Op::Z, Complex64::new(1.0, 0.0))
    }

    pub fn selected_thetas(theta: Option<f64>) -> Vec<f64> {
        theta.map_or_else(
            || vec![0.0, PI / 8.0, FRAC_PI_4, 3.0 * PI / 8.0, FRAC_PI_2],
            |theta| vec![theta],
        )
    }

    fn parse_theta(value: &str) -> Result<f64, Box<dyn Error>> {
        let normalized = value.trim().to_ascii_lowercase();
        if let Ok(theta) = normalized.parse::<f64>() {
            return Ok(theta);
        }

        let (coefficient, denominator) = match normalized.as_str() {
            "pi" | "π" => (1.0, 1.0),
            "pi/8" | "π/8" => (1.0, 8.0),
            "pi/4" | "π/4" => (1.0, 4.0),
            "pi/2" | "π/2" => (1.0, 2.0),
            "3pi/8" | "3π/8" => (3.0, 8.0),
            "3*pi/8" | "3*π/8" => (3.0, 8.0),
            _ => return Err(format!("unsupported theta value: {value}").into()),
        };
        Ok(coefficient * PI / denominator)
    }

    fn gate_count(circuit: &Circuit) -> usize {
        circuit
            .elements
            .iter()
            .filter(|element| matches!(element, CircuitElement::Gate(_)))
            .count()
    }

    fn format_complex(value: Complex64) -> String {
        format!("{:+.6e} {:+.6e}i", value.re, value.im)
    }
}

#[cfg(feature = "omeinsum")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    midterm::run()
}

#[cfg(all(test, feature = "omeinsum"))]
mod tests {
    use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_2, FRAC_PI_4};

    use num_complex::Complex64;
    use yao_rs::{apply, put, ArrayReg, Circuit, CircuitElement, Gate};

    use super::midterm::{kicked_ising_chain, rzz_gate, selected_thetas, zero_overlap_amplitude};

    fn assert_close(actual: Complex64, expected: Complex64) {
        assert!(
            (actual - expected).norm() < 1e-12,
            "actual={actual:?}, expected={expected:?}"
        );
    }

    #[test]
    fn rzz_gate_is_diagonal_with_expected_phases() {
        let gate = rzz_gate(FRAC_PI_2);
        assert!(gate.is_diagonal());

        let matrix = gate.matrix();
        let neg_phase = Complex64::from_polar(1.0, -FRAC_PI_4);
        let pos_phase = Complex64::from_polar(1.0, FRAC_PI_4);

        assert_close(matrix[[0, 0]], neg_phase);
        assert_close(matrix[[1, 1]], pos_phase);
        assert_close(matrix[[2, 2]], pos_phase);
        assert_close(matrix[[3, 3]], neg_phase);
        assert_close(matrix[[0, 1]], Complex64::new(0.0, 0.0));
    }

    #[test]
    fn kicked_ising_chain_keeps_demo_noise_outside_core_library() {
        let clean = kicked_ising_chain(4, 2, FRAC_PI_4, 0.0);
        let noisy = kicked_ising_chain(4, 2, FRAC_PI_4, 0.005);

        assert_eq!(clean.num_sites(), 4);
        assert_eq!(clean.elements.len(), 14);
        assert_eq!(noisy.elements.len(), 22);
        assert_eq!(
            noisy
                .elements
                .iter()
                .filter(|element| matches!(element, CircuitElement::Channel(_)))
                .count(),
            8
        );
    }

    #[test]
    fn zero_overlap_amplitude_matches_exact_statevector() {
        let circuit = Circuit::qubits(1, vec![put(vec![0], Gate::H)]).unwrap();
        let exact = apply(&circuit, &ArrayReg::zero_state(1)).state_vec()[0];
        let tn = zero_overlap_amplitude(&circuit).unwrap();

        assert_close(exact, Complex64::new(FRAC_1_SQRT_2, 0.0));
        assert_close(tn, exact);
    }

    #[test]
    fn selected_thetas_can_limit_the_full_sweep_to_one_angle() {
        assert_eq!(selected_thetas(Some(FRAC_PI_4)), vec![FRAC_PI_4]);
        assert_eq!(selected_thetas(None).len(), 5);
    }
}
