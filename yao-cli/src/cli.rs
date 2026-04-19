use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "yao",
    about = "Quantum circuit simulation toolkit",
    version,
    after_help = "\
Output is human-readable in a terminal, JSON when piped. Use --json to force JSON.

Typical workflow:
  yao inspect circuit.json
  yao run circuit.json --shots 1024
  yao run circuit.json --op \"Z(0)Z(1)\"

Pipeline (no intermediate files):
  yao simulate circuit.json | yao measure - --shots 1024
  yao simulate circuit.json | yao probs -

Tensor network pipeline:
  yao toeinsum circuit.json --mode state | yao optimize - | yao contract -

Use `yao <command> --help` for detailed usage of each command.

Enable tab completion:
  eval \"$(yao completions)\"     # add to ~/.bashrc or ~/.zshrc"
)]
pub struct Cli {
    /// Output file path
    #[arg(long, short, global = true)]
    pub output: Option<PathBuf>,

    /// Suppress informational messages on stderr
    #[arg(long, short, global = true)]
    pub quiet: bool,

    /// Force JSON output
    #[arg(long, global = true)]
    pub json: bool,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Display circuit information (gate count, qubit count, gate list)
    #[command(after_help = "\
Examples:
  yao inspect circuit.json
  yao inspect circuit.json --json
  cat circuit.json | yao inspect -")]
    Inspect {
        /// Circuit JSON file (use - for stdin)
        input: String,
    },

    /// Simulate a circuit and output the resulting state
    #[command(after_help = "\
Examples:
  yao simulate circuit.json --output state.bin
  yao simulate circuit.json --input initial.bin --output final.bin
  yao simulate circuit.json | yao measure - --shots 100")]
    Simulate {
        /// Circuit JSON file (use - for stdin)
        circuit: String,
        /// Input state file (defaults to |0...0>)
        #[arg(long)]
        input: Option<String>,
    },

    /// Sample measurement outcomes from a state
    #[command(after_help = "\
Examples:
  yao measure state.bin --shots 1024
  yao measure state.bin --shots 100 --locs 0,1
  yao simulate circuit.json | yao measure - --shots 1024")]
    Measure {
        /// State file (use - for stdin)
        input: String,
        /// Number of measurement shots
        #[arg(long, default_value = "1024")]
        shots: usize,
        /// Qubit indices for partial measurement (comma-separated)
        #[arg(long, value_delimiter = ',')]
        locs: Option<Vec<usize>>,
    },

    /// Compute probability distribution from a state
    #[command(after_help = "\
Examples:
  yao probs state.bin
  yao probs state.bin --locs 0,1
  yao simulate circuit.json | yao probs -")]
    Probs {
        /// State file (use - for stdin)
        input: String,
        /// Qubit indices for marginal probabilities (comma-separated)
        #[arg(long, value_delimiter = ',')]
        locs: Option<Vec<usize>>,
    },

    /// Compute expectation value of an operator on a state
    #[command(after_help = "\
Operators: I, X, Y, Z, P0(=|0><0|), P1(=|1><1|), Pu(=|0><1| raising), Pd(=|1><0| lowering)
Syntax: coeff * Op(site)Op(site) [+/- ...]

Examples:
  yao expect state.bin --op \"Z(0)\"
  yao expect state.bin --op \"0.5*Z(0)Z(1) + X(0)\"
  yao simulate circuit.json | yao expect - --op \"Z(0)\"")]
    Expect {
        /// State file (use - for stdin)
        input: String,
        /// Operator expression (e.g. "Z(0)Z(1) + 0.5*X(0)")
        #[arg(long, allow_hyphen_values = true)]
        op: String,
    },

    /// Simulate and post-process in one step (no intermediate files)
    #[command(after_help = "\
Operators: I, X, Y, Z, P0(=|0><0|), P1(=|1><1|), Pu(=|0><1| raising), Pd(=|1><0| lowering)

Examples:
  yao run circuit.json --shots 1024
  yao run circuit.json --op \"Z(0)Z(1)\"
  yao run circuit.json --shots 100 --locs 0,1
  yao run circuit.json --output state.bin")]
    Run {
        /// Circuit JSON file (use - for stdin)
        circuit: String,
        /// Input state file (defaults to |0...0>)
        #[arg(long)]
        input: Option<String>,
        /// Number of measurement shots (mutually exclusive with --op)
        #[arg(long, conflicts_with = "op")]
        shots: Option<usize>,
        /// Operator expression for expectation (mutually exclusive with --shots)
        #[arg(long, conflicts_with = "shots", allow_hyphen_values = true)]
        op: Option<String>,
        /// Qubit indices for partial measurement (comma-separated, used with --shots)
        #[arg(long, value_delimiter = ',')]
        locs: Option<Vec<usize>>,
    },

    /// Contract a pre-optimized tensor network
    #[cfg(feature = "omeinsum")]
    #[command(after_help = "\
Examples:
  yao toeinsum circuit.json | yao optimize - | yao contract -
  yao toeinsum circuit.json --mode overlap | yao optimize - | yao contract -
  yao toeinsum circuit.json --op \"Z(0)Z(1)\" | yao optimize - | yao contract -")]
    Contract {
        /// Tensor network JSON file with contraction_order (use - for stdin)
        input: String,
    },

    /// Optimize contraction order for a tensor network
    #[cfg(feature = "omeinsum")]
    #[command(after_help = "\
Examples:
  yao optimize tn.json
  yao optimize tn.json --method treesa --ntrials 20
  yao toeinsum circuit.json --mode overlap | yao optimize -")]
    Optimize {
        /// Tensor network JSON file (use - for stdin)
        input: String,
        /// Optimization method: greedy (default) or treesa
        #[arg(long, default_value = "greedy")]
        method: String,
        /// [greedy] Weight for output-vs-input size balance (default: 0.0)
        #[arg(long)]
        alpha: Option<f64>,
        /// [greedy] Temperature for stochastic selection; 0 = deterministic (default: 0.0)
        #[arg(long)]
        temperature: Option<f64>,
        /// [treesa] Number of independent SA trials (default: 10)
        #[arg(long)]
        ntrials: Option<usize>,
        /// [treesa] Iterations per temperature level (default: 50)
        #[arg(long)]
        niters: Option<usize>,
        /// [treesa] Inverse temperature schedule as "start:step:stop" (default: "0.01:0.05:15.0")
        #[arg(long)]
        betas: Option<String>,
        /// [treesa] Space complexity target threshold (default: 20.0)
        #[arg(long)]
        sc_target: Option<f64>,
        /// [treesa] Time complexity weight (default: 1.0)
        #[arg(long)]
        tc_weight: Option<f64>,
        /// [treesa] Space complexity weight (default: 1.0)
        #[arg(long)]
        sc_weight: Option<f64>,
        /// [treesa] Read-write complexity weight (default: 0.0)
        #[arg(long)]
        rw_weight: Option<f64>,
    },

    /// Export circuit as tensor network (einsum)
    #[command(after_help = "\
Examples:
  yao toeinsum circuit.json
  yao toeinsum circuit.json --mode dm
  yao toeinsum circuit.json --mode overlap
  yao toeinsum circuit.json --mode state
  yao toeinsum circuit.json --op \"Z(0)Z(1)\"")]
    Toeinsum {
        /// Circuit JSON file (use - for stdin)
        circuit: String,
        /// Export mode: pure (default), dm, overlap, or state
        #[arg(long, value_enum, default_value_t = TnMode::Pure)]
        mode: TnMode,
        /// Operator expression for expectation TN (overrides --mode)
        #[arg(long, allow_hyphen_values = true)]
        op: Option<String>,
    },

    /// Render circuit diagram as SVG
    #[command(after_help = "\
Examples:
  yao visualize circuit.json --output circuit.svg")]
    Visualize {
        /// Circuit JSON file
        circuit: String,
    },

    /// Convert OpenQASM 2.0 file to circuit JSON
    #[cfg(feature = "qasm")]
    #[command(after_help = "\
Examples:
  yao fromqasm circuit.qasm
  yao fromqasm circuit.qasm --output circuit.json
  yao fromqasm circuit.qasm | yao run - --shots 1024")]
    Fromqasm {
        /// QASM file (use - for stdin)
        input: String,
    },

    /// Export circuit as OpenQASM 2.0
    #[cfg(feature = "qasm")]
    #[command(after_help = "\
Examples:
  yao toqasm circuit.json
  yao example bell | yao toqasm -")]
    Toqasm {
        /// Circuit JSON file (use - for stdin)
        input: String,
    },

    /// Download benchmark circuits from online repositories
    #[command(after_help = "\
Sources: qasmbench

Examples:
  yao fetch qasmbench list                  # List all circuits (queries GitHub)
  yao fetch qasmbench list --scale small    # List only small circuits
  yao fetch qasmbench grover               # Download by name (auto-detect scale)
  yao fetch qasmbench qft_n4 -o qft.qasm   # Save to file
  yao fetch qasmbench medium/shor_n5        # Explicit scale/name path

Pipeline:
  yao fetch qasmbench grover | yao fromqasm - | yao run - --shots 100")]
    Fetch {
        /// Source repository (qasmbench)
        source: String,
        /// Circuit name or 'list'
        name: String,
        /// Filter by scale: small, medium, large (used with 'list')
        #[arg(long)]
        scale: Option<String>,
    },

    /// Print example circuit JSON to stdout
    #[command(after_help = "\
Available examples: bell, ghz, qft, phase-estimation, hadamard-test, swap-test,
  bernstein-vazirani, grover, qaoa-maxcut, qcbm

Examples:
  yao example bell
  yao example bell > bell.json
  yao example qft --nqubits 6
  yao example phase-estimation --nqubits 3 --preset z")]
    Example {
        /// Example name: bell, ghz, qft, phase-estimation, hadamard-test, swap-test,
        /// bernstein-vazirani, grover, qaoa-maxcut, qcbm
        name: String,
        /// Number of qubits or register qubits, depending on the example.
        #[arg(long)]
        nqubits: Option<usize>,
        /// Preset name for examples that need a small built-in unitary or graph.
        #[arg(long)]
        preset: Option<String>,
        /// Secret bit string for Bernstein-Vazirani, e.g. 10101.
        #[arg(long)]
        secret: Option<String>,
        /// Marked basis-state index for Grover.
        #[arg(long)]
        marked: Option<usize>,
        /// Grover iteration count, or "auto".
        #[arg(long)]
        iterations: Option<String>,
        /// Ansatz depth for QAOA and QCBM.
        #[arg(long)]
        depth: Option<usize>,
        /// Phase parameter for phase-estimation, hadamard-test, and swap-test presets.
        #[arg(long)]
        phase: Option<f64>,
        /// Number of qubits per compared state for swap-test.
        #[arg(long)]
        nqubits_per_state: Option<usize>,
        /// Number of states for swap-test.
        #[arg(long)]
        nstates: Option<usize>,
    },

    /// Generate shell completion scripts
    #[command(after_help = "\
Examples:
  eval \"$(yao completions)\"
  yao completions zsh > _yao")]
    Completions {
        /// Shell to generate completions for (auto-detected if omitted)
        shell: Option<clap_complete::Shell>,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum TnMode {
    /// Pure-state tensor network
    Pure,
    /// Density-matrix tensor network
    Dm,
    /// Scalar overlap ⟨0|U|0⟩
    Overlap,
    /// State vector with |0⟩ boundary tensors
    State,
}
