mod cli;
mod commands;
mod operator_parser;
mod output;
mod state_io;
mod tn_dto;

use clap::{CommandFactory, Parser};
use cli::{Cli, Commands};
use output::OutputConfig;

fn main() -> anyhow::Result<()> {
    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        Err(e) => {
            if e.kind() == clap::error::ErrorKind::DisplayHelp
                || e.kind() == clap::error::ErrorKind::DisplayVersion
            {
                e.exit();
            }
            eprint!("{e}");
            std::process::exit(e.exit_code());
        }
    };

    let mut auto_json = matches!(
        cli.command,
        Commands::Inspect { .. }
            | Commands::Simulate { .. }
            | Commands::Measure { .. }
            | Commands::Probs { .. }
            | Commands::Expect { .. }
            | Commands::Run { .. }
            | Commands::Toeinsum { .. }
            | Commands::Example { .. }
            | Commands::Fetch { .. }
    );
    #[cfg(feature = "omeinsum")]
    {
        auto_json |= matches!(cli.command, Commands::Contract { .. });
    }
    #[cfg(feature = "qasm")]
    {
        auto_json |= matches!(
            cli.command,
            Commands::Fromqasm { .. } | Commands::Toqasm { .. }
        );
    }

    let out = OutputConfig {
        output: cli.output,
        quiet: cli.quiet,
        json: cli.json,
        auto_json,
    };

    match cli.command {
        Commands::Inspect { input } => commands::inspect::inspect(&input, &out),
        Commands::Simulate { circuit, input } => {
            commands::simulate::simulate(&circuit, input.as_deref(), &out)
        }
        Commands::Run {
            circuit,
            input,
            shots,
            op,
            locs,
        } => commands::run::run(
            &circuit,
            input.as_deref(),
            shots,
            op.as_deref(),
            locs.as_deref(),
            &out,
        ),
        Commands::Measure { input, shots, locs } => {
            commands::measure::measure(&input, shots, locs.as_deref(), &out)
        }
        Commands::Probs { input, locs } => commands::probs::probs(&input, locs.as_deref(), &out),
        Commands::Expect { input, op } => commands::expect::expect(&input, &op, &out),
        #[cfg(feature = "omeinsum")]
        Commands::Contract { circuit, mode, op } => {
            commands::contract::contract_cmd(&circuit, mode, op.as_deref(), &out)
        }
        Commands::Toeinsum { circuit, mode } => commands::toeinsum::toeinsum(&circuit, mode, &out),
        #[cfg(feature = "qasm")]
        Commands::Fromqasm { input } => commands::fromqasm::fromqasm(&input, &out),
        #[cfg(feature = "qasm")]
        Commands::Toqasm { input } => commands::toqasm::toqasm(&input, &out),
        #[cfg(feature = "typst")]
        Commands::Visualize { circuit } => commands::visualize::visualize(&circuit, &out),
        Commands::Fetch {
            source,
            name,
            scale,
        } => commands::fetch::fetch(&source, &name, scale.as_deref(), &out),
        Commands::Example { name, nqubits } => commands::example::example(&name, nqubits, &out),
        Commands::Completions { shell } => {
            let shell = shell
                .or_else(clap_complete::Shell::from_env)
                .unwrap_or(clap_complete::Shell::Bash);
            let mut cmd = Cli::command();
            clap_complete::generate(shell, &mut cmd, "yao", &mut std::io::stdout());
            Ok(())
        }
    }
}
