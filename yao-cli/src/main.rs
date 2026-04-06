mod cli;
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

    let auto_json = matches!(
        cli.command,
        Commands::Simulate { .. }
            | Commands::Measure { .. }
            | Commands::Probs { .. }
            | Commands::Expect { .. }
            | Commands::Run { .. }
            | Commands::Toeinsum { .. }
    );

    let _out = OutputConfig {
        output: cli.output.clone(),
        quiet: cli.quiet,
        json: cli.json,
        auto_json,
    };

    match cli.command {
        Commands::Inspect { input: _input } => {
            eprintln!("inspect: not yet implemented");
            Ok(())
        }
        Commands::Simulate {
            circuit: _circuit,
            input: _input,
        } => {
            eprintln!("simulate: not yet implemented");
            Ok(())
        }
        Commands::Measure {
            input: _input,
            shots: _shots,
            locs: _locs,
        } => {
            eprintln!("measure: not yet implemented");
            Ok(())
        }
        Commands::Probs {
            input: _input,
            locs: _locs,
        } => {
            eprintln!("probs: not yet implemented");
            Ok(())
        }
        Commands::Expect {
            input: _input,
            op: _op,
        } => {
            eprintln!("expect: not yet implemented");
            Ok(())
        }
        Commands::Run {
            circuit: _circuit,
            input: _input,
            shots: _shots,
            op: _op,
            locs: _locs,
        } => {
            eprintln!("run: not yet implemented");
            Ok(())
        }
        Commands::Toeinsum {
            circuit: _circuit,
            mode: _mode,
        } => {
            eprintln!("toeinsum: not yet implemented");
            Ok(())
        }
        #[cfg(feature = "typst")]
        Commands::Visualize { circuit: _circuit } => {
            eprintln!("visualize: not yet implemented");
            Ok(())
        }
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
