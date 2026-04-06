use clap::Parser;

#[derive(Parser)]
#[command(name = "yao", about = "Quantum circuit simulation toolkit", version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(clap::Subcommand)]
enum Commands {
    /// Display circuit information
    Inspect,
}

fn main() -> anyhow::Result<()> {
    let _cli = Cli::parse();
    println!("yao CLI placeholder");
    Ok(())
}
