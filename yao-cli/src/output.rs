use anyhow::Context;
use owo_colors::OwoColorize;
use std::io::IsTerminal;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct OutputConfig {
    pub output: Option<PathBuf>,
    pub quiet: bool,
    pub json: bool,
    pub auto_json: bool,
}

impl OutputConfig {
    pub fn info(&self, msg: &str) {
        if !self.quiet {
            eprintln!("{msg}");
        }
    }

    pub fn should_json(&self) -> bool {
        self.json || (self.auto_json && !std::io::stdout().is_terminal())
    }

    pub fn emit(
        &self,
        human_text: &str,
        json_value: &serde_json::Value,
    ) -> anyhow::Result<()> {
        if let Some(ref path) = self.output {
            let content =
                serde_json::to_string_pretty(json_value).context("Failed to serialize JSON")?;
            std::fs::write(path, &content)
                .with_context(|| format!("Failed to write {}", path.display()))?;
            self.info(&format!("Wrote {}", path.display()));
        } else if self.should_json() {
            println!(
                "{}",
                serde_json::to_string_pretty(json_value).context("Failed to serialize JSON")?
            );
        } else {
            println!("{human_text}");
        }
        Ok(())
    }
}

pub fn use_color() -> bool {
    std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none()
}

pub fn fmt_bold(text: &str) -> String {
    if use_color() {
        format!("{}", text.bold())
    } else {
        text.to_string()
    }
}

pub fn fmt_dim(text: &str) -> String {
    if use_color() {
        format!("{}", text.dimmed())
    } else {
        text.to_string()
    }
}

pub fn fmt_cyan(text: &str) -> String {
    if use_color() {
        format!("{}", text.cyan())
    } else {
        text.to_string()
    }
}
