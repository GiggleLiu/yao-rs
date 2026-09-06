use yao_tenferro_probe::{Result, trajectories as tr};
fn main() -> Result<()> {
    let output = std::env::args().nth(1).ok_or("expected output JSON")?;
    let cases=[4,6,8].map(|n|->Result<_>{Ok(serde_json::json!({"id":format!("noisy_expectation_{n}"),"mode":"expectation_dm","initial":"zero","tensor":false,
        "circuit":serde_json::from_str::<serde_json::Value>(&yao_rs::circuit_to_json(&tr::circuit(n,true)?))?,"operator":tr::observable(n,true)}))}).into_iter().collect::<Result<Vec<_>>>()?;
    std::fs::write(output, serde_json::to_string_pretty(&cases)? + "\n")?;
    Ok(())
}
