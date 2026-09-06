//! Shared unitary/custom-loss fixtures for the circuit AD comparison.
use yao_tenferro_probe::{Result, circuit_ad};
fn main() -> Result<()> {
    let output = std::env::args()
        .nth(1)
        .ok_or("usage: ad_cases OUTPUT_JSON")?;
    let mut cases = Vec::new();
    for n in [8, 12, 16] {
        for depth in [10, 100] {
            let c = circuit_ad::circuit(n, depth)?;
            let circuit: serde_json::Value =
                serde_json::from_str(&yao_rs::circuit_to_json(c.template().circuit()))?;
            cases.push(serde_json::json!({"id":format!("custom_gradient_{n}_depth{depth}"),"mode":"custom_gradient","tensor":false,"initial":"deterministic","circuit":circuit,"custom_loss":{"kind":"squared_state_distance","target":"normalized cos(0.23*k)+i*sin(0.17*k)","depth":depth,"gates":4*depth,"parameters":3*depth,"active_sites":[n-2,n-1]}}));
        }
    }
    std::fs::write(output, serde_json::to_string_pretty(&cases)? + "\n")?;
    Ok(())
}
