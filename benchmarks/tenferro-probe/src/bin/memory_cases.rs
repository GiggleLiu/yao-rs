use yao_tenferro_probe::{Result, tensor_memory as tm};
fn main() -> Result<()> {
    let output = std::env::args().nth(1).ok_or("expected output JSON")?;
    let mut cases = Vec::new();
    for n in [4, 6] {
        for noisy in [false, true] {
            let terms = 5;
            let circuit = tm::circuit(n, noisy)?;
            let mode = if noisy {
                "expectation_dm"
            } else {
                "expectation"
            };
            cases.push(serde_json::json!({"id":format!("{mode}_{n}_terms{terms}"),"mode":mode,"initial":"zero","tensor":false,
            "circuit":serde_json::from_str::<serde_json::Value>(&yao_rs::circuit_to_json(&circuit))?,"operator":tm::observable(n,terms)}));
        }
    }
    std::fs::write(output, serde_json::to_string_pretty(&cases)? + "\n")?;
    Ok(())
}
