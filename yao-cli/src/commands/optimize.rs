use crate::output::OutputConfig;
use crate::tn_dto::TensorNetworkDto;
use anyhow::{Result, bail};
use omeco::json::NestedEinsumTree;
use omeco::{EinCode, GreedyMethod, TreeSA, optimize_code};

#[allow(clippy::too_many_arguments)]
pub fn optimize_cmd(
    input_path: &str,
    method: &str,
    alpha: Option<f64>,
    temperature: Option<f64>,
    ntrials: Option<usize>,
    niters: Option<usize>,
    betas: Option<&str>,
    sc_target: Option<f64>,
    tc_weight: Option<f64>,
    sc_weight: Option<f64>,
    rw_weight: Option<f64>,
    out: &OutputConfig,
) -> Result<()> {
    let json = super::load_stdin_or_file(input_path)?;
    let mut dto: TensorNetworkDto =
        serde_json::from_str(&json).map_err(|e| anyhow::anyhow!("Failed to parse TN JSON: {e}"))?;

    if dto.mode == "dm" {
        bail!(
            "DM mode is not supported by optimize/contract. \
             Density matrix tensor networks (mode=dm) are not supported. \
             Use mode=pure, overlap, or state."
        );
    }

    if dto.contraction_order.is_some() {
        out.info("Warning: replacing existing contraction order");
    }

    let tn = dto.to_tensor_network()?;
    let code = EinCode::new(tn.code.ixs.clone(), tn.code.iy.clone());

    let tree = match method {
        "greedy" => {
            let optimizer = GreedyMethod::new(alpha.unwrap_or(0.0), temperature.unwrap_or(0.0));
            optimize_code(&code, &tn.size_dict, &optimizer)
        }
        "treesa" => {
            let mut optimizer = TreeSA::default();
            if let Some(v) = ntrials {
                optimizer.ntrials = v;
            }
            if let Some(v) = niters {
                optimizer.niters = v;
            }
            if let Some(s) = betas {
                optimizer.betas = parse_betas(s)?;
            }
            if let Some(v) = sc_target {
                optimizer.score.sc_target = v;
            }
            if let Some(v) = tc_weight {
                optimizer.score.tc_weight = v;
            }
            if let Some(v) = sc_weight {
                optimizer.score.sc_weight = v;
            }
            if let Some(v) = rw_weight {
                optimizer.score.rw_weight = v;
            }
            optimize_code(&code, &tn.size_dict, &optimizer)
        }
        _ => bail!("Unknown method '{method}': expected 'greedy' or 'treesa'"),
    };

    let tree = tree.ok_or_else(|| anyhow::anyhow!("Optimization produced no contraction tree"))?;
    dto.contraction_order = Some(NestedEinsumTree::from(&tree));

    let json_value = serde_json::to_value(&dto)?;
    let human = format!(
        "Optimized (method={method}):\n  Tensors: {}\n  Labels: {}\n",
        dto.tensors.len(),
        dto.size_dict.len(),
    );

    out.emit(&human, &json_value)
}

fn parse_betas(s: &str) -> Result<Vec<f64>> {
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() != 3 {
        bail!("Invalid betas '{s}': expected 'start:step:stop'");
    }
    let start: f64 = parts[0]
        .parse()
        .map_err(|_| anyhow::anyhow!("Invalid betas start"))?;
    let step: f64 = parts[1]
        .parse()
        .map_err(|_| anyhow::anyhow!("Invalid betas step"))?;
    let stop: f64 = parts[2]
        .parse()
        .map_err(|_| anyhow::anyhow!("Invalid betas stop"))?;
    if step <= 0.0 {
        bail!("Betas step must be positive");
    }
    let mut betas = Vec::new();
    let mut v = start;
    while v <= stop {
        betas.push(v);
        v += step;
    }
    if betas.is_empty() {
        bail!("Betas '{s}' produced an empty schedule");
    }
    Ok(betas)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use yao_rs::{Circuit, Gate, circuit_to_einsum, put};

    fn temp_path(prefix: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "yao-cli-{prefix}-{}-{}.json",
            std::process::id(),
            rand::random::<u64>()
        ))
    }

    #[test]
    fn test_optimize_adds_contraction_order() {
        let circuit = Circuit::new(
            vec![2, 2],
            vec![put(vec![0], Gate::H), put(vec![0, 1], Gate::SWAP)],
        )
        .unwrap();
        let tn = circuit_to_einsum(&circuit);
        let dto = TensorNetworkDto::from_pure(&tn);
        assert!(dto.contraction_order.is_none());

        let json = serde_json::to_string(&dto).unwrap();
        let mut parsed: TensorNetworkDto = serde_json::from_str(&json).unwrap();
        let tn_back = parsed.to_tensor_network().unwrap();

        let code = EinCode::new(tn_back.code.ixs.clone(), tn_back.code.iy.clone());
        let tree = optimize_code(&code, &tn_back.size_dict, &GreedyMethod::default());
        assert!(tree.is_some());

        let tree = tree.unwrap();
        parsed.contraction_order = Some(NestedEinsumTree::from(&tree));

        let json2 = serde_json::to_string(&parsed).unwrap();
        assert!(json2.contains("contraction_order"));
        let reparsed: TensorNetworkDto = serde_json::from_str(&json2).unwrap();
        assert!(reparsed.contraction_order.is_some());
    }

    #[test]
    fn test_parse_betas() {
        let betas = parse_betas("0.01:0.05:0.20").unwrap();
        assert_eq!(betas.len(), 4); // 0.01, 0.06, 0.11, 0.16
        assert!((betas[0] - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_parse_betas_invalid() {
        assert!(parse_betas("1:2").is_err());
        assert!(parse_betas("1:-1:5").is_err());
        assert!(parse_betas("5:1:1").is_err());
    }

    #[test]
    fn test_optimize_rejects_density_matrix_mode() {
        let circuit = Circuit::new(vec![2], vec![put(vec![0], Gate::H)]).unwrap();
        let tn = yao_rs::circuit_to_einsum_dm(&circuit);
        let dto = TensorNetworkDto::from_dm(&tn);

        let input_path = temp_path("optimize-dm-input");
        let output_path = temp_path("optimize-dm-output");
        fs::write(&input_path, serde_json::to_string(&dto).unwrap()).unwrap();

        let out = OutputConfig {
            output: Some(output_path.clone()),
            quiet: true,
            json: false,
            auto_json: false,
        };

        let err = optimize_cmd(
            input_path.to_str().unwrap(),
            "greedy",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            &out,
        )
        .unwrap_err();

        assert!(
            err.to_string().contains("Density matrix tensor networks"),
            "unexpected error: {err}"
        );

        let _ = fs::remove_file(input_path);
        let _ = fs::remove_file(output_path);
    }
}
