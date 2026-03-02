/// Identifies where a HuggingFace weight tensor should be loaded into the model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WeightTarget {
    Embedding,
    LmHead,
    FinalNorm,
    LayerAttnNorm(usize),
    LayerMlpNorm(usize),
    LayerQProj(usize),
    LayerKProj(usize),
    LayerVProj(usize),
    LayerOProj(usize),
    LayerGateProj(usize),
    LayerUpProj(usize),
    LayerDownProj(usize),
}

/// Parse a HuggingFace weight name to a WeightTarget.
///
/// Expected patterns:
/// - `model.embed_tokens.weight`
/// - `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight`
/// - `model.layers.{i}.mlp.{gate,up,down}_proj.weight`
/// - `model.layers.{i}.input_layernorm.weight`
/// - `model.layers.{i}.post_attention_layernorm.weight`
/// - `model.norm.weight`
/// - `lm_head.weight`
pub fn parse_weight_name(name: &str) -> Option<WeightTarget> {
    if name == "model.embed_tokens.weight" {
        return Some(WeightTarget::Embedding);
    }
    if name == "lm_head.weight" {
        return Some(WeightTarget::LmHead);
    }
    if name == "model.norm.weight" {
        return Some(WeightTarget::FinalNorm);
    }

    // model.layers.{i}.*
    if let Some(rest) = name.strip_prefix("model.layers.") {
        let dot_pos = rest.find('.')?;
        let layer_idx: usize = rest[..dot_pos].parse().ok()?;
        let suffix = &rest[dot_pos + 1..];

        return match suffix {
            "self_attn.q_proj.weight" => Some(WeightTarget::LayerQProj(layer_idx)),
            "self_attn.k_proj.weight" => Some(WeightTarget::LayerKProj(layer_idx)),
            "self_attn.v_proj.weight" => Some(WeightTarget::LayerVProj(layer_idx)),
            "self_attn.o_proj.weight" => Some(WeightTarget::LayerOProj(layer_idx)),
            "mlp.gate_proj.weight" => Some(WeightTarget::LayerGateProj(layer_idx)),
            "mlp.up_proj.weight" => Some(WeightTarget::LayerUpProj(layer_idx)),
            "mlp.down_proj.weight" => Some(WeightTarget::LayerDownProj(layer_idx)),
            "input_layernorm.weight" => Some(WeightTarget::LayerAttnNorm(layer_idx)),
            "post_attention_layernorm.weight" => Some(WeightTarget::LayerMlpNorm(layer_idx)),
            _ => None,
        };
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding() {
        assert_eq!(
            parse_weight_name("model.embed_tokens.weight"),
            Some(WeightTarget::Embedding)
        );
    }

    #[test]
    fn test_lm_head() {
        assert_eq!(
            parse_weight_name("lm_head.weight"),
            Some(WeightTarget::LmHead)
        );
    }

    #[test]
    fn test_final_norm() {
        assert_eq!(
            parse_weight_name("model.norm.weight"),
            Some(WeightTarget::FinalNorm)
        );
    }

    #[test]
    fn test_layer_attn_projections() {
        assert_eq!(
            parse_weight_name("model.layers.0.self_attn.q_proj.weight"),
            Some(WeightTarget::LayerQProj(0))
        );
        assert_eq!(
            parse_weight_name("model.layers.21.self_attn.k_proj.weight"),
            Some(WeightTarget::LayerKProj(21))
        );
        assert_eq!(
            parse_weight_name("model.layers.5.self_attn.v_proj.weight"),
            Some(WeightTarget::LayerVProj(5))
        );
        assert_eq!(
            parse_weight_name("model.layers.10.self_attn.o_proj.weight"),
            Some(WeightTarget::LayerOProj(10))
        );
    }

    #[test]
    fn test_layer_mlp_projections() {
        assert_eq!(
            parse_weight_name("model.layers.3.mlp.gate_proj.weight"),
            Some(WeightTarget::LayerGateProj(3))
        );
        assert_eq!(
            parse_weight_name("model.layers.3.mlp.up_proj.weight"),
            Some(WeightTarget::LayerUpProj(3))
        );
        assert_eq!(
            parse_weight_name("model.layers.3.mlp.down_proj.weight"),
            Some(WeightTarget::LayerDownProj(3))
        );
    }

    #[test]
    fn test_layer_norms() {
        assert_eq!(
            parse_weight_name("model.layers.0.input_layernorm.weight"),
            Some(WeightTarget::LayerAttnNorm(0))
        );
        assert_eq!(
            parse_weight_name("model.layers.0.post_attention_layernorm.weight"),
            Some(WeightTarget::LayerMlpNorm(0))
        );
    }

    #[test]
    fn test_unknown_name() {
        assert_eq!(parse_weight_name("model.layers.0.self_attn.rotary_emb.inv_freq"), None);
        assert_eq!(parse_weight_name("some.random.tensor"), None);
    }
}
