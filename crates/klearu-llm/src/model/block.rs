use crate::config::LlmConfig;

use super::attention::Attention;
use super::mlp::Mlp;
use super::rms_norm::RmsNorm;

/// A single transformer block: pre-norm -> attention -> residual -> pre-norm -> MLP -> residual.
pub struct TransformerBlock {
    pub attn_norm: RmsNorm,
    pub attention: Attention,
    pub mlp_norm: RmsNorm,
    pub mlp: Mlp,
}

impl TransformerBlock {
    pub fn new(config: &LlmConfig) -> Self {
        Self {
            attn_norm: RmsNorm::new(config.hidden_size, config.rms_norm_eps),
            attention: Attention::new(
                config.hidden_size,
                config.num_heads,
                config.num_kv_heads,
                config.head_dim,
            ),
            mlp_norm: RmsNorm::new(config.hidden_size, config.rms_norm_eps),
            mlp: Mlp::new(config.hidden_size, config.intermediate_size),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_creation() {
        let config = LlmConfig {
            hidden_size: 32,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 8,
            intermediate_size: 64,
            ..LlmConfig::default()
        };
        let block = TransformerBlock::new(&config);
        assert_eq!(block.attention.q_proj.in_features(), 32);
        assert_eq!(block.attention.q_proj.out_features(), 32); // 4 heads * 8 dim
        assert_eq!(block.mlp.gate_proj.in_features(), 32);
        assert_eq!(block.mlp.gate_proj.out_features(), 64);
    }
}
