use serde::{Deserialize, Serialize};

use crate::error::{LlmError, Result};

/// LLaMA-compatible model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    #[serde(alias = "num_attention_heads")]
    pub num_heads: usize,
    #[serde(alias = "num_key_value_heads")]
    pub num_kv_heads: usize,
    #[serde(default)]
    pub head_dim: usize,
    pub intermediate_size: usize,
    #[serde(alias = "num_hidden_layers")]
    pub num_layers: usize,
    #[serde(alias = "max_position_embeddings", default = "default_max_seq_len")]
    pub max_seq_len: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

fn default_max_seq_len() -> usize {
    2048
}
fn default_rope_theta() -> f32 {
    10000.0
}
fn default_rms_norm_eps() -> f32 {
    1e-5
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 2048,
            num_heads: 32,
            num_kv_heads: 4,
            head_dim: 64,
            intermediate_size: 5632,
            num_layers: 22,
            max_seq_len: 2048,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            tie_word_embeddings: false,
        }
    }
}

impl LlmConfig {
    /// Load config from a HuggingFace `config.json` string.
    pub fn from_hf_config(json: &str) -> Result<Self> {
        let mut config: LlmConfig =
            serde_json::from_str(json).map_err(LlmError::Json)?;
        // Compute head_dim if not explicitly set
        if config.head_dim == 0 {
            if config.num_heads == 0 {
                return Err(LlmError::InvalidConfig("num_heads is 0".into()));
            }
            config.head_dim = config.hidden_size / config.num_heads;
        }
        config.validate()?;
        Ok(config)
    }

    /// Load config from a HuggingFace `config.json` file.
    pub fn from_file(path: &std::path::Path) -> Result<Self> {
        let json = std::fs::read_to_string(path)?;
        Self::from_hf_config(&json)
    }

    pub fn validate(&self) -> Result<()> {
        if self.hidden_size == 0 {
            return Err(LlmError::InvalidConfig("hidden_size is 0".into()));
        }
        if self.num_heads == 0 {
            return Err(LlmError::InvalidConfig("num_heads is 0".into()));
        }
        if self.num_kv_heads == 0 {
            return Err(LlmError::InvalidConfig("num_kv_heads is 0".into()));
        }
        if !self.num_heads.is_multiple_of(self.num_kv_heads) {
            return Err(LlmError::InvalidConfig(format!(
                "num_heads ({}) must be divisible by num_kv_heads ({})",
                self.num_heads, self.num_kv_heads
            )));
        }
        if self.head_dim * self.num_heads != self.hidden_size {
            return Err(LlmError::InvalidConfig(format!(
                "head_dim ({}) * num_heads ({}) != hidden_size ({})",
                self.head_dim, self.num_heads, self.hidden_size
            )));
        }
        Ok(())
    }

    /// Number of query heads sharing each KV head.
    pub fn gqa_group_size(&self) -> usize {
        self.num_heads / self.num_kv_heads
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = LlmConfig::default();
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.gqa_group_size(), 8);
        config.validate().unwrap();
    }

    #[test]
    fn test_from_hf_json() {
        let json = r#"{
            "vocab_size": 32000,
            "hidden_size": 2048,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "intermediate_size": 5632,
            "num_hidden_layers": 22,
            "max_position_embeddings": 2048,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-5,
            "tie_word_embeddings": false
        }"#;
        let config = LlmConfig::from_hf_config(json).unwrap();
        assert_eq!(config.num_heads, 32);
        assert_eq!(config.num_kv_heads, 4);
        assert_eq!(config.head_dim, 64);
        assert_eq!(config.num_layers, 22);
    }

    #[test]
    fn test_invalid_gqa() {
        let json = r#"{
            "vocab_size": 32000,
            "hidden_size": 2048,
            "num_attention_heads": 32,
            "num_key_value_heads": 5,
            "intermediate_size": 5632,
            "num_hidden_layers": 22
        }"#;
        assert!(LlmConfig::from_hf_config(json).is_err());
    }

    #[test]
    fn test_roundtrip_serde() {
        let config = LlmConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let _: serde_json::Value = serde_json::from_str(&json).unwrap();
    }
}
