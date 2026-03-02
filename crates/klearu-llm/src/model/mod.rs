pub mod attention;
pub mod block;
pub mod embedding;
pub mod kv_cache;
pub mod linear;
pub mod mlp;
pub mod rms_norm;
pub mod rope;

use crate::config::LlmConfig;
use block::TransformerBlock;
use embedding::Embedding;
use kv_cache::KvCacheStore;
use rms_norm::RmsNorm;
use rope::RotaryEmbedding;

/// Full LLaMA-style transformer model.
pub struct Model {
    pub config: LlmConfig,
    pub embedding: Embedding,
    pub layers: Vec<TransformerBlock>,
    pub final_norm: RmsNorm,
    pub lm_head: Option<linear::Linear>,
    pub rope: RotaryEmbedding,
    pub kv_caches: KvCacheStore,
}

impl Model {
    /// Create a zeroed model from config (weights must be loaded separately).
    pub fn new(config: LlmConfig) -> Self {
        let rope = RotaryEmbedding::new(config.head_dim, config.max_seq_len, config.rope_theta);
        let embedding = Embedding::new(config.vocab_size, config.hidden_size);

        let mut layers = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            layers.push(TransformerBlock::new(&config));
        }

        let final_norm = RmsNorm::new(config.hidden_size, config.rms_norm_eps);

        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(linear::Linear::new(config.hidden_size, config.vocab_size))
        };

        let kv_caches = KvCacheStore::new(
            config.num_layers,
            config.num_kv_heads,
            config.max_seq_len,
            config.head_dim,
        );

        Self {
            config,
            embedding,
            layers,
            final_norm,
            lm_head,
            rope,
            kv_caches,
        }
    }

    /// Reset all KV caches (e.g., for a new generation).
    pub fn reset_kv_caches(&mut self) {
        self.kv_caches.clear();
    }

    /// Decode a single token, returning logits over the vocabulary.
    ///
    /// `position` is the sequence position for this token (0-indexed).
    pub fn forward_decode(&mut self, token_id: u32, position: usize) -> Vec<f32> {
        let hidden_size = self.config.hidden_size;
        let vocab_size = self.config.vocab_size;

        // Embed the token
        let mut hidden = vec![0.0f32; hidden_size];
        self.embedding.forward(token_id, &mut hidden);

        // Scratch buffers
        let mut norm_buf = vec![0.0f32; hidden_size];

        for layer_idx in 0..self.config.num_layers {
            // Pre-attention norm
            norm_buf.copy_from_slice(&hidden);
            self.layers[layer_idx].attn_norm.forward(&mut norm_buf);

            // Attention
            let attn_out = self.layers[layer_idx].attention.forward(
                &norm_buf,
                position,
                &self.rope,
                self.kv_caches.layer_mut(layer_idx),
            );

            // Residual
            for (h, a) in hidden.iter_mut().zip(attn_out.iter()) {
                *h += a;
            }

            // Pre-MLP norm
            norm_buf.copy_from_slice(&hidden);
            self.layers[layer_idx].mlp_norm.forward(&mut norm_buf);

            // MLP
            let mlp_out = self.layers[layer_idx].mlp.forward(&norm_buf);

            // Residual
            for (h, m) in hidden.iter_mut().zip(mlp_out.iter()) {
                *h += m;
            }
        }

        // Final norm
        self.final_norm.forward(&mut hidden);

        // LM head
        let mut logits = vec![0.0f32; vocab_size];
        match &self.lm_head {
            Some(head) => head.forward(&hidden, &mut logits),
            None => self.embedding.lm_head_forward(&hidden, &mut logits),
        }

        logits
    }

    /// Prefill multiple tokens (initial prompt processing).
    /// Returns logits for the last token only.
    pub fn forward_prefill(&mut self, token_ids: &[u32]) -> Vec<f32> {
        let mut logits = Vec::new();
        for (pos, &token_id) in token_ids.iter().enumerate() {
            logits = self.forward_decode(token_id, pos);
        }
        logits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> LlmConfig {
        LlmConfig {
            vocab_size: 64,
            hidden_size: 32,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 8,
            intermediate_size: 64,
            num_layers: 2,
            max_seq_len: 32,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            tie_word_embeddings: true,
        }
    }

    #[test]
    fn test_model_creation() {
        let config = tiny_config();
        let model = Model::new(config);
        assert_eq!(model.layers.len(), 2);
        assert!(model.lm_head.is_none()); // tied embeddings
    }

    #[test]
    fn test_model_forward_produces_finite_logits() {
        let config = tiny_config();
        let mut model = Model::new(config.clone());

        // Set embedding weights to small non-zero values
        for i in 0..config.vocab_size {
            let row = model.embedding.weights.get_weights_mut(i);
            for (j, v) in row.iter_mut().enumerate() {
                *v = ((i * 7 + j * 3) % 17) as f32 * 0.01;
            }
        }

        // Set norm weights to 1.0
        for w in model.final_norm.weight.iter_mut() {
            *w = 1.0;
        }
        for layer in &mut model.layers {
            for w in layer.attn_norm.weight.iter_mut() {
                *w = 1.0;
            }
            for w in layer.mlp_norm.weight.iter_mut() {
                *w = 1.0;
            }
        }

        let logits = model.forward_decode(1, 0);
        assert_eq!(logits.len(), config.vocab_size);
        assert!(logits.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_prefill_decode_consistency() {
        let config = tiny_config();

        // Run prefill
        let mut model_prefill = Model::new(config.clone());
        // Set some weights
        for layer in &mut model_prefill.layers {
            for w in layer.attn_norm.weight.iter_mut() {
                *w = 1.0;
            }
            for w in layer.mlp_norm.weight.iter_mut() {
                *w = 1.0;
            }
        }
        for w in model_prefill.final_norm.weight.iter_mut() {
            *w = 1.0;
        }
        let _ = model_prefill.forward_prefill(&[0, 1, 2]);

        // Run sequential decode
        let mut model_decode = Model::new(config.clone());
        for layer in &mut model_decode.layers {
            for w in layer.attn_norm.weight.iter_mut() {
                *w = 1.0;
            }
            for w in layer.mlp_norm.weight.iter_mut() {
                *w = 1.0;
            }
        }
        for w in model_decode.final_norm.weight.iter_mut() {
            *w = 1.0;
        }
        let _ = model_decode.forward_decode(0, 0);
        let _ = model_decode.forward_decode(1, 1);
        let _ = model_decode.forward_decode(2, 2);

        // KV caches should have same content
        for layer_idx in 0..config.num_layers {
            let kv_pf = model_prefill.kv_caches.layer(layer_idx);
            let kv_dc = model_decode.kv_caches.layer(layer_idx);
            assert_eq!(kv_pf.current_len(), kv_dc.current_len());
            for h in 0..config.num_kv_heads {
                let k_pf = kv_pf.k_head_positions(h, 3);
                let k_dc = kv_dc.k_head_positions(h, 3);
                for (a, b) in k_pf.iter().zip(k_dc.iter()) {
                    assert!((a - b).abs() < 1e-6, "KV cache mismatch");
                }
            }
        }
    }
}
