pub mod calibrate;
pub mod predictor_store;
pub mod sparse_attention;
pub mod sparse_mlp;

use crate::model::Model;

use predictor_store::PredictorStore;

/// Model wrapper that applies Deja Vu sparsity during inference.
pub struct SparseModel {
    pub model: Model,
    pub predictor_store: PredictorStore,
    pub head_sparsity: f32,   // fraction of heads to keep (e.g. 0.5)
    pub neuron_sparsity: f32, // fraction of neurons to keep (e.g. 0.5)
}

impl SparseModel {
    pub fn new(model: Model, predictor_store: PredictorStore, head_sparsity: f32, neuron_sparsity: f32) -> Self {
        Self {
            model,
            predictor_store,
            head_sparsity,
            neuron_sparsity,
        }
    }

    /// Sparse decode: predict important heads/neurons, then only compute those.
    pub fn forward_decode_sparse(&mut self, token_id: u32, position: usize) -> Vec<f32> {
        let config = &self.model.config;
        let hidden_size = config.hidden_size;
        let vocab_size = config.vocab_size;

        let mut hidden = vec![0.0f32; hidden_size];
        self.model.embedding.forward(token_id, &mut hidden);

        let mut norm_buf = vec![0.0f32; hidden_size];

        for layer_idx in 0..config.num_layers {
            // Pre-attention norm
            norm_buf.copy_from_slice(&hidden);
            self.model.layers[layer_idx].attn_norm.forward(&mut norm_buf);

            // Predict important heads
            let num_heads_to_keep = ((config.num_heads as f32) * self.head_sparsity).ceil() as usize;
            let head_indices = self
                .predictor_store
                .predict_heads(layer_idx, &norm_buf, num_heads_to_keep);

            // Sparse attention (falls back to dense if no predictor)
            let attn_out = if head_indices.len() < config.num_heads {
                sparse_attention::forward_sparse(
                    &self.model.layers[layer_idx].attention,
                    &norm_buf,
                    position,
                    &self.model.rope,
                    self.model.kv_caches.layer_mut(layer_idx),
                    &head_indices,
                )
            } else {
                self.model.layers[layer_idx].attention.forward(
                    &norm_buf,
                    position,
                    &self.model.rope,
                    self.model.kv_caches.layer_mut(layer_idx),
                )
            };

            for (h, a) in hidden.iter_mut().zip(attn_out.iter()) {
                *h += a;
            }

            // Pre-MLP norm
            norm_buf.copy_from_slice(&hidden);
            self.model.layers[layer_idx].mlp_norm.forward(&mut norm_buf);

            // Predict important neurons
            let num_neurons_to_keep =
                ((config.intermediate_size as f32) * self.neuron_sparsity).ceil() as usize;
            let neuron_indices = self
                .predictor_store
                .predict_neurons(layer_idx, &norm_buf, num_neurons_to_keep);

            // Sparse MLP
            let mlp_out = if neuron_indices.len() < config.intermediate_size {
                sparse_mlp::forward_sparse(
                    &self.model.layers[layer_idx].mlp,
                    &norm_buf,
                    &neuron_indices,
                )
            } else {
                self.model.layers[layer_idx].mlp.forward(&norm_buf)
            };

            for (h, m) in hidden.iter_mut().zip(mlp_out.iter()) {
                *h += m;
            }
        }

        self.model.final_norm.forward(&mut hidden);

        let mut logits = vec![0.0f32; vocab_size];
        match &self.model.lm_head {
            Some(head) => head.forward(&hidden, &mut logits),
            None => self.model.embedding.lm_head_forward(&hidden, &mut logits),
        }

        logits
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::LlmConfig;

    #[test]
    fn test_sparse_model_creation() {
        let config = LlmConfig {
            vocab_size: 32,
            hidden_size: 16,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 4,
            intermediate_size: 32,
            num_layers: 2,
            max_seq_len: 16,
            tie_word_embeddings: true,
            ..LlmConfig::default()
        };
        let model = Model::new(config.clone());
        let store = PredictorStore::new(config.num_layers);
        let sparse = SparseModel::new(model, store, 0.5, 0.5);
        assert_eq!(sparse.head_sparsity, 0.5);
    }
}
