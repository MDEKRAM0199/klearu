use crate::transformer::{Linear, TransformerConfig};

/// Sparse attention that computes only selected heads per input.
pub struct SparseAttention {
    pub config: TransformerConfig,
}

impl SparseAttention {
    pub fn new(config: TransformerConfig) -> Self {
        Self { config }
    }

    /// Forward pass computing only the selected heads.
    /// `input`: hidden state [hidden_size]
    /// `q_proj`, `k_proj`, `v_proj`, `o_proj`: projection layers
    /// `selected_heads`: which attention heads to compute
    /// Returns: attention output [hidden_size]
    pub fn forward(
        &self,
        input: &[f32],
        q_proj: &Linear,
        k_proj: &Linear,
        v_proj: &Linear,
        o_proj: &Linear,
        selected_heads: &[usize],
    ) -> Vec<f32> {
        let head_dim = self.config.head_dim;
        let num_heads = self.config.num_heads;

        // Compute Q, K, V only for selected heads
        let mut q_selected = vec![0.0f32; num_heads * head_dim];
        let mut k_selected = vec![0.0f32; num_heads * head_dim];
        let mut v_selected = vec![0.0f32; num_heads * head_dim];

        for &head in selected_heads {
            let start = head * head_dim;
            let end = start + head_dim;
            let indices: Vec<usize> = (start..end).collect();

            for (idx, val) in q_proj.forward_sparse(input, &indices) {
                q_selected[idx] = val;
            }
            for (idx, val) in k_proj.forward_sparse(input, &indices) {
                k_selected[idx] = val;
            }
            for (idx, val) in v_proj.forward_sparse(input, &indices) {
                v_selected[idx] = val;
            }
        }

        // Compute attention per selected head (single-token simplified)
        let scale = (head_dim as f32).sqrt();
        let mut attn_output = vec![0.0f32; num_heads * head_dim];

        for &head in selected_heads {
            let start = head * head_dim;
            let end = start + head_dim;

            let score: f32 = q_selected[start..end]
                .iter()
                .zip(k_selected[start..end].iter())
                .map(|(q, k)| q * k)
                .sum::<f32>()
                / scale;
            let score = 1.0 / (1.0 + (-score).exp());

            for i in start..end {
                attn_output[i] = v_selected[i] * score;
            }
        }

        // Project back through o_proj (dense since output is full hidden_size)
        o_proj.forward(&attn_output)
    }

    /// Calculate the fraction of FLOPs saved by sparse attention.
    pub fn flops_savings(&self, num_selected: usize) -> f32 {
        1.0 - (num_selected as f32 / self.config.num_heads as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transformer::TransformerConfig;

    fn make_config() -> TransformerConfig {
        TransformerConfig {
            hidden_size: 64,
            num_heads: 4,
            head_dim: 16,
            intermediate_size: 128,
            num_layers: 1,
            max_seq_len: 32,
            layer_norm_eps: 1e-5,
            head_sparsity: 0.5,
            mlp_sparsity: 0.1,
        }
    }

    #[test]
    fn test_sparse_attention_output_shape() {
        let config = make_config();
        let sa = SparseAttention::new(config.clone());

        let q_proj = Linear::new(config.hidden_size, config.num_heads * config.head_dim, 10);
        let k_proj = Linear::new(config.hidden_size, config.num_heads * config.head_dim, 11);
        let v_proj = Linear::new(config.hidden_size, config.num_heads * config.head_dim, 12);
        let o_proj = Linear::new(config.num_heads * config.head_dim, config.hidden_size, 13);

        let input = vec![0.5f32; config.hidden_size];
        let selected_heads = vec![0, 2];
        let output = sa.forward(&input, &q_proj, &k_proj, &v_proj, &o_proj, &selected_heads);

        assert_eq!(output.len(), config.hidden_size);
    }

    #[test]
    fn test_sparse_matches_dense_all_heads() {
        let config = make_config();
        let sa = SparseAttention::new(config.clone());

        let q_proj = Linear::new(config.hidden_size, config.num_heads * config.head_dim, 10);
        let k_proj = Linear::new(config.hidden_size, config.num_heads * config.head_dim, 11);
        let v_proj = Linear::new(config.hidden_size, config.num_heads * config.head_dim, 12);
        let o_proj = Linear::new(config.num_heads * config.head_dim, config.hidden_size, 13);

        let input = vec![0.5f32; config.hidden_size];
        let all_heads: Vec<usize> = (0..config.num_heads).collect();

        let sparse_out =
            sa.forward(&input, &q_proj, &k_proj, &v_proj, &o_proj, &all_heads);

        // Compute dense: full Q, K, V projections
        let q_full = q_proj.forward(&input);
        let k_full = k_proj.forward(&input);
        let v_full = v_proj.forward(&input);

        let head_dim = config.head_dim;
        let scale = (head_dim as f32).sqrt();
        let mut attn_output = vec![0.0f32; config.num_heads * head_dim];
        for head in 0..config.num_heads {
            let start = head * head_dim;
            let end = start + head_dim;
            let score: f32 = q_full[start..end]
                .iter()
                .zip(k_full[start..end].iter())
                .map(|(q, k)| q * k)
                .sum::<f32>()
                / scale;
            let score = 1.0 / (1.0 + (-score).exp());
            for i in start..end {
                attn_output[i] = v_full[i] * score;
            }
        }
        let dense_out = o_proj.forward(&attn_output);

        for (i, (s, d)) in sparse_out.iter().zip(dense_out.iter()).enumerate() {
            assert!(
                (s - d).abs() < 1e-5,
                "Mismatch at index {}: sparse={}, dense={}",
                i,
                s,
                d
            );
        }
    }

    #[test]
    fn test_flops_savings() {
        let config = make_config();
        let sa = SparseAttention::new(config.clone());

        // Using 2 of 4 heads => 50% savings
        let savings = sa.flops_savings(2);
        assert!((savings - 0.5).abs() < 1e-6);

        // Using all heads => 0% savings
        let savings = sa.flops_savings(4);
        assert!(savings.abs() < 1e-6);
    }
}
