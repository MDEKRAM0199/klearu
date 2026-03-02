use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub max_seq_len: usize,
    pub layer_norm_eps: f64,
    /// Fraction of heads to keep active (0.0 to 1.0).
    pub head_sparsity: f32,
    /// Fraction of MLP neurons to keep active (0.0 to 1.0).
    pub mlp_sparsity: f32,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            hidden_size: 768,
            num_heads: 12,
            head_dim: 64,
            intermediate_size: 3072,
            num_layers: 12,
            max_seq_len: 512,
            layer_norm_eps: 1e-5,
            head_sparsity: 0.5,
            mlp_sparsity: 0.1,
        }
    }
}

/// Simple layer normalization.
#[derive(Debug, Clone)]
pub struct LayerNorm {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    pub eps: f64,
}

impl LayerNorm {
    pub fn new(size: usize, eps: f64) -> Self {
        Self {
            weight: vec![1.0; size],
            bias: vec![0.0; size],
            eps,
        }
    }

    /// Apply layer normalization to a vector.
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let n = input.len() as f64;
        let mean: f64 = input.iter().map(|&x| x as f64).sum::<f64>() / n;
        let var: f64 = input
            .iter()
            .map(|&x| {
                let d = x as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / n;
        let std = (var + self.eps).sqrt();

        input
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let normalized = ((x as f64 - mean) / std) as f32;
                normalized * self.weight[i] + self.bias[i]
            })
            .collect()
    }
}

/// Linear projection layer.
#[derive(Debug, Clone)]
pub struct Linear {
    /// Weight matrix stored as [out_features][in_features] (row-major).
    pub weight: Vec<Vec<f32>>,
    pub bias: Vec<f32>,
    pub in_features: usize,
    pub out_features: usize,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, seed: u64) -> Self {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        use rand_distr::{Distribution, Normal};

        let mut rng = StdRng::seed_from_u64(seed);
        let std = (2.0 / (in_features + out_features) as f64).sqrt();
        let dist = Normal::new(0.0, std).unwrap();

        let weight: Vec<Vec<f32>> = (0..out_features)
            .map(|_| {
                (0..in_features)
                    .map(|_| dist.sample(&mut rng) as f32)
                    .collect()
            })
            .collect();
        let bias = vec![0.0; out_features];

        Self {
            weight,
            bias,
            in_features,
            out_features,
        }
    }

    /// Full forward pass: output = weight * input + bias
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        self.weight
            .iter()
            .enumerate()
            .map(|(i, row)| {
                let dot: f32 = row.iter().zip(input.iter()).map(|(w, x)| w * x).sum();
                dot + self.bias[i]
            })
            .collect()
    }

    /// Sparse forward: only compute selected output indices.
    pub fn forward_sparse(&self, input: &[f32], indices: &[usize]) -> Vec<(usize, f32)> {
        indices
            .iter()
            .map(|&i| {
                let dot: f32 = self.weight[i].iter().zip(input.iter()).map(|(w, x)| w * x).sum();
                (i, dot + self.bias[i])
            })
            .collect()
    }
}

/// A single transformer layer with optional sparse attention and MLP.
#[derive(Debug, Clone)]
pub struct TransformerLayer {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    pub gate_proj: Linear,
    pub up_proj: Linear,
    pub down_proj: Linear,
    pub attn_norm: LayerNorm,
    pub mlp_norm: LayerNorm,
    pub config: TransformerConfig,
}

impl TransformerLayer {
    pub fn new(config: TransformerConfig, seed: u64) -> Self {
        let h = config.hidden_size;
        let nh = config.num_heads * config.head_dim;
        let inter = config.intermediate_size;

        Self {
            q_proj: Linear::new(h, nh, seed),
            k_proj: Linear::new(h, nh, seed + 1),
            v_proj: Linear::new(h, nh, seed + 2),
            o_proj: Linear::new(nh, h, seed + 3),
            gate_proj: Linear::new(h, inter, seed + 4),
            up_proj: Linear::new(h, inter, seed + 5),
            down_proj: Linear::new(inter, h, seed + 6),
            attn_norm: LayerNorm::new(h, config.layer_norm_eps),
            mlp_norm: LayerNorm::new(h, config.layer_norm_eps),
            config,
        }
    }

    /// Dense forward pass (no sparsity, for reference/comparison).
    pub fn forward_dense(&self, input: &[f32]) -> Vec<f32> {
        // Attention block
        let normed = self.attn_norm.forward(input);
        let q = self.q_proj.forward(&normed);
        let k = self.k_proj.forward(&normed);
        let v = self.v_proj.forward(&normed);

        // Simple single-token attention (no sequence dimension for simplicity)
        // score = q*k / sqrt(head_dim), output = score * v
        let scale = (self.config.head_dim as f32).sqrt();
        let score: f32 = q.iter().zip(k.iter()).map(|(a, b)| a * b).sum::<f32>() / scale;
        let score = 1.0 / (1.0 + (-score).exp()); // sigmoid for single token
        let attn_out: Vec<f32> = v.iter().map(|&vi| vi * score).collect();

        let proj = self.o_proj.forward(&attn_out);
        let residual: Vec<f32> = input.iter().zip(proj.iter()).map(|(a, b)| a + b).collect();

        // MLP block (SwiGLU)
        let normed2 = self.mlp_norm.forward(&residual);
        let gate = self.gate_proj.forward(&normed2);
        let up = self.up_proj.forward(&normed2);

        // SwiGLU: gate * sigmoid(gate) * up
        let mlp_hidden: Vec<f32> = gate
            .iter()
            .zip(up.iter())
            .map(|(g, u)| {
                let silu = g / (1.0 + (-g).exp());
                silu * u
            })
            .collect();

        let mlp_out = self.down_proj.forward(&mlp_hidden);
        residual
            .iter()
            .zip(mlp_out.iter())
            .map(|(a, b)| a + b)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_norm_shape() {
        let ln = LayerNorm::new(64, 1e-5);
        let input = vec![1.0f32; 64];
        let output = ln.forward(&input);
        assert_eq!(output.len(), 64);
    }

    #[test]
    fn test_layer_norm_normalizes() {
        let ln = LayerNorm::new(128, 1e-5);
        // Create non-uniform input
        let input: Vec<f32> = (0..128).map(|i| i as f32 * 0.1 - 6.4).collect();
        let output = ln.forward(&input);

        // Check mean is approximately 0
        let mean: f64 = output.iter().map(|&x| x as f64).sum::<f64>() / output.len() as f64;
        assert!(
            mean.abs() < 1e-5,
            "Mean should be ~0 after layer norm, got {}",
            mean
        );

        // Check variance is approximately 1
        let var: f64 = output
            .iter()
            .map(|&x| {
                let d = x as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / output.len() as f64;
        assert!(
            (var - 1.0).abs() < 1e-3,
            "Variance should be ~1 after layer norm, got {}",
            var
        );
    }

    #[test]
    fn test_linear_forward_shape() {
        let linear = Linear::new(32, 64, 42);
        let input = vec![1.0f32; 32];
        let output = linear.forward(&input);
        assert_eq!(output.len(), 64);
    }

    #[test]
    fn test_linear_sparse_forward() {
        let linear = Linear::new(32, 64, 42);
        let input = vec![1.0f32; 32];
        let indices = vec![0, 5, 10, 63];
        let sparse_out = linear.forward_sparse(&input, &indices);
        let dense_out = linear.forward(&input);

        assert_eq!(sparse_out.len(), indices.len());
        for (idx, val) in &sparse_out {
            assert!(
                (val - dense_out[*idx]).abs() < 1e-6,
                "Sparse and dense should match for index {}",
                idx
            );
        }
    }

    #[test]
    fn test_transformer_layer_dense_forward_shape() {
        let config = TransformerConfig {
            hidden_size: 64,
            num_heads: 4,
            head_dim: 16,
            intermediate_size: 128,
            num_layers: 1,
            max_seq_len: 32,
            layer_norm_eps: 1e-5,
            head_sparsity: 0.5,
            mlp_sparsity: 0.1,
        };
        let layer = TransformerLayer::new(config.clone(), 42);
        let input = vec![0.1f32; config.hidden_size];
        let output = layer.forward_dense(&input);
        assert_eq!(output.len(), config.hidden_size);
    }

    #[test]
    fn test_transformer_layer_dense_forward_deterministic() {
        let config = TransformerConfig {
            hidden_size: 64,
            num_heads: 4,
            head_dim: 16,
            intermediate_size: 128,
            num_layers: 1,
            max_seq_len: 32,
            layer_norm_eps: 1e-5,
            head_sparsity: 0.5,
            mlp_sparsity: 0.1,
        };
        let layer = TransformerLayer::new(config.clone(), 42);
        let input = vec![0.1f32; config.hidden_size];
        let out1 = layer.forward_dense(&input);
        let out2 = layer.forward_dense(&input);
        assert_eq!(out1, out2, "Dense forward should be deterministic");
    }
}
