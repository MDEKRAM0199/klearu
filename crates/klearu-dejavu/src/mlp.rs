use crate::transformer::Linear;

/// Sparse MLP that only computes selected neurons in the intermediate layer.
/// For SwiGLU architecture: gate_proj, up_proj -> activation -> down_proj.
pub struct SparseMlp {
    pub intermediate_size: usize,
    pub hidden_size: usize,
}

impl SparseMlp {
    pub fn new(hidden_size: usize, intermediate_size: usize) -> Self {
        Self {
            intermediate_size,
            hidden_size,
        }
    }

    /// Forward pass computing only selected intermediate neurons.
    /// `input`: [hidden_size]
    /// `gate_proj`, `up_proj`, `down_proj`: linear layers
    /// `selected_neurons`: which intermediate neurons to compute
    /// Returns: [hidden_size] output
    pub fn forward(
        &self,
        input: &[f32],
        gate_proj: &Linear,
        up_proj: &Linear,
        down_proj: &Linear,
        selected_neurons: &[usize],
    ) -> Vec<f32> {
        // Compute gate and up projections only for selected neurons
        let gate_vals = gate_proj.forward_sparse(input, selected_neurons);
        let up_vals = up_proj.forward_sparse(input, selected_neurons);

        // SwiGLU activation for selected neurons
        let mut sparse_hidden = vec![0.0f32; self.intermediate_size];
        for (&(gi, gv), &(_, uv)) in gate_vals.iter().zip(up_vals.iter()) {
            let silu = gv / (1.0 + (-gv).exp());
            sparse_hidden[gi] = silu * uv;
        }

        // down_proj: only accumulate contributions from selected neurons
        // output[j] = sum_i(down_proj.weight[j][i] * sparse_hidden[i]) for i in selected
        let mut output = vec![0.0f32; self.hidden_size];
        for &(idx, _) in &gate_vals {
            let val = sparse_hidden[idx];
            if val != 0.0 {
                for j in 0..self.hidden_size {
                    output[j] += down_proj.weight[j][idx] * val;
                }
            }
        }
        for j in 0..self.hidden_size {
            output[j] += down_proj.bias[j];
        }

        output
    }

    /// Calculate the fraction of FLOPs saved.
    pub fn flops_savings(&self, num_selected: usize) -> f32 {
        1.0 - (num_selected as f32 / self.intermediate_size as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_mlp_output_shape() {
        let hidden_size = 64;
        let intermediate_size = 128;
        let smlp = SparseMlp::new(hidden_size, intermediate_size);

        let gate_proj = Linear::new(hidden_size, intermediate_size, 20);
        let up_proj = Linear::new(hidden_size, intermediate_size, 21);
        let down_proj = Linear::new(intermediate_size, hidden_size, 22);

        let input = vec![0.5f32; hidden_size];
        let selected = vec![0, 10, 50, 100, 127];
        let output = smlp.forward(&input, &gate_proj, &up_proj, &down_proj, &selected);

        assert_eq!(output.len(), hidden_size);
    }

    #[test]
    fn test_sparse_matches_dense_all_neurons() {
        let hidden_size = 32;
        let intermediate_size = 64;
        let smlp = SparseMlp::new(hidden_size, intermediate_size);

        let gate_proj = Linear::new(hidden_size, intermediate_size, 20);
        let up_proj = Linear::new(hidden_size, intermediate_size, 21);
        let down_proj = Linear::new(intermediate_size, hidden_size, 22);

        let input = vec![0.3f32; hidden_size];
        let all_neurons: Vec<usize> = (0..intermediate_size).collect();

        let sparse_out =
            smlp.forward(&input, &gate_proj, &up_proj, &down_proj, &all_neurons);

        // Dense forward: compute full gate, up, silu, down
        let gate_full = gate_proj.forward(&input);
        let up_full = up_proj.forward(&input);
        let mlp_hidden: Vec<f32> = gate_full
            .iter()
            .zip(up_full.iter())
            .map(|(g, u)| {
                let silu = g / (1.0 + (-g).exp());
                silu * u
            })
            .collect();
        let dense_out = down_proj.forward(&mlp_hidden);

        for (i, (s, d)) in sparse_out.iter().zip(dense_out.iter()).enumerate() {
            assert!(
                (s - d).abs() < 1e-4,
                "Mismatch at index {}: sparse={}, dense={}",
                i,
                s,
                d
            );
        }
    }

    #[test]
    fn test_mlp_flops_savings() {
        let smlp = SparseMlp::new(64, 256);

        // Using 64 of 256 neurons => 75% savings
        let savings = smlp.flops_savings(64);
        assert!((savings - 0.75).abs() < 1e-6);

        // Using all neurons => 0% savings
        let savings = smlp.flops_savings(256);
        assert!(savings.abs() < 1e-6);
    }
}
