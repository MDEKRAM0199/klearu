use crate::optim::OptimizerState;
use crate::tensor::SparseVector;

/// A single neuron with a weight vector, bias, and optimizer state.
#[derive(Debug, Clone)]
pub struct Neuron {
    pub weights: Vec<f32>,
    pub bias: f32,
    pub optimizer_state: OptimizerState,
}

impl Neuron {
    /// Create a new neuron with Xavier initialization.
    ///
    /// Weights are drawn from N(0, sqrt(2/dim)).
    pub fn new(dim: usize) -> Self {
        Self::new_with_seed(dim, rand::random::<u64>())
    }

    /// Create a new neuron with deterministic Xavier initialization.
    ///
    /// Weights are drawn from N(0, sqrt(2/dim)) using the given seed.
    pub fn new_with_seed(dim: usize, seed: u64) -> Self {
        use rand::SeedableRng;
        use rand_distr::{Distribution, Normal};

        let stddev = if dim > 0 {
            (2.0 / dim as f64).sqrt() as f32
        } else {
            1.0
        };

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0f32, stddev).unwrap();

        let weights: Vec<f32> = (0..dim).map(|_| normal.sample(&mut rng)).collect();

        Self {
            weights,
            bias: 0.0,
            optimizer_state: OptimizerState::None,
        }
    }

    /// Compute the forward pass: dot(weights, input) + bias.
    #[inline]
    pub fn forward(&self, input: &[f32]) -> f32 {
        debug_assert_eq!(
            self.weights.len(),
            input.len(),
            "input dim {} != weight dim {}",
            input.len(),
            self.weights.len(),
        );
        let dot: f32 = self
            .weights
            .iter()
            .zip(input.iter())
            .map(|(w, x)| w * x)
            .sum();
        dot + self.bias
    }

    /// Compute the forward pass for a sparse input.
    #[inline]
    pub fn forward_sparse(&self, input: &SparseVector) -> f32 {
        input.dot_dense(&self.weights) + self.bias
    }

    /// The dimensionality (number of weights) of this neuron.
    #[inline]
    pub fn dim(&self) -> usize {
        self.weights.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neuron_new_with_seed_deterministic() {
        let n1 = Neuron::new_with_seed(10, 42);
        let n2 = Neuron::new_with_seed(10, 42);
        assert_eq!(n1.weights, n2.weights);
        assert_eq!(n1.bias, n2.bias);
    }

    #[test]
    fn test_neuron_new_with_seed_different_seeds() {
        let n1 = Neuron::new_with_seed(10, 42);
        let n2 = Neuron::new_with_seed(10, 43);
        assert_ne!(n1.weights, n2.weights);
    }

    #[test]
    fn test_neuron_dim() {
        let n = Neuron::new_with_seed(16, 0);
        assert_eq!(n.dim(), 16);
    }

    #[test]
    fn test_neuron_dim_zero() {
        let n = Neuron::new_with_seed(0, 0);
        assert_eq!(n.dim(), 0);
    }

    #[test]
    fn test_neuron_xavier_scale() {
        // Xavier init: weights ~ N(0, sqrt(2/dim))
        // For dim=100, stddev = sqrt(2/100) = sqrt(0.02) ~ 0.1414
        let n = Neuron::new_with_seed(100, 42);
        let mean: f32 = n.weights.iter().sum::<f32>() / n.weights.len() as f32;
        let variance: f32 = n
            .weights
            .iter()
            .map(|w| (w - mean).powi(2))
            .sum::<f32>()
            / n.weights.len() as f32;
        let stddev = variance.sqrt();

        // Expected stddev ~ 0.1414
        assert!(
            (stddev - 0.1414).abs() < 0.05,
            "stddev {stddev} not close to expected 0.1414"
        );
    }

    #[test]
    fn test_neuron_forward() {
        let mut n = Neuron::new_with_seed(3, 0);
        n.weights = vec![1.0, 2.0, 3.0];
        n.bias = 0.5;

        let input = [1.0, 1.0, 1.0];
        // dot = 1*1 + 2*1 + 3*1 = 6, + 0.5 = 6.5
        assert!((n.forward(&input) - 6.5).abs() < 1e-6);
    }

    #[test]
    fn test_neuron_forward_zero_input() {
        let mut n = Neuron::new_with_seed(3, 0);
        n.weights = vec![1.0, 2.0, 3.0];
        n.bias = 0.5;

        let input = [0.0, 0.0, 0.0];
        assert!((n.forward(&input) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_neuron_forward_sparse() {
        let mut n = Neuron::new_with_seed(5, 0);
        n.weights = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        n.bias = 1.0;

        let sparse = SparseVector::from_pairs(5, vec![(1, 3.0), (3, 2.0)]);
        // dot = 2*3 + 4*2 = 14, + 1.0 = 15.0
        assert!((n.forward_sparse(&sparse) - 15.0).abs() < 1e-6);
    }

    #[test]
    fn test_neuron_forward_sparse_empty() {
        let mut n = Neuron::new_with_seed(5, 0);
        n.weights = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        n.bias = 1.0;

        let sparse = SparseVector::new(5);
        assert!((n.forward_sparse(&sparse) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_neuron_forward_dense_sparse_agree() {
        let n = Neuron::new_with_seed(10, 99);

        let dense = vec![0.0, 1.0, 0.0, -2.0, 0.0, 0.5, 0.0, 0.0, 3.0, 0.0];
        let sparse = SparseVector::from_dense(&dense);

        let f_dense = n.forward(&dense);
        let f_sparse = n.forward_sparse(&sparse);

        assert!(
            (f_dense - f_sparse).abs() < 1e-5,
            "dense={f_dense}, sparse={f_sparse}"
        );
    }

    #[test]
    fn test_neuron_initial_bias_zero() {
        let n = Neuron::new_with_seed(10, 0);
        assert_eq!(n.bias, 0.0);
    }

    #[test]
    fn test_neuron_initial_optimizer_state_none() {
        let n = Neuron::new_with_seed(10, 0);
        assert!(matches!(n.optimizer_state, OptimizerState::None));
    }

    #[test]
    fn test_neuron_clone() {
        let n1 = Neuron::new_with_seed(5, 42);
        let n2 = n1.clone();
        assert_eq!(n1.weights, n2.weights);
        assert_eq!(n1.bias, n2.bias);
    }
}
