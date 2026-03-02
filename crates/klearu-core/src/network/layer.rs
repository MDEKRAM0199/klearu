use crate::bucket::NeuronId;
use crate::config::{ActivationType, LayerConfig, SamplingType};
use crate::lsh::{create_lsh_index, LshIndexTrait};
use crate::optim::Optimizer;
use crate::sampling::{SamplingStrategy, ThresholdSampling, TopKSampling, VanillaSampling};

use super::neuron::Neuron;

/// A single layer in the SLIDE network.
pub struct Layer {
    pub neurons: Vec<Neuron>,
    pub lsh_index: Box<dyn LshIndexTrait>,
    pub sampling: Box<dyn SamplingStrategy>,
    pub config: LayerConfig,
}

/// Result of a forward pass through a layer.
#[derive(Debug, Clone)]
pub struct LayerOutput {
    /// Dense output vector (full dimension, zeros for inactive neurons).
    pub activations: Vec<f32>,
    /// Which neurons were activated.
    pub active_neurons: Vec<NeuronId>,
    /// Pre-activation values for active neurons only (in order of active_neurons).
    pub pre_activations: Vec<f32>,
}

impl Layer {
    /// Create a new layer from configuration.
    pub fn new(config: LayerConfig, seed: u64) -> Self {
        let num_neurons = config.num_neurons;
        let input_dim = config.input_dim;

        // Create neurons with Xavier initialization, each with a different seed.
        let neurons: Vec<Neuron> = (0..num_neurons)
            .map(|i| Neuron::new_with_seed(input_dim, seed.wrapping_add(i as u64)))
            .collect();

        // Create LSH index and insert all neurons.
        let mut lsh_index = create_lsh_index(&config.lsh, input_dim, seed.wrapping_add(1_000_000));

        for (i, neuron) in neurons.iter().enumerate() {
            lsh_index.insert(i as NeuronId, &neuron.weights);
        }

        // Create sampling strategy based on config.
        let sampling: Box<dyn SamplingStrategy> = create_sampling_strategy(
            config.sampling,
            config.top_k,
            config.sampling_threshold,
        );

        Self {
            neurons,
            lsh_index,
            sampling,
            config,
        }
    }

    /// Forward pass through this layer.
    ///
    /// Uses LSH to select active neurons, computes their outputs, and applies
    /// the activation function.
    pub fn forward(&self, input: &[f32]) -> LayerOutput {
        let num_neurons = self.neurons.len();

        // Query LSH for candidates.
        let candidates = self.lsh_index.query(input);

        // Convert LshCandidate to (NeuronId, u32) for sampling.
        let candidate_pairs: Vec<(NeuronId, u32)> = candidates
            .iter()
            .map(|c| (c.neuron_id, c.count))
            .collect();

        // Apply sampling strategy.
        let mut active_neurons = self.sampling.select(&candidate_pairs, num_neurons);

        // If no neurons were selected (e.g., LSH found nothing), fall back to
        // activating all neurons to avoid producing an all-zero output.
        if active_neurons.is_empty() {
            active_neurons = (0..num_neurons as NeuronId).collect();
        }

        // Filter out any out-of-bounds neuron IDs.
        active_neurons.retain(|&id| (id as usize) < num_neurons);

        // Compute pre-activations for active neurons.
        let pre_activations: Vec<f32> = active_neurons
            .iter()
            .map(|&id| self.neurons[id as usize].forward(input))
            .collect();

        // Apply activation function.
        let post_activations = match self.config.activation {
            ActivationType::Softmax => {
                // Softmax over active neurons only, using log-sum-exp trick.
                apply_softmax(&pre_activations)
            }
            act => {
                pre_activations
                    .iter()
                    .map(|&x| apply_activation(x, act))
                    .collect()
            }
        };

        // Build dense output vector.
        let mut activations = vec![0.0f32; num_neurons];
        for (i, &id) in active_neurons.iter().enumerate() {
            activations[id as usize] = post_activations[i];
        }

        LayerOutput {
            activations,
            active_neurons,
            pre_activations,
        }
    }

    /// Backward pass: compute gradients for active neurons, update weights, and
    /// return gradient w.r.t. input for backpropagation to the previous layer.
    pub fn backward(
        &mut self,
        input: &[f32],
        output: &LayerOutput,
        grad_output: &[f32],
        optimizer: &dyn Optimizer,
        lr: f64,
        step: u64,
    ) -> Vec<f32> {
        let input_dim = self.config.input_dim;
        let mut grad_input = vec![0.0f32; input_dim];

        if output.active_neurons.is_empty() {
            return grad_input;
        }

        // Compute activation derivatives for each active neuron.
        let local_grads: Vec<f32> = match self.config.activation {
            ActivationType::Softmax => {
                // For softmax, use Jacobian-based gradient.
                // d(loss)/d(pre_act_i) = sum_j (grad_output_j * d(softmax_j)/d(pre_act_i))
                // For cross-entropy + softmax: d/d(z_i) = softmax(z_i) - y_i, but
                // here we receive generic grad_output and compute the Jacobian.
                let softmax_out = apply_softmax(&output.pre_activations);
                let n = softmax_out.len();
                let mut result = vec![0.0f32; n];

                for i in 0..n {
                    let neuron_id = output.active_neurons[i] as usize;
                    let go = if neuron_id < grad_output.len() {
                        grad_output[neuron_id]
                    } else {
                        0.0
                    };
                    for j in 0..n {
                        let s_j = softmax_out[j];
                        let neuron_id_j = output.active_neurons[j] as usize;
                        let go_j = if neuron_id_j < grad_output.len() {
                            grad_output[neuron_id_j]
                        } else {
                            0.0
                        };
                        if i == j {
                            result[i] += go_j * s_j * (1.0 - s_j);
                        } else {
                            result[i] += go_j * (-softmax_out[i] * s_j);
                        }
                    }
                    // Add the direct component.
                    let _ = go; // Jacobian already accounts for all outputs.
                }
                result
            }
            act => {
                output
                    .active_neurons
                    .iter()
                    .enumerate()
                    .map(|(i, &id)| {
                        let pre_act = output.pre_activations[i];
                        let post_act = output.activations[id as usize];
                        let go = if (id as usize) < grad_output.len() {
                            grad_output[id as usize]
                        } else {
                            0.0
                        };
                        go * activation_derivative(pre_act, post_act, act)
                    })
                    .collect()
            }
        };

        // Update weights and compute input gradients.
        for (i, &neuron_id) in output.active_neurons.iter().enumerate() {
            let idx = neuron_id as usize;
            let local_grad = local_grads[i];

            // Compute weight gradients: dL/dw = local_grad * input
            let grad_weights: Vec<f32> = input.iter().map(|&x| local_grad * x).collect();
            let grad_bias = local_grad;

            // Accumulate input gradient: dL/dinput += local_grad * weights
            let neuron = &self.neurons[idx];
            for (j, &w) in neuron.weights.iter().enumerate() {
                grad_input[j] += local_grad * w;
            }

            // Update neuron weights using optimizer.
            let neuron = &mut self.neurons[idx];
            optimizer.update(
                &mut neuron.weights,
                &grad_weights,
                &mut neuron.bias,
                grad_bias,
                lr,
                step,
                &mut neuron.optimizer_state,
            );
        }

        grad_input
    }

    /// Rebuild the LSH index by rehashing all neuron weights.
    pub fn rebuild_index(&mut self) {
        self.lsh_index.clear();
        for (i, neuron) in self.neurons.iter().enumerate() {
            self.lsh_index.insert(i as NeuronId, &neuron.weights);
        }
    }
}

/// Create a sampling strategy from configuration parameters.
fn create_sampling_strategy(
    sampling_type: SamplingType,
    top_k: usize,
    threshold: usize,
) -> Box<dyn SamplingStrategy> {
    match sampling_type {
        SamplingType::Vanilla => Box::new(VanillaSampling),
        SamplingType::TopK => Box::new(TopKSampling::new(top_k)),
        SamplingType::Threshold => Box::new(ThresholdSampling::new(threshold as u32)),
    }
}

/// Apply an activation function to a single value.
fn apply_activation(x: f32, act: ActivationType) -> f32 {
    match act {
        ActivationType::Relu => x.max(0.0),
        ActivationType::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        ActivationType::Tanh => x.tanh(),
        ActivationType::Identity => x,
        ActivationType::Softmax => {
            // Softmax is handled at the vector level; for single values, identity.
            x
        }
    }
}

/// Compute the derivative of an activation function.
///
/// `pre_act` is the pre-activation value and `post_act` is the post-activation value.
fn activation_derivative(pre_act: f32, post_act: f32, act: ActivationType) -> f32 {
    match act {
        ActivationType::Relu => {
            if pre_act > 0.0 {
                1.0
            } else {
                0.0
            }
        }
        ActivationType::Sigmoid => post_act * (1.0 - post_act),
        ActivationType::Tanh => 1.0 - post_act * post_act,
        ActivationType::Identity => 1.0,
        ActivationType::Softmax => {
            // Softmax derivative is Jacobian-based, handled separately.
            1.0
        }
    }
}

/// Apply softmax to a vector using the log-sum-exp trick for numerical stability.
fn apply_softmax(values: &[f32]) -> Vec<f32> {
    if values.is_empty() {
        return Vec::new();
    }

    // Find max for numerical stability.
    let max_val = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Compute exp(x - max) for each value.
    let exps: Vec<f32> = values.iter().map(|&x| (x - max_val).exp()).collect();

    // Sum of exponentials.
    let sum: f32 = exps.iter().sum();

    // Normalize.
    if sum > 0.0 {
        exps.iter().map(|&e| e / sum).collect()
    } else {
        // Fallback: uniform distribution.
        let uniform = 1.0 / values.len() as f32;
        vec![uniform; values.len()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::LshConfig;

    fn make_hidden_config(input_dim: usize, num_neurons: usize) -> LayerConfig {
        LayerConfig {
            input_dim,
            num_neurons,
            activation: ActivationType::Relu,
            lsh: LshConfig {
                num_tables: 3,
                num_hashes: 4,
                ..LshConfig::default()
            },
            sampling: SamplingType::TopK,
            sampling_threshold: 1,
            top_k: num_neurons, // activate all for testing
            is_output: false,
        }
    }

    fn make_output_config(input_dim: usize, num_neurons: usize) -> LayerConfig {
        LayerConfig {
            input_dim,
            num_neurons,
            activation: ActivationType::Softmax,
            lsh: LshConfig {
                num_tables: 3,
                num_hashes: 4,
                ..LshConfig::default()
            },
            sampling: SamplingType::TopK,
            sampling_threshold: 1,
            top_k: num_neurons, // activate all for testing
            is_output: true,
        }
    }

    #[test]
    fn test_layer_new() {
        let config = make_hidden_config(4, 8);
        let layer = Layer::new(config, 42);
        assert_eq!(layer.neurons.len(), 8);
        for n in &layer.neurons {
            assert_eq!(n.dim(), 4);
        }
    }

    #[test]
    fn test_layer_forward_hidden() {
        let config = make_hidden_config(4, 8);
        let layer = Layer::new(config, 42);
        let input = vec![1.0, 0.5, -0.3, 0.2];
        let output = layer.forward(&input);

        // Output activations should have the right dimension.
        assert_eq!(output.activations.len(), 8);
        // At least some neurons should be active.
        assert!(!output.active_neurons.is_empty());
        // All ReLU outputs should be >= 0.
        for &a in &output.activations {
            assert!(a >= 0.0, "ReLU output should be non-negative, got {a}");
        }
    }

    #[test]
    fn test_layer_forward_output_softmax() {
        let config = make_output_config(4, 3);
        let layer = Layer::new(config, 42);
        let input = vec![1.0, 0.5, -0.3, 0.2];
        let output = layer.forward(&input);

        assert_eq!(output.activations.len(), 3);

        // Softmax outputs of active neurons should sum to ~1.
        let active_sum: f32 = output
            .active_neurons
            .iter()
            .map(|&id| output.activations[id as usize])
            .sum();
        assert!(
            (active_sum - 1.0).abs() < 1e-5,
            "softmax should sum to 1.0, got {active_sum}"
        );

        // All softmax values should be in [0, 1].
        for &a in &output.activations {
            assert!(a >= 0.0 && a <= 1.0, "softmax value {a} not in [0,1]");
        }
    }

    #[test]
    fn test_layer_backward() {
        let config = make_hidden_config(4, 8);
        let mut layer = Layer::new(config, 42);

        // Initialize optimizer states.
        let optimizer = crate::optim::Sgd::new(0.0);
        for neuron in &mut layer.neurons {
            neuron.optimizer_state = optimizer.create_state(neuron.dim());
        }

        let input = vec![1.0, 0.5, -0.3, 0.2];
        let output = layer.forward(&input);

        let grad_output = vec![0.1; 8];
        let grad_input = layer.backward(&input, &output, &grad_output, &optimizer, 0.01, 1);

        assert_eq!(grad_input.len(), 4);
        // Gradient should be non-trivial (at least some non-zero values).
        let any_nonzero = grad_input.iter().any(|&g| g.abs() > 1e-10);
        assert!(any_nonzero, "gradient should have some non-zero values");
    }

    #[test]
    fn test_layer_rebuild_index() {
        let config = make_hidden_config(4, 8);
        let mut layer = Layer::new(config, 42);

        // Modify a neuron's weights.
        layer.neurons[0].weights = vec![10.0, 10.0, 10.0, 10.0];

        // Rebuild index.
        layer.rebuild_index();

        // The layer should still work after rebuild.
        let input = vec![1.0, 0.5, -0.3, 0.2];
        let output = layer.forward(&input);
        assert_eq!(output.activations.len(), 8);
    }

    #[test]
    fn test_apply_activation_relu() {
        assert_eq!(apply_activation(1.0, ActivationType::Relu), 1.0);
        assert_eq!(apply_activation(-1.0, ActivationType::Relu), 0.0);
        assert_eq!(apply_activation(0.0, ActivationType::Relu), 0.0);
    }

    #[test]
    fn test_apply_activation_sigmoid() {
        let s = apply_activation(0.0, ActivationType::Sigmoid);
        assert!((s - 0.5).abs() < 1e-6);

        let s_pos = apply_activation(100.0, ActivationType::Sigmoid);
        assert!((s_pos - 1.0).abs() < 1e-6);

        let s_neg = apply_activation(-100.0, ActivationType::Sigmoid);
        assert!(s_neg.abs() < 1e-6);
    }

    #[test]
    fn test_apply_activation_tanh() {
        let t = apply_activation(0.0, ActivationType::Tanh);
        assert!(t.abs() < 1e-6);

        let t_pos = apply_activation(100.0, ActivationType::Tanh);
        assert!((t_pos - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_apply_activation_identity() {
        assert_eq!(apply_activation(42.0, ActivationType::Identity), 42.0);
        assert_eq!(apply_activation(-3.25, ActivationType::Identity), -3.25);
    }

    #[test]
    fn test_activation_derivative_relu() {
        assert_eq!(
            activation_derivative(1.0, 1.0, ActivationType::Relu),
            1.0
        );
        assert_eq!(
            activation_derivative(-1.0, 0.0, ActivationType::Relu),
            0.0
        );
        assert_eq!(
            activation_derivative(0.0, 0.0, ActivationType::Relu),
            0.0
        );
    }

    #[test]
    fn test_activation_derivative_sigmoid() {
        // sigmoid(0) = 0.5, derivative = 0.5 * 0.5 = 0.25
        let d = activation_derivative(0.0, 0.5, ActivationType::Sigmoid);
        assert!((d - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_activation_derivative_tanh() {
        // tanh(0) = 0, derivative = 1 - 0^2 = 1
        let d = activation_derivative(0.0, 0.0, ActivationType::Tanh);
        assert!((d - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_apply_softmax() {
        let values = vec![1.0, 2.0, 3.0];
        let result = apply_softmax(&values);

        assert_eq!(result.len(), 3);
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Values should be in increasing order.
        assert!(result[0] < result[1]);
        assert!(result[1] < result[2]);
    }

    #[test]
    fn test_apply_softmax_empty() {
        let result = apply_softmax(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_apply_softmax_single() {
        let result = apply_softmax(&[5.0]);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_apply_softmax_numerical_stability() {
        // Large values should not cause overflow.
        let values = vec![1000.0, 1001.0, 1002.0];
        let result = apply_softmax(&values);
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(result.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_apply_softmax_negative_values() {
        let values = vec![-1000.0, -999.0, -998.0];
        let result = apply_softmax(&values);
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(result.iter().all(|&x| x.is_finite()));
    }
}
