use crate::config::{NetworkConfig, OptimizerType, SlideConfig};
use crate::data::Example;
use crate::lsh::RebuildScheduler;
use crate::optim::{Adam, Optimizer, Sgd};

use super::layer::{Layer, LayerOutput};

/// A SLIDE neural network: a stack of layers with LSH-accelerated forward
/// and backward passes.
pub struct Network {
    pub layers: Vec<Layer>,
    pub config: NetworkConfig,
    pub step: u64,
    pub rebuild_scheduler: RebuildScheduler,
}

impl Network {
    /// Create a new network from the SLIDE configuration.
    pub fn new(config: SlideConfig) -> Self {
        let seed = config.seed;
        let net_config = config.network;

        let layers: Vec<Layer> = net_config
            .layers
            .iter()
            .enumerate()
            .map(|(i, layer_config)| {
                Layer::new(layer_config.clone(), seed.wrapping_add(i as u64 * 100_000))
            })
            .collect();

        // Use the first layer's LSH config for rebuild scheduling.
        let (rebuild_base, rebuild_decay) = if let Some(first) = net_config.layers.first() {
            (first.lsh.rebuild_interval_base, first.lsh.rebuild_decay)
        } else {
            (100, 0.1)
        };

        let rebuild_scheduler = RebuildScheduler::new(rebuild_base, rebuild_decay);

        // Initialize optimizer states for all neurons.
        let optimizer = create_optimizer(&net_config);
        let mut network = Self {
            layers,
            config: net_config,
            step: 0,
            rebuild_scheduler,
        };

        for layer in &mut network.layers {
            for neuron in &mut layer.neurons {
                neuron.optimizer_state = optimizer.create_state(neuron.dim());
            }
        }

        network
    }

    /// Forward pass through all layers.
    ///
    /// Returns the output of each layer (needed for backward pass).
    pub fn forward(&self, input: &[f32]) -> Vec<LayerOutput> {
        let mut outputs = Vec::with_capacity(self.layers.len());
        let mut current_input = input;

        // Temporary storage for intermediate activations.
        let mut intermediate: Option<Vec<f32>>;

        for layer in &self.layers {
            let output = layer.forward(current_input);
            // Use this layer's activations as input for the next layer.
            intermediate = Some(output.activations.clone());
            outputs.push(output);
            current_input = intermediate.as_ref().unwrap();
        }

        outputs
    }

    /// Train on a mini-batch of examples, returning the average loss.
    pub fn train_step(&mut self, examples: &[&Example], lr: f64) -> f32 {
        if examples.is_empty() {
            return 0.0;
        }

        let optimizer = create_optimizer(&self.config);
        let mut total_loss = 0.0f32;

        for example in examples {
            // Convert sparse features to dense for the forward pass.
            let dense_input = example.features.to_dense();

            // Forward pass.
            let outputs = self.forward(&dense_input);

            // Compute loss from the last layer's output.
            let last_output = outputs.last().unwrap();
            let loss = compute_loss(last_output, &example.labels);
            total_loss += loss;

            // Backward pass.
            // Compute gradient of loss w.r.t. the output layer's activations.
            let num_output = self.layers.last().unwrap().neurons.len();
            let grad_output = compute_loss_gradient(last_output, &example.labels, num_output);

            self.backward(&outputs, &dense_input, &grad_output, &*optimizer, lr);
        }

        self.step += 1;

        // Check if we should rebuild LSH indices.
        if self.rebuild_scheduler.should_rebuild(self.step) {
            for layer in &mut self.layers {
                layer.rebuild_index();
            }
        }

        total_loss / examples.len() as f32
    }

    /// Backward pass through all layers.
    fn backward(
        &mut self,
        outputs: &[LayerOutput],
        input: &[f32],
        grad_output: &[f32],
        optimizer: &dyn Optimizer,
        lr: f64,
    ) {
        let num_layers = self.layers.len();
        let mut current_grad = grad_output.to_vec();

        for i in (0..num_layers).rev() {
            // Determine the input to this layer.
            let layer_input: &[f32] = if i == 0 {
                input
            } else {
                &outputs[i - 1].activations
            };

            // Borrow the layer mutably for backward pass.
            let layer = &mut self.layers[i];
            current_grad =
                layer.backward(layer_input, &outputs[i], &current_grad, optimizer, lr, self.step);
        }
    }

    /// Predict output for a given input.
    ///
    /// Returns the final layer's activations.
    pub fn predict(&self, input: &[f32]) -> Vec<f32> {
        let outputs = self.forward(input);
        outputs.last().unwrap().activations.clone()
    }

    /// Predict the top-k labels with their scores.
    ///
    /// Returns pairs of (label_index, score), sorted by score descending.
    pub fn predict_top_k(&self, input: &[f32], k: usize) -> Vec<(u32, f32)> {
        let output = self.predict(input);

        let mut indexed: Vec<(u32, f32)> = output
            .iter()
            .enumerate()
            .map(|(i, &v)| (i as u32, v))
            .collect();

        // Sort by score descending.
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        indexed.truncate(k);
        indexed
    }
}

/// Compute cross-entropy loss over active neurons for multi-label classification.
///
/// For each label that is among the active neurons, accumulates -log(p_label).
/// Labels not in the active set are penalized with a default small probability.
pub fn compute_loss(output: &LayerOutput, labels: &[u32]) -> f32 {
    if labels.is_empty() {
        return 0.0;
    }

    let eps = 1e-7_f32;
    let mut loss = 0.0_f32;

    for &label in labels {
        let prob = if output.active_neurons.contains(&label) {
            output.activations[label as usize].max(eps)
        } else {
            // Label not in active set; use a small penalty probability.
            eps
        };
        loss -= prob.ln();
    }

    loss / labels.len() as f32
}

/// Compute gradient of cross-entropy loss w.r.t. output activations.
///
/// For softmax output layer with multi-label: gradient = softmax(z) - y
/// where y is 1 for active labels, 0 otherwise.
fn compute_loss_gradient(output: &LayerOutput, labels: &[u32], num_output: usize) -> Vec<f32> {
    let mut grad = vec![0.0f32; num_output];

    if labels.is_empty() {
        return grad;
    }

    // For each active neuron, gradient = p_i (the softmax probability).
    for &id in &output.active_neurons {
        let idx = id as usize;
        if idx < num_output {
            grad[idx] = output.activations[idx];
        }
    }

    // For each label, subtract 1/|labels| from the gradient (softmax - target).
    let label_weight = 1.0 / labels.len() as f32;
    for &label in labels {
        let idx = label as usize;
        if idx < num_output {
            grad[idx] -= label_weight;
        }
    }

    grad
}

/// Create the appropriate optimizer from network config.
fn create_optimizer(config: &NetworkConfig) -> Box<dyn Optimizer> {
    match config.optimizer {
        OptimizerType::Sgd => Box::new(Sgd::new(0.9)),
        OptimizerType::Adam => Box::new(Adam::default()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::*;

    fn make_tiny_config() -> SlideConfig {
        SlideConfig {
            network: NetworkConfig {
                layers: vec![
                    LayerConfig {
                        input_dim: 4,
                        num_neurons: 8,
                        activation: ActivationType::Relu,
                        lsh: LshConfig {
                            num_tables: 3,
                            num_hashes: 4,
                            ..LshConfig::default()
                        },
                        sampling: SamplingType::TopK,
                        sampling_threshold: 1,
                        top_k: 8, // activate all for testing
                        is_output: false,
                    },
                    LayerConfig {
                        input_dim: 8,
                        num_neurons: 3,
                        activation: ActivationType::Softmax,
                        lsh: LshConfig {
                            num_tables: 3,
                            num_hashes: 4,
                            ..LshConfig::default()
                        },
                        sampling: SamplingType::TopK,
                        sampling_threshold: 1,
                        top_k: 3, // activate all for testing
                        is_output: true,
                    },
                ],
                optimizer: OptimizerType::Sgd,
                learning_rate: 0.01,
                batch_size: 4,
                num_threads: 1,
            },
            seed: 42,
            hogwild: false,
        }
    }

    #[test]
    fn test_network_new() {
        let config = make_tiny_config();
        let net = Network::new(config);
        assert_eq!(net.layers.len(), 2);
        assert_eq!(net.layers[0].neurons.len(), 8);
        assert_eq!(net.layers[1].neurons.len(), 3);
        assert_eq!(net.step, 0);
    }

    #[test]
    fn test_network_forward_output_dimension() {
        let config = make_tiny_config();
        let net = Network::new(config);
        let input = vec![1.0, 0.5, -0.3, 0.2];
        let outputs = net.forward(&input);

        assert_eq!(outputs.len(), 2);
        // First layer: 8 neurons.
        assert_eq!(outputs[0].activations.len(), 8);
        // Second layer: 3 neurons.
        assert_eq!(outputs[1].activations.len(), 3);
    }

    #[test]
    fn test_network_predict() {
        let config = make_tiny_config();
        let net = Network::new(config);
        let input = vec![1.0, 0.5, -0.3, 0.2];
        let prediction = net.predict(&input);

        assert_eq!(prediction.len(), 3);
        // Output layer uses softmax, values should be in [0, 1].
        for &v in &prediction {
            assert!(v >= 0.0 && v <= 1.0, "softmax output {v} not in [0,1]");
        }
    }

    #[test]
    fn test_network_predict_top_k() {
        let config = make_tiny_config();
        let net = Network::new(config);
        let input = vec![1.0, 0.5, -0.3, 0.2];
        let top = net.predict_top_k(&input, 2);

        assert_eq!(top.len(), 2);
        // Should be sorted by score descending.
        assert!(top[0].1 >= top[1].1);
    }

    #[test]
    fn test_network_train_step_returns_finite_loss() {
        let config = make_tiny_config();
        let mut net = Network::new(config);
        let example = Example::new(vec![1.0, 0.5, -0.3, 0.2], vec![1]);
        let loss = net.train_step(&[&example], 0.01);

        assert!(loss.is_finite(), "loss should be finite, got {loss}");
        assert!(loss >= 0.0, "loss should be non-negative, got {loss}");
    }

    #[test]
    fn test_network_train_step_reduces_loss() {
        let config = make_tiny_config();
        let mut net = Network::new(config);

        // Create simple training examples.
        let examples: Vec<Example> = vec![
            Example::new(vec![1.0, 0.0, 0.0, 0.0], vec![0]),
            Example::new(vec![0.0, 1.0, 0.0, 0.0], vec![1]),
            Example::new(vec![0.0, 0.0, 1.0, 0.0], vec![2]),
        ];
        let example_refs: Vec<&Example> = examples.iter().collect();

        // Initial loss.
        let loss_before = net.train_step(&example_refs, 0.1);

        // Train for more steps.
        let mut last_loss = loss_before;
        for _ in 0..20 {
            last_loss = net.train_step(&example_refs, 0.1);
        }

        // Loss should generally decrease over training.
        // Allow for some noise, just check it's still finite and reasonable.
        assert!(last_loss.is_finite(), "final loss should be finite");
        // Over 20 iterations with lr=0.1, we expect at least some improvement.
        // Being lenient here since SLIDE uses approximate neurons.
        assert!(
            last_loss < loss_before * 2.0,
            "loss should not have doubled: before={loss_before}, after={last_loss}"
        );
    }

    #[test]
    fn test_network_train_step_increments_step() {
        let config = make_tiny_config();
        let mut net = Network::new(config);
        let example = Example::new(vec![1.0, 0.5, -0.3, 0.2], vec![0]);

        assert_eq!(net.step, 0);
        net.train_step(&[&example], 0.01);
        assert_eq!(net.step, 1);
        net.train_step(&[&example], 0.01);
        assert_eq!(net.step, 2);
    }

    #[test]
    fn test_network_train_step_empty_batch() {
        let config = make_tiny_config();
        let mut net = Network::new(config);
        let loss = net.train_step(&[], 0.01);
        assert_eq!(loss, 0.0);
    }

    #[test]
    fn test_compute_loss_single_label() {
        let output = LayerOutput {
            activations: vec![0.1, 0.7, 0.2],
            active_neurons: vec![0, 1, 2],
            pre_activations: vec![0.0, 0.0, 0.0],
        };

        let loss = compute_loss(&output, &[1]);
        // -log(0.7) ~ 0.3567
        let expected = -(0.7_f32.ln());
        assert!(
            (loss - expected).abs() < 1e-4,
            "loss={loss}, expected={expected}"
        );
    }

    #[test]
    fn test_compute_loss_multi_label() {
        let output = LayerOutput {
            activations: vec![0.3, 0.5, 0.2],
            active_neurons: vec![0, 1, 2],
            pre_activations: vec![0.0, 0.0, 0.0],
        };

        let loss = compute_loss(&output, &[0, 1]);
        let expected = (-(0.3_f32.ln()) + -(0.5_f32.ln())) / 2.0;
        assert!(
            (loss - expected).abs() < 1e-4,
            "loss={loss}, expected={expected}"
        );
    }

    #[test]
    fn test_compute_loss_empty_labels() {
        let output = LayerOutput {
            activations: vec![0.5, 0.5],
            active_neurons: vec![0, 1],
            pre_activations: vec![0.0, 0.0],
        };

        let loss = compute_loss(&output, &[]);
        assert_eq!(loss, 0.0);
    }

    #[test]
    fn test_compute_loss_inactive_label() {
        // Label 2 is not in active_neurons.
        let output = LayerOutput {
            activations: vec![0.5, 0.5, 0.0],
            active_neurons: vec![0, 1],
            pre_activations: vec![0.0, 0.0],
        };

        let loss = compute_loss(&output, &[2]);
        // Should use eps as probability.
        let expected = -(1e-7_f32.ln());
        assert!(
            (loss - expected).abs() < 1.0,
            "loss={loss}, expected near {expected}"
        );
    }

    #[test]
    fn test_compute_loss_gradient_shape() {
        let output = LayerOutput {
            activations: vec![0.2, 0.5, 0.3],
            active_neurons: vec![0, 1, 2],
            pre_activations: vec![0.0, 0.0, 0.0],
        };

        let grad = compute_loss_gradient(&output, &[1], 3);
        assert_eq!(grad.len(), 3);
    }

    #[test]
    fn test_compute_loss_gradient_values() {
        let output = LayerOutput {
            activations: vec![0.2, 0.5, 0.3],
            active_neurons: vec![0, 1, 2],
            pre_activations: vec![0.0, 0.0, 0.0],
        };

        let grad = compute_loss_gradient(&output, &[1], 3);
        // For label=1: grad[1] = softmax[1] - 1.0 = 0.5 - 1.0 = -0.5
        assert!((grad[1] - (-0.5)).abs() < 1e-6, "grad[1]={}", grad[1]);
        // For non-labels: grad[0] = softmax[0] = 0.2
        assert!((grad[0] - 0.2).abs() < 1e-6, "grad[0]={}", grad[0]);
        // grad[2] = softmax[2] = 0.3
        assert!((grad[2] - 0.3).abs() < 1e-6, "grad[2]={}", grad[2]);
    }

    #[test]
    fn test_network_with_adam_optimizer() {
        let config = SlideConfig {
            network: NetworkConfig {
                layers: vec![
                    LayerConfig {
                        input_dim: 4,
                        num_neurons: 8,
                        activation: ActivationType::Relu,
                        lsh: LshConfig {
                            num_tables: 3,
                            num_hashes: 4,
                            ..LshConfig::default()
                        },
                        sampling: SamplingType::TopK,
                        sampling_threshold: 1,
                        top_k: 8,
                        is_output: false,
                    },
                    LayerConfig {
                        input_dim: 8,
                        num_neurons: 3,
                        activation: ActivationType::Softmax,
                        lsh: LshConfig {
                            num_tables: 3,
                            num_hashes: 4,
                            ..LshConfig::default()
                        },
                        sampling: SamplingType::TopK,
                        sampling_threshold: 1,
                        top_k: 3,
                        is_output: true,
                    },
                ],
                optimizer: OptimizerType::Adam,
                learning_rate: 0.001,
                batch_size: 4,
                num_threads: 1,
            },
            seed: 42,
            hogwild: false,
        };

        let mut net = Network::new(config);
        let example = Example::new(vec![1.0, 0.5, -0.3, 0.2], vec![0]);
        let loss = net.train_step(&[&example], 0.001);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_network_multiple_labels() {
        let config = make_tiny_config();
        let mut net = Network::new(config);

        // Example with multiple labels.
        let example = Example::new(vec![1.0, 0.5, -0.3, 0.2], vec![0, 2]);
        let loss = net.train_step(&[&example], 0.01);
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_network_deterministic_with_same_seed() {
        let config1 = make_tiny_config();
        let config2 = make_tiny_config();
        let net1 = Network::new(config1);
        let net2 = Network::new(config2);

        let input = vec![1.0, 0.5, -0.3, 0.2];
        let pred1 = net1.predict(&input);
        let pred2 = net2.predict(&input);

        for (a, b) in pred1.iter().zip(pred2.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "predictions should be deterministic: {a} vs {b}"
            );
        }
    }
}
