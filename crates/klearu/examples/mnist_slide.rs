//! Demonstrates basic SLIDE training on a synthetic "MNIST-like" dataset.
//!
//! Creates a two-layer network (784 -> 128 -> 10), generates random sparse
//! training data that mimics MNIST dimensions, trains for a few epochs, and
//! runs prediction on a test sample.
//!
//! Run with:
//!   cargo run --example mnist_slide

use klearu::core::config::*;
use klearu::core::data::Example;
use klearu::core::network::Network;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn main() {
    println!("=== SLIDE MNIST-like Demo ===\n");

    // -----------------------------------------------------------------------
    // 1. Configure a two-layer SLIDE network: 784 -> 128 (ReLU) -> 10 (Softmax)
    // -----------------------------------------------------------------------
    let config = SlideConfig {
        network: NetworkConfig {
            layers: vec![
                LayerConfig {
                    input_dim: 784,
                    num_neurons: 128,
                    activation: ActivationType::Relu,
                    lsh: LshConfig {
                        num_tables: 10,
                        num_hashes: 6,
                        ..LshConfig::default()
                    },
                    sampling: SamplingType::TopK,
                    sampling_threshold: 2,
                    top_k: 32,
                    is_output: false,
                },
                LayerConfig {
                    input_dim: 128,
                    num_neurons: 10,
                    activation: ActivationType::Softmax,
                    lsh: LshConfig {
                        num_tables: 10,
                        num_hashes: 4,
                        ..LshConfig::default()
                    },
                    sampling: SamplingType::TopK,
                    sampling_threshold: 1,
                    top_k: 10, // activate all output neurons for 10-class problem
                    is_output: true,
                },
            ],
            optimizer: OptimizerType::Adam,
            learning_rate: 0.001,
            batch_size: 32,
            num_threads: 1,
        },
        seed: 42,
        hogwild: false,
    };

    let mut network = Network::new(config);
    println!("Network created: {} layers", network.layers.len());
    println!(
        "  Layer 0: {} neurons (input_dim=784)",
        network.layers[0].neurons.len()
    );
    println!(
        "  Layer 1: {} neurons (input_dim=128)",
        network.layers[1].neurons.len()
    );

    // -----------------------------------------------------------------------
    // 2. Generate synthetic training data (sparse 784-dim vectors, labels 0-9)
    // -----------------------------------------------------------------------
    let mut rng = StdRng::seed_from_u64(123);
    let num_train = 200;
    let num_test = 5;

    let mut train_examples: Vec<Example> = Vec::with_capacity(num_train);
    for _ in 0..num_train {
        let label = rng.gen_range(0u32..10);
        // Create a sparse vector: ~5% of pixels are non-zero.
        let mut features = vec![0.0f32; 784];
        for f in &mut features {
            if rng.gen::<f32>() < 0.05 {
                *f = rng.gen_range(0.0..1.0);
            }
        }
        train_examples.push(Example::new(features, vec![label]));
    }

    let mut test_examples: Vec<Example> = Vec::with_capacity(num_test);
    for _ in 0..num_test {
        let label = rng.gen_range(0u32..10);
        let mut features = vec![0.0f32; 784];
        for f in &mut features {
            if rng.gen::<f32>() < 0.05 {
                *f = rng.gen_range(0.0..1.0);
            }
        }
        test_examples.push(Example::new(features, vec![label]));
    }

    println!(
        "\nGenerated {} training examples, {} test examples",
        train_examples.len(),
        test_examples.len()
    );

    // -----------------------------------------------------------------------
    // 3. Train for a few epochs
    // -----------------------------------------------------------------------
    let lr = 0.01;
    let epochs = 5;
    let batch_size = 32;

    println!("\nTraining for {} epochs (lr={}, batch_size={})...", epochs, lr, batch_size);
    for epoch in 0..epochs {
        let mut epoch_loss = 0.0f32;
        let mut num_batches = 0;

        for batch_start in (0..train_examples.len()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(train_examples.len());
            let batch_refs: Vec<&Example> = train_examples[batch_start..batch_end].iter().collect();
            let loss = network.train_step(&batch_refs, lr);
            epoch_loss += loss;
            num_batches += 1;
        }

        let avg_loss = epoch_loss / num_batches as f32;
        println!("  Epoch {}: avg loss = {:.4}", epoch + 1, avg_loss);
    }

    // -----------------------------------------------------------------------
    // 4. Predict on test samples, showing top-k predictions
    // -----------------------------------------------------------------------
    println!("\nPredictions on test samples:");
    for (i, example) in test_examples.iter().enumerate() {
        let dense_input = example.features.to_dense();
        let top_k = network.predict_top_k(&dense_input, 3);

        print!("  Test {}: true_label={}, top-3 predictions: ", i, example.labels[0]);
        for (label, score) in &top_k {
            print!("[{}: {:.4}] ", label, score);
        }
        println!();
    }

    println!("\nDone.");
}
