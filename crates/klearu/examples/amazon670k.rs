//! Demonstrates extreme classification with many labels (Amazon-670K style).
//!
//! Creates a network with a large output layer (1000 labels) and uses TopK
//! sampling to avoid evaluating all neurons. Trains on synthetic sparse data
//! and shows that loss decreases over training steps.
//!
//! Run with:
//!   cargo run --example amazon670k

use klearu::core::config::*;
use klearu::core::data::Example;
use klearu::core::network::Network;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn main() {
    println!("=== Extreme Classification (Amazon-670K style) Demo ===\n");

    // -----------------------------------------------------------------------
    // 1. Configure network: 1000 -> 512 (hidden) -> 1000 (output, TopK=50)
    // -----------------------------------------------------------------------
    let config = SlideConfig {
        network: NetworkConfig {
            layers: vec![
                LayerConfig {
                    input_dim: 1000,
                    num_neurons: 512,
                    activation: ActivationType::Relu,
                    lsh: LshConfig {
                        num_tables: 20,
                        num_hashes: 6,
                        ..LshConfig::default()
                    },
                    sampling: SamplingType::TopK,
                    sampling_threshold: 2,
                    top_k: 50,
                    is_output: false,
                },
                LayerConfig {
                    input_dim: 512,
                    num_neurons: 1000,
                    activation: ActivationType::Softmax,
                    lsh: LshConfig {
                        num_tables: 20,
                        num_hashes: 6,
                        ..LshConfig::default()
                    },
                    sampling: SamplingType::TopK,
                    sampling_threshold: 2,
                    top_k: 50,
                    is_output: true,
                },
            ],
            optimizer: OptimizerType::Adam,
            learning_rate: 0.001,
            batch_size: 16,
            num_threads: 1,
        },
        seed: 7,
        hogwild: false,
    };

    let mut network = Network::new(config);
    println!("Network created:");
    println!(
        "  Hidden layer: {} neurons, top_k=50 (of 512)",
        network.layers[0].neurons.len()
    );
    println!(
        "  Output layer: {} neurons, top_k=50 (of 1000)",
        network.layers[1].neurons.len()
    );

    // -----------------------------------------------------------------------
    // 2. Generate synthetic sparse training data
    //    Each example has ~2% non-zero features and 1-3 active labels.
    // -----------------------------------------------------------------------
    let mut rng = StdRng::seed_from_u64(999);
    let num_examples = 100;

    let mut examples: Vec<Example> = Vec::with_capacity(num_examples);
    for _ in 0..num_examples {
        let mut features = vec![0.0f32; 1000];
        for f in &mut features {
            if rng.gen::<f32>() < 0.02 {
                *f = rng.gen_range(0.0..1.0);
            }
        }

        let num_labels = rng.gen_range(1u32..=3);
        let labels: Vec<u32> = (0..num_labels).map(|_| rng.gen_range(0u32..1000)).collect();

        examples.push(Example::new(features, labels));
    }

    println!("\nGenerated {} training examples (1000-dim, up to 1000 labels)", num_examples);

    // -----------------------------------------------------------------------
    // 3. Train a few steps, track that loss decreases
    // -----------------------------------------------------------------------
    let lr = 0.01;
    let num_steps = 20;
    let batch_size = 8;

    println!("\nTraining for {} steps (lr={}, batch_size={})...", num_steps, lr, batch_size);

    let mut losses: Vec<f32> = Vec::new();
    let mut batch_start = 0;

    for step in 0..num_steps {
        let batch_end = (batch_start + batch_size).min(examples.len());
        let batch_refs: Vec<&Example> = examples[batch_start..batch_end].iter().collect();
        let loss = network.train_step(&batch_refs, lr);
        losses.push(loss);

        if step % 5 == 0 || step == num_steps - 1 {
            println!("  Step {:>2}: loss = {:.4}", step, loss);
        }

        batch_start = (batch_start + batch_size) % examples.len();
    }

    let first_loss = losses[0];
    let last_loss = *losses.last().unwrap();
    println!(
        "\n  First loss: {:.4}, Last loss: {:.4} (ratio: {:.2}x)",
        first_loss,
        last_loss,
        last_loss / first_loss.max(1e-8)
    );

    // -----------------------------------------------------------------------
    // 4. Predict top-10 labels for a test input
    // -----------------------------------------------------------------------
    println!("\nPredicting top-10 labels on first training example...");
    let test_dense = examples[0].features.to_dense();
    let top10 = network.predict_top_k(&test_dense, 10);

    println!("  True labels: {:?}", examples[0].labels);
    println!("  Predicted top-10:");
    for (rank, (label, score)) in top10.iter().enumerate() {
        let marker = if examples[0].labels.contains(label) {
            " <-- true label"
        } else {
            ""
        };
        println!("    #{:>2}: label={:>4}, score={:.6}{}", rank + 1, label, score, marker);
    }

    println!("\nDone.");
}
