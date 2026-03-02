//! Demonstrates data loading from LibSVM format and batch iteration.
//!
//! Creates a temporary LibSVM-format file, loads it with LibSvmDataset,
//! iterates over batches, and prints statistics.
//!
//! Run with:
//!   cargo run --example data_loading

use klearu::core::data::{BatchIterator, Dataset, LibSvmDataset};

use std::io::Write;

fn main() {
    println!("=== Data Loading Demo ===\n");

    // -----------------------------------------------------------------------
    // 1. Create a temporary LibSVM-format file
    // -----------------------------------------------------------------------
    let tmp_dir = std::env::temp_dir();
    let tmp_path = tmp_dir.join("klearu_example_data.libsvm");
    let tmp_path_str = tmp_path.to_str().expect("invalid temp path");

    // LibSVM format: label feature_idx:value feature_idx:value ...
    // Multi-label: label1,label2 feature_idx:value ...
    let libsvm_data = "\
0 0:1.0 3:0.5 7:2.0
1 1:3.0 4:1.5
2 2:0.8 5:1.2 9:0.3
0,2 0:2.0 6:1.0 8:0.7
1 3:1.0 7:0.5 9:1.8
0 1:0.3 4:2.1 6:0.9
2 0:1.5 2:0.6 5:1.1 8:0.4
1,2 3:0.7 7:1.3
0 0:0.9 1:1.1 9:2.5
1 2:1.4 4:0.8 6:1.6 8:0.2
0 5:0.5 7:1.0 9:0.8
2 0:2.2 3:1.8 6:0.3
";

    {
        let mut file = std::fs::File::create(&tmp_path).expect("failed to create temp file");
        file.write_all(libsvm_data.as_bytes())
            .expect("failed to write temp file");
    }
    println!("Created temporary LibSVM file: {}", tmp_path_str);

    // -----------------------------------------------------------------------
    // 2. Load with LibSvmDataset
    // -----------------------------------------------------------------------
    let feature_dim = 10;
    let num_labels = 3;

    let dataset = LibSvmDataset::load(tmp_path_str, feature_dim, num_labels)
        .expect("failed to load dataset");

    println!(
        "Loaded dataset: {} examples, feature_dim={}, num_labels={}",
        dataset.len(),
        dataset.feature_dim(),
        dataset.num_labels()
    );

    // -----------------------------------------------------------------------
    // 3. Inspect individual examples
    // -----------------------------------------------------------------------
    println!("\nFirst 3 examples:");
    for i in 0..3.min(dataset.len()) {
        let example = dataset.get(i);
        println!(
            "  Example {}: labels={:?}, nnz={}, features={:?}",
            i,
            example.labels,
            example.features.nnz(),
            example.features.iter().collect::<Vec<_>>()
        );
    }

    // -----------------------------------------------------------------------
    // 4. Iterate with BatchIterator
    // -----------------------------------------------------------------------
    let batch_size = 4;
    let seed = 42u64;

    println!(
        "\nBatch iteration (batch_size={}, seed={}):",
        batch_size, seed
    );

    let iter = BatchIterator::new(&dataset, batch_size, seed);
    let mut total_examples = 0;
    let mut batch_count = 0;

    for batch in iter {
        batch_count += 1;
        let batch_labels: Vec<Vec<u32>> = batch.iter().map(|ex| ex.labels.clone()).collect();
        let avg_nnz: f32 =
            batch.iter().map(|ex| ex.features.nnz() as f32).sum::<f32>() / batch.len() as f32;

        println!(
            "  Batch {}: size={}, avg_nnz={:.1}, labels={:?}",
            batch_count,
            batch.len(),
            avg_nnz,
            batch_labels
        );
        total_examples += batch.len();
    }

    println!(
        "\n  Total: {} batches, {} examples",
        batch_count, total_examples
    );
    assert_eq!(
        total_examples,
        dataset.len(),
        "all examples should be covered"
    );

    // -----------------------------------------------------------------------
    // 5. Second epoch with reset
    // -----------------------------------------------------------------------
    println!("\nSecond epoch (different seed):");
    let mut iter2 = BatchIterator::new(&dataset, batch_size, seed);

    // Consume first epoch.
    let epoch1_labels: Vec<Vec<u32>> = (&mut iter2)
        .flat_map(|batch| batch.iter().map(|ex| ex.labels.clone()).collect::<Vec<_>>())
        .collect();

    // Reset for second epoch with a different seed.
    iter2.reset(99);
    let epoch2_labels: Vec<Vec<u32>> = iter2
        .flat_map(|batch| batch.iter().map(|ex| ex.labels.clone()).collect::<Vec<_>>())
        .collect();

    println!(
        "  Epoch 1 label order (first 5): {:?}",
        &epoch1_labels[..5.min(epoch1_labels.len())]
    );
    println!(
        "  Epoch 2 label order (first 5): {:?}",
        &epoch2_labels[..5.min(epoch2_labels.len())]
    );
    println!(
        "  Orders differ: {}",
        epoch1_labels != epoch2_labels
    );

    // -----------------------------------------------------------------------
    // 6. Dataset statistics
    // -----------------------------------------------------------------------
    println!("\nDataset statistics:");
    let total_nnz: usize = (0..dataset.len())
        .map(|i| dataset.get(i).features.nnz())
        .sum();
    let avg_nnz = total_nnz as f64 / dataset.len() as f64;
    let avg_sparsity = 1.0 - avg_nnz / feature_dim as f64;

    let multi_label_count = (0..dataset.len())
        .filter(|&i| dataset.get(i).labels.len() > 1)
        .count();

    println!("  Total non-zeros: {}", total_nnz);
    println!("  Avg nnz per example: {:.1}", avg_nnz);
    println!("  Avg sparsity: {:.1}%", avg_sparsity * 100.0);
    println!(
        "  Multi-label examples: {} / {}",
        multi_label_count,
        dataset.len()
    );

    // Clean up temp file.
    let _ = std::fs::remove_file(&tmp_path);
    println!("\nCleaned up temp file.");
    println!("Done.");
}
