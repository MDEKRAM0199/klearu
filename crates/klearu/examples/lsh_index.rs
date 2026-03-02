//! Demonstrates the LSH index: inserting neurons, querying, and rebuilding.
//!
//! Creates an LSH index with SimHash, inserts 100 neurons with random weights,
//! queries for similar neurons, and demonstrates query_union vs query_with_counts.
//!
//! Run with:
//!   cargo run --example lsh_index

use klearu::core::config::*;
use klearu::core::lsh::{create_lsh_index, LshIndexTrait};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn main() {
    println!("=== LSH Index Demo ===\n");

    let dim = 64;

    // -----------------------------------------------------------------------
    // 1. Create an LSH index with SimHash
    // -----------------------------------------------------------------------
    let lsh_config = LshConfig {
        hash_function: HashFunctionType::SimHash,
        bucket_type: BucketType::Fifo,
        num_tables: 10,
        range_pow: 6,
        num_hashes: 6,
        bucket_capacity: 64,
        rebuild_interval_base: 100,
        rebuild_decay: 0.1,
    };

    let mut index = create_lsh_index(&lsh_config, dim, 42);
    println!(
        "Created LSH index: {} tables, SimHash, FIFO buckets",
        index.num_tables()
    );

    // -----------------------------------------------------------------------
    // 2. Insert 100 neurons with random weights
    // -----------------------------------------------------------------------
    let mut rng = StdRng::seed_from_u64(77);
    let num_neurons = 100u32;

    let mut all_weights: Vec<(u32, Vec<f32>)> = Vec::with_capacity(num_neurons as usize);

    for id in 0..num_neurons {
        let weights: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        index.insert(id, &weights);
        all_weights.push((id, weights));
    }

    println!("Inserted {} neurons with {}-dim random weights\n", num_neurons, dim);

    // -----------------------------------------------------------------------
    // 3. Query for similar neurons to neuron 0
    // -----------------------------------------------------------------------
    let query_weights = &all_weights[0].1;
    println!("Querying for neurons similar to neuron 0...");

    let union_results = index.query_union(query_weights);
    println!(
        "  query_union: found {} candidate neurons",
        union_results.len()
    );
    println!(
        "  Neuron 0 in results: {}",
        union_results.contains(&0)
    );

    // Show first few candidates
    let display_count = union_results.len().min(10);
    println!(
        "  First {} candidates: {:?}",
        display_count,
        &union_results[..display_count]
    );

    // -----------------------------------------------------------------------
    // 4. query_with_counts: see how many tables each candidate matched
    // -----------------------------------------------------------------------
    println!("\nquery_with_counts for neuron 0's weights:");
    let mut counted_results = index.query_with_counts(query_weights);
    // Sort by count descending for display.
    counted_results.sort_by(|a, b| b.count.cmp(&a.count));

    let display_count = counted_results.len().min(15);
    println!("  Top {} candidates (by table count):", display_count);
    for candidate in &counted_results[..display_count] {
        let self_marker = if candidate.neuron_id == 0 { " <-- self" } else { "" };
        println!(
            "    neuron {:>3}: matched in {}/{} tables{}",
            candidate.neuron_id,
            candidate.count,
            index.num_tables(),
            self_marker,
        );
    }

    // -----------------------------------------------------------------------
    // 5. Query with a random vector (should find fewer matches)
    // -----------------------------------------------------------------------
    let random_query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let random_union = index.query_union(&random_query);
    let random_counts = index.query_with_counts(&random_query);
    let max_count = random_counts.iter().map(|c| c.count).max().unwrap_or(0);

    println!("\nRandom query vector:");
    println!("  Candidates found: {}", random_union.len());
    println!("  Max table matches: {}", max_count);

    // -----------------------------------------------------------------------
    // 6. Rebuild the index and query again
    // -----------------------------------------------------------------------
    println!("\nRebuilding index from scratch...");
    index.rebuild(&all_weights);
    println!("  Rebuild complete.");

    let post_rebuild = index.query_union(query_weights);
    println!(
        "  Post-rebuild query for neuron 0: {} candidates, self found: {}",
        post_rebuild.len(),
        post_rebuild.contains(&0)
    );

    // -----------------------------------------------------------------------
    // 7. Demonstrate clear
    // -----------------------------------------------------------------------
    index.clear();
    let after_clear = index.query_union(query_weights);
    println!("\nAfter clear: {} candidates (expected 0)", after_clear.len());

    println!("\nDone.");
}
