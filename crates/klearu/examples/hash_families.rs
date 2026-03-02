//! Demonstrates and compares all hash families in klearu.
//!
//! Creates SimHash, WtaHash, DwtaHash, MinHash, and SparseRandomProjection
//! with the same parameters, hashes dense and sparse vectors with each, and
//! checks collision properties.
//!
//! Run with:
//!   cargo run --example hash_families

use klearu::core::hash::{
    DwtaHash, HashFamily, MinHash, SimHash, SparseRandomProjection, WtaHash,
};
use klearu::core::tensor::SparseVector;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn main() {
    println!("=== Hash Families Comparison Demo ===\n");

    let dim = 64;
    let k = 4; // hash bits per table
    let num_tables = 8;
    let window_size = 8; // for WTA/DWTA
    let seed = 42u64;

    // -----------------------------------------------------------------------
    // 1. Create all five hash families
    // -----------------------------------------------------------------------
    let simhash = SimHash::new(dim, k, num_tables, seed);
    let wta = WtaHash::new(dim, k, num_tables, window_size, seed);
    let dwta = DwtaHash::new(dim, k, num_tables, window_size, seed);
    let minhash = MinHash::new(dim, k, num_tables, seed);
    let srp = SparseRandomProjection::new(dim, k, num_tables, 1.0 / 3.0, seed);

    let families: Vec<(&str, &dyn HashFamily)> = vec![
        ("SimHash", &simhash),
        ("WtaHash", &wta),
        ("DwtaHash", &dwta),
        ("MinHash", &minhash),
        ("SRP", &srp),
    ];

    println!(
        "Created 5 hash families: dim={}, k={}, tables={}\n",
        dim, k, num_tables
    );

    // -----------------------------------------------------------------------
    // 2. Generate a random dense vector and its sparse twin
    // -----------------------------------------------------------------------
    let mut rng = StdRng::seed_from_u64(100);
    let dense: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let sparse = SparseVector::from_dense(&dense);

    println!(
        "Test vector: dim={}, nnz={}\n",
        dense.len(),
        sparse.nnz()
    );

    // -----------------------------------------------------------------------
    // 3. Hash with each family, print results per table
    // -----------------------------------------------------------------------
    println!("Hash codes per table:");
    println!(
        "  {:>10} | {}",
        "Family",
        (0..num_tables)
            .map(|t| format!("T{}", t))
            .collect::<Vec<_>>()
            .join("  ")
    );
    println!("  {:-<10}-+-{:-<width$}", "", "", width = num_tables * 5);

    for (name, family) in &families {
        let hashes: Vec<u64> = (0..num_tables)
            .map(|t| family.hash_dense(&dense, t))
            .collect();
        println!(
            "  {:>10} | {}",
            name,
            hashes
                .iter()
                .map(|h| format!("{:>3}", h))
                .collect::<Vec<_>>()
                .join("  ")
        );
    }

    // -----------------------------------------------------------------------
    // 4. Verify dense and sparse hashing agree
    // -----------------------------------------------------------------------
    println!("\nDense vs Sparse hash agreement:");
    for (name, family) in &families {
        let mut all_agree = true;
        for t in 0..num_tables {
            let hd = family.hash_dense(&dense, t);
            let hs = family.hash_sparse(&sparse, t);
            if hd != hs {
                all_agree = false;
                println!("  {}: MISMATCH at table {} (dense={}, sparse={})", name, t, hd, hs);
            }
        }
        if all_agree {
            println!("  {}: all {} tables agree", name, num_tables);
        }
    }

    // -----------------------------------------------------------------------
    // 5. Collision rates: similar vs dissimilar vectors
    // -----------------------------------------------------------------------
    println!("\nCollision rate analysis:");

    // Similar vector: small perturbation
    let similar: Vec<f32> = dense
        .iter()
        .map(|&v| v + rng.gen_range(-0.05..0.05))
        .collect();

    // Dissimilar vector: completely random
    let dissimilar: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();

    for (name, family) in &families {
        let collisions_similar = (0..num_tables)
            .filter(|&t| family.hash_dense(&dense, t) == family.hash_dense(&similar, t))
            .count();
        let collisions_dissimilar = (0..num_tables)
            .filter(|&t| family.hash_dense(&dense, t) == family.hash_dense(&dissimilar, t))
            .count();

        println!(
            "  {:>10}: similar={}/{} ({:.0}%), dissimilar={}/{} ({:.0}%)",
            name,
            collisions_similar,
            num_tables,
            100.0 * collisions_similar as f64 / num_tables as f64,
            collisions_dissimilar,
            num_tables,
            100.0 * collisions_dissimilar as f64 / num_tables as f64,
        );
    }

    // -----------------------------------------------------------------------
    // 6. Hash range verification
    // -----------------------------------------------------------------------
    println!("\nHash range verification (all hashes < 2^k = {}):", 1u64 << k);
    for (name, family) in &families {
        let max_hash = 1u64 << family.k();
        let all_in_range = (0..num_tables)
            .all(|t| family.hash_dense(&dense, t) < max_hash);
        println!("  {}: k={}, max_value={}, all_in_range={}", name, family.k(), max_hash, all_in_range);
    }

    println!("\nDone.");
}
