//! Core LSH index data structure.
//!
//! Generic over the hash-function family `H: HashFamily`, and uses an enum
//! `BucketStorage` to support both FIFO and reservoir bucket types without
//! doubling the number of concrete types.

use std::collections::HashMap;

use crate::bucket::{Bucket, FifoBucket, NeuronId, ReservoirBucket};
use crate::config::{BucketType, HashFunctionType, LshConfig};
use crate::hash::{DwtaHash, HashFamily, MinHash, SimHash, SparseRandomProjection, WtaHash};
use crate::tensor::SparseVector;

use super::LshCandidate;

// ---------------------------------------------------------------------------
// LshIndexTrait -- object-safe trait for type-erased LSH index
// ---------------------------------------------------------------------------

/// Object-safe trait for LSH index operations.
pub trait LshIndexTrait: Send + Sync {
    /// Insert a neuron's dense weight vector into the index.
    fn insert(&mut self, id: NeuronId, dense_weights: &[f32]);

    /// Insert a neuron's sparse weight vector into the index.
    fn insert_sparse(&mut self, id: NeuronId, weights: &SparseVector);

    /// Remove a neuron that was previously inserted with `dense_weights`.
    /// Hashes the weights to locate the neuron's buckets for targeted removal.
    fn remove_with_weights(&mut self, id: NeuronId, dense_weights: &[f32]);

    /// Remove a neuron from all buckets across all tables.
    ///
    /// This performs a linear scan of every bucket and is O(num_tables *
    /// num_buckets * bucket_capacity).  Prefer `remove_with_weights` when
    /// the original weights are available.
    fn remove(&mut self, id: NeuronId);

    /// Query with a dense input, returning candidates with match counts.
    ///
    /// This is the primary query method used by the forward pass.
    fn query(&self, input: &[f32]) -> Vec<LshCandidate>;

    /// Query with a dense input, returning a deduplicated set of candidate
    /// neuron IDs (union across all tables).
    fn query_union(&self, input: &[f32]) -> Vec<NeuronId>;

    /// Query with a sparse input, returning a deduplicated set of candidates.
    fn query_sparse_union(&self, input: &SparseVector) -> Vec<NeuronId>;

    /// Query with a dense input, returning each candidate with the number
    /// of tables it appeared in.
    fn query_with_counts(&self, input: &[f32]) -> Vec<LshCandidate>;

    /// Query with a sparse input, returning each candidate with match counts.
    fn query_sparse_with_counts(&self, input: &SparseVector) -> Vec<LshCandidate>;

    /// Clear and rebuild the entire index from a set of `(id, dense_weights)`
    /// pairs.
    fn rebuild(&mut self, neurons: &[(NeuronId, Vec<f32>)]);

    /// Number of hash tables.
    fn num_tables(&self) -> usize;

    /// Clear all buckets without deallocating.
    fn clear(&mut self);
}

// ---------------------------------------------------------------------------
// BucketStorage -- enum to avoid being generic over Bucket
// ---------------------------------------------------------------------------

/// Type-erased bucket storage: either all-FIFO or all-Reservoir.
///
/// Layout: `tables[table_index][bucket_index]`.
enum BucketStorage {
    Fifo(Vec<Vec<FifoBucket>>),
    Reservoir(Vec<Vec<ReservoirBucket>>),
}

impl BucketStorage {
    fn new(
        bucket_type: BucketType,
        num_tables: usize,
        num_buckets: usize,
        capacity: usize,
    ) -> Self {
        match bucket_type {
            BucketType::Fifo => {
                let tables = (0..num_tables)
                    .map(|_| (0..num_buckets).map(|_| FifoBucket::new(capacity)).collect())
                    .collect();
                BucketStorage::Fifo(tables)
            }
            BucketType::Reservoir => {
                let tables = (0..num_tables)
                    .map(|_| {
                        (0..num_buckets)
                            .map(|_| ReservoirBucket::new(capacity))
                            .collect()
                    })
                    .collect();
                BucketStorage::Reservoir(tables)
            }
        }
    }

    fn insert(&mut self, table: usize, bucket: usize, id: NeuronId) {
        match self {
            BucketStorage::Fifo(tables) => tables[table][bucket].insert(id),
            BucketStorage::Reservoir(tables) => tables[table][bucket].insert(id),
        }
    }

    fn remove(&mut self, table: usize, bucket: usize, id: NeuronId) -> bool {
        match self {
            BucketStorage::Fifo(tables) => tables[table][bucket].remove(id),
            BucketStorage::Reservoir(tables) => tables[table][bucket].remove(id),
        }
    }

    fn contents(&self, table: usize, bucket: usize) -> &[NeuronId] {
        match self {
            BucketStorage::Fifo(tables) => tables[table][bucket].contents(),
            BucketStorage::Reservoir(tables) => tables[table][bucket].contents(),
        }
    }

    fn clear(&mut self) {
        match self {
            BucketStorage::Fifo(tables) => {
                for table in tables.iter_mut() {
                    for bucket in table.iter_mut() {
                        bucket.clear();
                    }
                }
            }
            BucketStorage::Reservoir(tables) => {
                for table in tables.iter_mut() {
                    for bucket in table.iter_mut() {
                        bucket.clear();
                    }
                }
            }
        }
    }

    /// Remove a neuron ID from every bucket in every table (brute force).
    fn remove_all(&mut self, id: NeuronId) {
        match self {
            BucketStorage::Fifo(tables) => {
                for table in tables.iter_mut() {
                    for bucket in table.iter_mut() {
                        bucket.remove(id);
                    }
                }
            }
            BucketStorage::Reservoir(tables) => {
                for table in tables.iter_mut() {
                    for bucket in table.iter_mut() {
                        bucket.remove(id);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// LshIndex<H: HashFamily>
// ---------------------------------------------------------------------------

/// LSH index parameterised by a hash-function family.
///
/// The hash family is generic, but bucket storage is handled via the
/// `BucketStorage` enum so that the type doesn't need to be generic over
/// the bucket implementation.
pub struct LshIndex<H: HashFamily> {
    hash_family: H,
    buckets: BucketStorage,
    num_tables: usize,
    num_buckets: usize,
}

impl<H: HashFamily> LshIndex<H> {
    /// Create a new `LshIndex`.
    ///
    /// # Arguments
    /// * `hash_family` - The hash function family (already constructed with
    ///   the correct `k` and `num_tables`).
    /// * `bucket_type` - Whether to use FIFO or reservoir buckets.
    /// * `bucket_capacity` - Maximum number of neuron IDs per bucket.
    pub fn new(hash_family: H, bucket_type: BucketType, bucket_capacity: usize) -> Self {
        let num_tables = hash_family.num_tables();
        let num_buckets = 1usize << hash_family.k();
        let buckets =
            BucketStorage::new(bucket_type, num_tables, num_buckets, bucket_capacity);
        Self {
            hash_family,
            buckets,
            num_tables,
            num_buckets,
        }
    }

    /// Compute the bucket index for a dense vector in a given table.
    #[inline]
    fn bucket_idx_dense(&self, input: &[f32], table: usize) -> usize {
        (self.hash_family.hash_dense(input, table) as usize) & (self.num_buckets - 1)
    }

    /// Compute the bucket index for a sparse vector in a given table.
    #[inline]
    fn bucket_idx_sparse(&self, input: &SparseVector, table: usize) -> usize {
        (self.hash_family.hash_sparse(input, table) as usize) & (self.num_buckets - 1)
    }

    /// Collect candidate counts from bucket contents returned by `get_bucket`.
    fn collect_counts(
        &self,
        get_bucket: impl Fn(usize) -> usize,
    ) -> Vec<LshCandidate> {
        let mut counts: HashMap<NeuronId, u32> = HashMap::new();
        for t in 0..self.num_tables {
            let b = get_bucket(t);
            for &id in self.buckets.contents(t, b) {
                *counts.entry(id).or_insert(0) += 1;
            }
        }
        counts
            .into_iter()
            .map(|(neuron_id, count)| LshCandidate { neuron_id, count })
            .collect()
    }

    /// Collect a deduplicated union of candidates.
    fn collect_union(
        &self,
        get_bucket: impl Fn(usize) -> usize,
    ) -> Vec<NeuronId> {
        let mut seen = HashMap::<NeuronId, ()>::new();
        for t in 0..self.num_tables {
            let b = get_bucket(t);
            for &id in self.buckets.contents(t, b) {
                seen.entry(id).or_insert(());
            }
        }
        seen.into_keys().collect()
    }
}

// -- Trait implementation -------------------------------------------------

// `H` must be `'static` so that `LshIndex<H>` can be wrapped in
// `Box<dyn LshIndexTrait>`.
impl<H: HashFamily + 'static> LshIndexTrait for LshIndex<H> {
    fn insert(&mut self, id: NeuronId, dense_weights: &[f32]) {
        for t in 0..self.num_tables {
            let b = self.bucket_idx_dense(dense_weights, t);
            self.buckets.insert(t, b, id);
        }
    }

    fn insert_sparse(&mut self, id: NeuronId, weights: &SparseVector) {
        for t in 0..self.num_tables {
            let b = self.bucket_idx_sparse(weights, t);
            self.buckets.insert(t, b, id);
        }
    }

    fn remove_with_weights(&mut self, id: NeuronId, dense_weights: &[f32]) {
        for t in 0..self.num_tables {
            let b = self.bucket_idx_dense(dense_weights, t);
            self.buckets.remove(t, b, id);
        }
    }

    fn remove(&mut self, id: NeuronId) {
        self.buckets.remove_all(id);
    }

    fn query(&self, input: &[f32]) -> Vec<LshCandidate> {
        self.collect_counts(|t| self.bucket_idx_dense(input, t))
    }

    fn query_union(&self, input: &[f32]) -> Vec<NeuronId> {
        self.collect_union(|t| self.bucket_idx_dense(input, t))
    }

    fn query_sparse_union(&self, input: &SparseVector) -> Vec<NeuronId> {
        self.collect_union(|t| self.bucket_idx_sparse(input, t))
    }

    fn query_with_counts(&self, input: &[f32]) -> Vec<LshCandidate> {
        self.collect_counts(|t| self.bucket_idx_dense(input, t))
    }

    fn query_sparse_with_counts(&self, input: &SparseVector) -> Vec<LshCandidate> {
        self.collect_counts(|t| self.bucket_idx_sparse(input, t))
    }

    fn rebuild(&mut self, neurons: &[(NeuronId, Vec<f32>)]) {
        self.buckets.clear();
        for &(id, ref weights) in neurons {
            self.insert(id, weights);
        }
    }

    fn num_tables(&self) -> usize {
        self.num_tables
    }

    fn clear(&mut self) {
        self.buckets.clear();
    }
}

// -- Send + Sync ----------------------------------------------------------

// `BucketStorage` contains only `Vec` of `FifoBucket` / `ReservoirBucket`,
// both of which are `Send + Sync`.
unsafe impl<H: HashFamily> Send for LshIndex<H> {}
unsafe impl<H: HashFamily> Sync for LshIndex<H> {}

// ---------------------------------------------------------------------------
// Factory function
// ---------------------------------------------------------------------------

/// Default window size for WTA / DWTA when the config doesn't specify one.
const DEFAULT_WINDOW_SIZE: usize = 8;

/// Default sparsity for SRP when the config doesn't specify one.
const DEFAULT_SRP_SPARSITY: f32 = 1.0 / 3.0;

/// Create an LSH index from a configuration.
///
/// Dispatches on `config.hash_function` to build the appropriate hash family
/// and wraps the resulting `LshIndex<H>` in a `Box<dyn LshIndexTrait>`.
pub fn create_lsh_index(
    config: &LshConfig,
    input_dim: usize,
    seed: u64,
) -> Box<dyn LshIndexTrait> {
    let k = config.num_hashes;
    let l = config.num_tables;
    let bt = config.bucket_type;
    let cap = config.bucket_capacity;

    match config.hash_function {
        HashFunctionType::SimHash => {
            let hf = SimHash::new(input_dim, k, l, seed);
            Box::new(LshIndex::new(hf, bt, cap))
        }
        HashFunctionType::WtaHash => {
            let ws = DEFAULT_WINDOW_SIZE.min(input_dim);
            let hf = WtaHash::new(input_dim, k, l, ws, seed);
            Box::new(LshIndex::new(hf, bt, cap))
        }
        HashFunctionType::DwtaHash => {
            let ws = DEFAULT_WINDOW_SIZE.min(input_dim);
            let hf = DwtaHash::new(input_dim, k, l, ws, seed);
            Box::new(LshIndex::new(hf, bt, cap))
        }
        HashFunctionType::MinHash => {
            let hf = MinHash::new(input_dim, k, l, seed);
            Box::new(LshIndex::new(hf, bt, cap))
        }
        HashFunctionType::SparseRandomProjection => {
            let hf =
                SparseRandomProjection::new(input_dim, k, l, DEFAULT_SRP_SPARSITY, seed);
            Box::new(LshIndex::new(hf, bt, cap))
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::LshConfig;

    /// Helper: build a default config (SimHash + FIFO).
    fn default_config() -> LshConfig {
        LshConfig {
            hash_function: HashFunctionType::SimHash,
            bucket_type: BucketType::Fifo,
            num_tables: 4,
            range_pow: 3,
            num_hashes: 3,
            bucket_capacity: 32,
            rebuild_interval_base: 100,
            rebuild_decay: 0.1,
        }
    }

    // -- Basic insert / query -----------------------------------------------

    #[test]
    fn insert_and_query_finds_self() {
        let config = default_config();
        let mut idx = create_lsh_index(&config, 8, 42);

        let weights = vec![1.0, 0.5, -0.3, 0.0, 2.0, -1.0, 0.1, 0.7];
        idx.insert(100, &weights);

        let results = idx.query_union(&weights);
        assert!(
            results.contains(&100),
            "inserted neuron should be found in its own query"
        );
    }

    #[test]
    fn insert_and_query_with_counts() {
        let config = default_config();
        let mut idx = create_lsh_index(&config, 8, 42);

        let weights = vec![1.0, 0.5, -0.3, 0.0, 2.0, -1.0, 0.1, 0.7];
        idx.insert(100, &weights);

        let results = idx.query_with_counts(&weights);
        let candidate = results.iter().find(|c| c.neuron_id == 100);
        assert!(candidate.is_some(), "should find the inserted neuron");
        // When querying with the exact same vector, it should appear in every
        // table.
        assert_eq!(
            candidate.unwrap().count,
            config.num_tables as u32,
            "exact match should appear in all tables"
        );
    }

    #[test]
    fn insert_multiple_and_query() {
        let config = default_config();
        let mut idx = create_lsh_index(&config, 4, 42);

        idx.insert(1, &[1.0, 0.0, 0.0, 0.0]);
        idx.insert(2, &[0.0, 1.0, 0.0, 0.0]);
        idx.insert(3, &[0.0, 0.0, 1.0, 0.0]);

        let results = idx.query_union(&[1.0, 0.0, 0.0, 0.0]);
        assert!(
            results.contains(&1),
            "should find the neuron with matching weights"
        );
    }

    // -- Remove -------------------------------------------------------------

    #[test]
    fn remove_neuron_with_weights() {
        let config = default_config();
        let mut idx = create_lsh_index(&config, 4, 42);

        let w = vec![1.0, 0.5, -0.3, 0.2];
        idx.insert(10, &w);
        assert!(idx.query_union(&w).contains(&10));

        idx.remove_with_weights(10, &w);
        assert!(
            !idx.query_union(&w).contains(&10),
            "removed neuron should not appear in queries"
        );
    }

    #[test]
    fn remove_neuron_brute_force() {
        let config = default_config();
        let mut idx = create_lsh_index(&config, 4, 42);

        let w = vec![1.0, 0.5, -0.3, 0.2];
        idx.insert(10, &w);
        assert!(idx.query_union(&w).contains(&10));

        idx.remove(10);
        assert!(
            !idx.query_union(&w).contains(&10),
            "removed neuron should not appear in queries"
        );
    }

    // -- Clear --------------------------------------------------------------

    #[test]
    fn clear_empties_index() {
        let config = default_config();
        let mut idx = create_lsh_index(&config, 4, 42);

        let w = vec![1.0, 0.5, -0.3, 0.2];
        idx.insert(10, &w);
        idx.clear();

        let results = idx.query_union(&w);
        assert!(
            results.is_empty(),
            "cleared index should return no results"
        );
    }

    // -- Rebuild ------------------------------------------------------------

    #[test]
    fn rebuild_from_scratch() {
        let config = default_config();
        let mut idx = create_lsh_index(&config, 4, 42);

        let w1 = vec![1.0, 0.0, 0.0, 0.0];
        let w2 = vec![0.0, 1.0, 0.0, 0.0];
        idx.insert(1, &w1);

        // Rebuild with different data.
        idx.rebuild(&[(2, w2.clone())]);

        assert!(
            !idx.query_union(&w1).contains(&1),
            "old neuron should be gone after rebuild"
        );
        assert!(
            idx.query_union(&w2).contains(&2),
            "new neuron should be present after rebuild"
        );
    }

    // -- Sparse queries -----------------------------------------------------

    #[test]
    fn sparse_insert_and_query() {
        let config = default_config();
        let mut idx = create_lsh_index(&config, 8, 42);

        let dense = vec![1.0, 0.0, 0.5, 0.0, 2.0, 0.0, 0.1, 0.3];
        let sparse = SparseVector::from_dense(&dense);

        idx.insert_sparse(50, &sparse);

        let results = idx.query_sparse_union(&sparse);
        assert!(
            results.contains(&50),
            "sparse-inserted neuron should be found"
        );
    }

    #[test]
    fn sparse_query_with_counts() {
        let config = default_config();
        let mut idx = create_lsh_index(&config, 8, 42);

        let dense = vec![1.0, 0.0, 0.5, 0.0, 2.0, 0.0, 0.1, 0.3];
        let sparse = SparseVector::from_dense(&dense);

        idx.insert_sparse(50, &sparse);

        let results = idx.query_sparse_with_counts(&sparse);
        let candidate = results.iter().find(|c| c.neuron_id == 50);
        assert!(candidate.is_some());
        assert_eq!(candidate.unwrap().count, config.num_tables as u32);
    }

    // -- Factory: all hash function types -----------------------------------

    #[test]
    fn factory_simhash() {
        let mut config = default_config();
        config.hash_function = HashFunctionType::SimHash;
        let mut idx = create_lsh_index(&config, 8, 42);
        let w = vec![1.0; 8];
        idx.insert(1, &w);
        assert!(idx.query_union(&w).contains(&1));
    }

    #[test]
    fn factory_wtahash() {
        let mut config = default_config();
        config.hash_function = HashFunctionType::WtaHash;
        let mut idx = create_lsh_index(&config, 16, 42);
        let w = vec![1.0; 16];
        idx.insert(1, &w);
        assert!(idx.query_union(&w).contains(&1));
    }

    #[test]
    fn factory_dwtahash() {
        let mut config = default_config();
        config.hash_function = HashFunctionType::DwtaHash;
        let mut idx = create_lsh_index(&config, 16, 42);
        let w = vec![1.0; 16];
        idx.insert(1, &w);
        assert!(idx.query_union(&w).contains(&1));
    }

    #[test]
    fn factory_minhash() {
        let mut config = default_config();
        config.hash_function = HashFunctionType::MinHash;
        let mut idx = create_lsh_index(&config, 16, 42);
        let w = vec![1.0; 16];
        idx.insert(1, &w);
        assert!(idx.query_union(&w).contains(&1));
    }

    #[test]
    fn factory_srp() {
        let mut config = default_config();
        config.hash_function = HashFunctionType::SparseRandomProjection;
        let mut idx = create_lsh_index(&config, 16, 42);
        let w = vec![1.0; 16];
        idx.insert(1, &w);
        assert!(idx.query_union(&w).contains(&1));
    }

    // -- Factory: both bucket types ----------------------------------------

    #[test]
    fn factory_fifo_bucket() {
        let mut config = default_config();
        config.bucket_type = BucketType::Fifo;
        let mut idx = create_lsh_index(&config, 8, 42);
        let w = vec![1.0; 8];
        idx.insert(1, &w);
        assert!(idx.query_union(&w).contains(&1));
    }

    #[test]
    fn factory_reservoir_bucket() {
        let mut config = default_config();
        config.bucket_type = BucketType::Reservoir;
        let mut idx = create_lsh_index(&config, 8, 42);
        let w = vec![1.0; 8];
        idx.insert(1, &w);
        assert!(idx.query_union(&w).contains(&1));
    }

    // -- num_tables accessor ------------------------------------------------

    #[test]
    fn num_tables_matches_config() {
        let config = default_config();
        let idx = create_lsh_index(&config, 8, 42);
        assert_eq!(idx.num_tables(), config.num_tables);
    }

    // -- Edge cases ---------------------------------------------------------

    #[test]
    fn query_empty_index() {
        let config = default_config();
        let idx = create_lsh_index(&config, 8, 42);
        let results = idx.query_union(&[0.0; 8]);
        assert!(results.is_empty());
    }

    #[test]
    fn insert_same_neuron_twice() {
        let config = default_config();
        let mut idx = create_lsh_index(&config, 4, 42);
        let w = vec![1.0, 0.5, -0.3, 0.2];
        idx.insert(10, &w);
        idx.insert(10, &w);
        // Should still find it; it may appear with double count or just be
        // present -- the important thing is no panic.
        assert!(idx.query_union(&w).contains(&10));
    }

    #[test]
    fn many_neurons() {
        let config = default_config();
        let mut idx = create_lsh_index(&config, 4, 42);
        for i in 0..200u32 {
            let w = vec![
                (i as f32).sin(),
                (i as f32).cos(),
                (i as f32 * 0.1).sin(),
                (i as f32 * 0.1).cos(),
            ];
            idx.insert(i, &w);
        }
        // Query for neuron 50 -- it should be among the results.
        let w50 = vec![
            (50.0f32).sin(),
            (50.0f32).cos(),
            (50.0f32 * 0.1).sin(),
            (50.0f32 * 0.1).cos(),
        ];
        let results = idx.query_union(&w50);
        assert!(
            results.contains(&50),
            "neuron 50 should be found among candidates"
        );
    }
}
