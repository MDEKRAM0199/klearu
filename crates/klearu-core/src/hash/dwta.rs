//! Densified Winner-Take-All (DWTA) hashing for sparse inputs.
//!
//! Like WTA, but when a window contains no non-zero entries in a sparse input
//! it borrows from a deterministic fallback rather than always returning 0.
//! This avoids the degenerate case where many windows hash to the same value
//! for very sparse vectors.

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::tensor::SparseVector;

use super::HashFamily;

/// Densified Winner-Take-All hash family for sparse inputs.
pub struct DwtaHash {
    /// `windows[table][hash_fn]` -- each inner Vec has `window_size` dimension indices.
    windows: Vec<Vec<Vec<usize>>>,
    input_dim: usize,
    k: usize,
    num_tables: usize,
    window_size: usize,
    /// Number of bits needed to encode one window argmax position.
    bits_per_window: u32,
}

impl DwtaHash {
    /// Create a new DwtaHash family.
    ///
    /// # Arguments
    /// * `input_dim` - Dimensionality of input vectors.
    /// * `k` - Number of hash functions (windows) per table.
    /// * `num_tables` - Number of hash tables (L).
    /// * `window_size` - Number of dimensions sampled per window.
    /// * `seed` - RNG seed for reproducibility.
    pub fn new(
        input_dim: usize,
        k: usize,
        num_tables: usize,
        window_size: usize,
        seed: u64,
    ) -> Self {
        assert!(
            window_size <= input_dim,
            "window_size ({window_size}) must be <= input_dim ({input_dim})"
        );
        let bits_per_window = if window_size <= 1 {
            1
        } else {
            (window_size as f64).log2().ceil() as u32
        };
        assert!(
            (k as u32) * bits_per_window <= 64,
            "k * bits_per_window = {} exceeds 64",
            (k as u32) * bits_per_window
        );

        let mut rng = StdRng::seed_from_u64(seed);
        let all_dims: Vec<usize> = (0..input_dim).collect();
        let mut windows = Vec::with_capacity(num_tables);

        for _ in 0..num_tables {
            let mut table_windows = Vec::with_capacity(k);
            for _ in 0..k {
                let mut perm = all_dims.clone();
                perm.shuffle(&mut rng);
                perm.truncate(window_size);
                table_windows.push(perm);
            }
            windows.push(table_windows);
        }

        Self {
            windows,
            input_dim,
            k,
            num_tables,
            window_size,
            bits_per_window,
        }
    }

    /// Find the argmax position within a window for a dense input.
    #[inline]
    fn argmax_dense(&self, input: &[f32], window: &[usize]) -> usize {
        let mut best_pos = 0;
        let mut best_val = f32::NEG_INFINITY;
        for (pos, &dim_idx) in window.iter().enumerate() {
            let v = input[dim_idx];
            if v > best_val {
                best_val = v;
                best_pos = pos;
            }
        }
        best_pos
    }

    /// Find the argmax among window positions whose dimension indices have
    /// non-zero values in the sparse input.  If no window position is non-zero,
    /// return a deterministic fallback based on (table, hash_fn_index).
    #[inline]
    fn argmax_sparse(
        &self,
        input: &SparseVector,
        window: &[usize],
        table: usize,
        hash_fn: usize,
    ) -> usize {
        let mut best_pos: Option<usize> = None;
        let mut best_val = f32::NEG_INFINITY;

        for (pos, &dim_idx) in window.iter().enumerate() {
            // Binary search in the sorted sparse indices.
            if let Ok(sparse_pos) = input.indices.binary_search(&(dim_idx as u32)) {
                let v = input.values[sparse_pos];
                if v > best_val {
                    best_val = v;
                    best_pos = Some(pos);
                }
            }
        }

        match best_pos {
            Some(p) => p,
            None => {
                // Deterministic fallback: hash the (table, hash_fn) pair to
                // produce a position in [0, window_size).
                let mut hasher = DefaultHasher::new();
                table.hash(&mut hasher);
                hash_fn.hash(&mut hasher);
                (hasher.finish() as usize) % self.window_size
            }
        }
    }
}

impl HashFamily for DwtaHash {
    fn hash_dense(&self, input: &[f32], table: usize) -> u64 {
        debug_assert_eq!(input.len(), self.input_dim);
        let table_windows = &self.windows[table];
        let mut hash = 0u64;
        let mut shift = 0u32;
        for window in table_windows.iter() {
            let argmax = self.argmax_dense(input, window) as u64;
            hash |= argmax << shift;
            shift += self.bits_per_window;
        }
        hash
    }

    fn hash_sparse(&self, input: &SparseVector, table: usize) -> u64 {
        let table_windows = &self.windows[table];
        let mut hash = 0u64;
        let mut shift = 0u32;
        for (hash_fn, window) in table_windows.iter().enumerate() {
            let argmax = self.argmax_sparse(input, window, table, hash_fn) as u64;
            hash |= argmax << shift;
            shift += self.bits_per_window;
        }
        hash
    }

    fn k(&self) -> usize {
        self.k
    }

    fn input_dim(&self) -> usize {
        self.input_dim
    }

    fn num_tables(&self) -> usize {
        self.num_tables
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::test_helpers::*;
    use rand::rngs::StdRng;
    use rand::Rng;
    use rand::SeedableRng;

    const DIM: usize = 64;
    const K: usize = 4;
    const L: usize = 3;
    const WINDOW: usize = 8;
    const SEED: u64 = 42;

    fn random_dense(seed: u64, dim: usize) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect()
    }

    #[test]
    fn deterministic_with_same_seed() {
        let v = random_dense(99, DIM);
        assert_deterministic(
            || Box::new(DwtaHash::new(DIM, K, L, WINDOW, SEED)),
            &v,
        );
    }

    #[test]
    fn hash_in_range() {
        let h = DwtaHash::new(DIM, K, L, WINDOW, SEED);
        let v = random_dense(100, DIM);
        let total_bits = K as u32 * h.bits_per_window;
        let max = if total_bits >= 64 {
            u64::MAX
        } else {
            1u64 << total_bits
        };
        for t in 0..L {
            let hv = h.hash_dense(&v, t);
            assert!(hv < max, "table {t}: hash {hv} >= {max}");
        }
    }

    #[test]
    fn dense_sparse_agree_when_fully_dense() {
        // When the vector has no zero entries, DWTA and WTA give the same result.
        let h = DwtaHash::new(DIM, K, L, WINDOW, SEED);
        let v = random_dense(101, DIM);
        assert_dense_sparse_agree(&h, &v);
    }

    #[test]
    fn sparse_input_produces_valid_hash() {
        let h = DwtaHash::new(DIM, K, L, WINDOW, SEED);
        // Very sparse: only 3 non-zero
        let sparse = SparseVector::from_pairs(DIM, vec![(5, 1.0), (20, -2.0), (63, 0.5)]);
        let total_bits = K as u32 * h.bits_per_window;
        let max = if total_bits >= 64 {
            u64::MAX
        } else {
            1u64 << total_bits
        };
        for t in 0..L {
            let hv = h.hash_sparse(&sparse, t);
            assert!(hv < max, "table {t}: hash {hv} >= {max}");
        }
    }

    #[test]
    fn empty_sparse_still_hashes() {
        let h = DwtaHash::new(DIM, K, L, WINDOW, SEED);
        let sparse = SparseVector::new(DIM);
        // All windows are empty, so deterministic fallback kicks in.
        // Just verify it doesn't panic and is deterministic.
        let h1: Vec<u64> = (0..L).map(|t| h.hash_sparse(&sparse, t)).collect();
        let h2: Vec<u64> = (0..L).map(|t| h.hash_sparse(&sparse, t)).collect();
        assert_eq!(h1, h2);
    }

    #[test]
    fn different_inputs_different_hashes() {
        let h = DwtaHash::new(DIM, K, L, WINDOW, SEED);
        let v1 = random_dense(200, DIM);
        let v2 = random_dense(201, DIM);
        let any_differ = (0..L).any(|t| h.hash_dense(&v1, t) != h.hash_dense(&v2, t));
        assert!(any_differ, "all tables gave same hash for different inputs");
    }
}
