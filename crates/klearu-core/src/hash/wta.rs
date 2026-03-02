//! Winner-Take-All (WTA) hashing.
//!
//! For each of K hash functions, a random "window" of `window_size` dimension
//! indices is drawn.  The hash bit encodes the argmax within that window,
//! yielding an ordinal ranking-based hash that is invariant to monotone
//! transformations of the input.

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use crate::tensor::SparseVector;

use super::HashFamily;

/// Winner-Take-All locality-sensitive hash family.
///
/// For each table and each of the K hash functions, a random subset of
/// `window_size` dimension indices is stored.  The hash value for each
/// function is the position of the maximum-valued element within that window.
/// These positions are combined into a single hash code per table.
pub struct WtaHash {
    /// `windows[table][hash_fn]` is a `Vec<usize>` of `window_size` indices.
    windows: Vec<Vec<Vec<usize>>>,
    input_dim: usize,
    k: usize,
    num_tables: usize,
    window_size: usize,
    /// Number of bits needed to encode one window argmax.
    bits_per_window: u32,
}

impl WtaHash {
    /// Create a new WtaHash family.
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
        // We encode the argmax position (0..window_size) in ceil(log2(window_size)) bits.
        let bits_per_window = if window_size <= 1 {
            1
        } else {
            (window_size as f64).log2().ceil() as u32
        };
        // K windows each contributing `bits_per_window` bits must fit in u64.
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
}

impl HashFamily for WtaHash {
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
        // WTA needs the full vector to find argmax, so convert to dense.
        let dense = input.to_dense();
        self.hash_dense(&dense, table)
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
            || Box::new(WtaHash::new(DIM, K, L, WINDOW, SEED)),
            &v,
        );
    }

    #[test]
    fn hash_in_range() {
        let h = WtaHash::new(DIM, K, L, WINDOW, SEED);
        let v = random_dense(100, DIM);
        // Total bits = K * bits_per_window
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
    fn dense_sparse_agree() {
        let h = WtaHash::new(DIM, K, L, WINDOW, SEED);
        let v = random_dense(101, DIM);
        assert_dense_sparse_agree(&h, &v);
    }

    #[test]
    fn different_inputs_different_hashes() {
        let h = WtaHash::new(DIM, K, L, WINDOW, SEED);
        let v1 = random_dense(200, DIM);
        let v2 = random_dense(201, DIM);
        let any_differ = (0..L).any(|t| h.hash_dense(&v1, t) != h.hash_dense(&v2, t));
        assert!(any_differ, "all tables gave same hash for different inputs");
    }

    #[test]
    fn invariant_to_positive_scaling() {
        // WTA depends on ordering, not magnitudes, so scaling should not
        // change the hash.
        let h = WtaHash::new(DIM, K, L, WINDOW, SEED);
        let v: Vec<f32> = random_dense(300, DIM);
        let v_scaled: Vec<f32> = v.iter().map(|x| x * 3.0).collect();
        for t in 0..L {
            assert_eq!(
                h.hash_dense(&v, t),
                h.hash_dense(&v_scaled, t),
                "scaling changed hash at table {t}"
            );
        }
    }
}
