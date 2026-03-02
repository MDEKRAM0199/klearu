//! Sparse Random Projection (SRP) hashing.
//!
//! A variant of SimHash where the random projection vectors are stored in
//! sparse format.  Each component is drawn from the same ternary distribution
//! {-1, 0, +1} with probabilities (1/6, 2/3, 1/6) by default, so roughly
//! 1/3 of components are non-zero.  Storing only the non-zero components
//! reduces memory for high-dimensional inputs and accelerates both
//! dense-sparse and sparse-sparse dot products.

use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

use crate::tensor::SparseVector;

use super::HashFamily;

/// Sparse Random Projection locality-sensitive hash family.
///
/// Each random vector is stored as a sparse list of `(dim_index, sign)`
/// pairs, where sign is -1 or +1.
pub struct SparseRandomProjection {
    /// Sparse random vectors.
    /// Indexed as `projections[table * k + hash_fn]`.
    /// Each entry is a sorted Vec of (dimension_index, coefficient).
    projections: Vec<Vec<(u32, i8)>>,
    input_dim: usize,
    k: usize,
    num_tables: usize,
}

impl SparseRandomProjection {
    /// Create a new Sparse Random Projection family.
    ///
    /// # Arguments
    /// * `input_dim` - Dimensionality of input vectors.
    /// * `k` - Number of hash bits per table.
    /// * `num_tables` - Number of hash tables (L).
    /// * `sparsity` - Fraction of non-zero components (default ~1/3).
    ///   For example, `sparsity = 1.0/3.0` means each component has a 1/3
    ///   chance of being non-zero, drawn equally as -1 or +1.
    /// * `seed` - RNG seed for reproducibility.
    pub fn new(
        input_dim: usize,
        k: usize,
        num_tables: usize,
        sparsity: f32,
        seed: u64,
    ) -> Self {
        assert!(
            (0.0..=1.0).contains(&sparsity),
            "sparsity must be in [0, 1]"
        );

        let mut rng = StdRng::seed_from_u64(seed);
        let total = num_tables * k;
        let mut projections = Vec::with_capacity(total);

        // Each component: with probability (1-sparsity) it is 0;
        // with probability sparsity/2 it is -1; with probability sparsity/2 it is +1.
        let prob_neg = sparsity as f64 / 2.0;
        let prob_zero_end = prob_neg + (1.0 - sparsity as f64);

        for _ in 0..total {
            let mut sparse_vec: Vec<(u32, i8)> = Vec::new();
            for d in 0..input_dim {
                let r: f64 = rng.gen();
                if r < prob_neg {
                    sparse_vec.push((d as u32, -1));
                } else if r >= prob_zero_end {
                    sparse_vec.push((d as u32, 1));
                }
                // else: zero, skip
            }
            projections.push(sparse_vec);
        }

        Self {
            projections,
            input_dim,
            k,
            num_tables,
        }
    }

    /// Index into the flat `projections` array.
    #[inline]
    fn proj_index(&self, table: usize, hash_fn: usize) -> usize {
        table * self.k + hash_fn
    }

    /// Compute the sparse dot product of a sparse projection vector with a
    /// dense input.
    #[inline]
    fn dot_sparse_dense(proj: &[(u32, i8)], input: &[f32]) -> f32 {
        let mut sum = 0.0f32;
        for &(idx, coeff) in proj {
            let v = input[idx as usize];
            match coeff {
                1 => sum += v,
                -1 => sum -= v,
                _ => {}
            }
        }
        sum
    }

    /// Compute the sparse-sparse dot product between a sparse projection
    /// vector and a sparse input vector.  Both are sorted by index.
    #[inline]
    fn dot_sparse_sparse(proj: &[(u32, i8)], input: &SparseVector) -> f32 {
        let mut sum = 0.0f32;
        let mut i = 0;
        let mut j = 0;
        while i < proj.len() && j < input.indices.len() {
            let pi = proj[i].0;
            let si = input.indices[j];
            if pi == si {
                let coeff = proj[i].1;
                let val = input.values[j];
                match coeff {
                    1 => sum += val,
                    -1 => sum -= val,
                    _ => {}
                }
                i += 1;
                j += 1;
            } else if pi < si {
                i += 1;
            } else {
                j += 1;
            }
        }
        sum
    }
}

impl HashFamily for SparseRandomProjection {
    fn hash_dense(&self, input: &[f32], table: usize) -> u64 {
        debug_assert_eq!(input.len(), self.input_dim);
        let mut hash = 0u64;
        for bit in 0..self.k {
            let proj = &self.projections[self.proj_index(table, bit)];
            let dot = Self::dot_sparse_dense(proj, input);
            if dot >= 0.0 {
                hash |= 1u64 << bit;
            }
        }
        hash
    }

    fn hash_sparse(&self, input: &SparseVector, table: usize) -> u64 {
        let mut hash = 0u64;
        for bit in 0..self.k {
            let proj = &self.projections[self.proj_index(table, bit)];
            let dot = Self::dot_sparse_sparse(proj, input);
            if dot >= 0.0 {
                hash |= 1u64 << bit;
            }
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

    const DIM: usize = 128;
    const K: usize = 8;
    const L: usize = 4;
    const SPARSITY: f32 = 1.0 / 3.0;
    const SEED: u64 = 42;

    fn random_dense(seed: u64, dim: usize) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect()
    }

    #[test]
    fn deterministic_with_same_seed() {
        let v = random_dense(99, DIM);
        assert_deterministic(
            || Box::new(SparseRandomProjection::new(DIM, K, L, SPARSITY, SEED)),
            &v,
        );
    }

    #[test]
    fn hash_in_range() {
        let h = SparseRandomProjection::new(DIM, K, L, SPARSITY, SEED);
        let v = random_dense(100, DIM);
        assert_hash_in_range(&h, &v);
    }

    #[test]
    fn dense_sparse_agree() {
        let h = SparseRandomProjection::new(DIM, K, L, SPARSITY, SEED);
        let v = random_dense(101, DIM);
        assert_dense_sparse_agree(&h, &v);
    }

    #[test]
    fn different_inputs_different_hashes() {
        let h = SparseRandomProjection::new(DIM, K, L, SPARSITY, SEED);
        let v1 = random_dense(200, DIM);
        let v2 = random_dense(201, DIM);
        let any_differ = (0..L).any(|t| h.hash_dense(&v1, t) != h.hash_dense(&v2, t));
        assert!(any_differ, "all tables gave same hash for different inputs");
    }

    #[test]
    fn sparse_input_agrees() {
        let h = SparseRandomProjection::new(DIM, K, L, SPARSITY, SEED);
        let mut v = vec![0.0f32; DIM];
        v[3] = 1.0;
        v[17] = -0.5;
        v[50] = 2.0;
        v[99] = -1.0;
        v[127] = 0.3;
        assert_dense_sparse_agree(&h, &v);
    }

    #[test]
    fn zero_sparsity_all_projections_empty() {
        // sparsity=0 => all projection vectors are empty => all dots are 0 =>
        // sign(0) >= 0 => all bits set.
        let h = SparseRandomProjection::new(DIM, K, L, 0.0, SEED);
        let v = random_dense(300, DIM);
        for t in 0..L {
            assert_eq!(h.hash_dense(&v, t), (1u64 << K) - 1);
        }
    }

    #[test]
    fn full_sparsity_matches_dense_distribution() {
        // sparsity=1.0 => every component is -1 or +1
        let h = SparseRandomProjection::new(DIM, K, L, 1.0, SEED);
        // Verify all projection vectors have exactly `DIM` entries.
        for proj in &h.projections {
            assert_eq!(proj.len(), DIM);
        }
        let v = random_dense(400, DIM);
        assert_hash_in_range(&h, &v);
    }

    #[test]
    fn projection_sparsity_approximately_correct() {
        let h = SparseRandomProjection::new(DIM, K, L, SPARSITY, SEED);
        let total_components = h.projections.len() * DIM;
        let total_nonzero: usize = h.projections.iter().map(|p| p.len()).sum();
        let actual_sparsity = total_nonzero as f64 / total_components as f64;
        assert!(
            (actual_sparsity - SPARSITY as f64).abs() < 0.05,
            "expected sparsity ~{SPARSITY}, got {actual_sparsity}"
        );
    }
}
