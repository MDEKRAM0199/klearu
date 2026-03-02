//! SimHash -- random hyperplane hashing.
//!
//! h(x) = sign(r . x) where each component of r is drawn from {-1, 0, +1}
//! with probabilities (1/6, 2/3, 1/6).  The sparse ternary distribution
//! reduces both storage and computation while preserving angular LSH
//! properties.

use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

use crate::tensor::SparseVector;

use super::HashFamily;

/// Random-hyperplane (SimHash) locality-sensitive hash family.
///
/// Stores `num_tables * k` random ternary vectors of dimension `input_dim`.
/// Each vector's components are drawn from {-1, 0, +1} with probabilities
/// (1/6, 2/3, 1/6).
pub struct SimHash {
    /// Random hyperplanes: `hyperplanes[table][hash_fn][dim]`.
    hyperplanes: Vec<Vec<Vec<i8>>>,
    input_dim: usize,
    k: usize,
    num_tables: usize,
}

impl SimHash {
    /// Create a new SimHash family.
    ///
    /// # Arguments
    /// * `input_dim` - Dimensionality of input vectors.
    /// * `k` - Number of hash bits per table.
    /// * `num_tables` - Number of hash tables (L).
    /// * `seed` - RNG seed for reproducibility.
    pub fn new(input_dim: usize, k: usize, num_tables: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut hyperplanes = Vec::with_capacity(num_tables);

        for _ in 0..num_tables {
            let mut table_planes = Vec::with_capacity(k);
            for _ in 0..k {
                let plane: Vec<i8> = (0..input_dim)
                    .map(|_| sample_ternary(&mut rng))
                    .collect();
                table_planes.push(plane);
            }
            hyperplanes.push(table_planes);
        }

        Self {
            hyperplanes,
            input_dim,
            k,
            num_tables,
        }
    }
}

/// Sample from {-1, 0, +1} with probabilities (1/6, 2/3, 1/6).
fn sample_ternary(rng: &mut StdRng) -> i8 {
    let r: f64 = rng.gen();
    if r < 1.0 / 6.0 {
        -1
    } else if r < 1.0 / 6.0 + 2.0 / 3.0 {
        0
    } else {
        1
    }
}

/// Dense dot product of a ternary vector with a dense f32 vector.
#[inline]
fn dot_ternary_dense(plane: &[i8], input: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for (r, x) in plane.iter().zip(input.iter()) {
        match r {
            1 => sum += *x,
            -1 => sum -= *x,
            _ => {}
        }
    }
    sum
}

/// Sparse dot product of a ternary vector with a sparse vector.
#[inline]
fn dot_ternary_sparse(plane: &[i8], input: &SparseVector) -> f32 {
    let mut sum = 0.0f32;
    for (&idx, &val) in input.indices.iter().zip(input.values.iter()) {
        let r = plane[idx as usize];
        match r {
            1 => sum += val,
            -1 => sum -= val,
            _ => {}
        }
    }
    sum
}

impl HashFamily for SimHash {
    fn hash_dense(&self, input: &[f32], table: usize) -> u64 {
        debug_assert_eq!(input.len(), self.input_dim);
        let planes = &self.hyperplanes[table];
        let mut hash = 0u64;
        for (bit, plane) in planes.iter().enumerate() {
            let dot = dot_ternary_dense(plane, input);
            if dot >= 0.0 {
                hash |= 1u64 << bit;
            }
        }
        hash
    }

    fn hash_sparse(&self, input: &SparseVector, table: usize) -> u64 {
        let planes = &self.hyperplanes[table];
        let mut hash = 0u64;
        for (bit, plane) in planes.iter().enumerate() {
            let dot = dot_ternary_sparse(plane, input);
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
    const SEED: u64 = 42;

    fn random_dense(seed: u64, dim: usize) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect()
    }

    #[test]
    fn deterministic_with_same_seed() {
        let v = random_dense(99, DIM);
        assert_deterministic(|| Box::new(SimHash::new(DIM, K, L, SEED)), &v);
    }

    #[test]
    fn hash_in_range() {
        let h = SimHash::new(DIM, K, L, SEED);
        let v = random_dense(100, DIM);
        assert_hash_in_range(&h, &v);
    }

    #[test]
    fn dense_sparse_agree() {
        let h = SimHash::new(DIM, K, L, SEED);
        let v = random_dense(101, DIM);
        assert_dense_sparse_agree(&h, &v);
    }

    #[test]
    fn different_inputs_different_hashes() {
        let h = SimHash::new(DIM, K, L, SEED);
        let v1 = random_dense(200, DIM);
        let v2 = random_dense(201, DIM);
        // With K=8 and L=4 tables, the probability all 4 tables collide
        // for two random 128-d vectors is astronomically low.
        let any_differ = (0..L).any(|t| h.hash_dense(&v1, t) != h.hash_dense(&v2, t));
        assert!(any_differ, "all tables gave same hash for different inputs");
    }

    #[test]
    fn ternary_distribution() {
        // Verify the sampling distribution is roughly (1/6, 2/3, 1/6).
        let mut rng = StdRng::seed_from_u64(0);
        let n = 60_000;
        let mut counts = [0usize; 3]; // -1, 0, 1
        for _ in 0..n {
            let v = sample_ternary(&mut rng);
            counts[(v + 1) as usize] += 1;
        }
        let frac_neg = counts[0] as f64 / n as f64;
        let frac_zero = counts[1] as f64 / n as f64;
        let frac_pos = counts[2] as f64 / n as f64;
        assert!(
            (frac_neg - 1.0 / 6.0).abs() < 0.02,
            "P(-1) = {frac_neg}"
        );
        assert!(
            (frac_zero - 2.0 / 3.0).abs() < 0.02,
            "P(0) = {frac_zero}"
        );
        assert!(
            (frac_pos - 1.0 / 6.0).abs() < 0.02,
            "P(+1) = {frac_pos}"
        );
    }

    #[test]
    fn sparse_input_with_few_nonzeros() {
        let h = SimHash::new(DIM, K, L, SEED);
        // Mostly-zero vector: only 5 non-zero entries
        let mut v = vec![0.0f32; DIM];
        v[3] = 1.0;
        v[17] = -0.5;
        v[50] = 2.0;
        v[99] = -1.0;
        v[127] = 0.3;
        assert_dense_sparse_agree(&h, &v);
        assert_hash_in_range(&h, &v);
    }

    #[test]
    fn zero_vector() {
        let h = SimHash::new(DIM, K, L, SEED);
        let v = vec![0.0f32; DIM];
        // All dots are 0.0, so sign >= 0 => all bits set => hash = 2^K - 1
        for t in 0..L {
            let hv = h.hash_dense(&v, t);
            assert_eq!(hv, (1u64 << K) - 1);
        }
    }
}
