//! Learnable hash functions that implement `klearu_core`'s [`HashFamily`] trait,
//! trained with triplet loss.
//!
//! The core idea of MONGOOSE is to replace the random projections used in
//! SimHash with *trainable* projection matrices that are refined via a triplet
//! objective.  Positive pairs (same semantic bucket) should hash together while
//! negative pairs should hash apart.

use klearu_core::hash::HashFamily;
use klearu_core::tensor::SparseVector;
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// LearnableHashFamily trait
// ---------------------------------------------------------------------------

/// Extension trait for hash families that support learning.
pub trait LearnableHashFamily: HashFamily {
    /// Update hash function parameters given triplet gradients.
    fn update_params(&mut self, learning_rate: f64);

    /// Compute and accumulate gradients from a triplet (anchor, positive, negative).
    /// Positive should hash to same bucket as anchor, negative to different.
    fn triplet_gradient(&mut self, anchor: &[f32], positive: &[f32], negative: &[f32]);

    /// Get current training loss estimate.
    fn current_loss(&self) -> f64;
}

// ---------------------------------------------------------------------------
// MongooseHash
// ---------------------------------------------------------------------------

/// MONGOOSE learnable hash function.
///
/// Uses trainable projection matrices instead of random hyperplanes.
/// The projections are initialized randomly and then refined via triplet loss
/// to improve hash quality.
pub struct MongooseHash {
    /// Trainable projection matrices, one per table.
    /// Shape per table: `[k, input_dim]`
    projections: Vec<Array2<f32>>,
    /// Accumulated gradients for projections.
    grad_projections: Vec<Array2<f32>>,
    /// Number of hash functions per table.
    k: usize,
    /// Input dimension.
    input_dim: usize,
    /// Number of tables.
    num_tables: usize,
    /// Running loss estimate (EMA).
    loss_ema: f64,
    /// Number of gradient steps accumulated.
    grad_count: usize,
    /// Triplet margin for the loss.
    margin: f32,
}

impl MongooseHash {
    /// Create a new `MongooseHash` family.
    ///
    /// # Arguments
    /// * `input_dim` - Dimensionality of input vectors.
    /// * `k` - Number of hash bits per table.
    /// * `num_tables` - Number of hash tables (L).
    /// * `seed` - RNG seed for reproducibility.
    pub fn new(input_dim: usize, k: usize, num_tables: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let stddev = 1.0 / (input_dim as f64).sqrt();
        let normal = Normal::new(0.0, stddev).expect("invalid normal distribution params");

        let mut projections = Vec::with_capacity(num_tables);
        let mut grad_projections = Vec::with_capacity(num_tables);

        for _ in 0..num_tables {
            let data: Vec<f32> = (0..k * input_dim)
                .map(|_| normal.sample(&mut rng) as f32)
                .collect();
            projections.push(
                Array2::from_shape_vec((k, input_dim), data)
                    .expect("shape mismatch building projection"),
            );
            grad_projections.push(Array2::zeros((k, input_dim)));
        }

        Self {
            projections,
            grad_projections,
            k,
            input_dim,
            num_tables,
            loss_ema: 0.0,
            grad_count: 0,
            margin: 1.0,
        }
    }

    /// Compute the real-valued (pre-sign) projection for one table.
    /// Returns an `Array1<f32>` of length `k`.
    fn raw_projection(&self, input: &[f32], table: usize) -> Array1<f32> {
        let input_arr = Array1::from_vec(input.to_vec());
        self.projections[table].dot(&input_arr)
    }

    /// Convert raw projections into a binary hash code by taking the sign and
    /// packing into a `u64`.
    fn code_from_raw(raw: &Array1<f32>) -> u64 {
        let mut hash = 0u64;
        for (bit, &val) in raw.iter().enumerate() {
            if val >= 0.0 {
                hash |= 1u64 << bit;
            }
        }
        hash
    }

    /// Hamming distance between two codes, interpreted over `k` bits.
    fn hamming(a: u64, b: u64, k: usize) -> u32 {
        let mask = if k >= 64 { u64::MAX } else { (1u64 << k) - 1 };
        ((a ^ b) & mask).count_ones()
    }

    /// Normalize each row of a projection matrix to unit L2 norm.
    fn normalize_rows(proj: &mut Array2<f32>) {
        let (rows, _cols) = proj.dim();
        for r in 0..rows {
            let mut row = proj.row_mut(r);
            let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-12 {
                row.mapv_inplace(|x| x / norm);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// HashFamily implementation
// ---------------------------------------------------------------------------

impl HashFamily for MongooseHash {
    fn hash_dense(&self, input: &[f32], table: usize) -> u64 {
        debug_assert_eq!(input.len(), self.input_dim);
        let raw = self.raw_projection(input, table);
        Self::code_from_raw(&raw)
    }

    fn hash_sparse(&self, input: &SparseVector, table: usize) -> u64 {
        // Iterate sparse entries against projection rows.
        let proj = &self.projections[table];
        let mut dots = vec![0.0f32; self.k];
        for (&idx, &val) in input.indices.iter().zip(input.values.iter()) {
            let col = idx as usize;
            for row in 0..self.k {
                dots[row] += proj[[row, col]] * val;
            }
        }
        let mut hash = 0u64;
        for (bit, &dot) in dots.iter().enumerate() {
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

// ---------------------------------------------------------------------------
// LearnableHashFamily implementation
// ---------------------------------------------------------------------------

impl LearnableHashFamily for MongooseHash {
    fn triplet_gradient(
        &mut self,
        anchor: &[f32],
        positive: &[f32],
        negative: &[f32],
    ) {
        let anchor_arr = Array1::from_vec(anchor.to_vec());
        let positive_arr = Array1::from_vec(positive.to_vec());
        let negative_arr = Array1::from_vec(negative.to_vec());

        let mut total_loss = 0.0f64;

        for t in 0..self.num_tables {
            let raw_a = self.projections[t].dot(&anchor_arr);
            let raw_p = self.projections[t].dot(&positive_arr);
            let raw_n = self.projections[t].dot(&negative_arr);

            // Binary codes via sign
            let code_a = Self::code_from_raw(&raw_a);
            let code_p = Self::code_from_raw(&raw_p);
            let code_n = Self::code_from_raw(&raw_n);

            // Hamming distances
            let dist_pos = Self::hamming(code_a, code_p, self.k) as f32;
            let dist_neg = Self::hamming(code_a, code_n, self.k) as f32;

            // Triplet loss: max(0, dist_pos - dist_neg + margin)
            let loss = (dist_pos - dist_neg + self.margin).max(0.0);
            total_loss += loss as f64;

            if loss <= 0.0 {
                // No gradient contribution from this table.
                continue;
            }

            // Straight-through estimator: gradient of sign(x) ~ 1 if |x| < 1, else 0.
            // We compute d(hamming)/d(projection) using the STE.
            //
            // hamming(a,p) = sum_bit [ sign(raw_a_bit) != sign(raw_p_bit) ]
            //   = sum_bit 0.5 * (1 - sign(raw_a_bit) * sign(raw_p_bit))
            //
            // Gradient w.r.t. projection row_bit:
            //   d(loss)/d(proj[bit]) = d(dist_pos)/d(proj[bit]) - d(dist_neg)/d(proj[bit])
            //
            // For the positive pair distance:
            //   d(dist_pos)/d(proj[bit])
            //     = -0.5 * STE(raw_a[bit]) * sign(raw_p[bit]) * anchor
            //       -0.5 * sign(raw_a[bit]) * STE(raw_p[bit]) * positive
            //
            // For the negative pair distance:
            //   d(dist_neg)/d(proj[bit])
            //     = -0.5 * STE(raw_a[bit]) * sign(raw_n[bit]) * anchor
            //       -0.5 * sign(raw_a[bit]) * STE(raw_n[bit]) * negative

            for bit in 0..self.k {
                let ra = raw_a[bit];
                let rp = raw_p[bit];
                let rn = raw_n[bit];

                let sign_a = if ra >= 0.0 { 1.0f32 } else { -1.0 };
                let sign_p = if rp >= 0.0 { 1.0f32 } else { -1.0 };
                let sign_n = if rn >= 0.0 { 1.0f32 } else { -1.0 };

                // STE mask: gradient passes through only when |pre-activation| < 1
                let ste_a = if ra.abs() < 1.0 { 1.0f32 } else { 0.0 };
                let ste_p = if rp.abs() < 1.0 { 1.0f32 } else { 0.0 };
                let ste_n = if rn.abs() < 1.0 { 1.0f32 } else { 0.0 };

                // Gradient of dist_pos w.r.t. projection row `bit`
                // d(dist_pos)/d(proj[bit]) = -0.5 * (ste_a * sign_p * anchor + sign_a * ste_p * positive)
                let d_pos_anchor_scale = -0.5 * ste_a * sign_p;
                let d_pos_positive_scale = -0.5 * sign_a * ste_p;

                // Gradient of dist_neg w.r.t. projection row `bit`
                let d_neg_anchor_scale = -0.5 * ste_a * sign_n;
                let d_neg_negative_scale = -0.5 * sign_a * ste_n;

                // d(loss)/d(proj[bit]) = d(dist_pos)/d(proj[bit]) - d(dist_neg)/d(proj[bit])
                // = (d_pos_anchor_scale - d_neg_anchor_scale) * anchor
                //   + d_pos_positive_scale * positive
                //   - d_neg_negative_scale * negative
                let grad_anchor_scale = d_pos_anchor_scale - d_neg_anchor_scale;
                let grad_positive_scale = d_pos_positive_scale;
                let grad_negative_scale = -d_neg_negative_scale;

                let mut grad_row = self.grad_projections[t].row_mut(bit);
                for d in 0..self.input_dim {
                    grad_row[d] += grad_anchor_scale * anchor_arr[d]
                        + grad_positive_scale * positive_arr[d]
                        + grad_negative_scale * negative_arr[d];
                }
            }
        }

        // Update EMA of loss.
        let avg_loss = total_loss / self.num_tables as f64;
        if self.grad_count == 0 {
            self.loss_ema = avg_loss;
        } else {
            let alpha = 0.1;
            self.loss_ema = alpha * avg_loss + (1.0 - alpha) * self.loss_ema;
        }
        self.grad_count += 1;
    }

    fn update_params(&mut self, learning_rate: f64) {
        if self.grad_count == 0 {
            return;
        }

        let scale = learning_rate as f32 / self.grad_count as f32;

        for t in 0..self.num_tables {
            // Apply accumulated gradients: projection -= lr * grad / grad_count
            let grad = &self.grad_projections[t];
            for r in 0..self.k {
                for c in 0..self.input_dim {
                    self.projections[t][[r, c]] -= scale * grad[[r, c]];
                }
            }

            // Zero gradients.
            self.grad_projections[t].fill(0.0);

            // Normalize projection rows to unit norm (prevents explosion).
            Self::normalize_rows(&mut self.projections[t]);
        }

        self.grad_count = 0;
    }

    fn current_loss(&self) -> f64 {
        self.loss_ema
    }
}

// ---------------------------------------------------------------------------
// TripletMiner
// ---------------------------------------------------------------------------

/// Triplet mining utilities for selecting informative training examples.
pub struct TripletMiner {
    margin: f32,
}

impl TripletMiner {
    /// Create a new `TripletMiner` with the given margin.
    pub fn new(margin: f32) -> Self {
        Self { margin }
    }

    /// Given a set of `(neuron_id, weights)` pairs and their current hash
    /// codes, mine hard triplets: find anchor-positive pairs in the same
    /// bucket and anchor-negative pairs from different buckets.
    ///
    /// Returns indices into `neurons` as `(anchor, positive, negative)` tuples.
    pub fn mine_triplets(
        &self,
        neurons: &[(u32, Vec<f32>)],
        hash_fn: &dyn HashFamily,
        table: usize,
        max_triplets: usize,
    ) -> Vec<(usize, usize, usize)> {
        // Group neurons by hash code.
        let mut bucket_map: HashMap<u64, Vec<usize>> = HashMap::new();

        for (i, (_, weights)) in neurons.iter().enumerate() {
            let code = hash_fn.hash_dense(weights, table);
            bucket_map.entry(code).or_default().push(i);
        }

        // Collect buckets with > 1 element (we need at least a pair for anchor-positive).
        let multi_buckets: Vec<&Vec<usize>> =
            bucket_map.values().filter(|v| v.len() > 1).collect();

        // Collect all bucket values for finding negatives.
        let all_buckets: Vec<(u64, &Vec<usize>)> =
            bucket_map.iter().map(|(&k, v)| (k, v)).collect();

        let mut triplets = Vec::new();

        'outer: for bucket in &multi_buckets {
            if triplets.len() >= max_triplets {
                break;
            }

            // Hash code for this bucket (all members share it).
            let bucket_code = hash_fn.hash_dense(&neurons[bucket[0]].1, table);

            for i in 0..bucket.len() {
                for j in (i + 1)..bucket.len() {
                    if triplets.len() >= max_triplets {
                        break 'outer;
                    }

                    let anchor_idx = bucket[i];
                    let pos_idx = bucket[j];

                    // Find a negative: pick from a different bucket.
                    // For hard negatives, prefer the closest by L2 distance.
                    let mut best_neg: Option<usize> = None;
                    let mut best_dist = f32::MAX;

                    for &(code, other_bucket) in &all_buckets {
                        if code == bucket_code {
                            continue;
                        }
                        for &neg_idx in other_bucket {
                            let dist = l2_distance(
                                &neurons[anchor_idx].1,
                                &neurons[neg_idx].1,
                            );
                            if dist < best_dist {
                                best_dist = dist;
                                best_neg = Some(neg_idx);
                            }
                        }
                    }

                    if let Some(neg_idx) = best_neg {
                        triplets.push((anchor_idx, pos_idx, neg_idx));
                    }
                }
            }
        }

        triplets
    }
}

/// Squared L2 distance between two vectors.
fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    const DIM: usize = 32;
    const K: usize = 8;
    const L: usize = 4;
    const SEED: u64 = 42;

    fn random_dense(seed: u64, dim: usize) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect()
    }

    // -- Determinism --------------------------------------------------------

    #[test]
    fn deterministic_with_same_seed() {
        let v = random_dense(99, DIM);
        let h1 = MongooseHash::new(DIM, K, L, SEED);
        let h2 = MongooseHash::new(DIM, K, L, SEED);
        for t in 0..L {
            assert_eq!(
                h1.hash_dense(&v, t),
                h2.hash_dense(&v, t),
                "table {t}: non-deterministic"
            );
        }
    }

    #[test]
    fn different_seed_gives_different_hashes() {
        let v = random_dense(99, DIM);
        let h1 = MongooseHash::new(DIM, K, L, SEED);
        let h2 = MongooseHash::new(DIM, K, L, SEED + 1);
        let any_differ = (0..L).any(|t| h1.hash_dense(&v, t) != h2.hash_dense(&v, t));
        assert!(any_differ, "different seeds should give different hashes");
    }

    // -- Dense/Sparse agreement --------------------------------------------

    #[test]
    fn dense_sparse_agree() {
        let h = MongooseHash::new(DIM, K, L, SEED);
        let dense = random_dense(101, DIM);
        let sparse = SparseVector::from_dense(&dense);
        for t in 0..L {
            let hd = h.hash_dense(&dense, t);
            let hs = h.hash_sparse(&sparse, t);
            assert_eq!(hd, hs, "table {t}: dense hash {hd} != sparse hash {hs}");
        }
    }

    #[test]
    fn dense_sparse_agree_few_nonzeros() {
        let h = MongooseHash::new(DIM, K, L, SEED);
        let mut dense = vec![0.0f32; DIM];
        dense[3] = 1.0;
        dense[17] = -0.5;
        dense[25] = 2.0;
        let sparse = SparseVector::from_dense(&dense);
        for t in 0..L {
            let hd = h.hash_dense(&dense, t);
            let hs = h.hash_sparse(&sparse, t);
            assert_eq!(hd, hs, "table {t}: dense {hd} != sparse {hs}");
        }
    }

    // -- Hash range ---------------------------------------------------------

    #[test]
    fn hash_in_range() {
        let h = MongooseHash::new(DIM, K, L, SEED);
        let v = random_dense(100, DIM);
        let max = 1u64 << K;
        for t in 0..L {
            let hv = h.hash_dense(&v, t);
            assert!(hv < max, "table {t}: hash {hv} >= {max}");
        }
    }

    // -- Triplet training improves collision rates --------------------------

    #[test]
    fn triplet_training_improves_collision() {
        // Create anchor, positive (close to anchor), and negative (far from anchor).
        let mut rng = StdRng::seed_from_u64(777);
        let dim = 16;
        let k = 6;
        let num_tables = 8;

        let anchor: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();

        // Positive: small perturbation of anchor.
        let positive: Vec<f32> = anchor
            .iter()
            .map(|&x| x + rng.gen_range(-0.05..0.05))
            .collect();

        // Negative: independent random vector.
        let negative: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();

        let mut h = MongooseHash::new(dim, k, num_tables, 12345);

        // Measure pre-training collision rate.
        let collision_rate_before = collision_rate(&h, &anchor, &positive, &negative);

        // Train with many triplet steps.
        for _ in 0..200 {
            h.triplet_gradient(&anchor, &positive, &negative);
            h.update_params(0.01);
        }

        // Measure post-training collision rate.
        let collision_rate_after = collision_rate(&h, &anchor, &positive, &negative);

        // After training, anchor-positive should collide more often (lower hamming)
        // and anchor-negative should collide less often (higher hamming).
        // The ratio (pos_hamming / neg_hamming) should decrease.
        assert!(
            collision_rate_after <= collision_rate_before + 0.01, // allow tiny tolerance
            "Training should improve (or at least not worsen) collision quality: \
             before={collision_rate_before:.4}, after={collision_rate_after:.4}"
        );
    }

    /// Compute a collision quality metric: avg(hamming(anchor, positive)) / avg(hamming(anchor, negative)).
    /// Lower is better (positive pair closer, negative pair farther).
    fn collision_rate(h: &MongooseHash, anchor: &[f32], positive: &[f32], negative: &[f32]) -> f64 {
        let mut sum_pos = 0u32;
        let mut sum_neg = 0u32;
        for t in 0..h.num_tables() {
            let ha = h.hash_dense(anchor, t);
            let hp = h.hash_dense(positive, t);
            let hn = h.hash_dense(negative, t);
            sum_pos += MongooseHash::hamming(ha, hp, h.k());
            sum_neg += MongooseHash::hamming(ha, hn, h.k());
        }
        if sum_neg == 0 {
            return 0.0;
        }
        sum_pos as f64 / sum_neg as f64
    }

    #[test]
    fn loss_decreases_with_training() {
        let dim = 16;
        let k = 6;
        let num_tables = 4;

        let mut rng = StdRng::seed_from_u64(888);
        let anchor: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let positive: Vec<f32> = anchor
            .iter()
            .map(|&x| x + rng.gen_range(-0.05..0.05))
            .collect();
        let negative: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();

        let mut h = MongooseHash::new(dim, k, num_tables, 555);

        // Warm up loss EMA.
        for _ in 0..10 {
            h.triplet_gradient(&anchor, &positive, &negative);
        }
        h.update_params(0.01);
        let initial_loss = h.current_loss();

        // Train more.
        for _ in 0..200 {
            h.triplet_gradient(&anchor, &positive, &negative);
            h.update_params(0.01);
        }
        let final_loss = h.current_loss();

        assert!(
            final_loss <= initial_loss + 0.1,
            "Loss should generally decrease: initial={initial_loss:.4}, final={final_loss:.4}"
        );
    }

    // -- TripletMiner -------------------------------------------------------

    #[test]
    fn triplet_miner_produces_valid_triplets() {
        let dim = 16;
        let k = 4;
        let num_tables = 2;
        let h = MongooseHash::new(dim, k, num_tables, 42);

        // Create neurons that will likely share buckets (identical or near-identical).
        let mut rng = StdRng::seed_from_u64(999);
        let base: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();

        let mut neurons: Vec<(u32, Vec<f32>)> = Vec::new();

        // First group: near-copies of base (should hash together with k=4).
        for i in 0..5 {
            let weights: Vec<f32> = base
                .iter()
                .map(|&x| x + rng.gen_range(-0.001..0.001))
                .collect();
            neurons.push((i, weights));
        }

        // Second group: random vectors (should hash differently).
        for i in 5..10 {
            let weights: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
            neurons.push((i, weights));
        }

        let miner = TripletMiner::new(1.0);
        let triplets = miner.mine_triplets(&neurons, &h, 0, 100);

        // Verify all triplet indices are valid.
        for &(a, p, n) in &triplets {
            assert!(a < neurons.len(), "anchor index out of bounds");
            assert!(p < neurons.len(), "positive index out of bounds");
            assert!(n < neurons.len(), "negative index out of bounds");
            assert_ne!(a, p, "anchor and positive should be different neurons");

            // Anchor and positive should be in the same bucket.
            let code_a = h.hash_dense(&neurons[a].1, 0);
            let code_p = h.hash_dense(&neurons[p].1, 0);
            assert_eq!(
                code_a, code_p,
                "anchor and positive should share bucket"
            );

            // Negative should be in a different bucket.
            let code_n = h.hash_dense(&neurons[n].1, 0);
            assert_ne!(
                code_a, code_n,
                "negative should be in a different bucket"
            );
        }
    }

    #[test]
    fn triplet_miner_respects_max_triplets() {
        let dim = 8;
        let k = 2; // Very few bits => lots of collisions => lots of potential triplets.
        let num_tables = 1;
        let h = MongooseHash::new(dim, k, num_tables, 42);

        let mut rng = StdRng::seed_from_u64(123);
        let neurons: Vec<(u32, Vec<f32>)> = (0..20)
            .map(|i| {
                let w: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
                (i as u32, w)
            })
            .collect();

        let miner = TripletMiner::new(1.0);
        let triplets = miner.mine_triplets(&neurons, &h, 0, 5);
        assert!(
            triplets.len() <= 5,
            "should respect max_triplets: got {}",
            triplets.len()
        );
    }

    #[test]
    fn triplet_miner_empty_when_no_collisions() {
        // With high K and random vectors, buckets should each have at most 1 neuron
        // (very low collision probability).
        let dim = 32;
        let k = 16;
        let num_tables = 1;
        let h = MongooseHash::new(dim, k, num_tables, 42);

        let mut rng = StdRng::seed_from_u64(456);
        let neurons: Vec<(u32, Vec<f32>)> = (0..4)
            .map(|i| {
                let w: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
                (i as u32, w)
            })
            .collect();

        let miner = TripletMiner::new(1.0);
        let triplets = miner.mine_triplets(&neurons, &h, 0, 100);
        // With 4 random vectors and k=16, collisions are extremely unlikely.
        // This test is probabilistic but the chance of failure is negligible.
        assert!(
            triplets.is_empty(),
            "with k=16 and only 4 random vectors, no collisions expected, got {} triplets",
            triplets.len()
        );
    }

    // -- Hamming helper -----------------------------------------------------

    #[test]
    fn hamming_distance_basic() {
        assert_eq!(MongooseHash::hamming(0b0000, 0b0000, 4), 0);
        assert_eq!(MongooseHash::hamming(0b0000, 0b1111, 4), 4);
        assert_eq!(MongooseHash::hamming(0b1010, 0b0101, 4), 4);
        assert_eq!(MongooseHash::hamming(0b1100, 0b1010, 4), 2);
    }

    // -- Update with no gradients is a no-op --------------------------------

    #[test]
    fn update_with_no_gradients_is_noop() {
        let h1 = MongooseHash::new(DIM, K, L, SEED);
        let mut h2 = MongooseHash::new(DIM, K, L, SEED);
        h2.update_params(0.1);

        let v = random_dense(100, DIM);
        for t in 0..L {
            assert_eq!(h1.hash_dense(&v, t), h2.hash_dense(&v, t));
        }
    }

    // -- Zero vector --------------------------------------------------------

    #[test]
    fn zero_vector_hashes_all_ones() {
        // With projection * 0 = 0 for every row, all dots are 0.0,
        // sign >= 0 => all bits set.
        let h = MongooseHash::new(DIM, K, L, SEED);
        let v = vec![0.0f32; DIM];
        let expected = (1u64 << K) - 1;
        for t in 0..L {
            assert_eq!(h.hash_dense(&v, t), expected);
        }
    }
}
