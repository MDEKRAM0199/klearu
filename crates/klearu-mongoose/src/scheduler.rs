//! Adaptive rebuild scheduling based on weight drift monitoring.
//!
//! Instead of rebuilding the LSH tables on a fixed schedule, the
//! [`AdaptiveScheduler`] monitors how much neuron weights have drifted since
//! the last rebuild and only triggers a rebuild when hash codes would change
//! significantly.  This avoids unnecessary work during periods of slow
//! learning while ensuring hash quality during fast-moving phases.

use klearu_core::hash::HashFamily;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

// ---------------------------------------------------------------------------
// AdaptiveScheduler
// ---------------------------------------------------------------------------

/// Adaptive scheduler that monitors weight drift and triggers LSH rebuild
/// only when hash codes would change significantly.
pub struct AdaptiveScheduler {
    /// Minimum steps between rebuild checks.
    min_interval: usize,
    /// Maximum steps between rebuilds (forced rebuild).
    max_interval: usize,
    /// Fraction of neurons to sample for drift estimation.
    sample_fraction: f32,
    /// Threshold: fraction of sampled neurons whose hash changed to trigger rebuild.
    drift_threshold: f32,
    /// EMA of drift rate.
    drift_ema: f32,
    /// EMA decay factor.
    ema_alpha: f32,
    /// Steps since last rebuild.
    steps_since_rebuild: usize,
    /// Steps since last drift check.
    steps_since_check: usize,
    /// Stored hash codes from last rebuild for drift comparison.
    /// Each entry is `(neuron_index_into_slice, hash_codes_across_tables)`.
    baseline_codes: Vec<(u32, Vec<u64>)>,
    /// The indices (into the neuron slice) that were sampled for baseline.
    sampled_indices: Vec<usize>,
}

impl AdaptiveScheduler {
    /// Create a new `AdaptiveScheduler`.
    ///
    /// # Arguments
    /// * `min_interval` - Minimum steps between rebuild checks.
    /// * `max_interval` - Maximum steps between rebuilds (forced).
    /// * `sample_fraction` - Fraction of neurons to sample for drift estimation (0.0..=1.0).
    /// * `drift_threshold` - Fraction of sampled neurons whose hash changed to trigger rebuild.
    pub fn new(
        min_interval: usize,
        max_interval: usize,
        sample_fraction: f32,
        drift_threshold: f32,
    ) -> Self {
        assert!(
            min_interval <= max_interval,
            "min_interval ({min_interval}) must be <= max_interval ({max_interval})"
        );
        assert!(
            (0.0..=1.0).contains(&sample_fraction),
            "sample_fraction must be in [0, 1], got {sample_fraction}"
        );
        assert!(
            (0.0..=1.0).contains(&drift_threshold),
            "drift_threshold must be in [0, 1], got {drift_threshold}"
        );

        Self {
            min_interval,
            max_interval,
            sample_fraction,
            drift_threshold,
            drift_ema: 0.0,
            ema_alpha: 0.3,
            steps_since_rebuild: 0,
            steps_since_check: 0,
            baseline_codes: Vec::new(),
            sampled_indices: Vec::new(),
        }
    }

    /// Record baseline hash codes for sampled neurons.
    ///
    /// This should be called after every rebuild to snapshot the current
    /// hash codes for later drift comparison.
    pub fn record_baseline(
        &mut self,
        neurons: &[(u32, Vec<f32>)],
        hash_fn: &dyn HashFamily,
        seed: u64,
    ) {
        let num_tables = hash_fn.num_tables();
        let n = neurons.len();
        let sample_size = ((n as f32 * self.sample_fraction).ceil() as usize).max(1).min(n);

        // Randomly sample neuron indices.
        let mut rng = StdRng::seed_from_u64(seed);
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rng);
        indices.truncate(sample_size);
        indices.sort_unstable();

        // Store hash codes for sampled neurons.
        self.baseline_codes = indices
            .iter()
            .map(|&i| {
                let (neuron_id, ref weights) = neurons[i];
                let codes: Vec<u64> = (0..num_tables)
                    .map(|t| hash_fn.hash_dense(weights, t))
                    .collect();
                (neuron_id, codes)
            })
            .collect();

        self.sampled_indices = indices;
        self.steps_since_rebuild = 0;
        self.steps_since_check = 0;
        self.drift_ema = 0.0;
    }

    /// Check if rebuild should happen.  Call this every step.
    pub fn should_rebuild(
        &mut self,
        neurons: &[(u32, Vec<f32>)],
        hash_fn: &dyn HashFamily,
    ) -> bool {
        self.steps_since_rebuild += 1;
        self.steps_since_check += 1;

        // Forced rebuild at max_interval.
        if self.steps_since_rebuild >= self.max_interval {
            self.steps_since_rebuild = 0;
            self.steps_since_check = 0;
            return true;
        }

        // Don't check too frequently.
        if self.steps_since_check < self.min_interval {
            return false;
        }

        // Sample drift.
        let drift = self.estimate_drift(neurons, hash_fn);
        self.drift_ema = self.ema_alpha * drift + (1.0 - self.ema_alpha) * self.drift_ema;
        self.steps_since_check = 0;

        if self.drift_ema >= self.drift_threshold {
            self.steps_since_rebuild = 0;
            true
        } else {
            false
        }
    }

    /// Estimate fraction of sampled neurons whose hash codes have changed
    /// compared to the baseline.
    fn estimate_drift(
        &self,
        neurons: &[(u32, Vec<f32>)],
        hash_fn: &dyn HashFamily,
    ) -> f32 {
        if self.baseline_codes.is_empty() || self.sampled_indices.is_empty() {
            return 0.0;
        }

        let num_tables = hash_fn.num_tables();
        let mut changed = 0usize;

        for (baseline_idx, &neuron_pos) in self.sampled_indices.iter().enumerate() {
            if neuron_pos >= neurons.len() {
                continue;
            }

            let (_, ref baseline_hash) = self.baseline_codes[baseline_idx];
            let (_, ref weights) = neurons[neuron_pos];

            let any_changed = (0..num_tables).any(|t| {
                let current = hash_fn.hash_dense(weights, t);
                t < baseline_hash.len() && current != baseline_hash[t]
            });

            if any_changed {
                changed += 1;
            }
        }

        let total = self.sampled_indices.len();
        if total == 0 {
            0.0
        } else {
            changed as f32 / total as f32
        }
    }

    /// Notify that a rebuild was performed.
    pub fn notify_rebuild(&mut self) {
        self.steps_since_rebuild = 0;
        self.steps_since_check = 0;
    }

    /// Get the current drift EMA estimate.
    pub fn drift_estimate(&self) -> f32 {
        self.drift_ema
    }

    /// Get steps since last rebuild.
    pub fn steps_since_rebuild(&self) -> usize {
        self.steps_since_rebuild
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use klearu_core::hash::SimHash;

    const DIM: usize = 32;
    const K: usize = 8;
    const L: usize = 4;
    const SEED: u64 = 42;

    fn make_neurons(n: usize, dim: usize, seed: u64) -> Vec<(u32, Vec<f32>)> {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|i| {
                let w: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
                (i as u32, w)
            })
            .collect()
    }

    // -- Forced rebuild at max_interval ------------------------------------

    #[test]
    fn forced_rebuild_at_max_interval() {
        let hash_fn = SimHash::new(DIM, K, L, SEED);
        let neurons = make_neurons(20, DIM, 100);
        let mut sched = AdaptiveScheduler::new(5, 10, 0.5, 0.5);

        sched.record_baseline(&neurons, &hash_fn, 99);

        // Step through: no rebuild until max_interval.
        for step in 1..10 {
            let rebuild = sched.should_rebuild(&neurons, &hash_fn);
            // Before max_interval (10), drift is 0 (same neurons), so no rebuild.
            // Note: at step 5 a drift check happens but drift is 0.
            assert!(
                !rebuild,
                "unexpected rebuild at step {step}"
            );
        }

        // At step 10 = max_interval, forced rebuild.
        assert!(
            sched.should_rebuild(&neurons, &hash_fn),
            "should force rebuild at max_interval"
        );
    }

    // -- No rebuild before min_interval ------------------------------------

    #[test]
    fn no_rebuild_before_min_interval() {
        let hash_fn = SimHash::new(DIM, K, L, SEED);
        let neurons = make_neurons(20, DIM, 100);
        let mut sched = AdaptiveScheduler::new(5, 100, 0.5, 0.0);

        sched.record_baseline(&neurons, &hash_fn, 99);

        // Even with threshold=0.0 (any drift triggers rebuild), the scheduler
        // should not check (and thus not trigger) before min_interval steps.
        for step in 1..5 {
            let rebuild = sched.should_rebuild(&neurons, &hash_fn);
            assert!(
                !rebuild,
                "should not rebuild before min_interval (step {step})"
            );
        }
    }

    // -- High drift triggers rebuild ----------------------------------------

    #[test]
    fn high_drift_triggers_rebuild() {
        let hash_fn = SimHash::new(DIM, K, L, SEED);
        let neurons_before = make_neurons(50, DIM, 100);
        let mut sched = AdaptiveScheduler::new(1, 1000, 1.0, 0.3);

        sched.record_baseline(&neurons_before, &hash_fn, 99);

        // Now create completely different neurons (massive drift).
        let neurons_after = make_neurons(50, DIM, 999);

        // After min_interval (1 step), drift check should detect change.
        let rebuild = sched.should_rebuild(&neurons_after, &hash_fn);
        assert!(
            rebuild,
            "high drift should trigger rebuild; drift_ema={}",
            sched.drift_estimate()
        );
    }

    // -- No drift means no early rebuild ------------------------------------

    #[test]
    fn no_drift_no_early_rebuild() {
        let hash_fn = SimHash::new(DIM, K, L, SEED);
        let neurons = make_neurons(50, DIM, 100);
        let mut sched = AdaptiveScheduler::new(1, 1000, 1.0, 0.3);

        sched.record_baseline(&neurons, &hash_fn, 99);

        // Same neurons => zero drift => no rebuild.
        for _ in 0..20 {
            let rebuild = sched.should_rebuild(&neurons, &hash_fn);
            assert!(!rebuild, "no drift should not trigger rebuild");
        }
    }

    // -- notify_rebuild resets counters ------------------------------------

    #[test]
    fn notify_rebuild_resets_counters() {
        let mut sched = AdaptiveScheduler::new(5, 10, 0.5, 0.5);
        sched.steps_since_rebuild = 7;
        sched.steps_since_check = 3;

        sched.notify_rebuild();

        assert_eq!(sched.steps_since_rebuild(), 0);
    }

    // -- record_baseline overwrites old baseline ---------------------------

    #[test]
    fn record_baseline_updates_state() {
        let hash_fn = SimHash::new(DIM, K, L, SEED);
        let neurons1 = make_neurons(10, DIM, 100);
        let neurons2 = make_neurons(10, DIM, 200);

        let mut sched = AdaptiveScheduler::new(1, 100, 1.0, 0.3);
        sched.record_baseline(&neurons1, &hash_fn, 42);
        let first_baseline_len = sched.baseline_codes.len();

        sched.record_baseline(&neurons2, &hash_fn, 43);
        let second_baseline_len = sched.baseline_codes.len();

        assert_eq!(first_baseline_len, second_baseline_len);
        // Steps should be reset.
        assert_eq!(sched.steps_since_rebuild(), 0);
    }

    // -- Sample fraction controls sample size ------------------------------

    #[test]
    fn sample_fraction_controls_sample_size() {
        let hash_fn = SimHash::new(DIM, K, L, SEED);
        let neurons = make_neurons(100, DIM, 100);

        let mut sched = AdaptiveScheduler::new(1, 100, 0.1, 0.5);
        sched.record_baseline(&neurons, &hash_fn, 42);

        // 10% of 100 = 10 neurons sampled.
        assert_eq!(sched.baseline_codes.len(), 10);

        let mut sched2 = AdaptiveScheduler::new(1, 100, 0.5, 0.5);
        sched2.record_baseline(&neurons, &hash_fn, 42);
        assert_eq!(sched2.baseline_codes.len(), 50);
    }

    // -- Gradual drift: rebuild triggered after accumulation ----------------

    #[test]
    fn gradual_drift_triggers_rebuild_eventually() {
        let hash_fn = SimHash::new(DIM, K, L, SEED);
        let neurons_base = make_neurons(30, DIM, 100);
        let mut sched = AdaptiveScheduler::new(1, 10000, 1.0, 0.3);

        sched.record_baseline(&neurons_base, &hash_fn, 42);

        // Create increasingly drifted neurons.
        let neurons_drifted = make_neurons(30, DIM, 500);

        // Feed the drifted neurons until either rebuild triggers or we hit 100 steps.
        let mut triggered = false;
        for _ in 0..100 {
            if sched.should_rebuild(&neurons_drifted, &hash_fn) {
                triggered = true;
                break;
            }
        }

        assert!(
            triggered,
            "gradual drift should eventually trigger rebuild; drift_ema={}",
            sched.drift_estimate()
        );
    }

    // -- Edge case: empty neurons -------------------------------------------

    #[test]
    fn empty_neurons_does_not_panic() {
        let hash_fn = SimHash::new(DIM, K, L, SEED);
        let neurons: Vec<(u32, Vec<f32>)> = Vec::new();
        let mut sched = AdaptiveScheduler::new(1, 10, 0.5, 0.5);

        // Should not panic with empty neurons.
        sched.record_baseline(&neurons, &hash_fn, 42);
        let _ = sched.should_rebuild(&neurons, &hash_fn);
    }

    // -- Edge case: single neuron -------------------------------------------

    #[test]
    fn single_neuron_works() {
        let hash_fn = SimHash::new(DIM, K, L, SEED);
        let neurons = make_neurons(1, DIM, 100);
        let mut sched = AdaptiveScheduler::new(1, 10, 1.0, 0.5);

        sched.record_baseline(&neurons, &hash_fn, 42);

        for _ in 0..5 {
            let _ = sched.should_rebuild(&neurons, &hash_fn);
        }
    }
}
