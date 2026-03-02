/// Bitset-based parameter usage tracker.
/// Tracks which neurons/parameters were active in the current batch
/// to skip gradient computation for inactive parameters.
pub struct ParameterTracker {
    /// Bitset: one bit per neuron, 1 = active this batch.
    active: Vec<u64>,
    /// Total number of neurons tracked.
    num_neurons: usize,
}

impl ParameterTracker {
    pub fn new(num_neurons: usize) -> Self {
        let num_words = (num_neurons + 63) / 64;
        Self {
            active: vec![0u64; num_words],
            num_neurons,
        }
    }

    /// Mark a neuron as active.
    pub fn mark_active(&mut self, neuron_id: u32) {
        let idx = neuron_id as usize;
        if idx < self.num_neurons {
            self.active[idx / 64] |= 1u64 << (idx % 64);
        }
    }

    /// Check if a neuron is active.
    pub fn is_active(&self, neuron_id: u32) -> bool {
        let idx = neuron_id as usize;
        if idx >= self.num_neurons {
            return false;
        }
        (self.active[idx / 64] >> (idx % 64)) & 1 == 1
    }

    /// Get all active neuron IDs in ascending order.
    pub fn active_neurons(&self) -> Vec<u32> {
        let mut result = Vec::new();
        for (word_idx, &word) in self.active.iter().enumerate() {
            if word == 0 {
                continue;
            }
            let base = (word_idx * 64) as u32;
            let mut w = word;
            while w != 0 {
                let bit = w.trailing_zeros();
                let neuron_id = base + bit;
                if (neuron_id as usize) < self.num_neurons {
                    result.push(neuron_id);
                }
                w &= w - 1; // clear lowest set bit
            }
        }
        result
    }

    /// Count of active neurons.
    pub fn num_active(&self) -> usize {
        self.active.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Total number of neurons tracked.
    pub fn num_neurons(&self) -> usize {
        self.num_neurons
    }

    /// Reset all neurons to inactive.
    pub fn reset(&mut self) {
        self.active.fill(0);
    }

    /// Fraction of neurons that are active.
    pub fn sparsity(&self) -> f32 {
        if self.num_neurons == 0 {
            return 0.0;
        }
        self.num_active() as f32 / self.num_neurons as f32
    }

    /// Mark multiple neurons as active at once.
    pub fn mark_active_batch(&mut self, neuron_ids: &[u32]) {
        for &id in neuron_ids {
            self.mark_active(id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Construction ----

    #[test]
    fn test_new_empty() {
        let tracker = ParameterTracker::new(0);
        assert_eq!(tracker.num_neurons(), 0);
        assert_eq!(tracker.num_active(), 0);
        assert!(tracker.active_neurons().is_empty());
    }

    #[test]
    fn test_new_small() {
        let tracker = ParameterTracker::new(1);
        assert_eq!(tracker.num_neurons(), 1);
        assert_eq!(tracker.num_active(), 0);
        assert!(!tracker.is_active(0));
    }

    #[test]
    fn test_new_exact_word_boundary() {
        let tracker = ParameterTracker::new(64);
        assert_eq!(tracker.num_neurons(), 64);
        assert_eq!(tracker.active.len(), 1);
    }

    #[test]
    fn test_new_one_past_boundary() {
        let tracker = ParameterTracker::new(65);
        assert_eq!(tracker.num_neurons(), 65);
        assert_eq!(tracker.active.len(), 2);
    }

    #[test]
    fn test_new_two_words() {
        let tracker = ParameterTracker::new(128);
        assert_eq!(tracker.active.len(), 2);
    }

    // ---- mark_active / is_active ----

    #[test]
    fn test_mark_and_check_single() {
        let mut tracker = ParameterTracker::new(10);
        assert!(!tracker.is_active(5));
        tracker.mark_active(5);
        assert!(tracker.is_active(5));
        assert!(!tracker.is_active(4));
        assert!(!tracker.is_active(6));
    }

    #[test]
    fn test_mark_first_neuron() {
        let mut tracker = ParameterTracker::new(100);
        tracker.mark_active(0);
        assert!(tracker.is_active(0));
        assert_eq!(tracker.num_active(), 1);
    }

    #[test]
    fn test_mark_last_neuron() {
        let mut tracker = ParameterTracker::new(100);
        tracker.mark_active(99);
        assert!(tracker.is_active(99));
        assert!(!tracker.is_active(100)); // out of bounds
    }

    #[test]
    fn test_mark_out_of_bounds_ignored() {
        let mut tracker = ParameterTracker::new(10);
        tracker.mark_active(10); // out of bounds
        tracker.mark_active(100); // way out of bounds
        assert_eq!(tracker.num_active(), 0);
    }

    #[test]
    fn test_is_active_out_of_bounds_returns_false() {
        let tracker = ParameterTracker::new(10);
        assert!(!tracker.is_active(10));
        assert!(!tracker.is_active(1000));
        assert!(!tracker.is_active(u32::MAX));
    }

    #[test]
    fn test_mark_idempotent() {
        let mut tracker = ParameterTracker::new(10);
        tracker.mark_active(3);
        tracker.mark_active(3);
        tracker.mark_active(3);
        assert!(tracker.is_active(3));
        assert_eq!(tracker.num_active(), 1);
    }

    #[test]
    fn test_mark_multiple_neurons() {
        let mut tracker = ParameterTracker::new(200);
        tracker.mark_active(0);
        tracker.mark_active(63);
        tracker.mark_active(64);
        tracker.mark_active(127);
        tracker.mark_active(128);
        tracker.mark_active(199);

        assert!(tracker.is_active(0));
        assert!(tracker.is_active(63));
        assert!(tracker.is_active(64));
        assert!(tracker.is_active(127));
        assert!(tracker.is_active(128));
        assert!(tracker.is_active(199));

        assert!(!tracker.is_active(1));
        assert!(!tracker.is_active(62));
        assert!(!tracker.is_active(65));

        assert_eq!(tracker.num_active(), 6);
    }

    #[test]
    fn test_mark_all_in_first_word() {
        let mut tracker = ParameterTracker::new(64);
        for i in 0..64 {
            tracker.mark_active(i);
        }
        assert_eq!(tracker.num_active(), 64);
        for i in 0..64 {
            assert!(tracker.is_active(i));
        }
    }

    #[test]
    fn test_mark_at_word_boundaries() {
        let mut tracker = ParameterTracker::new(256);
        // Mark the last bit of each 64-bit word
        for word in 0..4 {
            tracker.mark_active((word * 64 + 63) as u32);
        }
        assert_eq!(tracker.num_active(), 4);
        assert!(tracker.is_active(63));
        assert!(tracker.is_active(127));
        assert!(tracker.is_active(191));
        assert!(tracker.is_active(255));
    }

    // ---- active_neurons ----

    #[test]
    fn test_active_neurons_empty() {
        let tracker = ParameterTracker::new(100);
        assert!(tracker.active_neurons().is_empty());
    }

    #[test]
    fn test_active_neurons_sorted() {
        let mut tracker = ParameterTracker::new(200);
        // Mark in random order
        tracker.mark_active(150);
        tracker.mark_active(10);
        tracker.mark_active(75);
        tracker.mark_active(0);

        let active = tracker.active_neurons();
        assert_eq!(active, vec![0, 10, 75, 150]);
    }

    #[test]
    fn test_active_neurons_across_words() {
        let mut tracker = ParameterTracker::new(130);
        tracker.mark_active(0);
        tracker.mark_active(63);
        tracker.mark_active(64);
        tracker.mark_active(129);

        let active = tracker.active_neurons();
        assert_eq!(active, vec![0, 63, 64, 129]);
    }

    #[test]
    fn test_active_neurons_all_active() {
        let n = 10;
        let mut tracker = ParameterTracker::new(n);
        for i in 0..n as u32 {
            tracker.mark_active(i);
        }
        let active = tracker.active_neurons();
        let expected: Vec<u32> = (0..n as u32).collect();
        assert_eq!(active, expected);
    }

    #[test]
    fn test_active_neurons_partial_last_word() {
        // 70 neurons = 1 full word (64) + 6 bits in second word
        let mut tracker = ParameterTracker::new(70);
        tracker.mark_active(63); // last bit of first word
        tracker.mark_active(64); // first bit of second word
        tracker.mark_active(69); // last valid bit of second word

        let active = tracker.active_neurons();
        assert_eq!(active, vec![63, 64, 69]);

        // Bit 70 would be out of bounds -- mark it and verify it's ignored
        tracker.mark_active(70);
        assert_eq!(tracker.active_neurons(), vec![63, 64, 69]);
    }

    // ---- num_active ----

    #[test]
    fn test_num_active_zero() {
        let tracker = ParameterTracker::new(100);
        assert_eq!(tracker.num_active(), 0);
    }

    #[test]
    fn test_num_active_increments() {
        let mut tracker = ParameterTracker::new(100);
        assert_eq!(tracker.num_active(), 0);
        tracker.mark_active(5);
        assert_eq!(tracker.num_active(), 1);
        tracker.mark_active(10);
        assert_eq!(tracker.num_active(), 2);
        // Duplicate doesn't increment
        tracker.mark_active(5);
        assert_eq!(tracker.num_active(), 2);
    }

    // ---- reset ----

    #[test]
    fn test_reset_clears_all() {
        let mut tracker = ParameterTracker::new(200);
        for i in 0..200 {
            tracker.mark_active(i);
        }
        assert_eq!(tracker.num_active(), 200);

        tracker.reset();
        assert_eq!(tracker.num_active(), 0);
        assert!(tracker.active_neurons().is_empty());
        for i in 0..200 {
            assert!(!tracker.is_active(i));
        }
    }

    #[test]
    fn test_reset_then_reuse() {
        let mut tracker = ParameterTracker::new(100);
        tracker.mark_active(50);
        assert_eq!(tracker.num_active(), 1);

        tracker.reset();
        assert_eq!(tracker.num_active(), 0);

        tracker.mark_active(25);
        assert_eq!(tracker.num_active(), 1);
        assert!(tracker.is_active(25));
        assert!(!tracker.is_active(50));
    }

    // ---- sparsity ----

    #[test]
    fn test_sparsity_empty_tracker() {
        let tracker = ParameterTracker::new(0);
        assert!((tracker.sparsity() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_sparsity_none_active() {
        let tracker = ParameterTracker::new(100);
        assert!((tracker.sparsity() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_sparsity_all_active() {
        let mut tracker = ParameterTracker::new(64);
        for i in 0..64 {
            tracker.mark_active(i);
        }
        assert!((tracker.sparsity() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_sparsity_half_active() {
        let mut tracker = ParameterTracker::new(100);
        for i in 0..50 {
            tracker.mark_active(i);
        }
        assert!((tracker.sparsity() - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_sparsity_one_of_many() {
        let mut tracker = ParameterTracker::new(1000);
        tracker.mark_active(500);
        assert!((tracker.sparsity() - 0.001).abs() < f32::EPSILON);
    }

    // ---- mark_active_batch ----

    #[test]
    fn test_mark_active_batch_empty() {
        let mut tracker = ParameterTracker::new(10);
        tracker.mark_active_batch(&[]);
        assert_eq!(tracker.num_active(), 0);
    }

    #[test]
    fn test_mark_active_batch_multiple() {
        let mut tracker = ParameterTracker::new(100);
        tracker.mark_active_batch(&[5, 10, 50, 99]);
        assert_eq!(tracker.num_active(), 4);
        assert!(tracker.is_active(5));
        assert!(tracker.is_active(10));
        assert!(tracker.is_active(50));
        assert!(tracker.is_active(99));
    }

    #[test]
    fn test_mark_active_batch_with_duplicates() {
        let mut tracker = ParameterTracker::new(10);
        tracker.mark_active_batch(&[3, 3, 3, 7, 7]);
        assert_eq!(tracker.num_active(), 2);
        assert!(tracker.is_active(3));
        assert!(tracker.is_active(7));
    }

    #[test]
    fn test_mark_active_batch_with_out_of_bounds() {
        let mut tracker = ParameterTracker::new(10);
        tracker.mark_active_batch(&[5, 100, 200]);
        assert_eq!(tracker.num_active(), 1);
        assert!(tracker.is_active(5));
    }

    // ---- Consistency checks ----

    #[test]
    fn test_active_neurons_matches_num_active() {
        let mut tracker = ParameterTracker::new(300);
        tracker.mark_active(0);
        tracker.mark_active(100);
        tracker.mark_active(200);
        tracker.mark_active(299);
        assert_eq!(tracker.active_neurons().len(), tracker.num_active());
    }

    #[test]
    fn test_active_neurons_all_report_is_active() {
        let mut tracker = ParameterTracker::new(200);
        tracker.mark_active_batch(&[7, 63, 64, 100, 199]);
        for &id in &tracker.active_neurons() {
            assert!(tracker.is_active(id), "neuron {} should be active", id);
        }
    }

    #[test]
    fn test_large_tracker() {
        let n = 10_000;
        let mut tracker = ParameterTracker::new(n);
        // Mark every 100th neuron
        for i in (0..n).step_by(100) {
            tracker.mark_active(i as u32);
        }
        assert_eq!(tracker.num_active(), 100);
        let active = tracker.active_neurons();
        assert_eq!(active.len(), 100);
        for (i, &id) in active.iter().enumerate() {
            assert_eq!(id, (i * 100) as u32);
        }
    }

    #[test]
    fn test_sparsity_after_reset_and_remark() {
        let mut tracker = ParameterTracker::new(200);
        for i in 0..200 {
            tracker.mark_active(i);
        }
        assert!((tracker.sparsity() - 1.0).abs() < f32::EPSILON);

        tracker.reset();
        assert!((tracker.sparsity() - 0.0).abs() < f32::EPSILON);

        tracker.mark_active(0);
        assert!((tracker.sparsity() - 0.005).abs() < f32::EPSILON);
    }
}
