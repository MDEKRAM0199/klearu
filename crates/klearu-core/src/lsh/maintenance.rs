//! Hash-table rebuild scheduling.
//!
//! The [`RebuildScheduler`] controls when the LSH index should be rebuilt from
//! scratch.  Rebuilds start at a base interval and grow exponentially with
//! each successive rebuild:
//!
//! ```text
//! rebuild_i_at = sum_{j=0}^{i} floor(base * e^(lambda * j))
//! ```
//!
//! This means early rebuilds happen frequently (to adapt to rapidly changing
//! weights) while later rebuilds become increasingly rare as the network
//! converges.

/// Exponential-decay rebuild scheduler.
///
/// Tracks a step counter and determines when the next rebuild should happen.
/// The interval between rebuilds grows exponentially by a factor of
/// `e^decay` with each successive rebuild.
pub struct RebuildScheduler {
    base_interval: usize,
    decay: f64,
    rebuild_count: usize,
    next_rebuild_step: usize,
    current_step: usize,
}

impl RebuildScheduler {
    /// Create a new scheduler.
    ///
    /// # Arguments
    /// * `base_interval` - Number of steps before the first rebuild.
    /// * `decay` - Exponential growth rate (`lambda`).  A value of 0.0
    ///   gives a fixed interval; positive values make rebuilds increasingly
    ///   rare.
    pub fn new(base_interval: usize, decay: f64) -> Self {
        Self {
            base_interval,
            decay,
            rebuild_count: 0,
            next_rebuild_step: base_interval,
            current_step: 0,
        }
    }

    /// Advance by one step.  Returns `true` if a rebuild should be performed
    /// at this step.
    pub fn step(&mut self) -> bool {
        self.current_step += 1;
        if self.current_step >= self.next_rebuild_step {
            self.rebuild_count += 1;
            // Next interval = base * e^(decay * rebuild_count)
            let next_interval =
                (self.base_interval as f64 * (self.decay * self.rebuild_count as f64).exp())
                    as usize;
            // Ensure at least 1 step between rebuilds.
            self.next_rebuild_step = self.current_step + next_interval.max(1);
            true
        } else {
            false
        }
    }

    /// Check if a rebuild should happen at the given step, and update internal
    /// state.
    ///
    /// Unlike [`step`](Self::step), this method jumps the internal step counter
    /// directly to `step` rather than incrementing by one.  This is useful when
    /// the caller tracks the global step externally.
    pub fn should_rebuild(&mut self, step: u64) -> bool {
        self.current_step = step as usize;
        if self.current_step >= self.next_rebuild_step {
            self.rebuild_count += 1;
            let next_interval =
                (self.base_interval as f64 * (self.decay * self.rebuild_count as f64).exp())
                    as usize;
            self.next_rebuild_step = self.current_step + next_interval.max(1);
            true
        } else {
            false
        }
    }

    /// The step number at which the next rebuild will happen.
    pub fn next_rebuild(&self) -> usize {
        self.next_rebuild_step
    }

    /// How many rebuilds have happened so far.
    pub fn rebuild_count(&self) -> usize {
        self.rebuild_count
    }

    /// The current step number.
    pub fn current_step(&self) -> usize {
        self.current_step
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_rebuild_at_base_interval() {
        let mut s = RebuildScheduler::new(10, 0.0);
        for _ in 0..9 {
            assert!(!s.step(), "should not rebuild before base_interval");
        }
        assert!(s.step(), "should rebuild at step 10");
    }

    #[test]
    fn zero_decay_gives_fixed_interval() {
        let mut s = RebuildScheduler::new(5, 0.0);
        let mut rebuild_steps = Vec::new();
        for _ in 0..30 {
            if s.step() {
                rebuild_steps.push(s.current_step());
            }
        }
        // With decay=0, e^(0*n) = 1 for all n, so interval stays at 5.
        assert_eq!(rebuild_steps, vec![5, 10, 15, 20, 25, 30]);
    }

    #[test]
    fn positive_decay_increases_interval() {
        let mut s = RebuildScheduler::new(10, 0.5);
        let mut intervals = Vec::new();
        let mut last_rebuild = 0;
        for _ in 0..500 {
            if s.step() {
                intervals.push(s.current_step() - last_rebuild);
                last_rebuild = s.current_step();
            }
        }
        // Each successive interval should be >= the previous one.
        for window in intervals.windows(2) {
            assert!(
                window[1] >= window[0],
                "interval should not decrease: {:?}",
                intervals
            );
        }
    }

    #[test]
    fn next_rebuild_accessor() {
        let mut s = RebuildScheduler::new(10, 0.0);
        assert_eq!(s.next_rebuild(), 10);
        for _ in 0..10 {
            s.step();
        }
        // After first rebuild, next should be at 20.
        assert_eq!(s.next_rebuild(), 20);
    }

    #[test]
    fn rebuild_count_increments() {
        let mut s = RebuildScheduler::new(5, 0.0);
        assert_eq!(s.rebuild_count(), 0);
        for _ in 0..5 {
            s.step();
        }
        assert_eq!(s.rebuild_count(), 1);
        for _ in 0..5 {
            s.step();
        }
        assert_eq!(s.rebuild_count(), 2);
    }

    #[test]
    fn base_interval_one() {
        let mut s = RebuildScheduler::new(1, 0.0);
        assert!(s.step(), "should rebuild immediately at step 1");
        assert!(s.step(), "should rebuild again at step 2");
    }

    #[test]
    fn large_decay_still_rebuilds() {
        let mut s = RebuildScheduler::new(5, 2.0);
        let mut count = 0;
        for _ in 0..10000 {
            if s.step() {
                count += 1;
            }
        }
        // Even with large decay, at least one rebuild should happen.
        assert!(count >= 1);
    }

    #[test]
    fn current_step_tracks_calls() {
        let mut s = RebuildScheduler::new(100, 0.0);
        assert_eq!(s.current_step(), 0);
        s.step();
        assert_eq!(s.current_step(), 1);
        for _ in 0..9 {
            s.step();
        }
        assert_eq!(s.current_step(), 10);
    }
}
