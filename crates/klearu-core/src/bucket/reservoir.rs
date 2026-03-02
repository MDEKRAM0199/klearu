use super::{Bucket, NeuronId};
use serde::{Deserialize, Serialize};

/// Reservoir-sampling bucket.
///
/// Uses Algorithm R (Vitter, 1985): the first `capacity` items are stored
/// directly; each subsequent item replaces a uniformly random existing item
/// with probability `capacity / count`.  This guarantees that at any point
/// every item seen so far had an equal probability of being in the bucket.
///
/// The RNG is deterministic and derived solely from the running insertion
/// count, so no RNG state needs to be serialised.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReservoirBucket {
    data: Vec<NeuronId>,
    capacity: usize,
    /// Total number of insertions ever performed (including evicted items).
    count: u64,
}

impl ReservoirBucket {
    /// Create a new reservoir bucket with the given maximum capacity.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "bucket capacity must be positive");
        Self {
            data: Vec::with_capacity(capacity),
            capacity,
            count: 0,
        }
    }

    /// Total number of items ever inserted (including those that were evicted
    /// or rejected by the reservoir sampling step).
    pub fn total_insertions(&self) -> u64 {
        self.count
    }

    /// Simple deterministic hash used as a lightweight PRNG seeded by the
    /// insertion counter.  Returns a value in `[0, bound)`.
    fn deterministic_rand(seed: u64, bound: u64) -> u64 {
        // SplitMix64 -- a fast, well-distributed bijective mixer.
        let mut z = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z = z ^ (z >> 31);
        z % bound
    }
}

impl Bucket for ReservoirBucket {
    fn insert(&mut self, id: NeuronId) {
        self.count += 1;

        if self.data.len() < self.capacity {
            self.data.push(id);
        } else {
            // With probability capacity / count, replace a random element.
            // Mix the neuron id into the seed so that different ids inserted
            // at the same count position yield different random outcomes.
            let seed = self.count.wrapping_mul(0x517cc1b727220a95).wrapping_add(id as u64);
            let j = Self::deterministic_rand(seed, self.count);
            if j < self.capacity as u64 {
                let slot = Self::deterministic_rand(seed.wrapping_add(1), self.capacity as u64);
                self.data[slot as usize] = id;
            }
        }
    }

    fn remove(&mut self, id: NeuronId) -> bool {
        if let Some(pos) = self.data.iter().position(|&x| x == id) {
            self.data.swap_remove(pos);
            true
        } else {
            false
        }
    }

    fn contents(&self) -> &[NeuronId] {
        &self.data
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn clear(&mut self) {
        self.data.clear();
        self.count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_bucket() {
        let b = ReservoirBucket::new(4);
        assert!(b.is_empty());
        assert_eq!(b.len(), 0);
        assert_eq!(b.total_insertions(), 0);
    }

    #[test]
    #[should_panic(expected = "bucket capacity must be positive")]
    fn test_zero_capacity_panics() {
        ReservoirBucket::new(0);
    }

    #[test]
    fn test_insert_within_capacity() {
        let mut b = ReservoirBucket::new(4);
        b.insert(1);
        b.insert(2);
        b.insert(3);
        assert_eq!(b.len(), 3);
        assert_eq!(b.total_insertions(), 3);
        // All three must be present when under capacity.
        assert!(b.contents().contains(&1));
        assert!(b.contents().contains(&2));
        assert!(b.contents().contains(&3));
    }

    #[test]
    fn test_insert_at_capacity_does_not_exceed() {
        let mut b = ReservoirBucket::new(3);
        for i in 0..100 {
            b.insert(i);
        }
        assert!(b.len() <= 3);
        assert_eq!(b.total_insertions(), 100);
    }

    #[test]
    fn test_deterministic_behaviour() {
        // Two buckets with the same capacity and same insertion sequence
        // must produce identical contents.
        let mut b1 = ReservoirBucket::new(4);
        let mut b2 = ReservoirBucket::new(4);
        for i in 0..50 {
            b1.insert(i);
            b2.insert(i);
        }
        assert_eq!(b1.contents(), b2.contents());
    }

    #[test]
    fn test_remove_existing() {
        let mut b = ReservoirBucket::new(4);
        b.insert(10);
        b.insert(20);
        b.insert(30);
        assert!(b.remove(20));
        assert_eq!(b.len(), 2);
        assert!(!b.contents().contains(&20));
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut b = ReservoirBucket::new(4);
        b.insert(1);
        assert!(!b.remove(99));
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_remove_from_empty() {
        let mut b = ReservoirBucket::new(4);
        assert!(!b.remove(1));
    }

    #[test]
    fn test_clear() {
        let mut b = ReservoirBucket::new(4);
        b.insert(1);
        b.insert(2);
        b.clear();
        assert!(b.is_empty());
        assert_eq!(b.len(), 0);
        assert_eq!(b.total_insertions(), 0);
    }

    #[test]
    fn test_capacity_one() {
        let mut b = ReservoirBucket::new(1);
        b.insert(10);
        assert_eq!(b.len(), 1);
        // Insert many more -- bucket should never exceed capacity 1.
        for i in 11..100 {
            b.insert(i);
        }
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_reservoir_sampling_fairness() {
        // Statistical test: insert 0..=9 into capacity-5 bucket 10000 times,
        // count how often each id appears.  Every id should appear roughly
        // 50% of the time (5/10).  We allow a generous margin.
        let trials = 10_000;
        let mut counts = [0u32; 10];

        for trial in 0..trials {
            let mut b = ReservoirBucket::new(5);
            // Use a different "universe" offset per trial to avoid identical
            // deterministic sequences.  We vary the IDs so the deterministic
            // RNG gets exercised differently.
            for i in 0..10u32 {
                // Inject trial number into the id so each trial is distinct.
                b.insert(i + trial * 1000);
            }
            for &id in b.contents() {
                let original = (id - trial * 1000) as usize;
                counts[original] += 1;
            }
        }

        let expected = (trials * 5 / 10) as f64; // 5000
        for (id, &c) in counts.iter().enumerate() {
            let ratio = c as f64 / expected;
            assert!(
                (0.5..=1.5).contains(&ratio),
                "id {id} appeared {c} times, expected ~{expected} (ratio {ratio:.2})"
            );
        }
    }

    #[test]
    fn test_serde_roundtrip() {
        let mut b = ReservoirBucket::new(4);
        b.insert(1);
        b.insert(2);
        b.insert(3);

        let json = serde_json::to_string(&b).unwrap();
        let b2: ReservoirBucket = serde_json::from_str(&json).unwrap();
        assert_eq!(b.contents(), b2.contents());
        assert_eq!(b.total_insertions(), b2.total_insertions());
    }
}
