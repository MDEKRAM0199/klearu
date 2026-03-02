use super::SamplingStrategy;
use crate::bucket::NeuronId;

/// Top-K sampling: return the K candidates with the highest hit counts.
///
/// If fewer than K candidates are available, all of them are returned.
pub struct TopKSampling {
    pub k: usize,
}

impl TopKSampling {
    /// Create a new top-K sampler.
    pub fn new(k: usize) -> Self {
        Self { k }
    }
}

impl SamplingStrategy for TopKSampling {
    fn select(&self, candidates: &[(NeuronId, u32)], _total_neurons: usize) -> Vec<NeuronId> {
        if candidates.is_empty() {
            return Vec::new();
        }

        let mut sorted: Vec<(NeuronId, u32)> = candidates.to_vec();
        // Sort descending by count. On ties, lower NeuronId first for
        // deterministic ordering.
        sorted.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        sorted
            .iter()
            .take(self.k)
            .map(|&(id, _)| id)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topk_basic() {
        let sampler = TopKSampling::new(2);
        let candidates = vec![(0, 3), (1, 5), (2, 1), (3, 4)];
        let selected = sampler.select(&candidates, 100);
        assert_eq!(selected, vec![1, 3]);
    }

    #[test]
    fn test_topk_fewer_than_k() {
        let sampler = TopKSampling::new(10);
        let candidates = vec![(0, 1), (1, 2)];
        let selected = sampler.select(&candidates, 100);
        assert_eq!(selected.len(), 2);
        // Sorted by count descending: 1 first, then 0.
        assert_eq!(selected, vec![1, 0]);
    }

    #[test]
    fn test_topk_exactly_k() {
        let sampler = TopKSampling::new(3);
        let candidates = vec![(10, 5), (20, 3), (30, 1)];
        let selected = sampler.select(&candidates, 100);
        assert_eq!(selected, vec![10, 20, 30]);
    }

    #[test]
    fn test_topk_empty_candidates() {
        let sampler = TopKSampling::new(5);
        let candidates: Vec<(NeuronId, u32)> = vec![];
        let selected = sampler.select(&candidates, 100);
        assert!(selected.is_empty());
    }

    #[test]
    fn test_topk_k_zero() {
        let sampler = TopKSampling::new(0);
        let candidates = vec![(0, 5), (1, 3)];
        let selected = sampler.select(&candidates, 100);
        assert!(selected.is_empty());
    }

    #[test]
    fn test_topk_ties_broken_by_id() {
        let sampler = TopKSampling::new(2);
        let candidates = vec![(5, 3), (2, 3), (8, 3), (1, 3)];
        let selected = sampler.select(&candidates, 100);
        // All have the same count; tiebreak by lower NeuronId first.
        assert_eq!(selected, vec![1, 2]);
    }

    #[test]
    fn test_topk_single_candidate() {
        let sampler = TopKSampling::new(5);
        let candidates = vec![(42, 10)];
        let selected = sampler.select(&candidates, 100);
        assert_eq!(selected, vec![42]);
    }

    #[test]
    fn test_topk_k_one() {
        let sampler = TopKSampling::new(1);
        let candidates = vec![(0, 1), (1, 10), (2, 5)];
        let selected = sampler.select(&candidates, 100);
        assert_eq!(selected, vec![1]);
    }

    #[test]
    fn test_topk_does_not_mutate_input() {
        let sampler = TopKSampling::new(2);
        let candidates = vec![(0, 3), (1, 5), (2, 1)];
        let _ = sampler.select(&candidates, 100);
        // Original slice is unchanged.
        assert_eq!(candidates, vec![(0, 3), (1, 5), (2, 1)]);
    }
}
