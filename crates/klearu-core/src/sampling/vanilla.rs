use super::SamplingStrategy;
use crate::bucket::NeuronId;

/// Vanilla sampling: return all candidates (the union of bucket matches),
/// ignoring hit counts entirely.
pub struct VanillaSampling;

impl SamplingStrategy for VanillaSampling {
    fn select(&self, candidates: &[(NeuronId, u32)], _total_neurons: usize) -> Vec<NeuronId> {
        candidates.iter().map(|&(id, _)| id).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vanilla_returns_all_candidates() {
        let sampler = VanillaSampling;
        let candidates = vec![(0, 3), (5, 1), (10, 2)];
        let selected = sampler.select(&candidates, 100);
        assert_eq!(selected, vec![0, 5, 10]);
    }

    #[test]
    fn test_vanilla_empty_candidates() {
        let sampler = VanillaSampling;
        let candidates: Vec<(NeuronId, u32)> = vec![];
        let selected = sampler.select(&candidates, 100);
        assert!(selected.is_empty());
    }

    #[test]
    fn test_vanilla_single_candidate() {
        let sampler = VanillaSampling;
        let candidates = vec![(42, 1)];
        let selected = sampler.select(&candidates, 1000);
        assert_eq!(selected, vec![42]);
    }

    #[test]
    fn test_vanilla_ignores_counts() {
        let sampler = VanillaSampling;
        let candidates = vec![(1, 100), (2, 0), (3, 50)];
        let selected = sampler.select(&candidates, 10);
        // All candidates returned regardless of count.
        assert_eq!(selected.len(), 3);
        assert_eq!(selected, vec![1, 2, 3]);
    }

    #[test]
    fn test_vanilla_ignores_total_neurons() {
        let sampler = VanillaSampling;
        let candidates = vec![(0, 1), (1, 1)];
        // total_neurons parameter should not affect the result.
        let a = sampler.select(&candidates, 10);
        let b = sampler.select(&candidates, 1_000_000);
        assert_eq!(a, b);
    }

    #[test]
    fn test_vanilla_preserves_order() {
        let sampler = VanillaSampling;
        let candidates = vec![(99, 1), (3, 5), (50, 2), (7, 10)];
        let selected = sampler.select(&candidates, 100);
        assert_eq!(selected, vec![99, 3, 50, 7]);
    }
}
