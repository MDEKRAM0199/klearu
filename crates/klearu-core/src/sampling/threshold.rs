use super::SamplingStrategy;
use crate::bucket::NeuronId;

/// Threshold sampling: return all candidates whose hit count meets or exceeds
/// a configured threshold.
///
/// As a fallback, if no candidate meets the threshold, the single candidate
/// with the highest hit count is returned (top-1).
pub struct ThresholdSampling {
    pub threshold: u32,
}

impl ThresholdSampling {
    /// Create a new threshold sampler.
    pub fn new(threshold: u32) -> Self {
        Self { threshold }
    }
}

impl SamplingStrategy for ThresholdSampling {
    fn select(&self, candidates: &[(NeuronId, u32)], _total_neurons: usize) -> Vec<NeuronId> {
        if candidates.is_empty() {
            return Vec::new();
        }

        let passing: Vec<NeuronId> = candidates
            .iter()
            .filter(|&&(_, count)| count >= self.threshold)
            .map(|&(id, _)| id)
            .collect();

        if !passing.is_empty() {
            return passing;
        }

        // Fallback: return the single best candidate (top-1).
        // On ties, pick the lowest NeuronId for determinism.
        let best = candidates
            .iter()
            .max_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.cmp(&a.0)))
            .unwrap();
        vec![best.0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_threshold_basic() {
        let sampler = ThresholdSampling::new(3);
        let candidates = vec![(0, 5), (1, 2), (2, 3), (3, 1)];
        let selected = sampler.select(&candidates, 100);
        // Only candidates with count >= 3: 0 (5) and 2 (3).
        assert_eq!(selected, vec![0, 2]);
    }

    #[test]
    fn test_threshold_all_pass() {
        let sampler = ThresholdSampling::new(1);
        let candidates = vec![(0, 5), (1, 2), (2, 3)];
        let selected = sampler.select(&candidates, 100);
        assert_eq!(selected, vec![0, 1, 2]);
    }

    #[test]
    fn test_threshold_none_pass_fallback_to_top1() {
        let sampler = ThresholdSampling::new(10);
        let candidates = vec![(0, 3), (1, 5), (2, 1)];
        let selected = sampler.select(&candidates, 100);
        // None meet threshold 10, so fallback to top-1 by count: neuron 1 (count=5).
        assert_eq!(selected, vec![1]);
    }

    #[test]
    fn test_threshold_fallback_tie_picks_lowest_id() {
        let sampler = ThresholdSampling::new(100);
        let candidates = vec![(5, 3), (2, 3), (8, 3)];
        let selected = sampler.select(&candidates, 100);
        // All tied at 3, none pass 100. Fallback picks lowest NeuronId = 2.
        assert_eq!(selected, vec![2]);
    }

    #[test]
    fn test_threshold_empty_candidates() {
        let sampler = ThresholdSampling::new(1);
        let candidates: Vec<(NeuronId, u32)> = vec![];
        let selected = sampler.select(&candidates, 100);
        assert!(selected.is_empty());
    }

    #[test]
    fn test_threshold_single_candidate_passes() {
        let sampler = ThresholdSampling::new(1);
        let candidates = vec![(42, 1)];
        let selected = sampler.select(&candidates, 100);
        assert_eq!(selected, vec![42]);
    }

    #[test]
    fn test_threshold_single_candidate_does_not_pass() {
        let sampler = ThresholdSampling::new(5);
        let candidates = vec![(42, 2)];
        let selected = sampler.select(&candidates, 100);
        // Fallback returns the only candidate.
        assert_eq!(selected, vec![42]);
    }

    #[test]
    fn test_threshold_zero_means_all_pass() {
        let sampler = ThresholdSampling::new(0);
        let candidates = vec![(0, 0), (1, 0), (2, 0)];
        let selected = sampler.select(&candidates, 100);
        assert_eq!(selected, vec![0, 1, 2]);
    }

    #[test]
    fn test_threshold_preserves_candidate_order() {
        let sampler = ThresholdSampling::new(2);
        let candidates = vec![(99, 5), (3, 2), (50, 10), (7, 3)];
        let selected = sampler.select(&candidates, 100);
        // Order follows the input order for those that pass.
        assert_eq!(selected, vec![99, 3, 50, 7]);
    }

    #[test]
    fn test_threshold_exact_boundary() {
        let sampler = ThresholdSampling::new(5);
        let candidates = vec![(0, 4), (1, 5), (2, 6)];
        let selected = sampler.select(&candidates, 100);
        // 4 < 5 fails, 5 >= 5 passes, 6 >= 5 passes.
        assert_eq!(selected, vec![1, 2]);
    }
}
