use klearu_core::bucket::NeuronId;
use klearu_core::config::LshConfig;
use klearu_core::lsh::{create_lsh_index, LshIndexTrait};

/// Sparsity predictor that uses LSH to select important heads/neurons.
/// Alternative to the MLP predictor, reusing core SLIDE infrastructure.
pub struct LshSparsityPredictor {
    /// LSH index mapping input representations to important neuron/head IDs.
    index: Box<dyn LshIndexTrait>,
    /// Number of items (heads or neurons) to select.
    top_k: usize,
}

impl LshSparsityPredictor {
    pub fn new(config: LshConfig, input_dim: usize, top_k: usize, seed: u64) -> Self {
        let index = create_lsh_index(&config, input_dim, seed);
        Self { index, top_k }
    }

    /// Build the LSH index from observed (input, important_indices) pairs.
    /// For each important index, insert the input vector under that index's ID.
    pub fn build_from_observations(&mut self, observations: &[(Vec<f32>, Vec<usize>)]) {
        self.index.clear();
        for (input, important_indices) in observations {
            for &idx in important_indices {
                self.index.insert(idx as NeuronId, input);
            }
        }
    }

    /// Predict which heads/neurons to activate for given input.
    pub fn predict(&self, input: &[f32]) -> Vec<usize> {
        let candidates = self.index.query(input);
        let mut sorted: Vec<(NeuronId, u32)> =
            candidates.iter().map(|c| (c.neuron_id, c.count)).collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        sorted
            .into_iter()
            .take(self.top_k)
            .map(|(id, _)| id as usize)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lsh_predictor_empty_returns_empty() {
        let config = LshConfig::default();
        let predictor = LshSparsityPredictor::new(config, 32, 5, 42);
        let input = vec![1.0f32; 32];
        let result = predictor.predict(&input);
        assert!(
            result.is_empty(),
            "Empty index should return no predictions"
        );
    }

    #[test]
    fn test_lsh_predictor_returns_relevant_indices() {
        let config = LshConfig {
            num_tables: 10,
            num_hashes: 4,
            ..LshConfig::default()
        };
        let dim = 16;
        let mut predictor = LshSparsityPredictor::new(config, dim, 3, 42);

        // Build observations: associate specific input patterns with specific indices
        // Use well-separated vectors for different neuron groups
        let obs: Vec<(Vec<f32>, Vec<usize>)> = vec![
            // Positive direction associated with neurons 0, 1
            (vec![1.0; dim], vec![0, 1]),
            (vec![0.9; dim], vec![0, 1]),
            (vec![0.8; dim], vec![0, 1]),
            // Negative direction associated with neurons 5, 6
            (vec![-1.0; dim], vec![5, 6]),
            (vec![-0.9; dim], vec![5, 6]),
            (vec![-0.8; dim], vec![5, 6]),
        ];
        predictor.build_from_observations(&obs);

        // Query with positive input => should find neurons 0, 1
        let result = predictor.predict(&vec![1.0; dim]);
        assert!(
            !result.is_empty(),
            "Should return at least some predictions"
        );

        // Check that the returned indices are from the set we inserted
        let all_inserted: Vec<usize> = vec![0, 1, 5, 6];
        for &idx in &result {
            assert!(
                all_inserted.contains(&idx),
                "Returned index {} should be one we inserted",
                idx
            );
        }
    }

    #[test]
    fn test_lsh_predictor_respects_top_k() {
        let config = LshConfig {
            num_tables: 10,
            num_hashes: 4,
            ..LshConfig::default()
        };
        let dim = 16;
        let mut predictor = LshSparsityPredictor::new(config, dim, 2, 42);

        let obs: Vec<(Vec<f32>, Vec<usize>)> = vec![
            (vec![1.0; dim], vec![0, 1, 2, 3, 4]),
            (vec![0.9; dim], vec![0, 1, 2, 3, 4]),
        ];
        predictor.build_from_observations(&obs);

        let result = predictor.predict(&vec![1.0; dim]);
        assert!(
            result.len() <= 2,
            "Should return at most top_k=2 results, got {}",
            result.len()
        );
    }

    #[test]
    fn test_lsh_predictor_rebuild() {
        let config = LshConfig {
            num_tables: 10,
            num_hashes: 4,
            ..LshConfig::default()
        };
        let dim = 16;
        let mut predictor = LshSparsityPredictor::new(config, dim, 5, 42);

        // Build initial index
        let obs1: Vec<(Vec<f32>, Vec<usize>)> =
            vec![(vec![1.0; dim], vec![0, 1])];
        predictor.build_from_observations(&obs1);

        // Rebuild with new observations
        let obs2: Vec<(Vec<f32>, Vec<usize>)> =
            vec![(vec![1.0; dim], vec![10, 11])];
        predictor.build_from_observations(&obs2);

        let result = predictor.predict(&vec![1.0; dim]);
        // After rebuild, old indices (0, 1) should not appear
        for &idx in &result {
            assert!(
                idx == 10 || idx == 11,
                "After rebuild, should only see new indices, got {}",
                idx
            );
        }
    }
}
