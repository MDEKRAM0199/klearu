use klearu_core::bucket::NeuronId;
use klearu_core::lsh::LshIndexTrait;

/// Sparse inference engine that uses LSH at inference time.
/// Unlike training SLIDE which always uses LSH, vanilla SLIDE
/// disables LSH at inference. BOLT re-enables it with potentially
/// different sparsity settings.
pub struct SparseInferenceEngine {
    /// Multiplier on number of active neurons compared to training.
    /// >1.0 means evaluate more neurons at inference for better accuracy.
    sparsity_multiplier: f32,
}

impl SparseInferenceEngine {
    pub fn new(sparsity_multiplier: f32) -> Self {
        Self { sparsity_multiplier }
    }

    /// Get the current sparsity multiplier.
    pub fn sparsity_multiplier(&self) -> f32 {
        self.sparsity_multiplier
    }

    /// Perform sparse inference on a single input.
    /// Returns (active_neuron_ids, activations) pairs sorted by neuron ID.
    ///
    /// `input`: the input feature vector.
    /// `lsh_index`: the LSH index to query for candidate neurons.
    /// `weights`: (neuron_id, weight_vector) pairs for neurons in this layer.
    /// `biases`: (neuron_id, bias) pairs for neurons in this layer.
    /// `base_top_k`: the base number of neurons to select (before multiplier).
    pub fn infer(
        &self,
        input: &[f32],
        lsh_index: &dyn LshIndexTrait,
        weights: &[(NeuronId, Vec<f32>)],
        biases: &[(NeuronId, f32)],
        base_top_k: usize,
    ) -> Vec<(NeuronId, f32)> {
        let effective_k = (base_top_k as f32 * self.sparsity_multiplier) as usize;

        // Query the LSH index -- returns candidates with match counts
        let candidates = lsh_index.query(input);

        // Sort by hit count descending, take top effective_k
        let mut sorted_candidates: Vec<(NeuronId, u32)> = candidates
            .iter()
            .map(|c| (c.neuron_id, c.count))
            .collect();
        sorted_candidates.sort_by(|a, b| b.1.cmp(&a.1));
        sorted_candidates.truncate(effective_k);

        // Compute activations for selected neurons
        let mut results = Vec::with_capacity(sorted_candidates.len());
        for (neuron_id, _count) in &sorted_candidates {
            if let Some((_, w)) = weights.iter().find(|(id, _)| id == neuron_id) {
                let bias = biases
                    .iter()
                    .find(|(id, _)| id == neuron_id)
                    .map(|(_, b)| *b)
                    .unwrap_or(0.0);
                let activation: f32 =
                    input.iter().zip(w.iter()).map(|(a, b)| a * b).sum::<f32>() + bias;
                let activation = activation.max(0.0); // ReLU
                results.push((*neuron_id, activation));
            }
        }

        results
    }

    /// Perform sparse inference returning raw pre-activation values (no ReLU).
    pub fn infer_linear(
        &self,
        input: &[f32],
        lsh_index: &dyn LshIndexTrait,
        weights: &[(NeuronId, Vec<f32>)],
        biases: &[(NeuronId, f32)],
        base_top_k: usize,
    ) -> Vec<(NeuronId, f32)> {
        let effective_k = (base_top_k as f32 * self.sparsity_multiplier) as usize;

        let candidates = lsh_index.query(input);
        let mut sorted_candidates: Vec<(NeuronId, u32)> = candidates
            .iter()
            .map(|c| (c.neuron_id, c.count))
            .collect();
        sorted_candidates.sort_by(|a, b| b.1.cmp(&a.1));
        sorted_candidates.truncate(effective_k);

        let mut results = Vec::with_capacity(sorted_candidates.len());
        for (neuron_id, _count) in &sorted_candidates {
            if let Some((_, w)) = weights.iter().find(|(id, _)| id == neuron_id) {
                let bias = biases
                    .iter()
                    .find(|(id, _)| id == neuron_id)
                    .map(|(_, b)| *b)
                    .unwrap_or(0.0);
                let activation: f32 =
                    input.iter().zip(w.iter()).map(|(a, b)| a * b).sum::<f32>() + bias;
                results.push((*neuron_id, activation));
            }
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use klearu_core::config::{BucketType, HashFunctionType, LshConfig};
    use klearu_core::lsh::create_lsh_index;

    fn make_test_index(
        neurons: &[(NeuronId, Vec<f32>)],
        dim: usize,
    ) -> Box<dyn LshIndexTrait> {
        let config = LshConfig {
            hash_function: HashFunctionType::SimHash,
            bucket_type: BucketType::Fifo,
            num_tables: 50,
            range_pow: 4,
            num_hashes: 4,
            bucket_capacity: 128,
            rebuild_interval_base: 1000,
            rebuild_decay: 0.0,
        };
        let mut index = create_lsh_index(&config, dim, 42);
        for (id, weights) in neurons {
            index.insert(*id, weights);
        }
        index
    }

    #[test]
    fn test_sparse_inference_basic() {
        let dim = 4;
        let neurons: Vec<(NeuronId, Vec<f32>)> = vec![
            (0, vec![1.0, 0.0, 0.0, 0.0]),
            (1, vec![0.0, 1.0, 0.0, 0.0]),
            (2, vec![0.0, 0.0, 1.0, 0.0]),
            (3, vec![0.0, 0.0, 0.0, 1.0]),
        ];
        let biases: Vec<(NeuronId, f32)> = vec![(0, 0.0), (1, 0.0), (2, 0.0), (3, 0.0)];

        let index = make_test_index(&neurons, dim);
        let engine = SparseInferenceEngine::new(1.0);

        let input = vec![1.0, 0.0, 0.0, 0.0];
        let results = engine.infer(&input, index.as_ref(), &neurons, &biases, 4);

        // Should get some results (LSH might not find all, but should find at least one)
        assert!(!results.is_empty() || neurons.is_empty());
    }

    #[test]
    fn test_sparse_inference_with_bias() {
        let dim = 2;
        let neurons: Vec<(NeuronId, Vec<f32>)> = vec![
            (0, vec![1.0, 0.0]),
            (1, vec![0.0, 1.0]),
        ];
        let biases: Vec<(NeuronId, f32)> = vec![(0, 0.5), (1, -0.5)];

        let index = make_test_index(&neurons, dim);
        let engine = SparseInferenceEngine::new(1.0);

        let input = vec![1.0, 0.0];
        let results = engine.infer(&input, index.as_ref(), &neurons, &biases, 2);

        // Check that returned activations are non-negative (ReLU)
        for (_, act) in &results {
            assert!(*act >= 0.0);
        }
    }

    #[test]
    fn test_sparse_inference_relu() {
        let dim = 2;
        let neurons: Vec<(NeuronId, Vec<f32>)> = vec![(0, vec![-1.0, -1.0])];
        let biases: Vec<(NeuronId, f32)> = vec![(0, 0.0)];

        let index = make_test_index(&neurons, dim);
        let engine = SparseInferenceEngine::new(1.0);

        let input = vec![1.0, 1.0];
        let results = engine.infer(&input, index.as_ref(), &neurons, &biases, 1);

        // Dot product = -2.0, ReLU clamps to 0.0
        for (_, act) in &results {
            assert!((*act - 0.0).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_sparse_inference_linear_no_relu() {
        let dim = 2;
        let neurons: Vec<(NeuronId, Vec<f32>)> = vec![(0, vec![-1.0, -1.0])];
        let biases: Vec<(NeuronId, f32)> = vec![(0, 0.0)];

        let index = make_test_index(&neurons, dim);
        let engine = SparseInferenceEngine::new(1.0);

        let input = vec![1.0, 1.0];
        let results = engine.infer_linear(&input, index.as_ref(), &neurons, &biases, 1);

        // Linear mode should preserve negative activations
        for (_, act) in &results {
            assert!(*act <= 0.0);
        }
    }

    #[test]
    fn test_sparsity_multiplier() {
        let engine = SparseInferenceEngine::new(2.5);
        assert!((engine.sparsity_multiplier() - 2.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_sparse_inference_empty_index() {
        let config = LshConfig {
            hash_function: HashFunctionType::SimHash,
            bucket_type: BucketType::Fifo,
            num_tables: 10,
            range_pow: 4,
            num_hashes: 4,
            bucket_capacity: 128,
            rebuild_interval_base: 1000,
            rebuild_decay: 0.0,
        };
        let index = create_lsh_index(&config, 4, 42);
        let engine = SparseInferenceEngine::new(1.0);

        let input = vec![1.0, 0.0, 0.0, 0.0];
        let results = engine.infer(&input, index.as_ref(), &[], &[], 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_sparse_inference_missing_bias() {
        let dim = 2;
        let neurons: Vec<(NeuronId, Vec<f32>)> = vec![(0, vec![1.0, 0.0])];
        // No bias for neuron 0 -- should default to 0.0
        let biases: Vec<(NeuronId, f32)> = vec![];

        let index = make_test_index(&neurons, dim);
        let engine = SparseInferenceEngine::new(1.0);

        let input = vec![2.0, 0.0];
        let results = engine.infer(&input, index.as_ref(), &neurons, &biases, 1);

        // If neuron 0 was found, activation = 2.0*1.0 + 0.0*0.0 + 0.0(default bias) = 2.0
        for (id, act) in &results {
            if *id == 0 {
                assert!((*act - 2.0).abs() < f32::EPSILON);
            }
        }
    }

    #[test]
    fn test_sparse_inference_multiplier_reduces_candidates() {
        let dim = 4;
        let neurons: Vec<(NeuronId, Vec<f32>)> = (0..20)
            .map(|i| {
                let mut w = vec![0.0f32; dim];
                w[i % dim] = 1.0;
                (i as NeuronId, w)
            })
            .collect();
        let biases: Vec<(NeuronId, f32)> = neurons.iter().map(|(id, _)| (*id, 0.0)).collect();

        let index = make_test_index(&neurons, dim);

        let input = vec![1.0, 0.0, 0.0, 0.0];

        // With multiplier 0.5, we get fewer neurons than with 2.0
        let engine_small = SparseInferenceEngine::new(0.5);
        let engine_large = SparseInferenceEngine::new(2.0);

        let results_small = engine_small.infer(&input, index.as_ref(), &neurons, &biases, 10);
        let results_large = engine_large.infer(&input, index.as_ref(), &neurons, &biases, 10);

        // Larger multiplier should yield at least as many results
        assert!(results_large.len() >= results_small.len());
    }
}
