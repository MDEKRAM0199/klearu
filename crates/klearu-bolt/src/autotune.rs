use klearu_core::config::{BucketType, HashFunctionType, LshConfig};
use klearu_core::lsh::create_lsh_index;

/// Result of autotuning.
#[derive(Debug, Clone)]
pub struct AutotuneResult {
    pub best_k: usize,
    pub best_l: usize,
    pub recall: f32,
    pub query_cost: f64,
    pub configs_evaluated: usize,
}

/// LSH autotuner that searches over K (num hashes per table) and L (num tables)
/// to find the cheapest configuration meeting a target recall.
pub struct LshAutotuner {
    /// Target recall (fraction of true nearest neighbors found).
    target_recall: f32,
    /// Safety factor c1: reliability (1.0 = 95th percentile).
    #[allow(dead_code)]
    safety_factor: f32,
    /// Speedup ratio c2: fraction of neurons to evaluate (0.1 = 10x potential speedup).
    speedup_ratio: f32,
    /// Range of K values to search.
    k_range: (usize, usize),
    /// Range of L values to search.
    l_range: (usize, usize),
    /// Number of sample queries for evaluation.
    #[allow(dead_code)]
    num_samples: usize,
}

impl LshAutotuner {
    pub fn new(target_recall: f32) -> Self {
        Self {
            target_recall,
            safety_factor: 1.0,
            speedup_ratio: 0.1,
            k_range: (4, 16),
            l_range: (10, 200),
            num_samples: 100,
        }
    }

    pub fn with_k_range(mut self, min: usize, max: usize) -> Self {
        self.k_range = (min, max);
        self
    }

    pub fn with_l_range(mut self, min: usize, max: usize) -> Self {
        self.l_range = (min, max);
        self
    }

    pub fn with_num_samples(mut self, n: usize) -> Self {
        self.num_samples = n;
        self
    }

    pub fn with_speedup_ratio(mut self, ratio: f32) -> Self {
        self.speedup_ratio = ratio;
        self
    }

    /// Run autotuning given a set of neuron weights and sample queries.
    /// `neurons`: (id, weight_vector) pairs for all neurons in the layer.
    /// `queries`: sample input vectors to evaluate recall on.
    /// Returns the best configuration found.
    pub fn autotune(
        &self,
        neurons: &[(u32, Vec<f32>)],
        queries: &[Vec<f32>],
        seed: u64,
    ) -> AutotuneResult {
        // Compute brute-force top-k for each query (ground truth)
        let k_neighbors = (neurons.len() as f32 * self.speedup_ratio).max(1.0) as usize;
        let ground_truth = self.brute_force_topk(neurons, queries, k_neighbors);

        let mut best = AutotuneResult {
            best_k: self.k_range.0,
            best_l: self.l_range.0,
            recall: 0.0,
            query_cost: f64::MAX,
            configs_evaluated: 0,
        };

        // Grid search over K and L
        let k_step = 2;
        let l_step = 10;
        let mut k = self.k_range.0;
        while k <= self.k_range.1 {
            let mut l = self.l_range.0;
            while l <= self.l_range.1 {
                let (recall, cost) = self.evaluate_config(
                    neurons,
                    queries,
                    &ground_truth,
                    k,
                    l,
                    seed,
                );
                best.configs_evaluated += 1;

                if recall >= self.target_recall && cost < best.query_cost {
                    best.best_k = k;
                    best.best_l = l;
                    best.recall = recall;
                    best.query_cost = cost;
                }

                l += l_step;
            }
            k += k_step;
        }

        best
    }

    fn brute_force_topk(
        &self,
        neurons: &[(u32, Vec<f32>)],
        queries: &[Vec<f32>],
        k: usize,
    ) -> Vec<Vec<u32>> {
        // For each query, compute dot product with all neurons, return top-k ids
        queries
            .iter()
            .map(|query| {
                let mut scores: Vec<(u32, f32)> = neurons
                    .iter()
                    .map(|(id, w)| {
                        let dot: f32 = query.iter().zip(w.iter()).map(|(a, b)| a * b).sum();
                        (*id, dot)
                    })
                    .collect();
                scores.sort_by(|a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                scores.iter().take(k).map(|(id, _)| *id).collect()
            })
            .collect()
    }

    fn evaluate_config(
        &self,
        neurons: &[(u32, Vec<f32>)],
        queries: &[Vec<f32>],
        ground_truth: &[Vec<u32>],
        k: usize,
        l: usize,
        seed: u64,
    ) -> (f32, f64) {
        // Build LSH index with this config
        let config = LshConfig {
            hash_function: HashFunctionType::SimHash,
            bucket_type: BucketType::Fifo,
            num_tables: l,
            range_pow: k,
            num_hashes: k,
            bucket_capacity: 128,
            rebuild_interval_base: 1000,
            rebuild_decay: 0.0,
        };

        let input_dim = if neurons.is_empty() {
            queries.first().map(|q| q.len()).unwrap_or(1)
        } else {
            neurons[0].1.len()
        };
        let mut index = create_lsh_index(&config, input_dim, seed);

        for (id, weights) in neurons {
            index.insert(*id, weights);
        }

        let mut total_recall = 0.0f32;
        let mut total_candidates = 0usize;

        for (query, truth) in queries.iter().zip(ground_truth.iter()) {
            let candidates = index.query(query);
            total_candidates += candidates.len();

            // Collect candidate neuron IDs into a set for fast lookup
            let candidate_ids: std::collections::HashSet<u32> =
                candidates.iter().map(|c| c.neuron_id).collect();

            let hits: usize = truth.iter().filter(|id| candidate_ids.contains(id)).count();
            total_recall += hits as f32 / truth.len().max(1) as f32;
        }

        let avg_recall = total_recall / queries.len().max(1) as f32;
        let avg_cost = total_candidates as f64 / queries.len().max(1) as f64;

        (avg_recall, avg_cost)
    }

    /// Apply autotuning result to an LSH config.
    pub fn apply_result(result: &AutotuneResult, config: &mut LshConfig) {
        config.num_hashes = result.best_k;
        config.range_pow = result.best_k;
        config.num_tables = result.best_l;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use rand::SeedableRng;

    /// Generate random neurons with the given dimensionality.
    fn random_neurons(
        count: usize,
        dim: usize,
        rng: &mut impl Rng,
    ) -> Vec<(u32, Vec<f32>)> {
        (0..count)
            .map(|i| {
                let w: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
                (i as u32, w)
            })
            .collect()
    }

    /// Generate random query vectors.
    fn random_queries(count: usize, dim: usize, rng: &mut impl Rng) -> Vec<Vec<f32>> {
        (0..count)
            .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect()
    }

    #[test]
    fn test_autotuner_creation() {
        let tuner = LshAutotuner::new(0.9);
        assert!((tuner.target_recall - 0.9).abs() < f32::EPSILON);
        assert_eq!(tuner.k_range, (4, 16));
        assert_eq!(tuner.l_range, (10, 200));
    }

    #[test]
    fn test_autotuner_builder() {
        let tuner = LshAutotuner::new(0.8)
            .with_k_range(2, 8)
            .with_l_range(5, 50)
            .with_num_samples(50)
            .with_speedup_ratio(0.2);
        assert_eq!(tuner.k_range, (2, 8));
        assert_eq!(tuner.l_range, (5, 50));
        assert_eq!(tuner.num_samples, 50);
        assert!((tuner.speedup_ratio - 0.2).abs() < f32::EPSILON);
    }

    #[test]
    fn test_brute_force_topk() {
        let tuner = LshAutotuner::new(0.9).with_speedup_ratio(0.5);
        // 4 neurons, dim=2
        let neurons = vec![
            (0, vec![1.0, 0.0]),
            (1, vec![0.0, 1.0]),
            (2, vec![-1.0, 0.0]),
            (3, vec![0.0, -1.0]),
        ];
        // Query aligned with neuron 0
        let queries = vec![vec![1.0, 0.0]];
        let topk = tuner.brute_force_topk(&neurons, &queries, 2);
        assert_eq!(topk.len(), 1);
        assert_eq!(topk[0].len(), 2);
        // Neuron 0 should be the best match (dot=1.0), neuron 1 and 3 have dot=0
        assert_eq!(topk[0][0], 0);
    }

    #[test]
    fn test_brute_force_topk_ordering() {
        let tuner = LshAutotuner::new(0.9);
        let neurons = vec![
            (0, vec![0.1, 0.0]),
            (1, vec![0.5, 0.5]),
            (2, vec![1.0, 0.0]),
        ];
        let queries = vec![vec![1.0, 0.0]];
        let topk = tuner.brute_force_topk(&neurons, &queries, 3);
        // Dots: neuron0=0.1, neuron1=0.5, neuron2=1.0
        // Order: 2, 1, 0
        assert_eq!(topk[0], vec![2, 1, 0]);
    }

    #[test]
    fn test_autotune_small() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let neurons = random_neurons(20, 8, &mut rng);
        let queries = random_queries(5, 8, &mut rng);

        // Use a very low target recall so we find at least one valid config
        let tuner = LshAutotuner::new(0.0)
            .with_k_range(4, 6)
            .with_l_range(10, 20)
            .with_speedup_ratio(0.5);

        let result = tuner.autotune(&neurons, &queries, 42);
        assert!(result.configs_evaluated > 0);
        // With recall target 0.0, we should always find a config
        assert!(result.query_cost < f64::MAX);
    }

    #[test]
    fn test_autotune_evaluates_all_configs() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        let neurons = random_neurons(10, 4, &mut rng);
        let queries = random_queries(3, 4, &mut rng);

        let tuner = LshAutotuner::new(0.5)
            .with_k_range(4, 6)   // k=4,6 -> 2 values
            .with_l_range(10, 20) // l=10,20 -> 2 values
            .with_speedup_ratio(0.5);

        let result = tuner.autotune(&neurons, &queries, 42);
        // 2 K values * 2 L values = 4 configs
        assert_eq!(result.configs_evaluated, 4);
    }

    #[test]
    fn test_autotune_prefers_lower_cost() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(999);
        let neurons = random_neurons(30, 8, &mut rng);
        let queries = random_queries(5, 8, &mut rng);

        // With target recall 0.0, all configs pass; the one with lowest cost wins
        let tuner = LshAutotuner::new(0.0)
            .with_k_range(4, 8)
            .with_l_range(10, 30)
            .with_speedup_ratio(0.3);

        let result = tuner.autotune(&neurons, &queries, 42);
        assert!(result.query_cost < f64::MAX);
        assert!(result.query_cost >= 0.0);
    }

    #[test]
    fn test_apply_result() {
        let result = AutotuneResult {
            best_k: 8,
            best_l: 50,
            recall: 0.95,
            query_cost: 100.0,
            configs_evaluated: 20,
        };
        let mut config = LshConfig::default();
        LshAutotuner::apply_result(&result, &mut config);
        assert_eq!(config.num_hashes, 8);
        assert_eq!(config.range_pow, 8);
        assert_eq!(config.num_tables, 50);
    }

    #[test]
    fn test_autotune_empty_neurons() {
        let tuner = LshAutotuner::new(0.5)
            .with_k_range(4, 4)
            .with_l_range(10, 10);
        let result = tuner.autotune(&[], &[vec![1.0, 2.0]], 42);
        assert_eq!(result.configs_evaluated, 1);
    }

    #[test]
    fn test_autotune_empty_queries() {
        let neurons = vec![(0, vec![1.0, 0.0])];
        let tuner = LshAutotuner::new(0.5)
            .with_k_range(4, 4)
            .with_l_range(10, 10);
        let result = tuner.autotune(&neurons, &[], 42);
        assert_eq!(result.configs_evaluated, 1);
    }

    #[test]
    fn test_autotune_result_recall_range() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(77);
        let neurons = random_neurons(20, 4, &mut rng);
        let queries = random_queries(5, 4, &mut rng);

        let tuner = LshAutotuner::new(0.0)
            .with_k_range(4, 6)
            .with_l_range(10, 20)
            .with_speedup_ratio(0.5);

        let result = tuner.autotune(&neurons, &queries, 77);
        assert!(result.recall >= 0.0);
        assert!(result.recall <= 1.0);
    }
}
