use klearu_core::bucket::NeuronId;
use klearu_core::lsh::LshIndexTrait;
use std::collections::HashMap;

/// Tracks label co-occurrence statistics and inserts correct labels
/// into non-selected hash buckets to improve recall.
pub struct LabelAwareInserter {
    /// Co-occurrence count: (label_a, label_b) -> count
    /// Keys are canonicalized so that label_a < label_b.
    cooccurrence: HashMap<(u32, u32), u32>,
    /// Total occurrences per label.
    label_counts: HashMap<u32, u32>,
    /// Minimum co-occurrence to trigger insertion.
    min_cooccurrence: u32,
}

impl LabelAwareInserter {
    pub fn new(min_cooccurrence: u32) -> Self {
        Self {
            cooccurrence: HashMap::new(),
            label_counts: HashMap::new(),
            min_cooccurrence,
        }
    }

    /// Return the minimum co-occurrence threshold.
    pub fn min_cooccurrence(&self) -> u32 {
        self.min_cooccurrence
    }

    /// Return the count for a single label.
    pub fn label_count(&self, label: u32) -> u32 {
        self.label_counts.get(&label).copied().unwrap_or(0)
    }

    /// Return the co-occurrence count for a pair of labels.
    pub fn pair_count(&self, a: u32, b: u32) -> u32 {
        let key = if a < b { (a, b) } else { (b, a) };
        self.cooccurrence.get(&key).copied().unwrap_or(0)
    }

    /// Update co-occurrence statistics from a training example's labels.
    pub fn observe_labels(&mut self, labels: &[u32]) {
        for &label in labels {
            *self.label_counts.entry(label).or_insert(0) += 1;
        }
        for i in 0..labels.len() {
            for j in (i + 1)..labels.len() {
                let key = if labels[i] < labels[j] {
                    (labels[i], labels[j])
                } else {
                    (labels[j], labels[i])
                };
                *self.cooccurrence.entry(key).or_insert(0) += 1;
            }
        }
    }

    /// Get labels that frequently co-occur with the given label.
    /// Returns up to `top_k` labels sorted by co-occurrence count descending.
    pub fn get_cooccurring_labels(&self, label: u32, top_k: usize) -> Vec<u32> {
        let mut related: Vec<(u32, u32)> = self
            .cooccurrence
            .iter()
            .filter_map(|(&(a, b), &count)| {
                if count >= self.min_cooccurrence {
                    if a == label {
                        Some((b, count))
                    } else if b == label {
                        Some((a, count))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();

        related.sort_by(|a, b| b.1.cmp(&a.1));
        related.into_iter().take(top_k).map(|(id, _)| id).collect()
    }

    /// Insert correct labels into the LSH index for neurons that weren't selected
    /// but should have been (based on co-occurrence with actual labels).
    ///
    /// `lsh_index`: the mutable LSH index to augment.
    /// `selected_neurons`: neurons already selected by LSH query.
    /// `correct_labels`: the ground-truth labels for the current example.
    /// `neuron_weights`: (neuron_id, weight_vector) pairs for all neurons.
    /// `top_k`: maximum number of co-occurring labels to consider per label.
    pub fn augment_index(
        &self,
        lsh_index: &mut dyn LshIndexTrait,
        selected_neurons: &[NeuronId],
        correct_labels: &[u32],
        neuron_weights: &[(NeuronId, Vec<f32>)],
        top_k: usize,
    ) {
        for &label in correct_labels {
            let cooccurring = self.get_cooccurring_labels(label, top_k);
            for co_label in cooccurring {
                let neuron_id = co_label as NeuronId;
                if !selected_neurons.contains(&neuron_id) {
                    if let Some((_, weights)) =
                        neuron_weights.iter().find(|(id, _)| *id == neuron_id)
                    {
                        lsh_index.insert(neuron_id, weights);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use klearu_core::config::{BucketType, HashFunctionType, LshConfig};
    use klearu_core::lsh::create_lsh_index;

    #[test]
    fn test_new_inserter() {
        let inserter = LabelAwareInserter::new(3);
        assert_eq!(inserter.min_cooccurrence(), 3);
        assert_eq!(inserter.label_count(0), 0);
        assert_eq!(inserter.pair_count(0, 1), 0);
    }

    #[test]
    fn test_observe_single_label() {
        let mut inserter = LabelAwareInserter::new(1);
        inserter.observe_labels(&[5]);
        assert_eq!(inserter.label_count(5), 1);
        assert_eq!(inserter.label_count(0), 0);
    }

    #[test]
    fn test_observe_pair() {
        let mut inserter = LabelAwareInserter::new(1);
        inserter.observe_labels(&[2, 7]);
        assert_eq!(inserter.label_count(2), 1);
        assert_eq!(inserter.label_count(7), 1);
        assert_eq!(inserter.pair_count(2, 7), 1);
        assert_eq!(inserter.pair_count(7, 2), 1); // symmetric
    }

    #[test]
    fn test_observe_triple() {
        let mut inserter = LabelAwareInserter::new(1);
        inserter.observe_labels(&[1, 3, 5]);
        // 3 pairs: (1,3), (1,5), (3,5)
        assert_eq!(inserter.pair_count(1, 3), 1);
        assert_eq!(inserter.pair_count(1, 5), 1);
        assert_eq!(inserter.pair_count(3, 5), 1);
        // Each label observed once
        assert_eq!(inserter.label_count(1), 1);
        assert_eq!(inserter.label_count(3), 1);
        assert_eq!(inserter.label_count(5), 1);
    }

    #[test]
    fn test_observe_accumulates() {
        let mut inserter = LabelAwareInserter::new(1);
        inserter.observe_labels(&[1, 2]);
        inserter.observe_labels(&[1, 2]);
        inserter.observe_labels(&[1, 3]);
        assert_eq!(inserter.label_count(1), 3);
        assert_eq!(inserter.label_count(2), 2);
        assert_eq!(inserter.label_count(3), 1);
        assert_eq!(inserter.pair_count(1, 2), 2);
        assert_eq!(inserter.pair_count(1, 3), 1);
        assert_eq!(inserter.pair_count(2, 3), 0);
    }

    #[test]
    fn test_observe_empty_labels() {
        let mut inserter = LabelAwareInserter::new(1);
        inserter.observe_labels(&[]);
        assert_eq!(inserter.label_count(0), 0);
    }

    #[test]
    fn test_get_cooccurring_labels_basic() {
        let mut inserter = LabelAwareInserter::new(1);
        inserter.observe_labels(&[10, 20]);
        inserter.observe_labels(&[10, 30]);
        inserter.observe_labels(&[10, 20]);

        let related = inserter.get_cooccurring_labels(10, 5);
        // (10,20) has count 2, (10,30) has count 1
        assert_eq!(related.len(), 2);
        assert_eq!(related[0], 20); // higher count first
        assert_eq!(related[1], 30);
    }

    #[test]
    fn test_get_cooccurring_labels_min_threshold() {
        let mut inserter = LabelAwareInserter::new(3);
        inserter.observe_labels(&[1, 2]);
        inserter.observe_labels(&[1, 2]);
        // pair (1,2) has count 2, below threshold 3
        let related = inserter.get_cooccurring_labels(1, 10);
        assert!(related.is_empty());

        // One more observation pushes it over
        inserter.observe_labels(&[1, 2]);
        let related = inserter.get_cooccurring_labels(1, 10);
        assert_eq!(related, vec![2]);
    }

    #[test]
    fn test_get_cooccurring_labels_top_k_limit() {
        let mut inserter = LabelAwareInserter::new(1);
        // Create many co-occurring labels with label 0
        for i in 1..=10 {
            for _ in 0..i {
                inserter.observe_labels(&[0, i]);
            }
        }

        // Request only top 3
        let related = inserter.get_cooccurring_labels(0, 3);
        assert_eq!(related.len(), 3);
        // Should be labels 10, 9, 8 (highest counts)
        assert_eq!(related[0], 10);
        assert_eq!(related[1], 9);
        assert_eq!(related[2], 8);
    }

    #[test]
    fn test_get_cooccurring_labels_unknown_label() {
        let mut inserter = LabelAwareInserter::new(1);
        inserter.observe_labels(&[1, 2]);
        let related = inserter.get_cooccurring_labels(99, 10);
        assert!(related.is_empty());
    }

    #[test]
    fn test_augment_index() {
        let dim = 4;
        let config = LshConfig {
            hash_function: HashFunctionType::SimHash,
            bucket_type: BucketType::Fifo,
            num_tables: 20,
            range_pow: 4,
            num_hashes: 4,
            bucket_capacity: 128,
            rebuild_interval_base: 1000,
            rebuild_decay: 0.0,
        };
        let mut index = create_lsh_index(&config, dim, 42);

        let neuron_weights: Vec<(NeuronId, Vec<f32>)> = vec![
            (0, vec![1.0, 0.0, 0.0, 0.0]),
            (1, vec![0.0, 1.0, 0.0, 0.0]),
            (2, vec![0.0, 0.0, 1.0, 0.0]),
            (3, vec![0.0, 0.0, 0.0, 1.0]),
        ];

        // Build co-occurrence: labels 0 and 2 co-occur frequently
        let mut inserter = LabelAwareInserter::new(2);
        for _ in 0..5 {
            inserter.observe_labels(&[0, 2]);
        }

        // Suppose only neuron 0 was selected by LSH
        let selected = vec![0u32];
        let correct_labels = &[0u32];

        // Augment should insert neuron 2 (co-occurs with label 0)
        inserter.augment_index(
            index.as_mut(),
            &selected,
            correct_labels,
            &neuron_weights,
            5,
        );

        // Query with a vector aligned to neuron 2 -- it should now be findable
        let query = vec![0.0, 0.0, 1.0, 0.0];
        let candidates = index.query(&query);
        let found_ids: Vec<NeuronId> = candidates.iter().map(|c| c.neuron_id).collect();
        // Neuron 2 was inserted, so it should appear in candidates
        assert!(
            found_ids.contains(&2),
            "Expected neuron 2 in candidates after augmentation, got {:?}",
            found_ids
        );
    }

    #[test]
    fn test_augment_skips_already_selected() {
        let dim = 2;
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
        let mut index = create_lsh_index(&config, dim, 42);

        let neuron_weights: Vec<(NeuronId, Vec<f32>)> = vec![
            (0, vec![1.0, 0.0]),
            (1, vec![0.0, 1.0]),
        ];

        let mut inserter = LabelAwareInserter::new(1);
        for _ in 0..3 {
            inserter.observe_labels(&[0, 1]);
        }

        // Both neurons already selected -- augment should not re-insert
        let selected = vec![0u32, 1u32];
        let correct_labels = &[0u32];

        // This should be a no-op (neuron 1 co-occurs with 0 but is already selected)
        inserter.augment_index(
            index.as_mut(),
            &selected,
            correct_labels,
            &neuron_weights,
            5,
        );
        // No assertion needed -- just verify it doesn't panic
    }

    #[test]
    fn test_augment_skips_missing_weights() {
        let dim = 2;
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
        let mut index = create_lsh_index(&config, dim, 42);

        // Co-occurring label 5 has no weights in neuron_weights
        let neuron_weights: Vec<(NeuronId, Vec<f32>)> = vec![(0, vec![1.0, 0.0])];

        let mut inserter = LabelAwareInserter::new(1);
        for _ in 0..3 {
            inserter.observe_labels(&[0, 5]);
        }

        let selected = vec![0u32];
        let correct_labels = &[0u32];

        // Should not panic even though neuron 5 has no weights
        inserter.augment_index(
            index.as_mut(),
            &selected,
            correct_labels,
            &neuron_weights,
            5,
        );
    }

    #[test]
    fn test_pair_count_symmetry() {
        let mut inserter = LabelAwareInserter::new(1);
        inserter.observe_labels(&[3, 7]);
        assert_eq!(inserter.pair_count(3, 7), inserter.pair_count(7, 3));
    }

    #[test]
    fn test_duplicate_labels_in_observation() {
        let mut inserter = LabelAwareInserter::new(1);
        // Labels [1, 1, 2] -- label 1 appears twice
        inserter.observe_labels(&[1, 1, 2]);
        // label_count(1) should be 2 (counted twice)
        assert_eq!(inserter.label_count(1), 2);
        assert_eq!(inserter.label_count(2), 1);
        // pair (1,1) is same label -- counted once by loop structure
        // pair (1,2) counted twice (i=0,j=2 and i=1,j=2)
        assert_eq!(inserter.pair_count(1, 2), 2);
    }
}
