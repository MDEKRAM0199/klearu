use crate::tensor::SparseVector;

/// A single training example.
#[derive(Debug, Clone)]
pub struct Example {
    pub features: SparseVector,
    pub labels: Vec<u32>,
}

impl Example {
    /// Create an example from a dense feature vector and label list.
    pub fn new(dense_features: Vec<f32>, labels: Vec<u32>) -> Self {
        let features = SparseVector::from_dense(&dense_features);
        Self { features, labels }
    }
}

/// Trait for datasets.
pub trait Dataset: Send + Sync {
    /// Number of examples in the dataset.
    fn len(&self) -> usize;

    /// Whether the dataset is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the example at the given index.
    fn get(&self, index: usize) -> &Example;

    /// Dimensionality of the feature vectors.
    fn feature_dim(&self) -> usize;

    /// Number of distinct labels.
    fn num_labels(&self) -> usize;
}

/// Iterator that yields shuffled mini-batches of examples.
pub struct BatchIterator<'a> {
    dataset: &'a dyn Dataset,
    indices: Vec<usize>,
    batch_size: usize,
    pos: usize,
}

impl<'a> BatchIterator<'a> {
    /// Create a new batch iterator over `dataset` with the given `batch_size`.
    ///
    /// The iteration order is shuffled using a seeded Fisher-Yates shuffle so
    /// that results are reproducible for a given `seed`.
    pub fn new(dataset: &'a dyn Dataset, batch_size: usize, seed: u64) -> Self {
        let mut indices: Vec<usize> = (0..dataset.len()).collect();
        Self::fisher_yates_shuffle(&mut indices, seed);
        Self {
            dataset,
            indices,
            batch_size,
            pos: 0,
        }
    }

    /// Reshuffle for the next epoch with a new seed.
    pub fn reset(&mut self, seed: u64) {
        // Reinitialize to sequential order before shuffling so that the
        // same seed always produces the same permutation.
        for (i, idx) in self.indices.iter_mut().enumerate() {
            *idx = i;
        }
        Self::fisher_yates_shuffle(&mut self.indices, seed);
        self.pos = 0;
    }

    /// Fisher-Yates shuffle using a simple SplitMix64-based PRNG for
    /// reproducibility without pulling in heavy RNG dependencies.
    fn fisher_yates_shuffle(indices: &mut [usize], seed: u64) {
        let mut rng_state = seed;
        for i in (1..indices.len()).rev() {
            rng_state = splitmix64(rng_state);
            let j = (rng_state as usize) % (i + 1);
            indices.swap(i, j);
        }
    }
}

/// SplitMix64 mixing function -- fast, well-distributed bijective mixer.
fn splitmix64(state: u64) -> u64 {
    let mut z = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

impl<'a> Iterator for BatchIterator<'a> {
    type Item = Vec<&'a Example>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.indices.len() {
            return None;
        }

        let end = (self.pos + self.batch_size).min(self.indices.len());
        let batch: Vec<&'a Example> = self.indices[self.pos..end]
            .iter()
            .map(|&idx| self.dataset.get(idx))
            .collect();
        self.pos = end;

        Some(batch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A simple in-memory dataset for testing.
    struct TestDataset {
        examples: Vec<Example>,
        feature_dim: usize,
        num_labels: usize,
    }

    impl TestDataset {
        fn new(n: usize, feature_dim: usize, num_labels: usize) -> Self {
            let examples = (0..n)
                .map(|i| Example {
                    features: SparseVector::from_pairs(
                        feature_dim,
                        vec![(i as u32 % feature_dim as u32, 1.0)],
                    ),
                    labels: vec![(i % num_labels) as u32],
                })
                .collect();
            Self {
                examples,
                feature_dim,
                num_labels,
            }
        }
    }

    impl Dataset for TestDataset {
        fn len(&self) -> usize {
            self.examples.len()
        }
        fn get(&self, index: usize) -> &Example {
            &self.examples[index]
        }
        fn feature_dim(&self) -> usize {
            self.feature_dim
        }
        fn num_labels(&self) -> usize {
            self.num_labels
        }
    }

    #[test]
    fn test_dataset_trait_basics() {
        let ds = TestDataset::new(10, 100, 5);
        assert_eq!(ds.len(), 10);
        assert!(!ds.is_empty());
        assert_eq!(ds.feature_dim(), 100);
        assert_eq!(ds.num_labels(), 5);
    }

    #[test]
    fn test_empty_dataset() {
        let ds = TestDataset::new(0, 100, 5);
        assert_eq!(ds.len(), 0);
        assert!(ds.is_empty());
    }

    #[test]
    fn test_batch_iterator_covers_all_examples() {
        let ds = TestDataset::new(10, 100, 5);
        let iter = BatchIterator::new(&ds, 3, 42);
        let batches: Vec<Vec<&Example>> = iter.collect();

        // 10 examples with batch_size=3: 4 batches (3+3+3+1).
        assert_eq!(batches.len(), 4);
        assert_eq!(batches[0].len(), 3);
        assert_eq!(batches[1].len(), 3);
        assert_eq!(batches[2].len(), 3);
        assert_eq!(batches[3].len(), 1);

        // Total number of examples yielded must equal dataset size.
        let total: usize = batches.iter().map(|b| b.len()).sum();
        assert_eq!(total, 10);
    }

    #[test]
    fn test_batch_iterator_exact_division() {
        let ds = TestDataset::new(9, 100, 3);
        let iter = BatchIterator::new(&ds, 3, 0);
        let batches: Vec<Vec<&Example>> = iter.collect();
        assert_eq!(batches.len(), 3);
        for batch in &batches {
            assert_eq!(batch.len(), 3);
        }
    }

    #[test]
    fn test_batch_iterator_batch_size_larger_than_dataset() {
        let ds = TestDataset::new(5, 100, 3);
        let iter = BatchIterator::new(&ds, 100, 0);
        let batches: Vec<Vec<&Example>> = iter.collect();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].len(), 5);
    }

    #[test]
    fn test_batch_iterator_empty_dataset() {
        let ds = TestDataset::new(0, 100, 3);
        let iter = BatchIterator::new(&ds, 10, 0);
        let batches: Vec<Vec<&Example>> = iter.collect();
        assert!(batches.is_empty());
    }

    #[test]
    fn test_batch_iterator_deterministic() {
        let ds = TestDataset::new(20, 100, 5);
        let iter1 = BatchIterator::new(&ds, 5, 42);
        let iter2 = BatchIterator::new(&ds, 5, 42);

        let batches1: Vec<Vec<&Example>> = iter1.collect();
        let batches2: Vec<Vec<&Example>> = iter2.collect();
        assert_eq!(batches1.len(), batches2.len());
        for (b1, b2) in batches1.iter().zip(batches2.iter()) {
            for (e1, e2) in b1.iter().zip(b2.iter()) {
                assert_eq!(e1.labels, e2.labels);
            }
        }
    }

    #[test]
    fn test_batch_iterator_different_seeds_differ() {
        let ds = TestDataset::new(20, 100, 20);
        let iter1 = BatchIterator::new(&ds, 5, 1);
        let iter2 = BatchIterator::new(&ds, 5, 2);

        let batches1: Vec<Vec<&Example>> = iter1.collect();
        let batches2: Vec<Vec<&Example>> = iter2.collect();

        // With 20 distinct labels and different seeds, the order should differ.
        let labels1: Vec<u32> = batches1
            .iter()
            .flat_map(|b| b.iter().map(|e| e.labels[0]))
            .collect();
        let labels2: Vec<u32> = batches2
            .iter()
            .flat_map(|b| b.iter().map(|e| e.labels[0]))
            .collect();
        assert_ne!(labels1, labels2);
    }

    #[test]
    fn test_batch_iterator_reset() {
        let ds = TestDataset::new(10, 100, 10);
        let mut iter = BatchIterator::new(&ds, 3, 42);

        // Consume first epoch.
        let epoch1: Vec<Vec<&Example>> = (&mut iter).collect();
        assert_eq!(epoch1.len(), 4);

        // Reset for second epoch with a different seed.
        iter.reset(99);
        let epoch2: Vec<Vec<&Example>> = iter.collect();
        assert_eq!(epoch2.len(), 4);

        // The two epochs should (very likely) have different orders since
        // different seeds were used.
        let labels1: Vec<u32> = epoch1
            .iter()
            .flat_map(|b| b.iter().map(|e| e.labels[0]))
            .collect();
        let labels2: Vec<u32> = epoch2
            .iter()
            .flat_map(|b| b.iter().map(|e| e.labels[0]))
            .collect();
        assert_ne!(labels1, labels2);
    }

    #[test]
    fn test_batch_iterator_reset_same_seed_reproduces() {
        let ds = TestDataset::new(10, 100, 10);
        let mut iter = BatchIterator::new(&ds, 3, 42);

        let epoch1: Vec<Vec<&Example>> = (&mut iter).collect();
        iter.reset(42);
        let epoch2: Vec<Vec<&Example>> = iter.collect();

        let labels1: Vec<u32> = epoch1
            .iter()
            .flat_map(|b| b.iter().map(|e| e.labels[0]))
            .collect();
        let labels2: Vec<u32> = epoch2
            .iter()
            .flat_map(|b| b.iter().map(|e| e.labels[0]))
            .collect();
        assert_eq!(labels1, labels2);
    }

    #[test]
    fn test_batch_size_one() {
        let ds = TestDataset::new(3, 10, 3);
        let iter = BatchIterator::new(&ds, 1, 0);
        let batches: Vec<Vec<&Example>> = iter.collect();
        assert_eq!(batches.len(), 3);
        for batch in &batches {
            assert_eq!(batch.len(), 1);
        }
    }

    #[test]
    fn test_all_indices_appear_exactly_once() {
        let ds = TestDataset::new(15, 100, 15);
        let iter = BatchIterator::new(&ds, 4, 123);
        let batches: Vec<Vec<&Example>> = iter.collect();

        let mut all_labels: Vec<u32> = batches
            .iter()
            .flat_map(|b| b.iter().map(|e| e.labels[0]))
            .collect();
        all_labels.sort();
        let expected: Vec<u32> = (0..15).collect();
        assert_eq!(all_labels, expected);
    }
}
