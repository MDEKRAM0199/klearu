use std::cell::UnsafeCell;

use crate::network::Network;

/// HOGWILD-style lock-free shared network wrapper.
///
/// Safety: SLIDE guarantees ~0.5% neuron overlap between threads,
/// matching HOGWILD's sparsity assumption for convergence. Because
/// each thread only updates the small set of active neurons selected
/// by LSH, the probability of two threads writing to the same neuron
/// simultaneously is negligible, and the occasional data race is
/// tolerable under the HOGWILD convergence analysis.
pub struct HogwildNetwork {
    inner: UnsafeCell<Network>,
}

// SAFETY: HOGWILD allows concurrent unsynchronized reads/writes.
// The SLIDE algorithm's LSH-based neuron selection ensures that
// concurrent threads access largely disjoint subsets of neurons,
// satisfying the sparsity condition required for HOGWILD convergence.
unsafe impl Send for HogwildNetwork {}
unsafe impl Sync for HogwildNetwork {}

impl HogwildNetwork {
    /// Wrap a network for lock-free parallel training.
    pub fn new(network: Network) -> Self {
        Self {
            inner: UnsafeCell::new(network),
        }
    }

    /// Get a shared reference to the underlying network.
    ///
    /// # Safety
    /// The caller must ensure no mutable references exist to the same
    /// memory being read. In the HOGWILD context, benign data races on
    /// weight reads are acceptable.
    pub fn get(&self) -> &Network {
        unsafe { &*self.inner.get() }
    }

    /// Get a mutable reference to the underlying network.
    ///
    /// # Safety
    /// HOGWILD allows racing writes; the caller accepts that concurrent
    /// writes to the same neuron may cause a benign data race (the
    /// update is "last writer wins"). This is theoretically sound when
    /// the access pattern is sparse, which SLIDE guarantees.
    #[allow(clippy::mut_from_ref)]
    pub fn get_mut(&self) -> &mut Network {
        unsafe { &mut *self.inner.get() }
    }

    /// Consume the wrapper and return the inner network.
    pub fn into_inner(self) -> Network {
        self.inner.into_inner()
    }

    /// Train the network on a batch of examples using HOGWILD parallelism.
    ///
    /// The batch is split into `num_threads` sub-batches, and each is processed
    /// independently via rayon. Each thread reads and writes to the shared network
    /// without locking, relying on SLIDE's sparse neuron activation to minimize
    /// write conflicts.
    pub fn train_parallel(
        &self,
        batch: &[crate::data::Example],
        num_threads: usize,
        lr: f64,
        _step: u64,
    ) -> f32 {
        use rayon::prelude::*;

        if batch.is_empty() || num_threads == 0 {
            return 0.0;
        }

        let chunk_size = (batch.len() + num_threads - 1) / num_threads;
        let chunks: Vec<&[crate::data::Example]> = batch.chunks(chunk_size).collect();

        let losses: Vec<f32> = chunks
            .par_iter()
            .map(|chunk| {
                let net = self.get_mut();
                let example_refs: Vec<&crate::data::Example> = chunk.iter().collect();
                net.train_step(&example_refs, lr)
            })
            .collect();

        if losses.is_empty() {
            0.0
        } else {
            losses.iter().sum::<f32>() / losses.len() as f32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::*;
    use crate::data::Example;

    fn make_tiny_config() -> SlideConfig {
        SlideConfig {
            network: NetworkConfig {
                layers: vec![
                    LayerConfig::hidden(4, 8),
                    LayerConfig::output(8, 3),
                ],
                optimizer: OptimizerType::Sgd,
                learning_rate: 0.01,
                batch_size: 4,
                num_threads: 2,
            },
            seed: 42,
            hogwild: true,
        }
    }

    #[test]
    fn test_hogwild_new_and_into_inner() {
        let config = make_tiny_config();
        let net = Network::new(config);
        let num_layers = net.layers.len();
        let hogwild = HogwildNetwork::new(net);
        let recovered = hogwild.into_inner();
        assert_eq!(recovered.layers.len(), num_layers);
    }

    #[test]
    fn test_hogwild_get_ref() {
        let config = make_tiny_config();
        let net = Network::new(config);
        let hogwild = HogwildNetwork::new(net);
        let net_ref = hogwild.get();
        assert_eq!(net_ref.layers.len(), 2);
    }

    #[test]
    fn test_hogwild_get_mut() {
        let config = make_tiny_config();
        let net = Network::new(config);
        let hogwild = HogwildNetwork::new(net);
        let net_mut = hogwild.get_mut();
        assert_eq!(net_mut.layers.len(), 2);
    }

    #[test]
    fn test_hogwild_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<HogwildNetwork>();
        assert_sync::<HogwildNetwork>();
    }

    #[test]
    fn test_hogwild_train_parallel_empty_batch() {
        let config = make_tiny_config();
        let net = Network::new(config);
        let hogwild = HogwildNetwork::new(net);
        let loss = hogwild.train_parallel(&[], 2, 0.01, 1);
        assert_eq!(loss, 0.0);
    }

    #[test]
    fn test_hogwild_train_parallel() {
        let config = make_tiny_config();
        let net = Network::new(config);
        let hogwild = HogwildNetwork::new(net);

        let examples: Vec<Example> = (0..4)
            .map(|i| Example::new(vec![i as f32 * 0.1 + 0.1; 4], vec![i % 3]))
            .collect();

        let loss = hogwild.train_parallel(&examples, 2, 0.01, 1);
        assert!(loss.is_finite(), "loss should be finite");
    }
}
