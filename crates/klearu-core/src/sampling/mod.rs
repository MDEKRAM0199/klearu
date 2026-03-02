mod vanilla;
mod topk;
mod threshold;

pub use vanilla::VanillaSampling;
pub use topk::TopKSampling;
pub use threshold::ThresholdSampling;

use crate::bucket::NeuronId;

/// Strategy for selecting active neurons from LSH candidates.
pub trait SamplingStrategy: Send + Sync {
    /// Select neurons from candidates with their hit counts.
    /// `total_neurons` is the total number of neurons in the layer.
    fn select(&self, candidates: &[(NeuronId, u32)], total_neurons: usize) -> Vec<NeuronId>;
}
