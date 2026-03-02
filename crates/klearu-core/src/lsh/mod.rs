mod index;
mod maintenance;

pub use index::{create_lsh_index, LshIndex, LshIndexTrait};
pub use maintenance::RebuildScheduler;

use crate::bucket::NeuronId;

/// Result of querying the LSH index: neuron id and how many tables matched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LshCandidate {
    pub neuron_id: NeuronId,
    pub count: u32,
}
