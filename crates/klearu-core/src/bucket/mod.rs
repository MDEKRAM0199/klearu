mod fifo;
mod reservoir;

pub use fifo::FifoBucket;
pub use reservoir::ReservoirBucket;

/// A neuron ID used throughout the system.
pub type NeuronId = u32;

/// Trait for LSH bucket storage.
pub trait Bucket: Send + Sync {
    /// Insert a neuron ID into this bucket.
    fn insert(&mut self, id: NeuronId);

    /// Remove a neuron ID from this bucket. Returns true if found and removed.
    fn remove(&mut self, id: NeuronId) -> bool;

    /// Get all neuron IDs currently in this bucket.
    fn contents(&self) -> &[NeuronId];

    /// Number of neurons in this bucket.
    fn len(&self) -> usize;

    /// Whether this bucket is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear all entries.
    fn clear(&mut self);
}
