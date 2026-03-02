use super::{Bucket, NeuronId};
use serde::{Deserialize, Serialize};

/// Fixed-capacity FIFO bucket.
///
/// When the bucket is full, the oldest entry is evicted (removed from index 0).
/// This is O(capacity) per eviction but capacity is small (typically 128), so
/// the constant overhead is negligible compared to hashing costs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FifoBucket {
    data: Vec<NeuronId>,
    capacity: usize,
}

impl FifoBucket {
    /// Create a new FIFO bucket with the given capacity.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "bucket capacity must be positive");
        Self {
            data: Vec::with_capacity(capacity),
            capacity,
        }
    }
}

impl Bucket for FifoBucket {
    fn insert(&mut self, id: NeuronId) {
        if self.data.len() >= self.capacity {
            // Evict the oldest entry (front of the Vec).
            self.data.remove(0);
        }
        self.data.push(id);
    }

    fn remove(&mut self, id: NeuronId) -> bool {
        if let Some(pos) = self.data.iter().position(|&x| x == id) {
            self.data.swap_remove(pos);
            true
        } else {
            false
        }
    }

    fn contents(&self) -> &[NeuronId] {
        &self.data
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn clear(&mut self) {
        self.data.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_bucket() {
        let b = FifoBucket::new(4);
        assert!(b.is_empty());
        assert_eq!(b.len(), 0);
    }

    #[test]
    #[should_panic(expected = "bucket capacity must be positive")]
    fn test_zero_capacity_panics() {
        FifoBucket::new(0);
    }

    #[test]
    fn test_insert_within_capacity() {
        let mut b = FifoBucket::new(4);
        b.insert(1);
        b.insert(2);
        b.insert(3);
        assert_eq!(b.len(), 3);
        assert_eq!(b.contents(), &[1, 2, 3]);
    }

    #[test]
    fn test_insert_evicts_oldest_when_full() {
        let mut b = FifoBucket::new(3);
        b.insert(10);
        b.insert(20);
        b.insert(30);
        assert_eq!(b.contents(), &[10, 20, 30]);

        // Inserting a 4th should evict 10 (oldest).
        b.insert(40);
        assert_eq!(b.len(), 3);
        assert!(!b.contents().contains(&10));
        assert!(b.contents().contains(&40));
    }

    #[test]
    fn test_fifo_order_after_multiple_evictions() {
        let mut b = FifoBucket::new(2);
        b.insert(1);
        b.insert(2);
        // Full: [1, 2]

        b.insert(3); // evict 1 -> [2, 3]
        assert_eq!(b.contents(), &[2, 3]);

        b.insert(4); // evict 2 -> [3, 4]
        assert_eq!(b.contents(), &[3, 4]);

        b.insert(5); // evict 3 -> [4, 5]
        assert_eq!(b.contents(), &[4, 5]);
    }

    #[test]
    fn test_remove_existing() {
        let mut b = FifoBucket::new(4);
        b.insert(1);
        b.insert(2);
        b.insert(3);
        assert!(b.remove(2));
        assert_eq!(b.len(), 2);
        assert!(!b.contents().contains(&2));
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut b = FifoBucket::new(4);
        b.insert(1);
        assert!(!b.remove(99));
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_remove_from_empty() {
        let mut b = FifoBucket::new(4);
        assert!(!b.remove(1));
    }

    #[test]
    fn test_clear() {
        let mut b = FifoBucket::new(4);
        b.insert(1);
        b.insert(2);
        b.clear();
        assert!(b.is_empty());
        assert_eq!(b.len(), 0);
        assert_eq!(b.contents(), &[] as &[u32]);
    }

    #[test]
    fn test_capacity_one() {
        let mut b = FifoBucket::new(1);
        b.insert(10);
        assert_eq!(b.contents(), &[10]);

        b.insert(20);
        assert_eq!(b.len(), 1);
        assert_eq!(b.contents(), &[20]);
    }

    #[test]
    fn test_serde_roundtrip() {
        let mut b = FifoBucket::new(4);
        b.insert(1);
        b.insert(2);
        b.insert(3);

        let json = serde_json::to_string(&b).unwrap();
        let b2: FifoBucket = serde_json::from_str(&json).unwrap();
        assert_eq!(b.contents(), b2.contents());
    }
}
