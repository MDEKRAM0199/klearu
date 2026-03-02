//! Memory layout optimizations for SLIDE neuron weights.
//!
//! Provides cache-line-aligned, contiguous weight storage so that vectorized
//! access (SIMD loads/stores) never straddle cache-line boundaries.

// ---------------------------------------------------------------------------
// Alignment helpers
// ---------------------------------------------------------------------------

/// Cache line size in bytes (typical on x86_64 and aarch64).
const CACHE_LINE: usize = 64;

/// Round `size` (in bytes) up to the next cache-line boundary.
///
/// ```
/// use klearu_accel::memory::align_to_cache_line;
/// assert_eq!(align_to_cache_line(1), 64);
/// assert_eq!(align_to_cache_line(64), 64);
/// assert_eq!(align_to_cache_line(65), 128);
/// ```
pub fn align_to_cache_line(size: usize) -> usize {
    (size + CACHE_LINE - 1) & !(CACHE_LINE - 1)
}

/// Number of `f32` elements per cache line (64 / 4 = 16).
const F32_PER_CACHE_LINE: usize = CACHE_LINE / std::mem::size_of::<f32>();

/// Round a count of `f32` elements up to the next multiple of
/// [`F32_PER_CACHE_LINE`] (16).
fn align_f32_count(count: usize) -> usize {
    (count + F32_PER_CACHE_LINE - 1) & !(F32_PER_CACHE_LINE - 1)
}

// ---------------------------------------------------------------------------
// ContiguousWeightStore
// ---------------------------------------------------------------------------

/// Contiguous weight store: all neuron weights packed sequentially with
/// cache-line-aligned stride for vectorized access.
///
/// Each neuron's weight slice starts at an offset that is a multiple of 16
/// f32s (64 bytes), so SIMD loads/stores are always aligned to a cache line.
/// The padding between `neuron_dim` and `stride` is zero-filled and must not
/// be read as meaningful data.
#[derive(Debug, Clone)]
pub struct ContiguousWeightStore {
    data: Vec<f32>,
    neuron_dim: usize,
    stride: usize,
    num_neurons: usize,
}

impl ContiguousWeightStore {
    /// Allocate storage for `num_neurons`, each with `neuron_dim` weights.
    ///
    /// The per-neuron stride is `neuron_dim` rounded up to the next multiple
    /// of 16 (`F32_PER_CACHE_LINE`). All weights are zero-initialized.
    pub fn new(num_neurons: usize, neuron_dim: usize) -> Self {
        let stride = align_f32_count(neuron_dim);
        let total = num_neurons
            .checked_mul(stride)
            .expect("ContiguousWeightStore: allocation size overflow");
        Self {
            data: vec![0.0f32; total],
            neuron_dim,
            stride,
            num_neurons,
        }
    }

    /// Return the weight slice for `neuron_id` (length = `neuron_dim`).
    ///
    /// # Panics
    ///
    /// Panics if `neuron_id >= num_neurons`.
    #[inline]
    pub fn get_weights(&self, neuron_id: usize) -> &[f32] {
        assert!(
            neuron_id < self.num_neurons,
            "neuron_id {neuron_id} out of range (num_neurons = {})",
            self.num_neurons,
        );
        let start = neuron_id * self.stride;
        &self.data[start..start + self.neuron_dim]
    }

    /// Return a mutable weight slice for `neuron_id` (length = `neuron_dim`).
    ///
    /// # Panics
    ///
    /// Panics if `neuron_id >= num_neurons`.
    #[inline]
    pub fn get_weights_mut(&mut self, neuron_id: usize) -> &mut [f32] {
        assert!(
            neuron_id < self.num_neurons,
            "neuron_id {neuron_id} out of range (num_neurons = {})",
            self.num_neurons,
        );
        let start = neuron_id * self.stride;
        &mut self.data[start..start + self.neuron_dim]
    }

    /// Overwrite the weights for `neuron_id`.
    ///
    /// # Panics
    ///
    /// Panics if `weights.len() != neuron_dim` or `neuron_id >= num_neurons`.
    pub fn set_weights(&mut self, neuron_id: usize, weights: &[f32]) {
        assert_eq!(
            weights.len(),
            self.neuron_dim,
            "set_weights: expected {} elements, got {}",
            self.neuron_dim,
            weights.len(),
        );
        let dst = self.get_weights_mut(neuron_id);
        dst.copy_from_slice(weights);
    }

    /// Number of neurons in the store.
    #[inline]
    pub fn num_neurons(&self) -> usize {
        self.num_neurons
    }

    /// Logical dimension of each neuron's weight vector.
    #[inline]
    pub fn neuron_dim(&self) -> usize {
        self.neuron_dim
    }

    /// Per-neuron stride in f32 elements (>= `neuron_dim`, rounded up to a
    /// multiple of 16 for cache-line alignment).
    #[inline]
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Total number of f32 elements in the backing store (including padding).
    #[inline]
    pub fn total_elements(&self) -> usize {
        self.data.len()
    }

    /// Raw view of the entire backing buffer.
    #[inline]
    pub fn as_raw_slice(&self) -> &[f32] {
        &self.data
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- align_to_cache_line ------------------------------------------------

    #[test]
    fn test_align_to_cache_line_exact() {
        assert_eq!(align_to_cache_line(64), 64);
        assert_eq!(align_to_cache_line(128), 128);
    }

    #[test]
    fn test_align_to_cache_line_rounds_up() {
        assert_eq!(align_to_cache_line(1), 64);
        assert_eq!(align_to_cache_line(63), 64);
        assert_eq!(align_to_cache_line(65), 128);
    }

    #[test]
    fn test_align_to_cache_line_zero() {
        assert_eq!(align_to_cache_line(0), 0);
    }

    // -- align_f32_count ----------------------------------------------------

    #[test]
    fn test_align_f32_count_exact() {
        assert_eq!(align_f32_count(16), 16);
        assert_eq!(align_f32_count(32), 32);
    }

    #[test]
    fn test_align_f32_count_rounds_up() {
        assert_eq!(align_f32_count(1), 16);
        assert_eq!(align_f32_count(15), 16);
        assert_eq!(align_f32_count(17), 32);
    }

    #[test]
    fn test_align_f32_count_zero() {
        assert_eq!(align_f32_count(0), 0);
    }

    // -- ContiguousWeightStore ----------------------------------------------

    #[test]
    fn test_new_basic() {
        let store = ContiguousWeightStore::new(10, 100);
        assert_eq!(store.num_neurons(), 10);
        assert_eq!(store.neuron_dim(), 100);
        // 100 rounded up to next multiple of 16 = 112
        assert_eq!(store.stride(), 112);
        assert_eq!(store.total_elements(), 10 * 112);
    }

    #[test]
    fn test_stride_is_multiple_of_16() {
        for dim in [1, 15, 16, 17, 31, 32, 33, 100, 128, 255, 256] {
            let store = ContiguousWeightStore::new(1, dim);
            assert!(
                store.stride() % 16 == 0,
                "stride {} is not a multiple of 16 for dim {dim}",
                store.stride(),
            );
            assert!(
                store.stride() >= dim,
                "stride {} < dim {dim}",
                store.stride(),
            );
        }
    }

    #[test]
    fn test_get_set_weights() {
        let mut store = ContiguousWeightStore::new(3, 4);
        let w0 = vec![1.0, 2.0, 3.0, 4.0];
        let w1 = vec![5.0, 6.0, 7.0, 8.0];
        let w2 = vec![-1.0, -2.0, -3.0, -4.0];

        store.set_weights(0, &w0);
        store.set_weights(1, &w1);
        store.set_weights(2, &w2);

        assert_eq!(store.get_weights(0), &w0[..]);
        assert_eq!(store.get_weights(1), &w1[..]);
        assert_eq!(store.get_weights(2), &w2[..]);
    }

    #[test]
    fn test_get_weights_mut() {
        let mut store = ContiguousWeightStore::new(2, 3);
        store.set_weights(0, &[1.0, 2.0, 3.0]);

        {
            let w = store.get_weights_mut(0);
            w[1] = 42.0;
        }

        assert_eq!(store.get_weights(0), &[1.0, 42.0, 3.0]);
    }

    #[test]
    fn test_zero_initialized() {
        let store = ContiguousWeightStore::new(5, 10);
        for nid in 0..5 {
            for &v in store.get_weights(nid) {
                assert_eq!(v, 0.0);
            }
        }
    }

    #[test]
    fn test_neurons_independent() {
        // Writing to one neuron must not clobber another.
        let mut store = ContiguousWeightStore::new(4, 5);
        store.set_weights(2, &[10.0, 20.0, 30.0, 40.0, 50.0]);

        // Neurons 0, 1, 3 must still be all zeros.
        for nid in [0, 1, 3] {
            assert!(
                store.get_weights(nid).iter().all(|&v| v == 0.0),
                "neuron {nid} was clobbered",
            );
        }
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn test_get_weights_oob() {
        let store = ContiguousWeightStore::new(3, 4);
        let _ = store.get_weights(3);
    }

    #[test]
    #[should_panic(expected = "expected 4 elements")]
    fn test_set_weights_wrong_len() {
        let mut store = ContiguousWeightStore::new(2, 4);
        store.set_weights(0, &[1.0, 2.0]); // too short
    }

    #[test]
    fn test_dim_exactly_16() {
        let store = ContiguousWeightStore::new(2, 16);
        assert_eq!(store.stride(), 16);
        assert_eq!(store.total_elements(), 32);
    }

    #[test]
    fn test_dim_one() {
        let mut store = ContiguousWeightStore::new(3, 1);
        assert_eq!(store.stride(), 16); // 1 rounds up to 16
        store.set_weights(0, &[42.0]);
        store.set_weights(2, &[99.0]);
        assert_eq!(store.get_weights(0), &[42.0]);
        assert_eq!(store.get_weights(1), &[0.0]);
        assert_eq!(store.get_weights(2), &[99.0]);
    }

    #[test]
    fn test_zero_neurons() {
        let store = ContiguousWeightStore::new(0, 100);
        assert_eq!(store.num_neurons(), 0);
        assert_eq!(store.total_elements(), 0);
    }
}
