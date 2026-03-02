use std::alloc::{self, Layout};
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::ptr;

/// Cache-line alignment (64 bytes) for SIMD-friendly dense storage.
const ALIGNMENT: usize = 64;

/// Cache-line aligned dense `f32` vector storage.
///
/// Guarantees 64-byte alignment of the backing buffer so that SIMD loads and
/// stores never cross cache-line boundaries.  The vector is *not* growable --
/// its length is fixed at construction time.
pub struct AlignedVec {
    ptr: *mut f32,
    len: usize,
    cap: usize,
}

// SAFETY: The internal pointer is exclusively owned and never aliased.
unsafe impl Send for AlignedVec {}
unsafe impl Sync for AlignedVec {}

impl AlignedVec {
    /// Allocate a zero-initialized aligned vector of `len` elements.
    ///
    /// If `len` is zero a dangling but properly aligned pointer is stored and
    /// no heap allocation occurs.
    pub fn new(len: usize) -> Self {
        if len == 0 {
            return Self {
                ptr: ALIGNMENT as *mut f32, // dangling, aligned
                len: 0,
                cap: 0,
            };
        }

        let layout = Self::layout_for(len);

        // SAFETY: layout has non-zero size (len > 0).
        let ptr = unsafe { alloc::alloc_zeroed(layout) } as *mut f32;
        if ptr.is_null() {
            alloc::handle_alloc_error(layout);
        }

        Self { ptr, len, cap: len }
    }

    /// Alias for [`Self::new`] -- allocate a zero-initialized aligned vector.
    #[inline]
    pub fn zeros(len: usize) -> Self {
        Self::new(len)
    }

    /// Copy `data` into a new aligned buffer.
    pub fn from_slice(data: &[f32]) -> Self {
        let v = Self::new(data.len());
        if !data.is_empty() {
            // SAFETY: both pointers are valid for `data.len()` elements and
            // do not overlap.
            unsafe {
                ptr::copy_nonoverlapping(data.as_ptr(), v.ptr, data.len());
            }
        }
        v
    }

    /// Returns an immutable slice over the stored elements.
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        if self.len == 0 {
            return &[];
        }
        // SAFETY: `ptr` is valid for `len` elements while `self` is alive.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Returns a mutable slice over the stored elements.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        if self.len == 0 {
            return &mut [];
        }
        // SAFETY: `ptr` is valid, aligned, and exclusively borrowed.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    /// Number of `f32` elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` when the vector contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Set every element to `value`.
    pub fn fill(&mut self, value: f32) {
        self.as_mut_slice().fill(value);
    }

    // -- internal -----------------------------------------------------------

    /// Compute the allocation [`Layout`] for `n` f32 elements at 64-byte
    /// alignment.
    fn layout_for(n: usize) -> Layout {
        let size = n.checked_mul(std::mem::size_of::<f32>()).expect("size overflow");
        Layout::from_size_align(size, ALIGNMENT).expect("invalid layout")
    }
}

// ---------------------------------------------------------------------------
// Trait impls
// ---------------------------------------------------------------------------

impl Drop for AlignedVec {
    fn drop(&mut self) {
        if self.cap > 0 {
            let layout = Self::layout_for(self.cap);
            // SAFETY: `ptr` was allocated with this layout and `cap > 0`.
            unsafe {
                alloc::dealloc(self.ptr as *mut u8, layout);
            }
        }
    }
}

impl Clone for AlignedVec {
    fn clone(&self) -> Self {
        Self::from_slice(self.as_slice())
    }
}

impl fmt::Debug for AlignedVec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AlignedVec")
            .field("len", &self.len)
            .field("data", &self.as_slice())
            .finish()
    }
}

impl Deref for AlignedVec {
    type Target = [f32];

    #[inline]
    fn deref(&self) -> &[f32] {
        self.as_slice()
    }
}

impl DerefMut for AlignedVec {
    #[inline]
    fn deref_mut(&mut self) -> &mut [f32] {
        self.as_mut_slice()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_zeroed() {
        let v = AlignedVec::new(16);
        assert_eq!(v.len(), 16);
        assert!(!v.is_empty());
        for &x in v.as_slice() {
            assert_eq!(x, 0.0);
        }
    }

    #[test]
    fn test_zeros_alias() {
        let v = AlignedVec::zeros(8);
        assert_eq!(v.len(), 8);
        assert!(v.as_slice().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_new_empty() {
        let v = AlignedVec::new(0);
        assert_eq!(v.len(), 0);
        assert!(v.is_empty());
        assert!(v.as_slice().is_empty());
    }

    #[test]
    fn test_from_slice() {
        let data = [1.0, 2.0, 3.0, 4.0];
        let v = AlignedVec::from_slice(&data);
        assert_eq!(v.as_slice(), &data);
    }

    #[test]
    fn test_from_empty_slice() {
        let v = AlignedVec::from_slice(&[]);
        assert!(v.is_empty());
    }

    #[test]
    fn test_alignment() {
        let v = AlignedVec::new(17);
        if v.len() > 0 {
            let addr = v.ptr as usize;
            assert_eq!(
                addr % ALIGNMENT,
                0,
                "pointer {addr:#x} is not {ALIGNMENT}-byte aligned",
            );
        }
    }

    #[test]
    fn test_alignment_various_sizes() {
        for &n in &[1, 2, 3, 4, 15, 16, 17, 31, 32, 33, 64, 100, 1024] {
            let v = AlignedVec::new(n);
            let addr = v.ptr as usize;
            assert_eq!(
                addr % ALIGNMENT,
                0,
                "size={n}: pointer {addr:#x} not aligned to {ALIGNMENT}",
            );
        }
    }

    #[test]
    fn test_fill() {
        let mut v = AlignedVec::new(5);
        v.fill(7.0);
        assert_eq!(v.as_slice(), &[7.0; 5]);
    }

    #[test]
    fn test_fill_empty() {
        let mut v = AlignedVec::new(0);
        v.fill(42.0); // should be a no-op, not panic
        assert!(v.is_empty());
    }

    #[test]
    fn test_as_mut_slice() {
        let mut v = AlignedVec::new(4);
        {
            let s = v.as_mut_slice();
            s[0] = 10.0;
            s[3] = 40.0;
        }
        assert_eq!(v.as_slice(), &[10.0, 0.0, 0.0, 40.0]);
    }

    #[test]
    fn test_deref() {
        let v = AlignedVec::from_slice(&[1.0, 2.0, 3.0]);
        // Deref allows using slice methods directly.
        assert_eq!(v.len(), 3);
        assert_eq!(v[0], 1.0);
        assert_eq!(v[2], 3.0);
        let sum: f32 = v.iter().sum();
        assert!((sum - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_deref_mut() {
        let mut v = AlignedVec::from_slice(&[1.0, 2.0]);
        v[0] = 99.0;
        assert_eq!(v[0], 99.0);
    }

    #[test]
    fn test_clone() {
        let original = AlignedVec::from_slice(&[1.0, 2.0, 3.0]);
        let cloned = original.clone();
        assert_eq!(cloned.as_slice(), original.as_slice());
        assert_eq!(cloned.len(), original.len());
        // Must be a distinct allocation.
        if original.len() > 0 {
            assert_ne!(
                original.ptr as usize,
                cloned.ptr as usize,
                "clone should have its own allocation",
            );
        }
    }

    #[test]
    fn test_clone_empty() {
        let v = AlignedVec::new(0);
        let c = v.clone();
        assert!(c.is_empty());
    }

    #[test]
    fn test_debug() {
        let v = AlignedVec::from_slice(&[1.0, 2.0]);
        let dbg = format!("{:?}", v);
        assert!(dbg.contains("AlignedVec"));
        assert!(dbg.contains("len: 2"));
    }

    #[test]
    fn test_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<AlignedVec>();
        assert_sync::<AlignedVec>();
    }

    #[test]
    fn test_large_allocation() {
        let n = 1_000_000;
        let v = AlignedVec::new(n);
        assert_eq!(v.len(), n);
        assert_eq!(v[0], 0.0);
        assert_eq!(v[n - 1], 0.0);
    }

    #[test]
    fn test_mutate_and_read_back() {
        let mut v = AlignedVec::new(8);
        for i in 0..8 {
            v[i] = i as f32;
        }
        let expected: Vec<f32> = (0..8).map(|i| i as f32).collect();
        assert_eq!(v.as_slice(), expected.as_slice());
    }

    #[test]
    fn test_drop_does_not_panic() {
        {
            let _v = AlignedVec::new(100);
            // Dropped here.
        }
        {
            let _v = AlignedVec::new(0);
            // Dropped here -- dangling pointer should not be freed.
        }
    }

    #[test]
    fn test_from_slice_independence() {
        let data = [1.0, 2.0, 3.0];
        let mut v = AlignedVec::from_slice(&data);
        v[0] = 999.0;
        // Original data is untouched (we copied into aligned storage).
        assert_eq!(data[0], 1.0);
        assert_eq!(v[0], 999.0);
    }
}
