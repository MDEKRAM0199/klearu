use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// SparseVector
// ---------------------------------------------------------------------------

/// A sparse vector stored as sorted (index, value) pairs.
///
/// Invariants maintained by the public API:
/// - `indices` is sorted in strictly ascending order.
/// - `indices.len() == values.len()`.
/// - Every index satisfies `index < dim as u32`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseVector {
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
    pub dim: usize,
}

impl SparseVector {
    /// Create an empty sparse vector with the given dimensionality.
    pub fn new(dim: usize) -> Self {
        Self {
            indices: Vec::new(),
            values: Vec::new(),
            dim,
        }
    }

    /// Build a sparse vector from `(index, value)` pairs.
    ///
    /// The pairs are sorted by index.  If duplicate indices appear their
    /// values are summed.  Zero-valued entries that result from cancellation
    /// are *not* pruned so that the caller can reason about which indices
    /// were explicitly provided.
    pub fn from_pairs(dim: usize, mut pairs: Vec<(u32, f32)>) -> Self {
        pairs.sort_by_key(|&(idx, _)| idx);

        let mut indices: Vec<u32> = Vec::with_capacity(pairs.len());
        let mut values: Vec<f32> = Vec::with_capacity(pairs.len());

        for (idx, val) in pairs {
            debug_assert!((idx as usize) < dim, "index {idx} out of bounds for dim {dim}");
            if let Some(last) = indices.last() {
                if *last == idx {
                    // Duplicate index -- accumulate.
                    let v = values.last_mut().unwrap();
                    *v += val;
                    continue;
                }
            }
            indices.push(idx);
            values.push(val);
        }

        Self { indices, values, dim }
    }

    /// Build a sparse vector from a dense slice, keeping only non-zero entries.
    pub fn from_dense(dense: &[f32]) -> Self {
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for (i, &v) in dense.iter().enumerate() {
            if v != 0.0 {
                indices.push(i as u32);
                values.push(v);
            }
        }
        Self {
            indices,
            values,
            dim: dense.len(),
        }
    }

    /// Expand the sparse representation into a full dense vector.
    pub fn to_dense(&self) -> Vec<f32> {
        let mut dense = vec![0.0_f32; self.dim];
        for (&idx, &val) in self.indices.iter().zip(self.values.iter()) {
            dense[idx as usize] = val;
        }
        dense
    }

    /// Gather-based dot product with a dense vector.
    ///
    /// Only the indices stored in `self` are touched in `dense`, making this
    /// O(nnz) rather than O(dim).
    pub fn dot_dense(&self, dense: &[f32]) -> f32 {
        debug_assert!(
            dense.len() >= self.dim,
            "dense vector length {} < sparse dim {}",
            dense.len(),
            self.dim,
        );
        self.indices
            .iter()
            .zip(self.values.iter())
            .map(|(&idx, &val)| val * dense[idx as usize])
            .sum()
    }

    /// Merge-based intersection dot product between two sparse vectors.
    ///
    /// Both vectors must have sorted indices (which is guaranteed by the
    /// constructors).  Only indices present in *both* vectors contribute.
    pub fn dot_sparse(&self, other: &SparseVector) -> f32 {
        let mut i = 0;
        let mut j = 0;
        let mut sum = 0.0_f32;

        while i < self.indices.len() && j < other.indices.len() {
            let a = self.indices[i];
            let b = other.indices[j];
            if a == b {
                sum += self.values[i] * other.values[j];
                i += 1;
                j += 1;
            } else if a < b {
                i += 1;
            } else {
                j += 1;
            }
        }
        sum
    }

    /// Scatter-add into a dense vector: `dense[idx] += scale * val` for every
    /// stored `(idx, val)`.
    pub fn add_to_dense(&self, dense: &mut [f32], scale: f32) {
        for (&idx, &val) in self.indices.iter().zip(self.values.iter()) {
            dense[idx as usize] += scale * val;
        }
    }

    /// Number of stored (non-zero) entries.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.indices.len()
    }

    /// Returns `true` when no entries are stored.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Iterate over `(index, value)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (u32, f32)> + '_ {
        self.indices.iter().copied().zip(self.values.iter().copied())
    }
}

// ---------------------------------------------------------------------------
// SparseBatchTensor
// ---------------------------------------------------------------------------

/// A batch of [`SparseVector`]s, e.g. a mini-batch of sparse input features.
#[derive(Debug, Clone)]
pub struct SparseBatchTensor {
    pub vectors: Vec<SparseVector>,
}

impl SparseBatchTensor {
    /// Create an empty batch.
    pub fn new() -> Self {
        Self {
            vectors: Vec::new(),
        }
    }

    /// Append a sparse vector to the batch.
    pub fn push(&mut self, vec: SparseVector) {
        self.vectors.push(vec);
    }

    /// Number of vectors in the batch.
    #[inline]
    pub fn batch_size(&self) -> usize {
        self.vectors.len()
    }

    /// Iterate over the vectors in the batch.
    pub fn iter(&self) -> impl Iterator<Item = &SparseVector> {
        self.vectors.iter()
    }
}

impl Default for SparseBatchTensor {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- SparseVector -------------------------------------------------------

    #[test]
    fn test_new_empty() {
        let v = SparseVector::new(10);
        assert_eq!(v.dim, 10);
        assert!(v.is_empty());
        assert_eq!(v.nnz(), 0);
    }

    #[test]
    fn test_from_pairs_basic() {
        let v = SparseVector::from_pairs(5, vec![(3, 1.0), (1, 2.0), (4, 3.0)]);
        assert_eq!(v.indices, vec![1, 3, 4]);
        assert_eq!(v.values, vec![2.0, 1.0, 3.0]);
        assert_eq!(v.dim, 5);
    }

    #[test]
    fn test_from_pairs_dedup_sums() {
        let v = SparseVector::from_pairs(5, vec![(2, 1.0), (2, 3.0), (0, 5.0)]);
        assert_eq!(v.indices, vec![0, 2]);
        assert_eq!(v.values, vec![5.0, 4.0]);
    }

    #[test]
    fn test_from_pairs_empty() {
        let v = SparseVector::from_pairs(8, vec![]);
        assert!(v.is_empty());
        assert_eq!(v.dim, 8);
    }

    #[test]
    fn test_from_dense() {
        let dense = [0.0, 1.0, 0.0, 3.0, 0.0];
        let v = SparseVector::from_dense(&dense);
        assert_eq!(v.indices, vec![1, 3]);
        assert_eq!(v.values, vec![1.0, 3.0]);
        assert_eq!(v.dim, 5);
    }

    #[test]
    fn test_from_dense_all_zero() {
        let dense = [0.0; 4];
        let v = SparseVector::from_dense(&dense);
        assert!(v.is_empty());
        assert_eq!(v.dim, 4);
    }

    #[test]
    fn test_from_dense_empty_slice() {
        let v = SparseVector::from_dense(&[]);
        assert!(v.is_empty());
        assert_eq!(v.dim, 0);
    }

    #[test]
    fn test_to_dense() {
        let v = SparseVector::from_pairs(5, vec![(1, 2.0), (3, 4.0)]);
        let dense = v.to_dense();
        assert_eq!(dense, vec![0.0, 2.0, 0.0, 4.0, 0.0]);
    }

    #[test]
    fn test_to_dense_empty() {
        let v = SparseVector::new(3);
        assert_eq!(v.to_dense(), vec![0.0; 3]);
    }

    #[test]
    fn test_roundtrip_dense() {
        let original = vec![0.0, 1.5, 0.0, -2.0, 0.0, 0.0, 7.0];
        let v = SparseVector::from_dense(&original);
        let recovered = v.to_dense();
        assert_eq!(original, recovered);
    }

    #[test]
    fn test_dot_dense() {
        let sparse = SparseVector::from_pairs(4, vec![(0, 1.0), (2, 3.0)]);
        let dense = [2.0, 5.0, 4.0, 9.0];
        // 1*2 + 3*4 = 14
        assert!((sparse.dot_dense(&dense) - 14.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_dense_empty_sparse() {
        let sparse = SparseVector::new(4);
        let dense = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(sparse.dot_dense(&dense), 0.0);
    }

    #[test]
    fn test_dot_sparse_overlap() {
        let a = SparseVector::from_pairs(10, vec![(1, 2.0), (3, 4.0), (7, 1.0)]);
        let b = SparseVector::from_pairs(10, vec![(0, 5.0), (3, 3.0), (7, 2.0)]);
        // overlap at 3: 4*3=12, at 7: 1*2=2 => 14
        assert!((a.dot_sparse(&b) - 14.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_sparse_no_overlap() {
        let a = SparseVector::from_pairs(10, vec![(0, 1.0), (2, 2.0)]);
        let b = SparseVector::from_pairs(10, vec![(1, 3.0), (3, 4.0)]);
        assert_eq!(a.dot_sparse(&b), 0.0);
    }

    #[test]
    fn test_dot_sparse_both_empty() {
        let a = SparseVector::new(5);
        let b = SparseVector::new(5);
        assert_eq!(a.dot_sparse(&b), 0.0);
    }

    #[test]
    fn test_dot_sparse_one_empty() {
        let a = SparseVector::from_pairs(5, vec![(0, 1.0)]);
        let b = SparseVector::new(5);
        assert_eq!(a.dot_sparse(&b), 0.0);
    }

    #[test]
    fn test_add_to_dense() {
        let sparse = SparseVector::from_pairs(4, vec![(0, 1.0), (2, 3.0)]);
        let mut dense = [10.0, 20.0, 30.0, 40.0];
        sparse.add_to_dense(&mut dense, 2.0);
        // dense[0] += 2*1 = 12, dense[2] += 2*3 = 36
        assert_eq!(dense, [12.0, 20.0, 36.0, 40.0]);
    }

    #[test]
    fn test_add_to_dense_zero_scale() {
        let sparse = SparseVector::from_pairs(3, vec![(0, 100.0)]);
        let mut dense = [1.0, 2.0, 3.0];
        sparse.add_to_dense(&mut dense, 0.0);
        assert_eq!(dense, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_add_to_dense_negative_scale() {
        let sparse = SparseVector::from_pairs(3, vec![(1, 5.0)]);
        let mut dense = [0.0, 10.0, 0.0];
        sparse.add_to_dense(&mut dense, -1.0);
        assert_eq!(dense, [0.0, 5.0, 0.0]);
    }

    #[test]
    fn test_nnz_and_is_empty() {
        let v = SparseVector::from_pairs(5, vec![(0, 1.0), (4, 2.0)]);
        assert_eq!(v.nnz(), 2);
        assert!(!v.is_empty());

        let empty = SparseVector::new(5);
        assert_eq!(empty.nnz(), 0);
        assert!(empty.is_empty());
    }

    #[test]
    fn test_iter() {
        let v = SparseVector::from_pairs(5, vec![(3, 1.0), (1, 2.0)]);
        let pairs: Vec<(u32, f32)> = v.iter().collect();
        assert_eq!(pairs, vec![(1, 2.0), (3, 1.0)]);
    }

    #[test]
    fn test_iter_empty() {
        let v = SparseVector::new(5);
        assert_eq!(v.iter().count(), 0);
    }

    #[test]
    fn test_serde_roundtrip() {
        let v = SparseVector::from_pairs(6, vec![(0, 1.0), (5, -3.5)]);
        let json = serde_json::to_string(&v).unwrap();
        let v2: SparseVector = serde_json::from_str(&json).unwrap();
        assert_eq!(v.indices, v2.indices);
        assert_eq!(v.values, v2.values);
        assert_eq!(v.dim, v2.dim);
    }

    // -- SparseBatchTensor --------------------------------------------------

    #[test]
    fn test_batch_new_empty() {
        let batch = SparseBatchTensor::new();
        assert_eq!(batch.batch_size(), 0);
    }

    #[test]
    fn test_batch_push_and_size() {
        let mut batch = SparseBatchTensor::new();
        batch.push(SparseVector::new(10));
        batch.push(SparseVector::from_pairs(10, vec![(0, 1.0)]));
        assert_eq!(batch.batch_size(), 2);
    }

    #[test]
    fn test_batch_iter() {
        let mut batch = SparseBatchTensor::new();
        batch.push(SparseVector::from_pairs(4, vec![(0, 1.0)]));
        batch.push(SparseVector::from_pairs(4, vec![(1, 2.0)]));

        let dims: Vec<usize> = batch.iter().map(|v| v.dim).collect();
        assert_eq!(dims, vec![4, 4]);
    }

    #[test]
    fn test_batch_default() {
        let batch = SparseBatchTensor::default();
        assert_eq!(batch.batch_size(), 0);
    }

    // -- Cross-method consistency ------------------------------------------

    #[test]
    fn test_dot_dense_matches_dot_sparse() {
        let a = SparseVector::from_pairs(8, vec![(1, 2.0), (3, -1.0), (5, 0.5)]);
        let b = SparseVector::from_pairs(8, vec![(0, 1.0), (1, 3.0), (5, 4.0)]);

        let via_sparse = a.dot_sparse(&b);
        let b_dense = b.to_dense();
        let via_dense = a.dot_dense(&b_dense);

        assert!(
            (via_sparse - via_dense).abs() < 1e-6,
            "dot_sparse={via_sparse} vs dot_dense={via_dense}",
        );
    }

    #[test]
    fn test_add_to_dense_equivalent_to_scaled_to_dense() {
        let v = SparseVector::from_pairs(5, vec![(1, 3.0), (4, -1.0)]);
        let scale = 2.5_f32;

        let mut dense = vec![0.0_f32; 5];
        v.add_to_dense(&mut dense, scale);

        let expected: Vec<f32> = v.to_dense().iter().map(|x| x * scale).collect();
        for (a, b) in dense.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_single_element() {
        let v = SparseVector::from_pairs(1, vec![(0, 42.0)]);
        assert_eq!(v.to_dense(), vec![42.0]);
        assert_eq!(v.dot_dense(&[2.0]), 84.0);
    }

    #[test]
    fn test_large_dim_sparse() {
        let dim = 1_000_000;
        let v = SparseVector::from_pairs(dim, vec![(999_999, 1.0)]);
        assert_eq!(v.nnz(), 1);
        let dense = v.to_dense();
        assert_eq!(dense.len(), dim);
        assert_eq!(dense[999_999], 1.0);
        assert_eq!(dense[0], 0.0);
    }
}
