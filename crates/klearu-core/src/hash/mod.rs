//! Locality-Sensitive Hashing families for the SLIDE algorithm.
//!
//! Each hash family implements the [`HashFamily`] trait, which provides both
//! dense and sparse hashing for a configurable number of tables and hash bits.

mod dwta;
mod minhash;
mod simhash;
mod srp;
mod wta;

pub use dwta::DwtaHash;
pub use minhash::MinHash;
pub use simhash::SimHash;
pub use srp::SparseRandomProjection;
pub use wta::WtaHash;

use crate::tensor::SparseVector;

/// A family of locality-sensitive hash functions.
///
/// Each family is constructed with L tables and K hash bits per table.
/// Hashing a vector produces a K-bit integer in `[0, 2^K)` for each table.
pub trait HashFamily: Send + Sync {
    /// Hash a dense vector, returning the hash code for a given table.
    fn hash_dense(&self, input: &[f32], table: usize) -> u64;

    /// Hash a sparse vector, returning the hash code for a given table.
    fn hash_sparse(&self, input: &SparseVector, table: usize) -> u64;

    /// Number of hash bits per table (K).
    fn k(&self) -> usize;

    /// Input dimensionality this family was constructed for.
    fn input_dim(&self) -> usize;

    /// Number of tables (L).
    fn num_tables(&self) -> usize;
}

#[cfg(test)]
pub(crate) mod test_helpers {
    use super::*;

    /// Build a dense vector and its sparse twin, hash both, assert equal.
    pub fn assert_dense_sparse_agree<H: HashFamily>(h: &H, dense: &[f32]) {
        let sparse = SparseVector::from_dense(dense);
        for t in 0..h.num_tables() {
            let hd = h.hash_dense(dense, t);
            let hs = h.hash_sparse(&sparse, t);
            assert_eq!(
                hd, hs,
                "table {t}: dense hash {hd} != sparse hash {hs}"
            );
        }
    }

    /// Verify hash values are in [0, 2^k).
    pub fn assert_hash_in_range<H: HashFamily>(h: &H, dense: &[f32]) {
        let max = 1u64 << h.k();
        for t in 0..h.num_tables() {
            let hv = h.hash_dense(dense, t);
            assert!(hv < max, "table {t}: hash {hv} >= {max}");
        }
    }

    /// Verify determinism: building twice from same seed gives same hashes.
    pub fn assert_deterministic<F>(build: F, dense: &[f32])
    where
        F: Fn() -> Box<dyn HashFamily>,
    {
        let h1 = build();
        let h2 = build();
        for t in 0..h1.num_tables() {
            assert_eq!(
                h1.hash_dense(dense, t),
                h2.hash_dense(dense, t),
                "table {t}: non-deterministic"
            );
        }
    }
}
