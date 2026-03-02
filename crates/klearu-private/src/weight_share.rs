//! Utilities for secret-sharing model weights between two parties.
//!
//! Converts floating-point model weights to fixed-point representation
//! and splits them into additive shares for 2PC inference.

#![allow(unused)]

use klearu_mpc::{SharedVec, to_fixed};

/// Convert a slice of f32 model weights to fixed-point and split into
/// two additive shares.
///
/// Returns `(share_0, share_1)` where `share_0 + share_1 = to_fixed(w)`
/// for each weight `w`.
pub fn share_weights(weights: &[f32], rng: &mut impl rand::Rng) -> (SharedVec, SharedVec) {
    let fixed: Vec<u32> = weights.iter().map(|&w| to_fixed(w)).collect();
    SharedVec::from_plaintext(&fixed, rng)
}

/// Reconstruct f32 weights from two additive shares (for testing / debugging).
pub fn reconstruct_weights(a: &SharedVec, b: &SharedVec) -> Vec<f32> {
    let plaintext = SharedVec::reconstruct(a, b);
    plaintext.iter().map(|&v| klearu_mpc::from_fixed(v)).collect()
}
