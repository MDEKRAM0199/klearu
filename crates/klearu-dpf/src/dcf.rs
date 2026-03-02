use crate::aes_prg::AesPrg;
use crate::dpf::{dpf_eval, dpf_gen, DpfKey};

/// DCF key for comparison function f(x) = beta if x < alpha, else 0.
///
/// Implemented via prefix decomposition into DPF keys. The comparison
/// function `[x < alpha]` can be expressed as a sum of at most `depth`
/// point functions, one for each bit position where alpha has a 1 bit.
///
/// For each level i where alpha's bit is 1:
///   The contribution is a "block" function that outputs beta for all
///   points whose top i bits match alpha's top i bits and whose bit i is 0.
///
/// We precompute this as a set of DPF keys, one per 1-bit in alpha.
#[derive(Clone, Debug)]
pub struct DcfKey {
    pub party: u8,
    /// One DPF key per 1-bit in alpha, at various depths
    pub dpf_keys: Vec<(u8, DpfKey)>, // (depth_of_this_dpf, key)
    pub total_depth: u8,
}

/// Generate a pair of DCF keys for f(x) = beta if x < alpha, else 0.
///
/// Decomposition: x < alpha iff there exists some bit position i (MSB first)
/// where x and alpha agree on bits 0..i-1, alpha has bit i = 1, and x has bit i = 0.
///
/// For each such bit position i where alpha_i = 1:
///   We create a DPF over a (i+1)-bit domain at the "prefix point" where
///   the top i bits match alpha and the (i+1)th bit is 0.
///   This DPF is evaluated by taking the top (i+1) bits of x.
///
/// Wait, that doesn't quite work because each "block" is actually an interval
/// [prefix0, prefix1) of size 2^(depth-i-1), not a single point.
///
/// Simpler approach: use `depth` DPF evaluations of 1-bit DPFs.
///
/// Actually, the simplest correct approach: implement DCF directly using the
/// DPF-tree walk, accumulating contributions from left subtrees when we go right
/// along alpha's path.
///
/// The tree structure is the same as a DPF keyed at alpha. When evaluating at x:
/// - Walk from root to x's leaf
/// - Whenever x goes RIGHT and alpha ALSO goes RIGHT at the same level:
///   the left subtree is entirely < alpha (for points sharing the same prefix)
///   → no contribution (these are less than alpha's path but we need to check)
///
/// Actually, let me use the simplest correct approach: decompose into DPFs.
///
/// f(x) = beta * [x < alpha] = beta * sum_{i : alpha_bit_i = 1} [top(x, i) == top(alpha, i) AND x_bit_i == 0]
///
/// where top(x, i) means the top i bits and x_bit_i is bit i (MSB-first).
///
/// Each indicator [top(x,i)==top(alpha,i) AND x_bit_i==0] is a "block" that
/// covers 2^(depth-i-1) consecutive points. This isn't a single point function.
///
/// Let's just implement DCF directly without DPF decomposition. Use a
/// tree-based approach where we track the state along alpha's path and
/// detect when x diverges.
pub fn dcf_gen(prg: &AesPrg, alpha: u32, beta: u32, depth: u8) -> (DcfKey, DcfKey) {
    assert!(depth > 0 && depth <= 32);
    if depth < 32 {
        assert!(alpha < (1u32 << depth));
    }

    // For each bit position where alpha has a 1, create a DPF at reduced depth.
    // The DPF is over the (depth - level) remaining bits, with the target at
    // the first point of alpha's left subtree at that level.
    //
    // Actually, the decomposition is simpler:
    // f(x) = beta * [x < alpha]
    //
    // In binary (MSB first), x < alpha iff at the first bit where they differ,
    // alpha has 1 and x has 0.
    //
    // Equivalently: [x < alpha] = sum over i from 0 to depth-1:
    //   [x_{0..i-1} == alpha_{0..i-1}] * [alpha_i == 1] * [x_i == 0]
    //
    // Each term is 1 for a "slab" of 2^(depth-1-i) consecutive x values.
    // These slabs are disjoint and their union is [0, alpha).
    //
    // To evaluate this using DPF: for each level i where alpha_i = 1,
    // we need to check if x's top i bits match alpha's top i bits AND x_i = 0.
    // This is equivalent to: x >> (depth - i) == alpha >> (depth - i) AND the
    // next bit of x is 0.
    // That is: (x >> (depth - 1 - i)) == 2*(alpha >> (depth - i))
    //        = (alpha >> (depth - 1 - i)) XOR 1  (flip the last bit from 1 to 0)
    //
    // For a DPF at depth (i+1): target = top (i+1) bits of alpha with bit i flipped to 0
    // = (alpha >> (depth - 1 - i)) ^ 1... no, alpha's bit i is 1 and we want 0,
    // so the target prefix is: (alpha >> (depth - 1 - i)) & ~1 = (alpha >> (depth - 1 - i)) ^ 1
    //
    // This means for each 1-bit in alpha at position i, we have a (i+1)-depth DPF
    // at target point = (alpha >> (depth - 1 - i)) ^ 1.
    //
    // When evaluating DCF at x, we evaluate each DPF at (x >> (depth - 1 - i)) and sum.
    //
    // Cost: one DPF per 1-bit in alpha, each at depth up to `depth`.
    // Key size: sum of DPF key sizes = O(depth^2) in the worst case.

    let mut dpf_keys_0 = Vec::new();
    let mut dpf_keys_1 = Vec::new();

    for i in 0..depth {
        let alpha_bit = ((alpha >> (depth - 1 - i)) & 1) != 0;
        if !alpha_bit {
            continue; // only create DPF for 1-bits
        }

        let sub_depth = i + 1;
        // Target prefix: alpha's top (i+1) bits with bit i flipped from 1 to 0
        let alpha_prefix = alpha >> (depth - 1 - i);
        let target = alpha_prefix ^ 1; // flip the lowest bit from 1 to 0

        let (k0, k1) = dpf_gen(prg, target as u32, beta, sub_depth);
        dpf_keys_0.push((sub_depth, k0));
        dpf_keys_1.push((sub_depth, k1));
    }

    (
        DcfKey { party: 0, dpf_keys: dpf_keys_0, total_depth: depth },
        DcfKey { party: 1, dpf_keys: dpf_keys_1, total_depth: depth },
    )
}

/// Evaluate DCF at point x: returns this party's share of f(x) = beta * [x < alpha].
pub fn dcf_eval(prg: &AesPrg, key: &DcfKey, x: u32) -> u32 {
    if key.total_depth < 32 {
        assert!(x < (1u32 << key.total_depth));
    }

    let mut result = 0u32;

    for (sub_depth, dpf_key) in &key.dpf_keys {
        // Extract the top sub_depth bits of x
        let x_prefix = x >> (key.total_depth - sub_depth);
        result = result.wrapping_add(dpf_eval(prg, dpf_key, x_prefix));
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aes_prg::AesPrg;

    fn make_prg() -> AesPrg {
        AesPrg::new(&[0u8; 16])
    }

    fn check_dcf(prg: &AesPrg, alpha: u32, beta: u32, depth: u8) {
        let (key0, key1) = dcf_gen(prg, alpha, beta, depth);
        let domain = 1u32 << depth;

        for x in 0..domain {
            let share0 = dcf_eval(prg, &key0, x);
            let share1 = dcf_eval(prg, &key1, x);
            let result = share0.wrapping_add(share1);
            let expected = if x < alpha { beta } else { 0 };
            assert_eq!(
                result, expected,
                "DCF(alpha={}, beta={}, depth={}) at x={}: got {}, expected {}",
                alpha, beta, depth, x, result, expected
            );
        }
    }

    #[test]
    fn test_dcf_depth_4() {
        let prg = make_prg();
        check_dcf(&prg, 10, 42, 4);
    }

    #[test]
    fn test_dcf_alpha_1() {
        let prg = make_prg();
        check_dcf(&prg, 1, 100, 4);
    }

    #[test]
    fn test_dcf_alpha_max() {
        let prg = make_prg();
        check_dcf(&prg, 15, 50, 4);
    }

    #[test]
    fn test_dcf_alpha_zero() {
        let prg = make_prg();
        check_dcf(&prg, 0, 50, 4);
    }

    #[test]
    fn test_dcf_large_beta() {
        let prg = make_prg();
        check_dcf(&prg, 5, 0xDEADBEEF, 4);
    }

    #[test]
    fn test_dcf_depth_8() {
        let prg = make_prg();
        check_dcf(&prg, 100, 7, 8);
    }

    #[test]
    fn test_dcf_multiple_thresholds() {
        let prg = make_prg();
        for alpha in [1u32, 3, 7, 8, 12, 15] {
            check_dcf(&prg, alpha, 1, 4);
        }
    }

    #[test]
    fn test_dcf_depth_8_multiple() {
        let prg = make_prg();
        for alpha in [0, 1, 50, 128, 200, 255] {
            for beta in [1u32, 42, 0xFFFFFFFF] {
                check_dcf(&prg, alpha, beta, 8);
            }
        }
    }
}
