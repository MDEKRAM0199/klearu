use crate::fixed_point::to_fixed;
use crate::multiply::beaver_multiply;
use crate::beaver::BeaverTriple;
use crate::sharing::SharedVec;
use crate::transport::Transport;
use std::io;

/// SiLU polynomial approximation coefficients (fitted over [-3, 3]).
///
/// silu(x) ≈ c0 + c1*x + c2*x^2
///
/// Max error over [-3, 3]: ~0.12
const SILU_C0: f32 = 0.07;
const SILU_C1: f32 = 0.5;
const SILU_C2: f32 = 0.157;

/// Evaluate SiLU polynomial approximation on shared values.
///
/// Needs 1 triple per element (for x^2).
pub fn silu_approx_shared(
    party: u8,
    x_share: &SharedVec,
    triples: &[BeaverTriple],
    transport: &mut impl Transport,
) -> io::Result<SharedVec> {
    let n = x_share.len();
    assert!(triples.len() >= n);

    // Compute x^2 shares
    let mut x2_shares = Vec::with_capacity(n);
    for i in 0..n {
        let x2 = beaver_multiply(party, x_share.0[i], x_share.0[i], &triples[i], transport)?;
        x2_shares.push(x2);
    }

    let c0_fp = to_fixed(SILU_C0);
    let c1_fp = to_fixed(SILU_C1);
    let c2_fp = to_fixed(SILU_C2);

    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let mut val = 0u32;

        // c0 (only party 0 adds public constants)
        if party == 0 {
            val = val.wrapping_add(c0_fp);
        }

        // c1 * x_share (public constant * share = local widening multiply + truncate)
        val = val.wrapping_add(
            ((c1_fp as i32 as i64 * x_share.0[i] as i32 as i64) >> 16) as i32 as u32
        );

        // c2 * x2_share
        val = val.wrapping_add(
            ((c2_fp as i32 as i64 * x2_shares[i] as i32 as i64) >> 16) as i32 as u32
        );

        result.push(val);
    }

    Ok(SharedVec(result))
}

/// SwiGLU: silu_approx(gate) * up, element-wise.
///
/// Needs 2 triples per element: 1 for SiLU's x^2 + 1 for gate*up multiply.
pub fn swiglu_shared(
    party: u8,
    gate_share: &SharedVec,
    up_share: &SharedVec,
    triples: &[BeaverTriple],
    transport: &mut impl Transport,
) -> io::Result<SharedVec> {
    let n = gate_share.len();
    assert_eq!(n, up_share.len());
    assert!(triples.len() >= 2 * n);

    let activated = silu_approx_shared(party, gate_share, &triples[..n], transport)?;

    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let val = beaver_multiply(
            party, activated.0[i], up_share.0[i], &triples[n + i], transport,
        )?;
        result.push(val);
    }

    Ok(SharedVec(result))
}

/// Reference SiLU: x * sigmoid(x).
pub fn silu_exact(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Reference SiLU polynomial approximation (plaintext).
pub fn silu_poly_approx(x: f32) -> f32 {
    SILU_C0 + SILU_C1 * x + SILU_C2 * x * x
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::beaver::{dummy_triple_pair, TripleGenerator};
    use crate::fixed_point::{from_fixed, to_fixed};
    use crate::transport::memory_transport_pair;

    #[test]
    fn test_silu_poly_approx_accuracy() {
        // Max error over [-3, 3] should be < 0.15
        for i in -30..=30 {
            let x = i as f32 / 10.0;
            let exact = silu_exact(x);
            let approx = silu_poly_approx(x);
            assert!(
                (exact - approx).abs() < 0.15,
                "SiLU approx at x={}: exact={}, approx={}, err={}",
                x, exact, approx, (exact - approx).abs()
            );
        }
    }

    #[test]
    fn test_silu_shared() {
        for &x in &[0.0f32, 1.0, -1.0, 2.0] {
            let x_fixed = to_fixed(x);
            let expected = silu_poly_approx(x);

            let (mut gen0, mut gen1) = dummy_triple_pair(500);
            let t0 = gen0.generate(1);
            let t1 = gen1.generate(1);

            let (mut trans_a, mut trans_b) = memory_transport_pair();

            let share0 = SharedVec(vec![x_fixed]);
            let share1 = SharedVec(vec![0u32]);

            let handle = std::thread::spawn(move || {
                silu_approx_shared(1, &share1, &t1, &mut trans_b).unwrap()
            });

            let result0 = silu_approx_shared(0, &share0, &t0, &mut trans_a).unwrap();
            let result1 = handle.join().unwrap();

            let result = from_fixed(result0.0[0].wrapping_add(result1.0[0]));
            assert!(
                (result - expected).abs() < 1.5,
                "shared SiLU({}) = {}, expected {} (exact silu: {})",
                x, result, expected, silu_exact(x)
            );
        }
    }

    #[test]
    fn test_swiglu_shared() {
        let gate = 1.0f32;
        let up = 2.0f32;
        let expected = silu_poly_approx(gate) * up;

        let (mut gen0, mut gen1) = dummy_triple_pair(600);
        let t0 = gen0.generate(2);
        let t1 = gen1.generate(2);

        let (mut trans_a, mut trans_b) = memory_transport_pair();

        let gate0 = SharedVec(vec![to_fixed(gate)]);
        let up0 = SharedVec(vec![to_fixed(up)]);
        let gate1 = SharedVec(vec![0u32]);
        let up1 = SharedVec(vec![0u32]);

        let handle = std::thread::spawn(move || {
            swiglu_shared(1, &gate1, &up1, &t1, &mut trans_b).unwrap()
        });

        let result0 = swiglu_shared(0, &gate0, &up0, &t0, &mut trans_a).unwrap();
        let result1 = handle.join().unwrap();

        let result = from_fixed(result0.0[0].wrapping_add(result1.0[0]));
        assert!(
            (result - expected).abs() < 2.0,
            "shared SwiGLU({}, {}) = {}, expected {}",
            gate, up, result, expected
        );
    }
}
