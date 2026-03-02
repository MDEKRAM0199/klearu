//! BF16 weight quantization with FP32 accumulation.
//!
//! Implements two modes from the SLIDE paper:
//!
//! - **Mode 2** ([`Bf16Weights`]): weights stored in BF16, gradients accumulated
//!   in FP32. This halves memory traffic while preserving gradient precision.
//! - **Mode 1** ([`Bf16FullMode`]): everything in BF16 for maximum throughput
//!   at the cost of some numerical precision.

use half::bf16;

// ---------------------------------------------------------------------------
// Mode 2: BF16 storage, FP32 accumulation
// ---------------------------------------------------------------------------

/// BF16 weight storage with FP32 accumulation (Mode 2).
///
/// Weights are stored in BF16 to halve memory bandwidth requirements.
/// Arithmetic (dot products, gradient updates) is performed by up-casting to
/// FP32 on the fly and accumulating in full precision.
#[derive(Debug, Clone)]
pub struct Bf16Weights {
    data: Vec<bf16>,
    len: usize,
}

impl Bf16Weights {
    /// Create a zero-initialized weight vector of the given length.
    pub fn new(len: usize) -> Self {
        Self {
            data: vec![bf16::ZERO; len],
            len,
        }
    }

    /// Quantize an FP32 slice into BF16 storage.
    pub fn from_f32(data: &[f32]) -> Self {
        let bf_data: Vec<bf16> = data.iter().map(|&v| bf16::from_f32(v)).collect();
        let len = bf_data.len();
        Self { data: bf_data, len }
    }

    /// Dequantize the full weight vector back to FP32.
    pub fn to_f32(&self) -> Vec<f32> {
        self.data.iter().map(|v| v.to_f32()).collect()
    }

    /// Read a single element, returned as f32.
    #[inline]
    pub fn get(&self, index: usize) -> f32 {
        self.data[index].to_f32()
    }

    /// Store a single f32 value, converting to BF16.
    #[inline]
    pub fn set(&mut self, index: usize, value: f32) {
        self.data[index] = bf16::from_f32(value);
    }

    /// Dot product of this BF16 vector with an FP32 slice.
    ///
    /// Each BF16 weight is up-cast to FP32 before multiplication; the running
    /// sum is maintained in FP32.
    pub fn dot_f32(&self, other: &[f32]) -> f32 {
        assert_eq!(
            self.len,
            other.len(),
            "Bf16Weights::dot_f32: length mismatch ({} vs {})",
            self.len,
            other.len(),
        );
        self.data
            .iter()
            .zip(other.iter())
            .map(|(w, &x)| w.to_f32() * x)
            .sum()
    }

    /// Read-modify-write a single element: `w[index] += delta`.
    ///
    /// The addition is performed in FP32 and the result is stored back as
    /// BF16. This avoids the double-rounding issue of doing arithmetic
    /// directly in BF16.
    #[inline]
    pub fn update_from_f32(&mut self, index: usize, delta: f32) {
        let current = self.data[index].to_f32();
        self.data[index] = bf16::from_f32(current + delta);
    }

    /// Number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the vector is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// View the raw BF16 data.
    #[inline]
    pub fn as_slice(&self) -> &[bf16] {
        &self.data
    }
}

// ---------------------------------------------------------------------------
// Mode 1: Full BF16
// ---------------------------------------------------------------------------

/// Full BF16 mode (Mode 1): everything in BF16 for maximum throughput.
///
/// Both operands are stored in BF16. For a dot product each pair is up-cast to
/// FP32, multiplied, and accumulated in FP32 to avoid catastrophic precision
/// loss during summation.
pub struct Bf16FullMode;

impl Bf16FullMode {
    /// Dot product of two BF16 slices, accumulated in FP32.
    pub fn dot(a: &[bf16], b: &[bf16]) -> f32 {
        assert_eq!(
            a.len(),
            b.len(),
            "Bf16FullMode::dot: length mismatch ({} vs {})",
            a.len(),
            b.len(),
        );
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| x.to_f32() * y.to_f32())
            .sum()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: approximate floating-point equality.
    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    // -- Bf16Weights --------------------------------------------------------

    #[test]
    fn test_new_zeros() {
        let w = Bf16Weights::new(16);
        assert_eq!(w.len(), 16);
        assert!(!w.is_empty());
        for &v in w.as_slice() {
            assert_eq!(v.to_f32(), 0.0);
        }
    }

    #[test]
    fn test_new_empty() {
        let w = Bf16Weights::new(0);
        assert_eq!(w.len(), 0);
        assert!(w.is_empty());
    }

    #[test]
    fn test_from_f32_roundtrip() {
        let original = vec![1.0f32, -2.5, 3.125, 0.0, 100.0, -0.001953125];
        let w = Bf16Weights::from_f32(&original);
        let recovered = w.to_f32();

        // BF16 has ~7 bits of mantissa, so we expect roughly 1% relative
        // error for values away from zero.
        for (o, r) in original.iter().zip(recovered.iter()) {
            if *o == 0.0 {
                assert_eq!(*r, 0.0);
            } else {
                let rel = (o - r).abs() / o.abs();
                assert!(
                    rel < 0.02,
                    "excessive roundtrip error: original={o}, recovered={r}, rel={rel}",
                );
            }
        }
    }

    #[test]
    fn test_get_set() {
        let mut w = Bf16Weights::new(4);
        w.set(0, 1.5);
        w.set(2, -3.0);
        assert!(approx_eq(w.get(0), 1.5, 0.1));
        assert!(approx_eq(w.get(1), 0.0, 1e-6));
        assert!(approx_eq(w.get(2), -3.0, 0.1));
    }

    #[test]
    fn test_dot_f32() {
        let f32_weights = vec![1.0f32, 2.0, 3.0, 4.0];
        let w = Bf16Weights::from_f32(&f32_weights);
        let other = vec![1.0f32, 1.0, 1.0, 1.0];
        // Expected: 1+2+3+4 = 10
        assert!(approx_eq(w.dot_f32(&other), 10.0, 0.1));
    }

    #[test]
    fn test_dot_f32_mixed() {
        let f32_weights = vec![0.5f32, -1.0, 2.0];
        let w = Bf16Weights::from_f32(&f32_weights);
        let other = vec![2.0f32, 3.0, 0.5];
        // Expected: 0.5*2 + (-1)*3 + 2*0.5 = 1 - 3 + 1 = -1
        assert!(approx_eq(w.dot_f32(&other), -1.0, 0.1));
    }

    #[test]
    fn test_update_from_f32() {
        let mut w = Bf16Weights::from_f32(&[10.0]);
        w.update_from_f32(0, 5.0);
        assert!(approx_eq(w.get(0), 15.0, 0.5));
        w.update_from_f32(0, -20.0);
        assert!(approx_eq(w.get(0), -5.0, 0.5));
    }

    #[test]
    fn test_update_from_f32_accumulation() {
        // Verify small increments accumulate correctly.
        let mut w = Bf16Weights::from_f32(&[0.0]);
        for _ in 0..100 {
            w.update_from_f32(0, 0.125);
        }
        // Expected: 12.5
        assert!(
            approx_eq(w.get(0), 12.5, 1.0),
            "accumulated value: {}",
            w.get(0),
        );
    }

    // -- Bf16FullMode -------------------------------------------------------

    #[test]
    fn test_full_mode_dot() {
        let a: Vec<bf16> = [1.0f32, 2.0, 3.0]
            .iter()
            .map(|&v| bf16::from_f32(v))
            .collect();
        let b: Vec<bf16> = [4.0f32, 5.0, 6.0]
            .iter()
            .map(|&v| bf16::from_f32(v))
            .collect();
        // Expected: 4+10+18 = 32
        assert!(approx_eq(Bf16FullMode::dot(&a, &b), 32.0, 0.5));
    }

    #[test]
    fn test_full_mode_dot_zeros() {
        let a = vec![bf16::ZERO; 10];
        let b: Vec<bf16> = (0..10).map(|i| bf16::from_f32(i as f32)).collect();
        assert!(approx_eq(Bf16FullMode::dot(&a, &b), 0.0, 1e-6));
    }

    #[test]
    fn test_full_mode_dot_empty() {
        assert!(approx_eq(Bf16FullMode::dot(&[], &[]), 0.0, 1e-6));
    }

    // -- Quantization error bounds ------------------------------------------

    #[test]
    fn test_bf16_quantization_error_bound() {
        // BF16 has 8 exponent bits and 7 mantissa bits, so relative error
        // should be <= 2^{-7} ~ 0.0078 for normal numbers.
        let test_values = [
            0.1, 0.5, 1.0, 3.25, 100.0, 1000.0, -0.1, -42.0, 65504.0,
        ];
        for &v in &test_values {
            let q = bf16::from_f32(v).to_f32();
            let rel = (v - q).abs() / v.abs();
            assert!(
                rel < 0.01,
                "BF16 relative error too large for {v}: quantized={q}, rel={rel}",
            );
        }
    }
}
