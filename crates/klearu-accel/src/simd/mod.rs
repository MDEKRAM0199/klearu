//! SIMD-accelerated operations for sparse and dense linear algebra.
//!
//! Provides runtime feature detection on x86_64 and compile-time dispatch on
//! aarch64 to select the fastest available implementation.

// ---------------------------------------------------------------------------
// Sparse dot product: indices/values against a dense vector
// ---------------------------------------------------------------------------

/// SIMD-accelerated sparse dot product with a dense vector.
///
/// Gathers values from `dense` at positions given by `indices`, multiplies
/// element-wise with `values`, and returns the sum.
///
/// Automatically dispatches to the best available instruction set:
/// - AVX2 on x86_64 (runtime detection)
/// - NEON on aarch64
/// - Scalar fallback otherwise
pub fn sparse_dot_dense_simd(indices: &[u32], values: &[f32], dense: &[f32]) -> f32 {
    assert_eq!(indices.len(), values.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { sparse_dot_dense_avx2(indices, values, dense) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { sparse_dot_dense_neon(indices, values, dense) };
    }

    #[allow(unreachable_code)]
    sparse_dot_dense_scalar(indices, values, dense)
}

/// Scalar fallback for sparse dot product.
pub fn sparse_dot_dense_scalar(indices: &[u32], values: &[f32], dense: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for (i, &idx) in indices.iter().enumerate() {
        sum += values[i] * dense[idx as usize];
    }
    sum
}

// ---------------------------------------------------------------------------
// Dense dot product
// ---------------------------------------------------------------------------

/// SIMD-accelerated dot product of two dense vectors.
///
/// The two slices must have the same length.
pub fn dense_dot_dense_simd(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dense_dot_dense_avx2(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { dense_dot_dense_neon(a, b) };
    }

    #[allow(unreachable_code)]
    dense_dot_dense_scalar(a, b)
}

/// Scalar fallback for dense dot product.
pub fn dense_dot_dense_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ---------------------------------------------------------------------------
// Scatter-add: add scaled values into a dense buffer at given indices
// ---------------------------------------------------------------------------

/// SIMD-accelerated scatter-add.
///
/// For each pair `(indices[i], values[i])`, performs
/// `dense[indices[i]] += values[i] * scale`.
pub fn scatter_add_simd(indices: &[u32], values: &[f32], dense: &mut [f32], scale: f32) {
    assert_eq!(indices.len(), values.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                scatter_add_avx2(indices, values, dense, scale);
            }
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            scatter_add_neon(indices, values, dense, scale);
        }
        return;
    }

    #[allow(unreachable_code)]
    scatter_add_scalar(indices, values, dense, scale);
}

/// Scalar fallback for scatter-add.
pub fn scatter_add_scalar(indices: &[u32], values: &[f32], dense: &mut [f32], scale: f32) {
    for (i, &idx) in indices.iter().enumerate() {
        dense[idx as usize] += values[i] * scale;
    }
}

// ===========================================================================
// x86_64 AVX2 implementations
// ===========================================================================

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Horizontal sum of all 8 lanes of a `__m256`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn hsum_avx2(v: __m256) -> f32 {
    // v = [a0 a1 a2 a3 | a4 a5 a6 a7]
    // high128 = [a4 a5 a6 a7]
    let high128 = _mm256_extractf128_ps(v, 1);
    let low128 = _mm256_castps256_ps128(v);
    // sum128 = [a0+a4, a1+a5, a2+a6, a3+a7]
    let sum128 = _mm_add_ps(low128, high128);
    // Horizontal add pairs: [a0+a4+a1+a5, a2+a6+a3+a7, ...]
    let shuf = _mm_movehdup_ps(sum128); // [a1+a5, a1+a5, a3+a7, a3+a7]
    let sums = _mm_add_ps(sum128, shuf); // [a0+a1+a4+a5, ?, a2+a3+a6+a7, ?]
    let shuf2 = _mm_movehl_ps(sums, sums); // [a2+a3+a6+a7, ?, ...]
    let result = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(result)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn sparse_dot_dense_avx2(indices: &[u32], values: &[f32], dense: &[f32]) -> f32 {
    let n = indices.len();
    let chunks = n / 8;
    let mut acc = _mm256_setzero_ps();

    for c in 0..chunks {
        let offset = c * 8;
        // Load 8 indices as i32 for gather
        let idx = _mm256_loadu_si256(indices.as_ptr().add(offset) as *const __m256i);
        // Gather 8 floats from dense using indices (scale = 4 bytes per f32)
        let gathered = _mm256_i32gather_ps::<4>(dense.as_ptr(), idx);
        // Load 8 values
        let vals = _mm256_loadu_ps(values.as_ptr().add(offset));
        // Fused multiply-accumulate
        acc = _mm256_fmadd_ps(vals, gathered, acc);
    }

    let mut sum = hsum_avx2(acc);

    // Handle remaining elements
    for i in (chunks * 8)..n {
        sum += values[i] * dense[indices[i] as usize];
    }

    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dense_dot_dense_avx2(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let chunks = n / 16;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();

    for c in 0..chunks {
        let offset = c * 16;
        let va0 = _mm256_loadu_ps(a.as_ptr().add(offset));
        let vb0 = _mm256_loadu_ps(b.as_ptr().add(offset));
        acc0 = _mm256_fmadd_ps(va0, vb0, acc0);
        let va1 = _mm256_loadu_ps(a.as_ptr().add(offset + 8));
        let vb1 = _mm256_loadu_ps(b.as_ptr().add(offset + 8));
        acc1 = _mm256_fmadd_ps(va1, vb1, acc1);
    }

    // Handle remaining 8-element chunk
    let remainder_start = chunks * 16;
    if remainder_start + 8 <= n {
        let va = _mm256_loadu_ps(a.as_ptr().add(remainder_start));
        let vb = _mm256_loadu_ps(b.as_ptr().add(remainder_start));
        acc0 = _mm256_fmadd_ps(va, vb, acc0);
    }

    let mut sum = hsum_avx2(_mm256_add_ps(acc0, acc1));

    // Scalar remainder
    let scalar_start = if remainder_start + 8 <= n { remainder_start + 8 } else { remainder_start };
    for i in scalar_start..n {
        sum += a[i] * b[i];
    }

    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn scatter_add_avx2(indices: &[u32], values: &[f32], dense: &mut [f32], scale: f32) {
    let n = indices.len();
    let scale_vec = _mm256_set1_ps(scale);
    let chunks = n / 8;

    for c in 0..chunks {
        let offset = c * 8;
        let vals = _mm256_loadu_ps(values.as_ptr().add(offset));
        let scaled = _mm256_mul_ps(vals, scale_vec);

        // AVX2 lacks scatter instructions, so we extract and do scalar stores.
        // We still benefit from the vectorized multiply above.
        let mut tmp = [0.0f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), scaled);

        for j in 0..8 {
            let idx = *indices.get_unchecked(offset + j) as usize;
            *dense.get_unchecked_mut(idx) += tmp[j];
        }
    }

    // Remainder
    for i in (chunks * 8)..n {
        dense[indices[i] as usize] += values[i] * scale;
    }
}

// ===========================================================================
// aarch64 NEON implementations
// ===========================================================================

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Horizontal sum of a NEON `float32x4_t`.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn hsum_neon(v: float32x4_t) -> f32 {
    let pair = vpaddq_f32(v, v); // [a0+a1, a2+a3, a0+a1, a2+a3]
    let total = vpaddq_f32(pair, pair); // [a0+a1+a2+a3, ...]
    vgetq_lane_f32(total, 0)
}

#[cfg(target_arch = "aarch64")]
unsafe fn sparse_dot_dense_neon(indices: &[u32], values: &[f32], dense: &[f32]) -> f32 {
    let n = indices.len();
    let chunks = n / 4;
    let mut acc = vdupq_n_f32(0.0);

    for c in 0..chunks {
        let offset = c * 4;
        // NEON lacks a gather instruction, so we do manual gathers.
        let g = vld1q_f32(
            [
                dense[indices[offset] as usize],
                dense[indices[offset + 1] as usize],
                dense[indices[offset + 2] as usize],
                dense[indices[offset + 3] as usize],
            ]
            .as_ptr(),
        );
        let vals = vld1q_f32(values.as_ptr().add(offset));
        acc = vfmaq_f32(acc, vals, g);
    }

    let mut sum = hsum_neon(acc);

    for i in (chunks * 4)..n {
        sum += values[i] * dense[indices[i] as usize];
    }

    sum
}

#[cfg(target_arch = "aarch64")]
unsafe fn dense_dot_dense_neon(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let chunks = n / 8;
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);

    for c in 0..chunks {
        let offset = c * 8;
        let va0 = vld1q_f32(a.as_ptr().add(offset));
        let vb0 = vld1q_f32(b.as_ptr().add(offset));
        acc0 = vfmaq_f32(acc0, va0, vb0);
        let va1 = vld1q_f32(a.as_ptr().add(offset + 4));
        let vb1 = vld1q_f32(b.as_ptr().add(offset + 4));
        acc1 = vfmaq_f32(acc1, va1, vb1);
    }

    // Handle remaining 4-element chunk
    let remainder_start = chunks * 8;
    if remainder_start + 4 <= n {
        let va = vld1q_f32(a.as_ptr().add(remainder_start));
        let vb = vld1q_f32(b.as_ptr().add(remainder_start));
        acc0 = vfmaq_f32(acc0, va, vb);
    }

    let mut sum = hsum_neon(vaddq_f32(acc0, acc1));

    // Scalar remainder
    let scalar_start = if remainder_start + 4 <= n { remainder_start + 4 } else { remainder_start };
    for i in scalar_start..n {
        sum += a[i] * b[i];
    }

    sum
}

#[cfg(target_arch = "aarch64")]
unsafe fn scatter_add_neon(indices: &[u32], values: &[f32], dense: &mut [f32], scale: f32) {
    let n = indices.len();
    let scale_vec = vdupq_n_f32(scale);
    let chunks = n / 4;

    for c in 0..chunks {
        let offset = c * 4;
        let vals = vld1q_f32(values.as_ptr().add(offset));
        let scaled = vmulq_f32(vals, scale_vec);

        // NEON lacks scatter instructions; extract lanes and store individually.
        let s0 = vgetq_lane_f32(scaled, 0);
        let s1 = vgetq_lane_f32(scaled, 1);
        let s2 = vgetq_lane_f32(scaled, 2);
        let s3 = vgetq_lane_f32(scaled, 3);

        dense[indices[offset] as usize] += s0;
        dense[indices[offset + 1] as usize] += s1;
        dense[indices[offset + 2] as usize] += s2;
        dense[indices[offset + 3] as usize] += s3;
    }

    for i in (chunks * 4)..n {
        dense[indices[i] as usize] += values[i] * scale;
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

    // -- Sparse dot product -------------------------------------------------

    #[test]
    fn test_sparse_dot_dense_scalar_basic() {
        let dense = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let indices = vec![0, 2, 4];
        let values = vec![1.0, 1.0, 1.0];
        // 1*1 + 1*3 + 1*5 = 9.0
        assert!(approx_eq(
            sparse_dot_dense_scalar(&indices, &values, &dense),
            9.0,
            1e-6,
        ));
    }

    #[test]
    fn test_sparse_dot_dense_scalar_weighted() {
        let dense = vec![10.0, 20.0, 30.0, 40.0];
        let indices = vec![1, 3];
        let values = vec![0.5, 0.25];
        // 0.5*20 + 0.25*40 = 10 + 10 = 20
        assert!(approx_eq(
            sparse_dot_dense_scalar(&indices, &values, &dense),
            20.0,
            1e-6,
        ));
    }

    #[test]
    fn test_sparse_dot_dense_simd_matches_scalar() {
        let dense: Vec<f32> = (0..256).map(|i| i as f32 * 0.1).collect();
        let indices: Vec<u32> = (0..37).map(|i| (i * 7) % 256).collect();
        let values: Vec<f32> = (0..37).map(|i| (i as f32 + 1.0) * 0.01).collect();

        let scalar = sparse_dot_dense_scalar(&indices, &values, &dense);
        let simd = sparse_dot_dense_simd(&indices, &values, &dense);
        assert!(
            approx_eq(scalar, simd, 1e-3),
            "scalar={scalar}, simd={simd}",
        );
    }

    #[test]
    fn test_sparse_dot_dense_simd_empty() {
        let dense = vec![1.0, 2.0, 3.0];
        assert!(approx_eq(
            sparse_dot_dense_simd(&[], &[], &dense),
            0.0,
            1e-6,
        ));
    }

    #[test]
    fn test_sparse_dot_dense_simd_exact_chunk() {
        // Exactly 8 elements (AVX2 chunk boundary) or 4 (NEON chunk boundary)
        let dense: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let indices: Vec<u32> = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let values = vec![1.0; 8];
        let expected: f32 = (0..8).map(|i| i as f32).sum();
        let result = sparse_dot_dense_simd(&indices, &values, &dense);
        assert!(approx_eq(result, expected, 1e-5), "result={result}, expected={expected}");
    }

    // -- Dense dot product --------------------------------------------------

    #[test]
    fn test_dense_dot_dense_scalar() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert!(approx_eq(dense_dot_dense_scalar(&a, &b), 32.0, 1e-6));
    }

    #[test]
    fn test_dense_dot_dense_simd_matches_scalar() {
        let a: Vec<f32> = (0..100).map(|i| (i as f32) * 0.3 - 15.0).collect();
        let b: Vec<f32> = (0..100).map(|i| (i as f32) * 0.7 + 2.0).collect();
        let scalar = dense_dot_dense_scalar(&a, &b);
        let simd = dense_dot_dense_simd(&a, &b);
        assert!(approx_eq(scalar, simd, 1e-2), "scalar={scalar}, simd={simd}");
    }

    #[test]
    fn test_dense_dot_dense_simd_empty() {
        assert!(approx_eq(dense_dot_dense_simd(&[], &[]), 0.0, 1e-6));
    }

    // -- Scatter-add --------------------------------------------------------

    #[test]
    fn test_scatter_add_scalar_basic() {
        let mut dense = vec![0.0; 8];
        let indices = vec![1, 3, 5];
        let values = vec![1.0, 2.0, 3.0];
        scatter_add_scalar(&indices, &values, &mut dense, 2.0);
        assert!(approx_eq(dense[1], 2.0, 1e-6));
        assert!(approx_eq(dense[3], 4.0, 1e-6));
        assert!(approx_eq(dense[5], 6.0, 1e-6));
        assert!(approx_eq(dense[0], 0.0, 1e-6));
    }

    #[test]
    fn test_scatter_add_simd_matches_scalar() {
        let mut dense_scalar = vec![1.0; 256];
        let mut dense_simd = dense_scalar.clone();
        let indices: Vec<u32> = (0..50).map(|i| (i * 5) % 256).collect();
        let values: Vec<f32> = (0..50).map(|i| (i as f32) * 0.1).collect();
        let scale = 0.5;

        scatter_add_scalar(&indices, &values, &mut dense_scalar, scale);
        scatter_add_simd(&indices, &values, &mut dense_simd, scale);

        for (i, (s, d)) in dense_scalar.iter().zip(dense_simd.iter()).enumerate() {
            assert!(
                approx_eq(*s, *d, 1e-4),
                "mismatch at {i}: scalar={s}, simd={d}",
            );
        }
    }

    #[test]
    fn test_scatter_add_simd_empty() {
        let mut dense = vec![5.0; 4];
        scatter_add_simd(&[], &[], &mut dense, 1.0);
        assert!(dense.iter().all(|&v| approx_eq(v, 5.0, 1e-6)));
    }
}
