use crate::beaver::BeaverTriple;
use crate::fixed_point::SCALE_64;
use crate::sharing::{SharedVec, SharedVec64};
use crate::transport::Transport;
use std::io;

/// Shared linear forward: y_share = W * x_share.
///
/// Weights W are public to both parties. Since W is public, the linear
/// operation is purely local: each party computes W * x_share independently.
/// No communication or triples needed.
///
/// `weights`: row-major with stride >= in_features (may include padding).
/// The stride is inferred as `weights.len() / out_features`.
pub fn shared_linear_forward(
    _party: u8,
    weights: &[f32],
    in_features: usize,
    out_features: usize,
    x_share: &SharedVec,
    _triples: &[BeaverTriple],
    _transport: &mut impl Transport,
) -> io::Result<SharedVec> {
    let stride = if out_features > 0 { weights.len() / out_features } else { in_features };
    assert!(stride >= in_features, "stride {} < in_features {}", stride, in_features);
    assert_eq!(weights.len(), out_features * stride);
    assert_eq!(x_share.len(), in_features);

    let mut output = Vec::with_capacity(out_features);

    for j in 0..out_features {
        let row_offset = j * stride;
        let mut acc = 0.0f64;
        for i in 0..in_features {
            acc += weights[row_offset + i] as f64 * x_share.0[i] as i32 as f64;
        }
        output.push(acc.round() as i64 as i32 as u32);
    }

    Ok(SharedVec(output))
}

/// Sparse variant: only compute selected output indices.
///
/// `weights`: full weight matrix, row-major. `total_rows` is the total number
/// of rows so stride can be computed as `weights.len() / total_rows`.
pub fn shared_linear_forward_sparse(
    _party: u8,
    weights: &[f32],
    in_features: usize,
    total_rows: usize,
    indices: &[usize],
    x_share: &SharedVec,
    _triples: &[BeaverTriple],
    _transport: &mut impl Transport,
) -> io::Result<SharedVec> {
    assert_eq!(x_share.len(), in_features);
    let stride = if total_rows > 0 { weights.len() / total_rows } else { in_features };
    assert!(stride >= in_features, "stride {} < in_features {}", stride, in_features);

    let mut output = Vec::with_capacity(indices.len());

    for &j in indices {
        let row_offset = j * stride;
        let mut acc = 0.0f64;
        for i in 0..in_features {
            acc += weights[row_offset + i] as f64 * x_share.0[i] as i32 as f64;
        }
        output.push(acc.round() as i64 as i32 as u32);
    }

    Ok(SharedVec(output))
}

/// Variant that takes f32 input shares (not Q16.16).
///
/// Used when the input is already in f32 (e.g., attention output) and we want
/// to avoid the quantization loss of converting to Q16.16 first.
/// Output is Q16.16 for compatibility with the rest of the pipeline.
pub fn shared_linear_forward_f32_input(
    weights: &[f32],
    in_features: usize,
    out_features: usize,
    x_f32: &[f32],
) -> SharedVec {
    let stride = if out_features > 0 { weights.len() / out_features } else { in_features };
    assert!(stride >= in_features, "stride {} < in_features {}", stride, in_features);
    assert_eq!(weights.len(), out_features * stride);
    assert_eq!(x_f32.len(), in_features);

    let scale = 65536.0f64;
    let mut output = Vec::with_capacity(out_features);

    for j in 0..out_features {
        let row_offset = j * stride;
        let mut acc = 0.0f64;
        for i in 0..in_features {
            acc += weights[row_offset + i] as f64 * x_f32[i] as f64;
        }
        output.push((acc * scale).round() as i64 as i32 as u32);
    }

    SharedVec(output)
}

// --- Q32.32 linear forward (u64 shares) ---

/// Q32.32 shared linear forward: y_share = W * x_share (u64 shares).
///
/// Weights are public f32, quantized to Q32.32 on the fly.
/// Accumulation uses i128 to avoid overflow.
pub fn shared_linear_forward_64(
    _party: u8,
    weights: &[f32],
    in_features: usize,
    out_features: usize,
    x_share: &SharedVec64,
    _triples: &[BeaverTriple],
    _transport: &mut impl Transport,
) -> io::Result<SharedVec64> {
    let stride = if out_features > 0 { weights.len() / out_features } else { in_features };
    assert!(stride >= in_features, "stride {} < in_features {}", stride, in_features);
    assert_eq!(weights.len(), out_features * stride);
    assert_eq!(x_share.len(), in_features);

    let mut output = Vec::with_capacity(out_features);

    for j in 0..out_features {
        let row_offset = j * stride;
        let mut acc = 0i128;
        for i in 0..in_features {
            let w_q32 = (weights[row_offset + i] as f64 * SCALE_64).round() as i64;
            acc += (w_q32 as i128) * (x_share.0[i] as i64 as i128);
        }
        output.push(((acc >> 32) as i64) as u64);
    }

    Ok(SharedVec64(output))
}

/// Q32.32 sparse linear forward: only compute selected output indices.
pub fn shared_linear_forward_sparse_64(
    _party: u8,
    weights: &[f32],
    in_features: usize,
    total_rows: usize,
    indices: &[usize],
    x_share: &SharedVec64,
    _triples: &[BeaverTriple],
    _transport: &mut impl Transport,
) -> io::Result<SharedVec64> {
    assert_eq!(x_share.len(), in_features);
    let stride = if total_rows > 0 { weights.len() / total_rows } else { in_features };
    assert!(stride >= in_features, "stride {} < in_features {}", stride, in_features);

    let mut output = Vec::with_capacity(indices.len());

    for &j in indices {
        let row_offset = j * stride;
        let mut acc = 0i128;
        for i in 0..in_features {
            let w_q32 = (weights[row_offset + i] as f64 * SCALE_64).round() as i64;
            acc += (w_q32 as i128) * (x_share.0[i] as i64 as i128);
        }
        output.push(((acc >> 32) as i64) as u64);
    }

    Ok(SharedVec64(output))
}

/// Q32.32 variant taking f32 input (not Q32.32 shares).
///
/// Used when the input is already in f32 (e.g., attention output after reveal)
/// and we want Q32.32 output.
pub fn shared_linear_forward_f32_input_64(
    weights: &[f32],
    in_features: usize,
    out_features: usize,
    x_f32: &[f32],
) -> SharedVec64 {
    let stride = if out_features > 0 { weights.len() / out_features } else { in_features };
    assert!(stride >= in_features, "stride {} < in_features {}", stride, in_features);
    assert_eq!(weights.len(), out_features * stride);
    assert_eq!(x_f32.len(), in_features);

    let mut output = Vec::with_capacity(out_features);

    for j in 0..out_features {
        let row_offset = j * stride;
        let mut acc = 0.0f64;
        for i in 0..in_features {
            acc += weights[row_offset + i] as f64 * x_f32[i] as f64;
        }
        output.push((acc * SCALE_64).round() as i64 as u64);
    }

    SharedVec64(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixed_point::{from_fixed, to_fixed, from_fixed64, to_fixed64};
    use crate::transport::memory_transport_pair;

    #[test]
    fn test_shared_linear_forward_identity() {
        let weights = vec![1.0f32, 0.0, 0.0, 1.0];
        let input = vec![3.0f32, 4.0];
        let x_fixed: Vec<u32> = input.iter().map(|&v| to_fixed(v)).collect();

        let (mut trans_a, _) = memory_transport_pair();
        let result = shared_linear_forward(0, &weights, 2, 2, &SharedVec(x_fixed), &[], &mut trans_a).unwrap();

        for i in 0..2 {
            let val = from_fixed(result.0[i]);
            assert!((val - input[i]).abs() < 0.01, "identity[{}]: got {}, expected {}", i, val, input[i]);
        }
    }

    #[test]
    fn test_shared_linear_forward_simple() {
        let weights = vec![1.0f32, 2.0, 3.0, 4.0];
        let input = vec![1.0f32, 1.0];
        // Expected: [3, 7]

        let (mut trans_a, _) = memory_transport_pair();
        let result = shared_linear_forward(
            0, &weights, 2, 2, &SharedVec(input.iter().map(|&v| to_fixed(v)).collect()), &[], &mut trans_a,
        ).unwrap();

        assert!((from_fixed(result.0[0]) - 3.0).abs() < 0.01);
        assert!((from_fixed(result.0[1]) - 7.0).abs() < 0.01);
    }

    #[test]
    fn test_shared_linear_forward_two_parties() {
        let weights = vec![1.0f32, 2.0, 3.0, 4.0];
        let input = vec![2.0f32, 3.0];
        let expected = [8.0f32, 18.0];

        let x_fixed: Vec<u32> = input.iter().map(|&v| to_fixed(v)).collect();

        let (mut trans_a, mut trans_b) = memory_transport_pair();

        let share0 = SharedVec(x_fixed);
        let share1 = SharedVec(vec![0u32; 2]);

        let weights_clone = weights.clone();
        let handle = std::thread::spawn(move || {
            shared_linear_forward(1, &weights_clone, 2, 2, &share1, &[], &mut trans_b).unwrap()
        });

        let result0 = shared_linear_forward(0, &weights, 2, 2, &share0, &[], &mut trans_a).unwrap();
        let result1 = handle.join().unwrap();

        for i in 0..2 {
            let val = from_fixed(result0.0[i].wrapping_add(result1.0[i]));
            assert!((val - expected[i]).abs() < 0.01, "linear[{}]: got {}, expected {}", i, val, expected[i]);
        }
    }

    #[test]
    fn test_shared_linear_forward_sparse() {
        let weights = vec![
            1.0f32, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
            1.0, 1.0, 1.0,
        ];
        let input = vec![2.0f32, 3.0, 4.0];
        let x_fixed: Vec<u32> = input.iter().map(|&v| to_fixed(v)).collect();

        let (mut trans_a, _) = memory_transport_pair();

        let indices = vec![1, 3];
        let result = shared_linear_forward_sparse(
            0, &weights, 3, 4, &indices, &SharedVec(x_fixed), &[], &mut trans_a,
        ).unwrap();

        assert_eq!(result.len(), 2);
        assert!((from_fixed(result.0[0]) - 3.0).abs() < 0.01); // row 1
        assert!((from_fixed(result.0[1]) - 9.0).abs() < 0.01); // row 3
    }

    // --- Q32.32 linear forward tests ---

    #[test]
    fn test_shared_linear_forward_64_identity() {
        let weights = vec![1.0f32, 0.0, 0.0, 1.0];
        let input = vec![3.0f32, 4.0];
        let x_fixed: Vec<u64> = input.iter().map(|&v| to_fixed64(v)).collect();

        let (mut trans_a, _) = memory_transport_pair();
        let result = shared_linear_forward_64(0, &weights, 2, 2, &SharedVec64(x_fixed), &[], &mut trans_a).unwrap();

        for i in 0..2 {
            let val = from_fixed64(result.0[i]);
            assert!((val - input[i]).abs() < 0.01, "Q32 identity[{}]: got {}, expected {}", i, val, input[i]);
        }
    }

    #[test]
    fn test_shared_linear_forward_64_two_parties() {
        let weights = vec![1.0f32, 2.0, 3.0, 4.0];
        let input = vec![2.0f32, 3.0];
        let expected = [8.0f32, 18.0];

        let x_fixed: Vec<u64> = input.iter().map(|&v| to_fixed64(v)).collect();

        let (mut trans_a, mut trans_b) = memory_transport_pair();

        let share0 = SharedVec64(x_fixed);
        let share1 = SharedVec64(vec![0u64; 2]);

        let weights_clone = weights.clone();
        let handle = std::thread::spawn(move || {
            shared_linear_forward_64(1, &weights_clone, 2, 2, &share1, &[], &mut trans_b).unwrap()
        });

        let result0 = shared_linear_forward_64(0, &weights, 2, 2, &share0, &[], &mut trans_a).unwrap();
        let result1 = handle.join().unwrap();

        for i in 0..2 {
            let val = from_fixed64(result0.0[i].wrapping_add(result1.0[i]));
            assert!((val - expected[i]).abs() < 0.01, "Q32 linear[{}]: got {}, expected {}", i, val, expected[i]);
        }
    }

    #[test]
    fn test_shared_linear_forward_sparse_64() {
        let weights = vec![
            1.0f32, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
            1.0, 1.0, 1.0,
        ];
        let input = vec![2.0f32, 3.0, 4.0];
        let x_fixed: Vec<u64> = input.iter().map(|&v| to_fixed64(v)).collect();

        let (mut trans_a, _) = memory_transport_pair();

        let indices = vec![1, 3];
        let result = shared_linear_forward_sparse_64(
            0, &weights, 3, 4, &indices, &SharedVec64(x_fixed), &[], &mut trans_a,
        ).unwrap();

        assert_eq!(result.len(), 2);
        assert!((from_fixed64(result.0[0]) - 3.0).abs() < 0.01);
        assert!((from_fixed64(result.0[1]) - 9.0).abs() < 0.01);
    }
}
