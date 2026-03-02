/// RMSNorm (LLaMA-style): `x_i = x_i / rms * weight_i`
/// where `rms = sqrt(mean(x^2) + eps)`.
///
/// No bias, no mean subtraction (unlike LayerNorm).
pub struct RmsNorm {
    pub weight: Vec<f32>,
    eps: f32,
}

impl RmsNorm {
    pub fn new(dim: usize, eps: f32) -> Self {
        Self {
            weight: vec![0.0; dim],
            eps,
        }
    }

    /// The epsilon used for numerical stability.
    pub fn eps(&self) -> f32 {
        self.eps
    }

    /// Apply RMSNorm in-place.
    pub fn forward(&self, x: &mut [f32]) {
        let n = x.len();
        debug_assert_eq!(n, self.weight.len());

        // Compute RMS
        let mut sum_sq = 0.0f32;
        for &v in x.iter() {
            sum_sq += v * v;
        }
        let rms = (sum_sq / n as f32 + self.eps).sqrt();
        let inv_rms = 1.0 / rms;

        // Normalize and scale
        for (xi, &wi) in x.iter_mut().zip(self.weight.iter()) {
            *xi = *xi * inv_rms * wi;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_norm_known_values() {
        let mut norm = RmsNorm::new(4, 1e-5);
        norm.weight = vec![1.0, 1.0, 1.0, 1.0];

        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        // rms = sqrt((1+4+9+16)/4 + 1e-5) = sqrt(7.5 + 1e-5) ≈ 2.7386
        let expected_rms = (7.5f32 + 1e-5).sqrt();
        let expected: Vec<f32> = [1.0, 2.0, 3.0, 4.0]
            .iter()
            .map(|&v| v / expected_rms)
            .collect();

        norm.forward(&mut x);
        for (a, b) in x.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} != {b}");
        }
    }

    #[test]
    fn test_rms_norm_with_weights() {
        let mut norm = RmsNorm::new(3, 1e-5);
        norm.weight = vec![2.0, 0.5, 1.0];

        let mut x = vec![3.0, 3.0, 3.0];
        // rms = sqrt(27/3 + 1e-5) = sqrt(9 + 1e-5) ≈ 3.0
        norm.forward(&mut x);

        let rms = (9.0f32 + 1e-5).sqrt();
        let normed = 3.0 / rms;
        assert!((x[0] - normed * 2.0).abs() < 1e-5);
        assert!((x[1] - normed * 0.5).abs() < 1e-5);
        assert!((x[2] - normed * 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_rms_norm_unit_rms() {
        let mut norm = RmsNorm::new(4, 1e-5);
        norm.weight = vec![1.0; 4];

        let mut x = vec![1.0, 0.0, 0.0, 0.0];
        norm.forward(&mut x);

        // After unit-weight RMS norm, the sum of squares should be ~dim
        let sum_sq: f32 = x.iter().map(|v| v * v).sum();
        assert!((sum_sq - 4.0).abs() < 0.01);
    }
}
