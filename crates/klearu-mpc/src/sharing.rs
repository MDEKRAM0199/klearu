use rand::Rng;

/// A single additive share in Z_{2^32}.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Share(pub u32);

/// A vector of additive shares in Z_{2^32}.
#[derive(Clone, Debug)]
pub struct SharedVec(pub Vec<u32>);

impl SharedVec {
    /// Split plaintext values into two additive shares.
    ///
    /// For each value v, generates random r and returns (r, v - r) mod 2^32.
    pub fn from_plaintext(values: &[u32], rng: &mut impl Rng) -> (SharedVec, SharedVec) {
        let n = values.len();
        let mut share_a = Vec::with_capacity(n);
        let mut share_b = Vec::with_capacity(n);

        for &v in values {
            let r: u32 = rng.gen();
            share_a.push(r);
            share_b.push(v.wrapping_sub(r));
        }

        (SharedVec(share_a), SharedVec(share_b))
    }

    /// Reconstruct plaintext from two additive shares.
    pub fn reconstruct(a: &SharedVec, b: &SharedVec) -> Vec<u32> {
        assert_eq!(a.len(), b.len());
        a.0.iter().zip(b.0.iter()).map(|(&x, &y)| x.wrapping_add(y)).collect()
    }

    /// Element-wise addition of two share vectors (local operation).
    pub fn add(&self, other: &SharedVec) -> SharedVec {
        assert_eq!(self.len(), other.len());
        SharedVec(
            self.0.iter().zip(other.0.iter())
                .map(|(&a, &b)| a.wrapping_add(b))
                .collect()
        )
    }

    /// Element-wise subtraction (local operation).
    pub fn sub(&self, other: &SharedVec) -> SharedVec {
        assert_eq!(self.len(), other.len());
        SharedVec(
            self.0.iter().zip(other.0.iter())
                .map(|(&a, &b)| a.wrapping_sub(b))
                .collect()
        )
    }

    /// Add a public constant to the share (only party 0 should add it).
    pub fn add_const(&self, c: u32) -> SharedVec {
        SharedVec(self.0.iter().map(|&x| x.wrapping_add(c)).collect())
    }

    /// Multiply each share element by a public constant (local operation).
    pub fn mul_const(&self, c: u32) -> SharedVec {
        SharedVec(self.0.iter().map(|&x| x.wrapping_mul(c)).collect())
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the vector is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// In-place element-wise addition (avoids allocation).
    pub fn add_assign(&mut self, other: &SharedVec) {
        assert_eq!(self.len(), other.len());
        for (a, &b) in self.0.iter_mut().zip(other.0.iter()) {
            *a = a.wrapping_add(b);
        }
    }

    /// Create a zero-filled shared vector.
    pub fn zeros(n: usize) -> SharedVec {
        SharedVec(vec![0u32; n])
    }
}

// --- Q32.32 sharing (u64) ---

/// A vector of additive shares in Z_{2^64} for Q32.32 fixed-point.
#[derive(Clone, Debug)]
pub struct SharedVec64(pub Vec<u64>);

impl SharedVec64 {
    /// Split plaintext values into two additive shares.
    pub fn from_plaintext(values: &[u64], rng: &mut impl Rng) -> (SharedVec64, SharedVec64) {
        let n = values.len();
        let mut share_a = Vec::with_capacity(n);
        let mut share_b = Vec::with_capacity(n);

        for &v in values {
            let r: u64 = rng.gen();
            share_a.push(r);
            share_b.push(v.wrapping_sub(r));
        }

        (SharedVec64(share_a), SharedVec64(share_b))
    }

    /// Reconstruct plaintext from two additive shares.
    pub fn reconstruct(a: &SharedVec64, b: &SharedVec64) -> Vec<u64> {
        assert_eq!(a.len(), b.len());
        a.0.iter().zip(b.0.iter()).map(|(&x, &y)| x.wrapping_add(y)).collect()
    }

    /// Element-wise addition of two share vectors (local operation).
    pub fn add(&self, other: &SharedVec64) -> SharedVec64 {
        assert_eq!(self.len(), other.len());
        SharedVec64(
            self.0.iter().zip(other.0.iter())
                .map(|(&a, &b)| a.wrapping_add(b))
                .collect()
        )
    }

    /// Element-wise subtraction (local operation).
    pub fn sub(&self, other: &SharedVec64) -> SharedVec64 {
        assert_eq!(self.len(), other.len());
        SharedVec64(
            self.0.iter().zip(other.0.iter())
                .map(|(&a, &b)| a.wrapping_sub(b))
                .collect()
        )
    }

    /// Add a public constant to the share (only party 0 should add it).
    pub fn add_const(&self, c: u64) -> SharedVec64 {
        SharedVec64(self.0.iter().map(|&x| x.wrapping_add(c)).collect())
    }

    /// Multiply each share element by a public constant (local operation).
    pub fn mul_const(&self, c: u64) -> SharedVec64 {
        SharedVec64(self.0.iter().map(|&x| x.wrapping_mul(c)).collect())
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the vector is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// In-place element-wise addition (avoids allocation).
    pub fn add_assign(&mut self, other: &SharedVec64) {
        assert_eq!(self.len(), other.len());
        for (a, &b) in self.0.iter_mut().zip(other.0.iter()) {
            *a = a.wrapping_add(b);
        }
    }

    /// Create a zero-filled shared vector.
    pub fn zeros(n: usize) -> SharedVec64 {
        SharedVec64(vec![0u64; n])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn test_rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    #[test]
    fn test_share_reconstruct_identity() {
        let mut rng = test_rng();
        let values = vec![0u32, 1, 42, 0xDEADBEEF, 0xFFFFFFFF];
        let (a, b) = SharedVec::from_plaintext(&values, &mut rng);
        let reconstructed = SharedVec::reconstruct(&a, &b);
        assert_eq!(reconstructed, values);
    }

    #[test]
    fn test_share_reconstruct_zeros() {
        let mut rng = test_rng();
        let values = vec![0u32; 10];
        let (a, b) = SharedVec::from_plaintext(&values, &mut rng);
        let reconstructed = SharedVec::reconstruct(&a, &b);
        assert_eq!(reconstructed, values);
    }

    #[test]
    fn test_additive_homomorphism() {
        let mut rng = test_rng();
        let x = vec![10u32, 20, 30];
        let y = vec![1u32, 2, 3];

        let (xa, xb) = SharedVec::from_plaintext(&x, &mut rng);
        let (ya, yb) = SharedVec::from_plaintext(&y, &mut rng);

        let sum_a = xa.add(&ya);
        let sum_b = xb.add(&yb);
        let result = SharedVec::reconstruct(&sum_a, &sum_b);

        let expected: Vec<u32> = x.iter().zip(y.iter()).map(|(a, b)| a.wrapping_add(*b)).collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_sub() {
        let mut rng = test_rng();
        let x = vec![100u32, 200, 300];
        let y = vec![10u32, 20, 30];

        let (xa, xb) = SharedVec::from_plaintext(&x, &mut rng);
        let (ya, yb) = SharedVec::from_plaintext(&y, &mut rng);

        let diff_a = xa.sub(&ya);
        let diff_b = xb.sub(&yb);
        let result = SharedVec::reconstruct(&diff_a, &diff_b);

        let expected: Vec<u32> = x.iter().zip(y.iter()).map(|(a, b)| a.wrapping_sub(*b)).collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_mul_const() {
        let mut rng = test_rng();
        let x = vec![10u32, 20, 30];
        let c = 5u32;

        let (xa, xb) = SharedVec::from_plaintext(&x, &mut rng);
        let prod_a = xa.mul_const(c);
        let prod_b = xb.mul_const(c);
        let result = SharedVec::reconstruct(&prod_a, &prod_b);

        let expected: Vec<u32> = x.iter().map(|&v| v.wrapping_mul(c)).collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_add_const_party0() {
        let mut rng = test_rng();
        let x = vec![10u32, 20, 30];
        let c = 100u32;

        let (xa, xb) = SharedVec::from_plaintext(&x, &mut rng);
        // Only party 0 adds the constant
        let xa_plus_c = xa.add_const(c);
        let result = SharedVec::reconstruct(&xa_plus_c, &xb);

        let expected: Vec<u32> = x.iter().map(|&v| v.wrapping_add(c)).collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_zeros() {
        let z = SharedVec::zeros(5);
        assert_eq!(z.len(), 5);
        assert!(z.0.iter().all(|&v| v == 0));
    }

    // --- SharedVec64 tests ---

    #[test]
    fn test_share64_reconstruct_identity() {
        let mut rng = test_rng();
        let values = vec![0u64, 1, 42, 0xDEADBEEF_CAFEBABE, 0xFFFFFFFF_FFFFFFFF];
        let (a, b) = SharedVec64::from_plaintext(&values, &mut rng);
        let reconstructed = SharedVec64::reconstruct(&a, &b);
        assert_eq!(reconstructed, values);
    }

    #[test]
    fn test_share64_additive_homomorphism() {
        let mut rng = test_rng();
        let x = vec![10u64, 20, 30];
        let y = vec![1u64, 2, 3];

        let (xa, xb) = SharedVec64::from_plaintext(&x, &mut rng);
        let (ya, yb) = SharedVec64::from_plaintext(&y, &mut rng);

        let sum_a = xa.add(&ya);
        let sum_b = xb.add(&yb);
        let result = SharedVec64::reconstruct(&sum_a, &sum_b);

        let expected: Vec<u64> = x.iter().zip(y.iter()).map(|(a, b)| a.wrapping_add(*b)).collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_share64_zeros() {
        let z = SharedVec64::zeros(5);
        assert_eq!(z.len(), 5);
        assert!(z.0.iter().all(|&v| v == 0));
    }
}
