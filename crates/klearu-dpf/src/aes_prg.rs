use aes::cipher::{BlockEncrypt, KeyInit};
use aes::Aes128;

/// AES-128 based pseudo-random generator for DPF/DCF tree expansion.
///
/// Uses fixed-key AES in the Matyas-Meyer-Oseas construction to expand
/// a 16-byte seed into two 16-byte children plus 2 control bits.
pub struct AesPrg {
    cipher_left: Aes128,
    cipher_right: Aes128,
}

impl AesPrg {
    pub fn new(key: &[u8; 16]) -> Self {
        let mut key_left = *key;
        for b in &mut key_left {
            *b ^= 0xAA;
        }
        let mut key_right = *key;
        for b in &mut key_right {
            *b ^= 0x55;
        }
        Self {
            cipher_left: Aes128::new((&key_left).into()),
            cipher_right: Aes128::new((&key_right).into()),
        }
    }

    /// Expand a seed into (left_seed, left_control_bit, right_seed, right_control_bit).
    ///
    /// The control bits are extracted from the LSB of the raw AES output before XOR.
    /// The seeds are MMO: `AES_k(seed) XOR seed`, with the LSB cleared.
    pub fn expand(&self, seed: &[u8; 16]) -> ([u8; 16], bool, [u8; 16], bool) {
        // Left child
        let mut left_block = (*seed).into();
        self.cipher_left.encrypt_block(&mut left_block);
        let left_enc: [u8; 16] = left_block.into();
        let left_control = (left_enc[0] & 1) != 0;
        let mut left_seed = [0u8; 16];
        for i in 0..16 {
            left_seed[i] = left_enc[i] ^ seed[i];
        }
        // Clear the control bit position so it doesn't interfere with seed
        left_seed[0] &= 0xFE;

        // Right child
        let mut right_block = (*seed).into();
        self.cipher_right.encrypt_block(&mut right_block);
        let right_enc: [u8; 16] = right_block.into();
        let right_control = (right_enc[0] & 1) != 0;
        let mut right_seed = [0u8; 16];
        for i in 0..16 {
            right_seed[i] = right_enc[i] ^ seed[i];
        }
        right_seed[0] &= 0xFE;

        (left_seed, left_control, right_seed, right_control)
    }

    /// Derive a u32 output from a seed (for leaf evaluation).
    pub fn derive_output(&self, seed: &[u8; 16]) -> u32 {
        let mut block = (*seed).into();
        self.cipher_left.encrypt_block(&mut block);
        let bytes: [u8; 16] = block.into();
        u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prg_deterministic() {
        let prg = AesPrg::new(&[0u8; 16]);
        let seed = [1u8; 16];
        let (l1, lc1, r1, rc1) = prg.expand(&seed);
        let (l2, lc2, r2, rc2) = prg.expand(&seed);
        assert_eq!(l1, l2);
        assert_eq!(lc1, lc2);
        assert_eq!(r1, r2);
        assert_eq!(rc1, rc2);
    }

    #[test]
    fn test_prg_different_seeds_different_outputs() {
        let prg = AesPrg::new(&[0u8; 16]);
        let (l1, _, r1, _) = prg.expand(&[0u8; 16]);
        let (l2, _, r2, _) = prg.expand(&[1u8; 16]);
        assert_ne!(l1, l2);
        assert_ne!(r1, r2);
    }

    #[test]
    fn test_prg_left_right_different() {
        let prg = AesPrg::new(&[0u8; 16]);
        let (left, _, right, _) = prg.expand(&[42u8; 16]);
        assert_ne!(left, right);
    }

    #[test]
    fn test_derive_output_deterministic() {
        let prg = AesPrg::new(&[0u8; 16]);
        let seed = [7u8; 16];
        assert_eq!(prg.derive_output(&seed), prg.derive_output(&seed));
    }
}
