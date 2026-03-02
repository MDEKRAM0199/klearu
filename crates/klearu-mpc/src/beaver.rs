use rand::Rng;

/// A Beaver multiplication triple over Z_{2^64}: shares of (a, b, c) where c = a * b mod 2^64.
///
/// Using u64 allows correct fixed-point multiplication without overflow:
/// fixed-point values are u32 (16.16 format), but their product needs 64 bits
/// to avoid losing the result to modular reduction before truncation.
///
/// In a 2PC protocol, each party holds their share of the triple.
#[derive(Clone, Debug)]
pub struct BeaverTriple {
    pub a: u64,
    pub b: u64,
    pub c: u64,
}

/// Trait for generating Beaver triples.
pub trait TripleGenerator {
    fn generate(&mut self, n: usize) -> Vec<BeaverTriple>;
}

/// Insecure dummy triple generator for testing.
///
/// Both parties share the same RNG state. NOT SECURE.
pub struct DummyTripleGen {
    party: u8,
    seed: u64,
    counter: u64,
}

impl DummyTripleGen {
    pub fn new(party: u8, seed: u64) -> Self {
        Self { party, seed, counter: 0 }
    }

    fn gen_one(&mut self) -> BeaverTriple {
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let mut rng = StdRng::seed_from_u64(self.seed.wrapping_add(self.counter));
        self.counter += 1;

        let a: u64 = rng.gen();
        let b: u64 = rng.gen();
        let c = a.wrapping_mul(b);

        let a0: u64 = rng.gen();
        let b0: u64 = rng.gen();
        let c0: u64 = rng.gen();

        let a1 = a.wrapping_sub(a0);
        let b1 = b.wrapping_sub(b0);
        let c1 = c.wrapping_sub(c0);

        if self.party == 0 {
            BeaverTriple { a: a0, b: b0, c: c0 }
        } else {
            BeaverTriple { a: a1, b: b1, c: c1 }
        }
    }
}

impl TripleGenerator for DummyTripleGen {
    fn generate(&mut self, n: usize) -> Vec<BeaverTriple> {
        (0..n).map(|_| self.gen_one()).collect()
    }
}

/// Create a pair of correlated DummyTripleGen instances.
pub fn dummy_triple_pair(seed: u64) -> (DummyTripleGen, DummyTripleGen) {
    (DummyTripleGen::new(0, seed), DummyTripleGen::new(1, seed))
}

// --- Q32.32 wide triples (u128) ---

/// A Beaver multiplication triple over Z_{2^128}: c = a * b mod 2^128.
///
/// Needed for Q32.32 Beaver multiply where u64 inputs produce i128 products.
/// The protocol works in Z_{2^128} and truncates >> 32 to get Q32.32 output.
#[derive(Clone, Debug)]
pub struct BeaverTriple128 {
    pub a: u128,
    pub b: u128,
    pub c: u128,
}

/// Trait for generating wide (128-bit) Beaver triples.
pub trait TripleGenerator128 {
    fn generate(&mut self, n: usize) -> Vec<BeaverTriple128>;
}

/// Insecure dummy generator for 128-bit triples. NOT SECURE.
pub struct DummyTripleGen128 {
    party: u8,
    seed: u64,
    counter: u64,
}

impl DummyTripleGen128 {
    pub fn new(party: u8, seed: u64) -> Self {
        Self { party, seed, counter: 0 }
    }

    fn gen_one(&mut self) -> BeaverTriple128 {
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let mut rng = StdRng::seed_from_u64(self.seed.wrapping_add(self.counter));
        self.counter += 1;

        // Generate random u128 values from two u64s
        let a = (rng.gen::<u64>() as u128) | ((rng.gen::<u64>() as u128) << 64);
        let b = (rng.gen::<u64>() as u128) | ((rng.gen::<u64>() as u128) << 64);
        let c = a.wrapping_mul(b);

        let a0 = (rng.gen::<u64>() as u128) | ((rng.gen::<u64>() as u128) << 64);
        let b0 = (rng.gen::<u64>() as u128) | ((rng.gen::<u64>() as u128) << 64);
        let c0 = (rng.gen::<u64>() as u128) | ((rng.gen::<u64>() as u128) << 64);

        let a1 = a.wrapping_sub(a0);
        let b1 = b.wrapping_sub(b0);
        let c1 = c.wrapping_sub(c0);

        if self.party == 0 {
            BeaverTriple128 { a: a0, b: b0, c: c0 }
        } else {
            BeaverTriple128 { a: a1, b: b1, c: c1 }
        }
    }
}

impl TripleGenerator128 for DummyTripleGen128 {
    fn generate(&mut self, n: usize) -> Vec<BeaverTriple128> {
        (0..n).map(|_| self.gen_one()).collect()
    }
}

/// Create a pair of correlated DummyTripleGen128 instances.
pub fn dummy_triple_pair_128(seed: u64) -> (DummyTripleGen128, DummyTripleGen128) {
    (DummyTripleGen128::new(0, seed), DummyTripleGen128::new(1, seed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dummy_triple_correctness() {
        let (mut gen0, mut gen1) = dummy_triple_pair(42);
        let triples0 = gen0.generate(10);
        let triples1 = gen1.generate(10);

        for (t0, t1) in triples0.iter().zip(triples1.iter()) {
            let a = t0.a.wrapping_add(t1.a);
            let b = t0.b.wrapping_add(t1.b);
            let c = t0.c.wrapping_add(t1.c);
            assert_eq!(c, a.wrapping_mul(b), "triple invariant violated");
        }
    }

    #[test]
    fn test_dummy_triple_deterministic() {
        let mut gen_a = DummyTripleGen::new(0, 123);
        let mut gen_b = DummyTripleGen::new(0, 123);

        let a = gen_a.generate(5);
        let b = gen_b.generate(5);

        for (ta, tb) in a.iter().zip(b.iter()) {
            assert_eq!(ta.a, tb.a);
            assert_eq!(ta.b, tb.b);
            assert_eq!(ta.c, tb.c);
        }
    }

    #[test]
    fn test_dummy_triple_128_correctness() {
        let (mut gen0, mut gen1) = dummy_triple_pair_128(42);
        let triples0 = gen0.generate(10);
        let triples1 = gen1.generate(10);

        for (t0, t1) in triples0.iter().zip(triples1.iter()) {
            let a = t0.a.wrapping_add(t1.a);
            let b = t0.b.wrapping_add(t1.b);
            let c = t0.c.wrapping_add(t1.c);
            assert_eq!(c, a.wrapping_mul(b), "wide triple invariant violated");
        }
    }
}
