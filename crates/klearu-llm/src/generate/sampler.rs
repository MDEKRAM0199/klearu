use rand::Rng;

/// Configuration for token sampling.
#[derive(Debug, Clone)]
pub struct SamplerConfig {
    /// Temperature for scaling logits. 0.0 = greedy.
    pub temperature: f32,
    /// Top-k filtering. 0 = disabled.
    pub top_k: usize,
    /// Top-p (nucleus) filtering. 1.0 = disabled.
    pub top_p: f32,
    /// Repetition penalty factor. 1.0 = disabled.
    pub repetition_penalty: f32,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
        }
    }
}

/// Sample a token ID from logits according to the sampling configuration.
pub fn sample(
    logits: &mut [f32],
    config: &SamplerConfig,
    prev_tokens: &[u32],
    rng: &mut impl Rng,
) -> u32 {
    // Apply repetition penalty
    if config.repetition_penalty != 1.0 {
        for &tok in prev_tokens {
            let idx = tok as usize;
            if idx < logits.len() {
                if logits[idx] > 0.0 {
                    logits[idx] /= config.repetition_penalty;
                } else {
                    logits[idx] *= config.repetition_penalty;
                }
            }
        }
    }

    // Greedy (temperature = 0)
    if config.temperature == 0.0 {
        return argmax(logits);
    }

    // Temperature scaling
    let inv_temp = 1.0 / config.temperature;
    for v in logits.iter_mut() {
        *v *= inv_temp;
    }

    // Top-k filtering
    if config.top_k > 0 && config.top_k < logits.len() {
        let threshold = top_k_threshold(logits, config.top_k);
        for v in logits.iter_mut() {
            if *v < threshold {
                *v = f32::NEG_INFINITY;
            }
        }
    }

    // Softmax
    softmax(logits);

    // Top-p filtering
    if config.top_p < 1.0 {
        apply_top_p(logits, config.top_p);
    }

    // Weighted random sampling
    sample_from_probs(logits, rng)
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

fn top_k_threshold(logits: &[f32], k: usize) -> f32 {
    let mut sorted: Vec<f32> = logits.to_vec();
    sorted.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap());
    sorted[k.min(sorted.len()) - 1]
}

fn softmax(logits: &mut [f32]) {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in logits.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in logits.iter_mut() {
        *v *= inv;
    }
}

fn apply_top_p(probs: &mut [f32], top_p: f32) {
    let mut indices: Vec<usize> = (0..probs.len()).collect();
    indices.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());

    let mut cumsum = 0.0f32;
    let mut cutoff_idx = indices.len();
    for (rank, &idx) in indices.iter().enumerate() {
        cumsum += probs[idx];
        if cumsum > top_p {
            cutoff_idx = rank + 1;
            break;
        }
    }

    for &idx in &indices[cutoff_idx..] {
        probs[idx] = 0.0;
    }

    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for v in probs.iter_mut() {
            *v *= inv;
        }
    }
}

fn sample_from_probs(probs: &[f32], rng: &mut impl Rng) -> u32 {
    let r: f32 = rng.gen();
    let mut acc = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if acc >= r {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn test_rng() -> rand::rngs::StdRng {
        rand::rngs::StdRng::seed_from_u64(42)
    }

    #[test]
    fn test_greedy_is_argmax() {
        let mut logits = vec![1.0, 5.0, 3.0, 2.0];
        let config = SamplerConfig::default(); // temperature = 0 => greedy
        let token = sample(&mut logits, &config, &[], &mut test_rng());
        assert_eq!(token, 1);
    }

    #[test]
    fn test_top_k_filters() {
        let mut rng = test_rng();
        for _ in 0..20 {
            let mut logits = vec![1.0, 10.0, 5.0, 3.0, 2.0];
            let config = SamplerConfig {
                temperature: 1.0,
                top_k: 2,
                ..Default::default()
            };
            let token = sample(&mut logits, &config, &[], &mut rng);
            assert!(token == 1 || token == 2, "Got token {token}, expected 1 or 2");
        }
    }

    #[test]
    fn test_repetition_penalty() {
        let mut logits = vec![5.0, 5.0, 5.0];
        let config = SamplerConfig {
            repetition_penalty: 2.0,
            ..Default::default()
        };
        let token = sample(&mut logits, &config, &[0], &mut test_rng());
        // Token 0 has logit 5.0/2.0 = 2.5, others stay at 5.0
        assert_ne!(token, 0);
    }

    #[test]
    fn test_temperature_sharpens() {
        let logits = vec![1.0, 2.0, 1.0];
        let mut l1 = logits.clone();
        softmax(&mut l1);

        let mut l2 = logits.clone();
        for v in l2.iter_mut() {
            *v *= 10.0;
        }
        softmax(&mut l2);

        assert!(l2[1] > l1[1]);
    }

    #[test]
    fn test_sampling_distribution() {
        // With temperature=1.0, token 1 (logit=10) should be picked most often
        let mut rng = test_rng();
        let mut counts = [0u32; 5];
        for _ in 0..100 {
            let mut logits = vec![1.0, 10.0, 1.0, 1.0, 1.0];
            let config = SamplerConfig {
                temperature: 1.0,
                ..Default::default()
            };
            let token = sample(&mut logits, &config, &[], &mut rng);
            counts[token as usize] += 1;
        }
        // Token 1 should dominate
        assert!(
            counts[1] > counts[0] + counts[2] + counts[3] + counts[4],
            "Token 1 should be most frequent: {counts:?}"
        );
    }
}
