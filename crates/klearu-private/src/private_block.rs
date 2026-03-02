//! Private (2PC) evaluation of a full transformer block.
//!
//! Composes private attention and MLP with shared RMSNorm
//! and residual connections to evaluate one transformer layer under
//! secret sharing.
//!
//! Flow: attn_norm → attention → residual → mlp_norm → MLP → residual

use klearu_llm::model::block::TransformerBlock;
use klearu_llm::model::kv_cache::KvCache;
use klearu_llm::model::rope::RotaryEmbedding;
use klearu_mpc::beaver::{TripleGenerator, TripleGenerator128};
use klearu_mpc::normalization::{rmsnorm_shared, rmsnorm_shared_64};
use klearu_mpc::transport::Transport;
use klearu_mpc::{SharedVec, SharedVec64};
use std::io;

use crate::private_attention::{private_attention_forward, private_attention_forward_secure};
use crate::private_mlp::{private_dense_mlp_forward, private_dense_mlp_forward_secure, private_sparse_mlp_forward};

/// Evaluate one private transformer block.
///
/// `block`: the plaintext block weights (both parties have identical copies).
/// `hidden_share`: this party's share of the hidden state (mutated in-place).
/// `position`: current sequence position.
/// `rope`: rotary embedding tables.
/// `kv_cache`: the KV cache for this layer.
/// `triples`: generator for Beaver triples.
/// `transport`: communication channel to the other party.
pub fn private_block_forward(
    party: u8,
    block: &TransformerBlock,
    hidden_share: &mut SharedVec,
    position: usize,
    rope: &RotaryEmbedding,
    kv_cache: &mut KvCache,
    triples: &mut impl TripleGenerator,
    transport: &mut impl Transport,
) -> io::Result<()> {
    // 1. Pre-attention RMSNorm
    let mut normed = hidden_share.clone();
    rmsnorm_shared(
        party,
        &mut normed,
        &block.attn_norm.weight,
        block.attn_norm.eps(),
        transport,
    )?;

    // 2. Private attention forward
    let attn_out = private_attention_forward(
        party,
        &block.attention,
        &normed,
        position,
        rope,
        kv_cache,
        transport,
    )?;

    // 3. Residual: hidden += attn_out (local, wrapping add of shares)
    *hidden_share = hidden_share.add(&attn_out);

    // 4. Pre-MLP RMSNorm
    let mut normed_mlp = hidden_share.clone();
    rmsnorm_shared(
        party,
        &mut normed_mlp,
        &block.mlp_norm.weight,
        block.mlp_norm.eps(),
        transport,
    )?;

    // 5. Private MLP forward (dense path — sparse variant would use neuron_indices)
    let mlp_out = private_dense_mlp_forward(
        party,
        &block.mlp,
        &normed_mlp,
        triples,
        transport,
    )?;

    // 6. Residual: hidden += mlp_out
    *hidden_share = hidden_share.add(&mlp_out);

    Ok(())
}

/// Sparse variant: uses a subset of MLP neurons.
pub fn private_block_forward_sparse(
    party: u8,
    block: &TransformerBlock,
    hidden_share: &mut SharedVec,
    position: usize,
    rope: &RotaryEmbedding,
    kv_cache: &mut KvCache,
    neuron_indices: &[usize],
    triples: &mut impl TripleGenerator,
    transport: &mut impl Transport,
) -> io::Result<()> {
    // 1. Pre-attention RMSNorm
    let mut normed = hidden_share.clone();
    rmsnorm_shared(
        party,
        &mut normed,
        &block.attn_norm.weight,
        block.attn_norm.eps(),
        transport,
    )?;

    // 2. Private attention forward (same as dense)
    let attn_out = private_attention_forward(
        party,
        &block.attention,
        &normed,
        position,
        rope,
        kv_cache,
        transport,
    )?;

    // 3. Residual
    *hidden_share = hidden_share.add(&attn_out);

    // 4. Pre-MLP RMSNorm
    let mut normed_mlp = hidden_share.clone();
    rmsnorm_shared(
        party,
        &mut normed_mlp,
        &block.mlp_norm.weight,
        block.mlp_norm.eps(),
        transport,
    )?;

    // 5. Private sparse MLP forward
    let mlp_out = private_sparse_mlp_forward(
        party,
        &block.mlp,
        &normed_mlp,
        neuron_indices,
        triples,
        transport,
    )?;

    // 6. Residual
    *hidden_share = hidden_share.add(&mlp_out);

    Ok(())
}

/// Secure (Q32.32) evaluation of one transformer block.
///
/// Uses privacy-preserving RMSNorm (Beaver squaring, only reveals sum(x²))
/// and Q32.32 attention/MLP.
pub fn private_block_forward_secure(
    party: u8,
    block: &TransformerBlock,
    hidden_share: &mut SharedVec64,
    position: usize,
    rope: &RotaryEmbedding,
    kv_cache: &mut KvCache,
    triples: &mut impl TripleGenerator128,
    transport: &mut impl Transport,
) -> io::Result<()> {
    // 1. Pre-attention RMSNorm (privacy-preserving)
    let mut normed = hidden_share.clone();
    rmsnorm_shared_64(
        party,
        &mut normed,
        &block.attn_norm.weight,
        block.attn_norm.eps(),
        triples,
        transport,
    )?;

    // 2. Secure attention forward
    let attn_out = private_attention_forward_secure(
        party,
        &block.attention,
        &normed,
        position,
        rope,
        kv_cache,
        transport,
    )?;

    // 3. Residual
    *hidden_share = hidden_share.add(&attn_out);

    // 4. Pre-MLP RMSNorm (privacy-preserving)
    let mut normed_mlp = hidden_share.clone();
    rmsnorm_shared_64(
        party,
        &mut normed_mlp,
        &block.mlp_norm.weight,
        block.mlp_norm.eps(),
        triples,
        transport,
    )?;

    // 5. Secure MLP forward
    let mlp_out = private_dense_mlp_forward_secure(
        party,
        &block.mlp,
        &normed_mlp,
        transport,
    )?;

    // 6. Residual
    *hidden_share = hidden_share.add(&mlp_out);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use klearu_llm::config::LlmConfig;
    use klearu_llm::model::block::TransformerBlock;
    use klearu_llm::model::kv_cache::KvCache;
    use klearu_llm::model::rope::RotaryEmbedding;
    use klearu_mpc::beaver::dummy_triple_pair;
    use klearu_mpc::fixed_point::{from_fixed, to_fixed};
    use klearu_mpc::transport::memory_transport_pair;

    fn tiny_config() -> LlmConfig {
        LlmConfig {
            vocab_size: 16,
            hidden_size: 8,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 4,
            intermediate_size: 16,
            num_layers: 1,
            max_seq_len: 8,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            tie_word_embeddings: true,
        }
    }

    #[test]
    fn test_private_block_runs_without_error() {
        let config = tiny_config();
        let mut block = TransformerBlock::new(&config);

        // Set norm weights to 1.0 so normalization is meaningful
        for w in block.attn_norm.weight.iter_mut() {
            *w = 1.0;
        }
        for w in block.mlp_norm.weight.iter_mut() {
            *w = 1.0;
        }

        let rope = RotaryEmbedding::new(config.head_dim, config.max_seq_len, config.rope_theta);

        let input = vec![0.1f32, 0.2, -0.1, 0.05, 0.3, -0.2, 0.15, 0.0];
        let x_fixed: Vec<u32> = input.iter().map(|&v| to_fixed(v)).collect();

        let (mut gen0, mut gen1) = dummy_triple_pair(10000);
        let (mut trans_a, mut trans_b) = memory_transport_pair();

        let mut share0 = SharedVec(x_fixed);
        let mut share1 = SharedVec(vec![0u32; config.hidden_size]);

        let block_clone = TransformerBlock::new(&config);
        // Set same norm weights
        let mut block1 = block_clone;
        for w in block1.attn_norm.weight.iter_mut() {
            *w = 1.0;
        }
        for w in block1.mlp_norm.weight.iter_mut() {
            *w = 1.0;
        }

        let rope_clone = RotaryEmbedding::new(config.head_dim, config.max_seq_len, config.rope_theta);
        let mut kv0 = KvCache::new(config.num_kv_heads, config.max_seq_len, config.head_dim);
        let mut kv1 = KvCache::new(config.num_kv_heads, config.max_seq_len, config.head_dim);

        let handle = std::thread::spawn(move || {
            private_block_forward(
                1, &block1, &mut share1, 0, &rope_clone, &mut kv1, &mut gen1, &mut trans_b,
            ).unwrap();
            share1
        });

        private_block_forward(
            0, &block, &mut share0, 0, &rope, &mut kv0, &mut gen0, &mut trans_a,
        ).unwrap();

        let share1 = handle.join().unwrap();

        // Reconstruct and verify finite
        for i in 0..config.hidden_size {
            let result = from_fixed(share0.0[i].wrapping_add(share1.0[i]));
            assert!(result.is_finite(), "block output[{}] is not finite: {}", i, result);
        }
    }
}
