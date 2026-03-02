//! Private (2PC) attention forward pass.
//!
//! Since weights are public, Q/K/V projections are local.
//! Q is revealed to compute attention scores via Q·K.
//! Partial attention scores (Q_plain · K_share) are exchanged and
//! reconstructed so both parties apply identical softmax.
//! Weighted V sum uses public softmax weights on V shares (local).

use klearu_llm::model::attention::Attention;
use klearu_llm::model::kv_cache::KvCache;
use klearu_llm::model::rope::RotaryEmbedding;
use klearu_mpc::fixed_point::{from_fixed, from_fixed64, to_fixed64};
use klearu_mpc::linear::{
    shared_linear_forward, shared_linear_forward_64, shared_linear_forward_f32_input,
    shared_linear_forward_f32_input_64,
};
use klearu_mpc::transport::Transport;
use klearu_mpc::{SharedVec, SharedVec64};
use std::io;

/// Private attention forward pass.
///
/// Leakage: attention score magnitudes (acceptable in semi-honest model).
///
/// `attention`: plaintext model weights (public to both parties).
/// `x_share`: this party's share of the input.
/// `kv_cache`: shared KV cache (both parties maintain identical structure
///   since K/V projections are deterministic from shared inputs).
///
/// Returns this party's share of the attention output.
pub fn private_attention_forward(
    party: u8,
    attention: &Attention,
    x_share: &SharedVec,
    position: usize,
    rope: &RotaryEmbedding,
    kv_cache: &mut KvCache,
    transport: &mut impl Transport,
) -> io::Result<SharedVec> {
    let hidden_size = attention.q_proj.in_features();
    let num_heads = attention.num_heads();
    let num_kv_heads = attention.num_kv_heads();
    let head_dim = attention.head_dim();
    let gqa_group_size = num_heads / num_kv_heads;

    // 1. Q/K/V projections (local: weights are public)
    let q_out = num_heads * head_dim;
    let kv_out = num_kv_heads * head_dim;

    let q_share = shared_linear_forward(
        party, attention.q_proj.weights.as_raw_slice(), hidden_size, q_out, x_share, &[], transport,
    )?;
    let k_share = shared_linear_forward(
        party, attention.k_proj.weights.as_raw_slice(), hidden_size, kv_out, x_share, &[], transport,
    )?;
    let v_share = shared_linear_forward(
        party, attention.v_proj.weights.as_raw_slice(), hidden_size, kv_out, x_share, &[], transport,
    )?;

    // 2. Apply RoPE to Q and K (linear operation, works on shares)
    let mut q_f32: Vec<f32> = q_share.0.iter().map(|&v| from_fixed(v)).collect();
    let mut k_f32: Vec<f32> = k_share.0.iter().map(|&v| from_fixed(v)).collect();

    for h in 0..num_heads {
        let offset = h * head_dim;
        rope.apply(&mut q_f32[offset..offset + head_dim], position);
    }
    for h in 0..num_kv_heads {
        let offset = h * head_dim;
        rope.apply(&mut k_f32[offset..offset + head_dim], position);
    }

    // 3. Append K/V to cache (both parties do this identically)
    // K: store f32 directly after RoPE (avoids lossy to_fixed→from_fixed round-trip)
    // V: convert from Q16.16 shares to f32 for cache storage
    let v_cache_f32: Vec<f32> = v_share.0.iter().map(|&v| from_fixed(v)).collect();
    kv_cache.append(&k_f32, &v_cache_f32);

    let seq_len = kv_cache.current_len();

    // 4. Reveal Q so both parties can compute Q·K partial scores
    //    Send as f32 bits to avoid lossy Q16.16 round-trip
    let q_bits: Vec<u32> = q_f32.iter().map(|&v| v.to_bits()).collect();
    transport.send_u32_slice(&q_bits)?;
    let q_other_bits = transport.recv_u32_slice(q_bits.len())?;
    let q_plain: Vec<f32> = q_f32.iter().zip(q_other_bits.iter())
        .map(|(&my, &other_bits)| my + f32::from_bits(other_bits))
        .collect();

    // 5. Compute partial attention scores: Q_plain · K_share_i for each head×position
    //    Each party's K_share is in the KV cache. The true score is partial_0 + partial_1.
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Collect all partial scores in a flat buffer: [head0_pos0, head0_pos1, ..., head1_pos0, ...]
    let total_scores = num_heads * seq_len;
    let mut partial_scores = Vec::with_capacity(total_scores);

    for h in 0..num_heads {
        let kv_h = h / gqa_group_size;
        let q_offset = h * head_dim;
        for t in 0..seq_len {
            let k_slice = kv_cache.k_at(kv_h, t);
            let mut score = 0.0f32;
            for d in 0..head_dim {
                score += q_plain[q_offset + d] * k_slice[d];
            }
            partial_scores.push(score * scale);
        }
    }

    // 6. Exchange partial scores to reconstruct true scores
    //    Send as f32 bits via u32 transport.
    let partial_bits: Vec<u32> = partial_scores.iter().map(|&s| s.to_bits()).collect();
    transport.send_u32_slice(&partial_bits)?;
    let other_bits = transport.recv_u32_slice(total_scores)?;

    let true_scores: Vec<f32> = partial_scores.iter().zip(other_bits.iter())
        .map(|(&my, &other_bits)| my + f32::from_bits(other_bits))
        .collect();

    // 7. Apply softmax per head on revealed (true) scores, then weighted V sum
    let mut output_f32 = vec![0.0f32; hidden_size];

    for h in 0..num_heads {
        let kv_h = h / gqa_group_size;
        let score_offset = h * seq_len;

        // Softmax (public, on true revealed scores)
        let scores = &true_scores[score_offset..score_offset + seq_len];
        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
        let sum: f32 = exp_scores.iter().sum();
        for s in &mut exp_scores {
            *s /= sum;
        }

        // Weighted sum of V: output_h = sum_t softmax[t] * V_h_t
        // V is shared, softmax weights are public → local operation on shares
        for t in 0..seq_len {
            let v_slice = kv_cache.v_at(kv_h, t);
            let w = exp_scores[t];
            for d in 0..head_dim {
                output_f32[h * head_dim + d] += w * v_slice[d];
            }
        }
    }

    // V values in the KV cache are this party's SHARES, not plaintext.
    // So output_f32 contains this party's share of the weighted sum.
    // Use f32-input linear forward to avoid lossy to_fixed quantization.
    let o_in = num_heads * head_dim;
    Ok(shared_linear_forward_f32_input(
        attention.o_proj.weights.as_raw_slice(), o_in, hidden_size, &output_f32,
    ))
}

/// Secure (Q32.32) attention forward pass.
///
/// Same protocol as the Q16.16 version but uses Q32.32 shares throughout.
/// Q is revealed (same leakage), K/V cached as f32 shares.
pub fn private_attention_forward_secure(
    party: u8,
    attention: &Attention,
    x_share: &SharedVec64,
    position: usize,
    rope: &RotaryEmbedding,
    kv_cache: &mut KvCache,
    transport: &mut impl Transport,
) -> io::Result<SharedVec64> {
    let hidden_size = attention.q_proj.in_features();
    let num_heads = attention.num_heads();
    let num_kv_heads = attention.num_kv_heads();
    let head_dim = attention.head_dim();
    let gqa_group_size = num_heads / num_kv_heads;

    // 1. Q/K/V projections (local: weights are public, Q32.32)
    let q_out = num_heads * head_dim;
    let kv_out = num_kv_heads * head_dim;

    let q_share = shared_linear_forward_64(
        party, attention.q_proj.weights.as_raw_slice(), hidden_size, q_out, x_share, &[], transport,
    )?;
    let k_share = shared_linear_forward_64(
        party, attention.k_proj.weights.as_raw_slice(), hidden_size, kv_out, x_share, &[], transport,
    )?;
    let v_share = shared_linear_forward_64(
        party, attention.v_proj.weights.as_raw_slice(), hidden_size, kv_out, x_share, &[], transport,
    )?;

    // 2. Convert to f32 for RoPE (same as Q16.16 version)
    let mut q_f32: Vec<f32> = q_share.0.iter().map(|&v| from_fixed64(v)).collect();
    let mut k_f32: Vec<f32> = k_share.0.iter().map(|&v| from_fixed64(v)).collect();

    for h in 0..num_heads {
        let offset = h * head_dim;
        rope.apply(&mut q_f32[offset..offset + head_dim], position);
    }
    for h in 0..num_kv_heads {
        let offset = h * head_dim;
        rope.apply(&mut k_f32[offset..offset + head_dim], position);
    }

    // 3. Append K/V to cache as f32 shares
    let v_cache_f32: Vec<f32> = v_share.0.iter().map(|&v| from_fixed64(v)).collect();
    kv_cache.append(&k_f32, &v_cache_f32);

    let seq_len = kv_cache.current_len();

    // 4. Reveal Q (send as f32 bits)
    let q_bits: Vec<u32> = q_f32.iter().map(|&v| v.to_bits()).collect();
    transport.send_u32_slice(&q_bits)?;
    let q_other_bits = transport.recv_u32_slice(q_bits.len())?;
    let q_plain: Vec<f32> = q_f32.iter().zip(q_other_bits.iter())
        .map(|(&my, &other_bits)| my + f32::from_bits(other_bits))
        .collect();

    // 5. Compute partial attention scores
    let scale = 1.0 / (head_dim as f32).sqrt();
    let total_scores = num_heads * seq_len;
    let mut partial_scores = Vec::with_capacity(total_scores);

    for h in 0..num_heads {
        let kv_h = h / gqa_group_size;
        let q_offset = h * head_dim;
        for t in 0..seq_len {
            let k_slice = kv_cache.k_at(kv_h, t);
            let mut score = 0.0f32;
            for d in 0..head_dim {
                score += q_plain[q_offset + d] * k_slice[d];
            }
            partial_scores.push(score * scale);
        }
    }

    // 6. Exchange partial scores
    let partial_bits: Vec<u32> = partial_scores.iter().map(|&s| s.to_bits()).collect();
    transport.send_u32_slice(&partial_bits)?;
    let other_bits = transport.recv_u32_slice(total_scores)?;

    let true_scores: Vec<f32> = partial_scores.iter().zip(other_bits.iter())
        .map(|(&my, &other_bits)| my + f32::from_bits(other_bits))
        .collect();

    // 7. Softmax + weighted V sum
    let mut output_f32 = vec![0.0f32; hidden_size];

    for h in 0..num_heads {
        let kv_h = h / gqa_group_size;
        let score_offset = h * seq_len;

        let scores = &true_scores[score_offset..score_offset + seq_len];
        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
        let sum: f32 = exp_scores.iter().sum();
        for s in &mut exp_scores {
            *s /= sum;
        }

        for t in 0..seq_len {
            let v_slice = kv_cache.v_at(kv_h, t);
            let w = exp_scores[t];
            for d in 0..head_dim {
                output_f32[h * head_dim + d] += w * v_slice[d];
            }
        }
    }

    // 8. O projection: f32 input → Q32.32 output
    let o_in = num_heads * head_dim;
    Ok(shared_linear_forward_f32_input_64(
        attention.o_proj.weights.as_raw_slice(), o_in, hidden_size, &output_f32,
    ))
}
