//! SIMD vectorization, BF16 quantization, and memory optimizations for SLIDE.
//!
//! This crate provides hardware-accelerated primitives used by the SLIDE
//! algorithm to achieve near-GPU throughput on commodity CPUs:
//!
//! - **simd**: Platform-adaptive SIMD routines (AVX2 / NEON / scalar fallback)
//!   for sparse and dense dot products and scatter-add operations.
//! - **bf16**: BF16 weight quantization with FP32 accumulation, implementing
//!   Mode 1 (full BF16) and Mode 2 (BF16 storage / FP32 gradients) from the
//!   SLIDE paper.
//! - **memory**: Cache-line-aligned contiguous weight storage for vectorized
//!   access patterns.

pub mod bf16;
pub mod memory;
pub mod simd;
