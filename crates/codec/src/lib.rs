//! Pure Rust codec decoder for the qwen3-tts TTS model.
//!
//! This crate implements the full codec decoder pipeline in pure Rust,
//! reading Q8_0 quantized weights from the codec GGUF file:
//!
//! ```text
//! codes [T, 16] → quantizer → pre_conv → transformer(8L) → upsample(2×) → DAC(4 blocks)
//! ```
//!
//! The goal is to eventually replace the C++ `pipeline_codec_decode` FFI call.

pub mod gguf;
pub mod q8_0;
